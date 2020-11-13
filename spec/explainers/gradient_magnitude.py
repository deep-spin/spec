import torch

from spec import constants
from spec.explainers.explainer import Explainer
from spec.explainers.utils import filter_word_ids_with_non_zero_probability


class GradientMagnitudeExplainer(Explainer):

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples)
        self.words_vocab_size = len(self.fields_dict['words'].vocab)
        self.explainer_attn_top_k = options.explainer_attn_top_k
        self.message_type = options.message_type
        # options.word_embeddings_size is updated in the classifier constructor
        # when a path to pretrained embeddings is passed
        self.emb_size = options.word_embeddings_size
        self.valid_top_word_ids = None

    def build_loss(self, loss_weights=None):
        self._loss = None

    def get_output_size(self):
        if self.message_type == 'bow':
            return self.words_vocab_size
        else:
            return self.emb_size

    def forward(self, batch, classifier):
        with torch.enable_grad():
            # zero out gradients
            classifier.zero_grad()

            # set classifier in train mode
            default_mode = classifier.training
            classifier.train(True)

            # set word embeddings to require grad
            default_emb_req_grad = classifier.word_emb.weight.requires_grad
            classifier.word_emb.weight.requires_grad_(True)

            # run the input through the network
            clf_log_pred = classifier(batch)

            # get the output of the embedding layer
            if hasattr(batch, 'words_hyp'):
                embeddings_out = classifier.embeddings_out_pre
            else:
                embeddings_out = classifier.embeddings_out

            # recover the gradients of pred with respect to embeddings_out
            gradients = self.get_gradients_with_respect_to_input(
                embeddings_out, clf_log_pred, multiply_by_inp=True
            )

            # calculate the gradients magnitude using their l1 norm
            # this could be configured to other stuff (inf-norm could be nice)
            # gradients_magnitude = gradients.norm(p=1, dim=-1)
            gradients_magnitude = gradients.sum(-1).abs()

            # gradients of the output with respect to the attention weights
            # self.get_gradients_with_respect_to_input(
            #     classifier.attn_weights, pred, multiply_by_itself=False
            # )
            # gradients_magnitude = gradients.squeeze().abs()

            # set word embeddings requires_grad to its default value
            classifier.word_emb.weight.requires_grad_(default_emb_req_grad)

            # put the classifier training mode back to its default value
            classifier.train(default_mode)

            # find the topk gradients magnitude using 1 < k < seq_len
            k = min(self.explainer_attn_top_k, gradients_magnitude.shape[-1])
            top_probas, top_idxs = torch.topk(gradients_magnitude, k, dim=-1)
            top_word_ids = batch.words.gather(1, top_idxs)

            # make the bow message
            valid_top_word_ids = filter_word_ids_with_non_zero_probability(
                top_word_ids, top_probas, pad_id=constants.PAD_ID
            )
            # save for getting the words later
            self.valid_top_word_ids = valid_top_word_ids

            # create the message
            message = self.make_message(
                valid_top_word_ids, top_probas, classifier.word_emb
            )

            # create a time dimension of size 1
            message = message.unsqueeze(1)

        return message

    def get_gradients_with_respect_to_input(
        self, inp, out, multiply_by_inp=True, split_by_classes=False, gold=None
    ):
        """
        Get the gradients of output (`out`) with respect to the input (`inp`).

        Args:
            inp (torch.Tensor): a tensor retrieved from the forward pass.
                Usually, this is the output of the embeddings layer or the
                attention weights. Clearly, the shape can vary.
            out (torch.Tensor): output of the forward pass. Usually, it is a
                log softmax output. Shape of (batch_size, out_len, nb_classes)
                out_len can be the sequence length for sequence classification
                or 1 for text classification.
            multiply_by_inp (bool): If true, the gradients of the input will by
                multiplied by the input itself. Default is True.
            split_by_classes (bool): If true, it will return the gradients of
                each output label with respect to the input. Default is False.
            gold (torch.Tensor): if not None, it will use NLLLoss between the
                output and the gold as the score function. This is unrealistic
                at test time, but it is useful for debugging purposes.
                Default is None.

        Returns:
            torch.Tensor with shape:
                inp.shape if split_by_classes = False (default)
                (*inp.shape, nb_classes) otherwise
        """
        output = out.view(-1, self.nb_classes)
        if split_by_classes:
            # gradients of each label with respect to the tensor
            gradients = torch.zeros(*inp.shape, self.nb_classes)
            for i in range(self.nb_labels):
                output = torch.exp(output[:, i]).mean()
                grad = torch.autograd.grad(output, inp, retain_graph=True)[0]
                grad = grad.data
                if multiply_by_inp:
                    grad = grad * inp
                gradients[:, :, i] = grad

        else:
            if gold is not None:
                # use the nll loss with the true gold as the score function
                # this is useful for debugging purposes
                scores = torch.nn.functional.nll_loss(
                    output, gold.view(-1), ignore_index=constants.TARGET_PAD_ID
                )

            else:
                # default: use the same score as Serrano and Smith (2019)
                max_scores, _ = torch.max(output, dim=-1)
                scores = torch.exp(max_scores) / torch.exp(output).sum(dim=-1)

            # get the gradients with of the scores with respect to the tensor
            gradients = torch.autograd.grad(scores.mean(), inp)[0].data

            # multiply the input by its gradients element-wise
            if multiply_by_inp:
                gradients = gradients * inp

        return gradients

