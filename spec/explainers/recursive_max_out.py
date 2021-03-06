import torch

from spec import constants
from spec.explainers.explainer import Explainer
from spec.explainers.utils import filter_word_ids_with_non_zero_probability


class RecursiveMaxOutExplainer(Explainer):

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
        """This is not a trainable module, so it does not have a loss."""
        self._loss = None

    def forward(self, batch, classifier):
        # recover attn_weights
        y_classes = classifier.predict_classes(batch)
        clf_attn_weights = classifier.attn_weights.squeeze()

        # find the topk attn weights using 1 < k < seq_len
        k = min(self.explainer_attn_top_k, clf_attn_weights.shape[-1])
        top_idxs, k_exs = classifier.recursive_maxout(batch, y_classes, max_k=k)

        # recover the word ids from the top indexes
        arange = torch.arange(batch.words.shape[0])
        arange = arange.to(batch.words.device).unsqueeze(-1)
        top_word_ids = batch.words[arange, top_idxs]
        top_word_ids[top_idxs == -1] = constants.PAD_ID
        top_probas_dummy = top_word_ids + 1

        # what to do when top ids map to pad ids? or when
        # it returns instances zeroed out by sparsity?
        # for now, hard coded in pure python: filter out these entries
        valid_top_word_ids = filter_word_ids_with_non_zero_probability(
            top_word_ids, top_probas_dummy, pad_id=constants.PAD_ID
        )
        # save for getting the words later
        self.valid_top_word_ids = valid_top_word_ids

        # create the message
        message = self.make_message(
            valid_top_word_ids, top_probas_dummy, classifier.word_emb
        )

        # create a time dimension of size 1
        message = message.unsqueeze(1)

        return message


