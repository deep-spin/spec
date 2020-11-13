import torch

from spec import constants
from spec.explainers.explainer import Explainer
from spec.explainers.utils import filter_word_ids_with_non_zero_probability, \
    make_bow_matrix


class EmbeddedAttentionSupervisedExplainer(Explainer):

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
        # pass the input once through the classifier to calculate attn_weights
        # _ = classifier(batch)

        # find the topk attn weights using 1 < k < seq_len
        # select only the first explanations
        # (bs, 3, nb_marks) -> (bs, nb_marks)
        if batch.marks.shape[1] > 1:
            top_idxs = batch.marks[:, 0]
        else:
            top_idxs = batch.marks[:, 0]

        # recover the word ids from the top indexes
        bs, ts = batch.words.shape
        arange_vec = torch.arange(bs).unsqueeze(1).to(batch.words.device)
        top_word_ids = batch.words[arange_vec, top_idxs]
        top_word_ids[top_idxs == -1] = constants.PAD_ID

        # testing explanations directly:
        # top_word_ids = batch.words_expl

        # what to do when top ids map to pad ids? or when
        # it returns instances zeroed out by sparsity?
        # for now, hard coded in pure python: filter out these entries
        dummy_probas = top_word_ids + 1
        valid_top_word_ids = filter_word_ids_with_non_zero_probability(
            top_word_ids, dummy_probas, pad_id=constants.PAD_ID
        )

        # save for getting the words later
        self.valid_top_word_ids = valid_top_word_ids

        # create the message
        message = make_bow_matrix(valid_top_word_ids,
                                  self.words_vocab_size,
                                  device=batch.marks.device)

        # create a time dimension of size 1
        message = message.unsqueeze(1)

        return message
