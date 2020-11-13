import torch

from spec import constants
from spec.explainers.explainer import Explainer
from spec.explainers.utils import filter_word_ids_with_non_zero_probability


class EraserMaxOutExplainer(Explainer):

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
        _ = classifier.erase_forward(batch, erase_ids=None)
        original_attn_weights = classifier.attn_weights.squeeze().clone()
        mask = torch.ne(batch.words, constants.PAD_ID)
        min_seq_len = torch.min(mask.int().sum(-1)).item()

        # iteratively erase the top features k times
        k = min(self.explainer_attn_top_k, min_seq_len)
        top_idxs = []
        for i in range(k):
            t_ids = classifier.attn_weights.squeeze().argmax(dim=-1)
            top_idxs.append(t_ids)
            if i < k - 1:
                erase_ids = torch.stack(top_idxs).t()
                _ = classifier.erase_forward(batch, erase_ids=erase_ids)

        # recover the word ids from the top indexes
        top_idxs = torch.stack(top_idxs).t()
        top_word_ids = batch.words.gather(1, top_idxs)
        top_probas = original_attn_weights.gather(1, top_idxs)

        # what to do when top ids map to pad ids? or when
        # it returns instances zeroed out by sparsity?
        # for now, hard coded in pure python: filter out these entries
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


