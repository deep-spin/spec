import torch

from spec import constants
from spec.explainers.explainer import Explainer
from spec.explainers.utils import filter_word_ids_with_non_zero_probability


class LeaveOneOutExplainer(Explainer):

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

        with torch.no_grad():
            # recover attn_weights
            y_log_proba = classifier.erase_forward(batch, erase_ids=None)
            full_proba = torch.exp(y_log_proba.detach())
            bs, ts = batch.words.shape
            batch_pos = torch.arange(ts).expand(bs, -1).to(batch.words.device)

            # iteratively remove ith feature
            tvds = []
            for i in range(ts):
                batch_i = batch_pos[:, i]
                y_log_proba = classifier.erase_forward(batch, erase_ids=batch_i)
                loo_proba = torch.exp(y_log_proba.detach())
                tvd = 0.5 * torch.sum(torch.abs(full_proba - loo_proba), dim=-1)
                tvds.append(tvd.squeeze(-1))

            # recover the word ids from the top indexes
            tvds = torch.stack(tvds).t()
            k = min(self.explainer_attn_top_k, tvds.shape[-1])
            _, top_idxs = torch.topk(tvds, k, dim=-1)
            top_word_ids = batch.words.gather(1, top_idxs)
            top_probas = top_word_ids + 1

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
