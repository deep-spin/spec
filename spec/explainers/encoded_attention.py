import torch

from spec import constants
from spec.explainers.explainer import Explainer
from spec.explainers.utils import filter_word_ids_with_non_zero_probability


class EncodedAttentionExplainer(Explainer):

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
        _ = classifier(batch)

        # recover attn_weights
        clf_attn_weights = classifier.attn_weights.squeeze()

        # find the topk attn weights using 1 < k < seq_len
        k = min(self.explainer_attn_top_k, clf_attn_weights.shape[-1])
        top_probas, top_idxs = torch.topk(clf_attn_weights, k, dim=-1)

        # recover the word ids from the top indexes
        top_word_ids = batch.words.gather(1, top_idxs)

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

    def fast_forward(self, batch, classifier):
        # without topk (for full bow)
        _ = classifier(batch)
        clf_attn_weights = classifier.attn_weights.squeeze()
        bs, ts = batch.words.shape
        device = batch.words.device
        bids = torch.arange(bs).unsqueeze(-1).expand(-1, ts).flatten()
        bids = bids.to(device)
        idxs = torch.stack((bids, batch.words.flatten()), dim=0)
        mask = torch.ne(batch.words, constants.PAD_ID).to(device).float()
        vals = (mask * torch.ceil(clf_attn_weights)).flatten()
        size = torch.Size([bs, self.words_vocab_size])
        message = torch.sparse.FloatTensor(idxs, vals, size).to_dense()
        message = message.to(device)
        return message.unsqueeze(1)
