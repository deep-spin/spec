import torch

from spec import constants
from spec.explainers.explainer import Explainer


class EmbeddedAttentionExplainer(Explainer):

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
        bs, ts = clf_attn_weights.shape
        device = clf_attn_weights.device

        # recover the word ids from the top indexes
        top_word_ids = batch.words.clone()
        mask_nonzero_probas = torch.gt(clf_attn_weights, 0.)
        top_word_ids[~mask_nonzero_probas] = constants.PAD_ID

        # save for getting the words later (filter pads later - slow!)
        self.valid_top_word_ids = top_word_ids

        # create a bag of words as the message
        mask_nonpad = torch.ne(batch.words, constants.PAD_ID)
        mask = mask_nonpad & mask_nonzero_probas

        bids = torch.arange(bs).unsqueeze(-1).expand(-1, ts).flatten()
        bids = bids.to(device)

        idxs = torch.stack((bids, batch.words.flatten()), dim=0)
        vals = mask.to(device).float().flatten()

        size = torch.Size([bs, self.words_vocab_size])
        message = torch.sparse.FloatTensor(idxs, vals, size).to_dense()
        message = message.to(device)

        # create a time dimension of size 1
        message = message.unsqueeze(1)

        return message


