import torch
import torch.nn as nn
import torch.nn.functional as F

from spec import constants
from spec.models.model import Model


class SimpleMLP(Model):
    """Simple MLP model for text classification."""

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples)

        #
        # Embeddings
        #
        embeddings_weight = None
        if self.fields_dict['words'].vocab.vectors is not None:
            embeddings_weight = self.fields_dict['words'].vocab.vectors
            options.word_embeddings_size = embeddings_weight.size(1)

        self.word_emb = nn.Embedding(
            num_embeddings=len(self.fields_dict['words'].vocab),
            embedding_dim=options.word_embeddings_size,
            padding_idx=constants.PAD_ID,
            _weight=embeddings_weight,
        )
        self.dropout_emb = nn.Dropout(options.emb_dropout)
        if options.freeze_embeddings:
            self.word_emb.weight.requires_grad = False

        #
        # Linear
        #
        self.linear_out = nn.Linear(options.word_embeddings_size,
                                    self.nb_classes)

        # stored variables
        self.embeddings_out = None
        self.hidden = None
        self.attn_weights = None
        self.logits = None

        self.init_weights()
        self.is_built = True

    def init_weights(self):
        pass
        # init_xavier(self.rnn, dist='uniform')
        # init_xavier(self.attn, dist='uniform')
        # init_xavier(self.linear_out, dist='uniform')

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        h = batch.words
        mask = torch.ne(h, constants.PAD_ID)
        lengths = mask.int().sum(dim=-1)

        # (bs, ts) -> (bs, ts, emb_dim)
        self.embeddings_out = self.word_emb(h)
        h = self.dropout_emb(self.embeddings_out)

        # (bs, ts, emb_dim) -> (bs, 1, emb_dim)
        h = (h * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        h = h.unsqueeze(1)

        # (bs, 1, hidden_size) -> (bs, 1, nb_classes)
        self.logits = self.linear_out(h)

        # (bs, 1, nb_classes) -> (bs, 1, nb_classes) in log simplex
        h = F.log_softmax(self.logits, dim=-1)

        return h
