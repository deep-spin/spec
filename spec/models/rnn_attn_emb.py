import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from spec import constants
from spec.initialization import init_xavier, init_kaiming
from spec.models.model import Model
from spec.modules.attention import Attention
from spec.modules.scorer import SelfAdditiveScorer, DotProductScorer, \
    GeneralScorer, OperationScorer, MLPScorer


class RNNAttentionEmb(Model):
    """Simple RNN with attention model over embs for text classification."""

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

        features_size = options.word_embeddings_size

        #
        # RNN
        #
        self.is_bidir = options.bidirectional
        self.sum_bidir = options.sum_bidir
        self.rnn_type = options.rnn_type

        rnn_map = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}
        rnn_class = rnn_map[self.rnn_type]
        hidden_size = options.hidden_size[0]
        self.rnn = rnn_class(features_size,
                             hidden_size,
                             bidirectional=self.is_bidir,
                             batch_first=True)
        self.dropout_rnn = nn.Dropout(options.rnn_dropout)

        n = 1 if not self.is_bidir or self.sum_bidir else 2
        features_size = n * hidden_size

        if options.attn_scorer == 'dot_product':
            self.scorer = DotProductScorer(scaled=True)
        elif options.attn_scorer == 'self_add':
            self.scorer = SelfAdditiveScorer(features_size,
                                             features_size // 2,
                                             scaled=False)
        elif options.attn_scorer == 'general':
            self.scorer = GeneralScorer(features_size, features_size)
        elif options.attn_scorer == 'add' or options.attn_scorer == 'concat':
            self.scorer = OperationScorer(features_size,
                                          options.word_embeddings_size,
                                          options.attn_hidden_size,
                                          op=options.attn_scorer)
        elif options.attn_scorer == 'mlp':
            self.scorer = MLPScorer(features_size, features_size,
                                    layer_sizes=[options.attn_hidden_size])
        self.attn = Attention(self.scorer,
                              dropout=options.attn_dropout,
                              max_activation=options.attn_max_activation)

        #
        # Linear
        #
        self.linear_out = nn.Linear(options.word_embeddings_size, self.nb_classes)

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

    def init_hidden(self, batch_size, hidden_size, device=None):
        # The axes semantics are (nb_layers, minibatch_size, hidden_dim)
        nb_layers = 2 if self.is_bidir else 1
        if self.rnn_type == 'lstm':
            return (torch.zeros(nb_layers, batch_size, hidden_size).to(device),
                    torch.zeros(nb_layers, batch_size, hidden_size).to(device))
        else:
            return torch.zeros(nb_layers, batch_size, hidden_size).to(device)

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None

        h = batch.words
        mask = h != constants.PAD_ID
        lengths = mask.int().sum(dim=-1)

        # initialize RNN hidden state
        self.hidden = self.init_hidden(
            h.shape[0], self.rnn.hidden_size, device=h.device
        )

        # (bs, ts) -> (bs, ts, emb_dim)
        self.embeddings_out = self.word_emb(h)
        h = self.dropout_emb(self.embeddings_out)

        # (bs, ts, emb_dim) -> (bs, ts, hidden_size)
        h = pack(h, lengths, batch_first=True, enforce_sorted=False)
        h, self.hidden = self.rnn(h, self.hidden)
        h, _ = unpack(h, batch_first=True)

        # (bs, ts, hidden_size)  -> (bs, 1, hidden_size)
        hid = torch.cat((self.hidden[0][0], self.hidden[0][1]), -1).unsqueeze(1)
        h, self.attn_weights = self.attn(hid, self.embeddings_out,
                                         values=self.embeddings_out, mask=mask)

        # (bs, 1, hidden_size) -> (bs, 1, nb_classes)
        self.logits = self.linear_out(h)

        # (bs, 1, nb_classes) -> (bs, 1, nb_classes) in log simplex
        h = F.log_softmax(self.logits, dim=-1)

        return h
