import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from spec import constants
from spec.initialization import init_xavier, init_kaiming
from spec.models.model import Model
from spec.modules.attention import Attention
from spec.modules.multi_headed_attention import MultiHeadedAttention
from spec.modules.scorer import SelfAdditiveScorer, DotProductScorer, \
    GeneralScorer, OperationScorer, MLPScorer


class RNNAttention(Model):
    """Simple RNN with attention model for text classification."""

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

        if options.attn_type == 'multihead':
            vector_size = options.attn_multihead_hidden_size // options.attn_nb_heads  # noqa
        else:
            vector_size = features_size

        if options.attn_scorer == 'dot_product':
            self.scorer = DotProductScorer(
                scaled=True
            )
        elif options.attn_scorer == 'self_add':
            self.scorer = SelfAdditiveScorer(
                vector_size,
                vector_size // 2,
                scaled=False
            )
        elif options.attn_scorer == 'general':
            self.scorer = GeneralScorer(
                vector_size,
                vector_size
            )
        elif options.attn_scorer == 'add' or options.attn_scorer == 'concat':
            self.scorer = OperationScorer(
                vector_size,
                vector_size,
                options.attn_hidden_size,
                op=options.attn_scorer
            )
        elif options.attn_scorer == 'mlp':
            self.scorer = MLPScorer(
                vector_size,
                vector_size,
                layer_sizes=[options.attn_hidden_size]
            )

        self.attn = Attention(
            self.scorer,
            dropout=options.attn_dropout,
            max_activation=options.attn_max_activation
        )

        if options.attn_type == 'multihead':
            self.attn = MultiHeadedAttention(
                self.attn,
                options.attn_nb_heads,
                features_size,
                features_size,
                features_size,
                options.attn_multihead_hidden_size
            )
            features_size = options.attn_multihead_hidden_size

        #
        # Linear
        #
        self.linear_out = nn.Linear(features_size, self.nb_classes)

        # stored variables
        self.embeddings_out = None
        self.lstm_out = None
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

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = (h[:, :, :self.rnn.hidden_size]
                 + h[:, :, self.rnn.hidden_size:])

        self.lstm_out = h

        # apply dropout
        h = self.dropout_rnn(h)

        # (bs, ts, hidden_size)  -> (bs, 1, hidden_size)
        h, self.attn_weights = self.attn(h, h, values=h, mask=mask)

        # (bs, 1, hidden_size) -> (bs, 1, nb_classes)
        self.logits = self.linear_out(h)

        # (bs, 1, nb_classes) -> (bs, 1, nb_classes) in log simplex
        h = F.log_softmax(self.logits, dim=-1)

        return h

    def recursive_maxout(self, batch, last_y_classes, max_k=5):
        """
        Recursive erase max-out attention probability and look for decision
        flips. Well, this is actually implemented in an iterative way, but
        the point is still the same. This approach is the one used by
        Serrano and Smith (2019).

        Args:
            batch (Tensor): input
            last_y_classes (Tensor):  output of self.predict_classes()
            max_k (int): maximum number of iterations

        Returns:

        """
        h = batch.words
        mask = h != constants.PAD_ID
        lengths = mask.int().sum(dim=-1)

        # initialize RNN hidden state
        hidden = self.init_hidden(
            h.shape[0], self.rnn.hidden_size, device=h.device
        )

        # (bs, ts) -> (bs, ts, emb_dim)
        embeddings_out = self.word_emb(h)
        h = self.dropout_emb(embeddings_out)

        # (bs, ts, emb_dim) -> (bs, ts, hidden_size)
        h = pack(h, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(h, hidden)
        h, _ = unpack(h, batch_first=True)

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = (h[:, :, :self.rnn.hidden_size]
                 + h[:, :, self.rnn.hidden_size:])

        # apply dropout
        lstm_out = self.dropout_rnn(h)

        # apply attention
        _, p_attn = self.attn(lstm_out, lstm_out, values=lstm_out, mask=mask)

        bs, ts, hdim = p_attn.shape
        finished_exs = torch.zeros(bs).to(p_attn.device).int()
        arange = torch.arange(bs).to(p_attn.device)
        top_idxs = []
        k = 0
        while not torch.all(finished_exs > 0).item() and k < max_k:
            p_attn = p_attn.view(bs*ts, hdim)
            t_ids = torch.argmax(p_attn, dim=-1)
            p_attn[arange, t_ids] = 0
            p_attn = p_attn / torch.sum(p_attn, dim=-1).unsqueeze(-1)
            p_attn = p_attn.view(bs, ts, hdim)

            p_attn = self.attn.dropout(p_attn)
            o_attn = torch.einsum('b...ts,b...sm->b...tm', [p_attn, lstm_out])
            logits = self.linear_out(o_attn)
            y_pred = F.log_softmax(logits, dim=-1)
            y_classes = torch.argmax(torch.exp(y_pred), dim=-1)

            last_not_equal = (y_classes.squeeze() != last_y_classes.squeeze())
            last_not_equal = last_not_equal.int()
            mask_unfinished = finished_exs == 0
            finished_exs[mask_unfinished] = last_not_equal[mask_unfinished] * (k + 1)
            k += 1

            last_y_classes = y_classes
            mask = last_not_equal * mask_unfinished.int()
            mask = mask.long()
            valid_ids = mask * t_ids + (1 - mask) * (-1)
            top_idxs.append(valid_ids.long())

        return torch.stack(top_idxs).t(), finished_exs

    def erase_forward(self, batch, erase_ids=None):
        h = batch.words
        mask = torch.ne(h, constants.PAD_ID)
        lengths = mask.int().sum(dim=-1)

        # initialize RNN hidden state
        self.hidden = self.init_hidden(
            h.shape[0], self.rnn.hidden_size, device=h.device
        )

        # (bs, ts) -> (bs, ts, emb_dim)
        self.embeddings_out = self.word_emb(h)

        # erase
        if erase_ids is not None:
            bs, ts = batch.words.shape
            device = batch.words.device
            a_vec = torch.arange(bs).unsqueeze(-1).to(device)
            # zero out embedding vector
            self.embeddings_out[a_vec, erase_ids] = 0
            # zero out attention weight
            mask[a_vec, erase_ids] = 0

        h = self.dropout_emb(self.embeddings_out)

        # (bs, ts, emb_dim) -> (bs, ts, hidden_size)
        h = pack(h, lengths, batch_first=True, enforce_sorted=False)
        h, self.hidden = self.rnn(h, self.hidden)
        h, _ = unpack(h, batch_first=True)

        # if you'd like to sum instead of concatenate:
        if self.sum_bidir:
            h = (h[:, :, :self.rnn.hidden_size]
                 + h[:, :, self.rnn.hidden_size:])

        # apply dropout
        h = self.dropout_rnn(h)

        # (bs, ts, hidden_size)  -> (bs, 1, hidden_size)
        h, self.attn_weights = self.attn(h, h, values=h, mask=mask)

        # (bs, 1, hidden_size) -> (bs, 1, nb_classes)
        self.logits = self.linear_out(h)

        # (bs, 1, nb_classes) -> (bs, 1, nb_classes) in log simplex
        h = F.log_softmax(self.logits, dim=-1)

        return h
