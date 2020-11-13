import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from spec import constants
from spec.initialization import init_xavier
from spec.models.model import Model


class LinearBoW(Model):
    """
    Simple linear bag of words classifier.
    This class replicates the top-k bow explainer + linear layman with k equal
    to the sequence length (i.e., a bow with all words in the document).
    """

    def __init__(self, fields_tuples, options):
        super().__init__(fields_tuples)
        self.words_vocab_size = len(self.fields_dict['words'].vocab)

        embeddings_weight = None
        if self.fields_dict['words'].vocab.vectors is not None:
            embeddings_weight = self.fields_dict['words'].vocab.vectors
            options.word_embeddings_size = embeddings_weight.size(1)

        self.word_emb_hyp = nn.Embedding(
            num_embeddings=self.words_vocab_size,
            embedding_dim=options.word_embeddings_size,
            padding_idx=constants.PAD_ID,
            _weight=embeddings_weight,
        )
        if options.freeze_embeddings:
            self.word_emb_hyp.weight.requires_grad = False

        self.is_bidir = options.bidirectional
        self.rnn_type = options.rnn_type
        rnn_map = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}
        rnn_class = rnn_map[self.rnn_type]
        hidden_size = options.hidden_size[0]
        self.rnn_hyp = rnn_class(options.word_embeddings_size,
                                 hidden_size,
                                 bidirectional=self.is_bidir,
                                 batch_first=True)
        n = 2 if self.is_bidir else 1
        self.linear_hyp = nn.Linear(n * hidden_size, self.nb_classes)
        self.linear_input = nn.Linear(self.words_vocab_size, self.nb_classes)
        self.linear_merge = nn.Linear(self.nb_classes, self.nb_classes)

        # stored variables
        self.logits = None

        self.init_weights()
        self.is_built = True

    def init_weights(self):
        init_xavier(self.linear_input, dist='uniform')
        init_xavier(self.rnn_hyp, dist='uniform')
        init_xavier(self.linear_hyp, dist='uniform')
        init_xavier(self.linear_merge, dist='uniform')

    def forward(self, batch):
        assert self.is_built
        assert self._loss is not None
        bow_words = self.get_bow(batch.words)
        words_logits = self.linear_input(bow_words)
        if hasattr(batch, 'words_hyp'):
            hyp_vec = self.get_hypothesis_logits(batch.words_hyp)
            hyp_logits = self.linear_hyp(hyp_vec)
            merged_logits = torch.tanh(words_logits + hyp_logits)
            self.logits = self.linear_merge(merged_logits)
        else:
            self.logits = words_logits

        return torch.log_softmax(self.logits, dim=-1)

    def get_bow(self, words):
        bs, ts = words.shape
        mask = torch.ne(words, constants.PAD_ID)
        bids = torch.arange(bs).unsqueeze(-1).expand(-1, ts).flatten()
        bids = bids.to(words.device)
        idxs = torch.stack((bids, words.flatten()), dim=0)
        vals = mask.int().to(words.device).flatten().float()
        size = torch.Size([bs, self.words_vocab_size])
        bow = torch.sparse.FloatTensor(idxs, vals, size).to_dense()
        bow = bow.to(words.device)
        return bow.unsqueeze(1)

    def get_hypothesis_logits(self, words_hyp):
        mask_hyp = torch.ne(words_hyp, constants.PAD_ID)
        h_hyp = self.word_emb_hyp(words_hyp)

        lengths = mask_hyp.int().sum(dim=-1)
        h_hyp = pack(h_hyp, lengths, batch_first=True, enforce_sorted=False)
        h_hyp, hidden_hyp = self.rnn_hyp(h_hyp)
        h_hyp, _ = unpack(h_hyp, batch_first=True)

        if self.rnn_type == 'lstm':
            hidden_hyp = hidden_hyp[0]

        if self.is_bidir:
            hidden_states = [hidden_hyp[0], hidden_hyp[1]]
        else:
            hidden_states = [hidden_hyp[0]]

        hyp_logits = torch.cat(hidden_states, dim=-1).unsqueeze(1)

        # get the last valid outputs (non pad) instead of the hidden state:
        # last_valid_idx = mask_hyp.int().sum(dim=-1) - 1
        # arange_vector = torch.arange(h_hyp.shape[0]).to(h_hyp.device)
        # hyp_logits = h_hyp[arange_vector, last_valid_idx].unsqueeze(1)

        return hyp_logits
