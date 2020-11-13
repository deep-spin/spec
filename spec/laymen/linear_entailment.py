import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from spec import constants
from spec.initialization import init_xavier
from spec.laymen.layman import Layman


class LinearEntailmentLayman(Layman):

    def __init__(self, fields_tuples, message_size, options):
        super().__init__(fields_tuples, message_size)
        self.linear_message = nn.Linear(self.message_size, self.nb_classes)

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
        if options.freeze_embeddings:
            self.word_emb.weight.requires_grad = False

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
        self.linear_merge = nn.Linear(self.nb_classes, self.nb_classes)
        self.init_weights()

    def init_weights(self):
        init_xavier(self.linear_message, dist='uniform')
        init_xavier(self.rnn_hyp, dist='uniform')
        init_xavier(self.linear_hyp, dist='uniform')
        init_xavier(self.linear_merge, dist='uniform')

    def forward(self, batch, message_vector):
        message_logits = self.linear_message(message_vector)
        hyp_vec = self.get_hypothesis_logits(batch.words_hyp)
        hyp_logits = self.linear_hyp(hyp_vec)
        merged_logits = torch.tanh(message_logits + hyp_logits)
        logits = self.linear_merge(merged_logits)
        return torch.log_softmax(logits, dim=-1)

    def get_hypothesis_logits(self, words_hyp):
        mask_hyp = torch.ne(words_hyp, constants.PAD_ID)
        h_hyp = self.word_emb(words_hyp)
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

        # get last valid outputs instead of the last hidden state:
        # last_valid_idx = mask_hyp.int().sum(dim=-1) - 1
        # arange_vector = torch.arange(h_hyp.shape[0]).to(h_hyp.device)
        # hyp_logits = h_hyp[arange_vector, last_valid_idx].unsqueeze(1)

        return hyp_logits
