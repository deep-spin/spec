import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from spec import constants
from spec.initialization import init_xavier
from spec.laymen.layman import Layman


class TranslationLinearLayman(Layman):

    def __init__(self, fields_tuples, message_size, options):
        super().__init__(fields_tuples, message_size)
        hidden_size = options.hidden_size[0]
        self.linear_message = nn.Linear(self.message_size, hidden_size)
        self.linear_mid = nn.Linear(2 * hidden_size, self.nb_classes)

        embeddings_weight = None
        if self.fields_dict['words_hyp'].vocab.vectors is not None:
            embeddings_weight = self.fields_dict['words_hyp'].vocab.vectors
            options.word_embeddings_size = embeddings_weight.size(1)

        self.word_emb = nn.Embedding(
            num_embeddings=len(self.fields_dict['words_hyp'].vocab),
            embedding_dim=options.word_embeddings_size,
            padding_idx=constants.TARGET_PAD_ID,
            _weight=embeddings_weight
        )
        if options.freeze_embeddings:
            self.word_emb.weight.requires_grad = False

        self.rnn_hyp = nn.LSTM(options.word_embeddings_size, hidden_size,
                               bidirectional=False, batch_first=True)
        # self.linear_hyp = nn.Linear(hidden_size, self.nb_classes)
        # self.linear_merge = nn.Linear(self.nb_classes, self.nb_classes)
        self.start_vector = torch.rand(1, hidden_size)
        self.start_vector = self.start_vector.to(options.gpu_id)
        self.init_weights()

    def init_weights(self):
        init_xavier(self.linear_mid, dist='uniform')
        init_xavier(self.linear_message, dist='uniform')
        init_xavier(self.rnn_hyp, dist='uniform')
        # init_xavier(self.linear_hyp, dist='uniform')
        # init_xavier(self.linear_merge, dist='uniform')

    def forward(self, batch, message_vector):
        message_vec = self.linear_message(message_vector)
        hyp_vec = self.get_hypothesis_logits(batch.words_hyp)
        start_vector = self.start_vector.expand(hyp_vec.shape[0], -1)
        hyp_vec = torch.cat((start_vector.unsqueeze(1), hyp_vec[:, :-1]), dim=1)
        merged_vec = torch.cat((message_vec, hyp_vec), dim=-1)
        logits = self.linear_mid(torch.tanh(merged_vec))
        return torch.log_softmax(logits, dim=-1)

        # without looking at the message:
        # logits = self.linear_mid(torch.tanh(hyp_vec))
        # return torch.log_softmax(logits, dim=-1)

        # source x target mapping (huge matrix)
        # h = self.linear_mid(torch.tanh(self.linear_message(message_vector)))
        # return torch.log_softmax(h, dim=-1)

    def get_hypothesis_logits(self, words_hyp):
        mask_hyp = torch.ne(words_hyp, constants.TARGET_PAD_ID)
        lengths = mask_hyp.int().sum(dim=-1)
        # last examples in a very few batches are filled with pad :-/
        lengths[lengths == 0] = 1
        h_hyp = self.word_emb(words_hyp)
        h_hyp = pack(h_hyp, lengths, batch_first=True, enforce_sorted=False)
        h_hyp, hidden_hyp = self.rnn_hyp(h_hyp)
        h_hyp, _ = unpack(h_hyp, batch_first=True)
        return h_hyp
