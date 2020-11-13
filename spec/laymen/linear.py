import torch
from torch import nn

from spec.initialization import init_xavier
from spec.laymen.layman import Layman


class LinearLayman(Layman):

    def __init__(self, fields_tuples, message_size, options):
        super().__init__(fields_tuples, message_size)
        self.linear_message = nn.Linear(self.message_size, self.nb_classes)
        self.init_weights()

    def init_weights(self):
        init_xavier(self.linear_message, dist='uniform')

    def forward(self, batch, message_vector):
        logits = self.linear_message(message_vector)
        return torch.log_softmax(logits, dim=-1)
