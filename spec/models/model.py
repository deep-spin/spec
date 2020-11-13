import logging
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from spec import constants

logger = logging.getLogger(__name__)


class Model(torch.nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, fields_tuples):
        super().__init__()
        # Default fields and embeddings
        self.fields_dict = dict(fields_tuples)
        self.target_field = self.fields_dict['target']
        # Building flag
        self.is_built = False
        # Loss function has to be defined later!
        self._loss = None

    @property
    def nb_classes(self):
        pad_shift = int(constants.PAD in self.target_field.vocab.stoi)
        return len(self.target_field.vocab) - pad_shift  # remove pad index

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, path):
        logger.info("Loading model weights from {}".format(path))
        self.load_state_dict(
            torch.load(str(path), map_location=lambda storage, loc: storage)
        )

    def save(self, path):
        logger.info("Saving model weights to {}".format(path))
        torch.save(self.state_dict(), str(path))

    def build_loss(self, loss_weights=None):
        if loss_weights is not None:
            loss_weights = torch.tensor(loss_weights).float()
        self._loss = nn.NLLLoss(weight=loss_weights,
                                ignore_index=constants.TARGET_PAD_ID)

    def loss(self, pred, gold):
        # (bs, ts, nb_classes) -> (bs*ts, nb_classes)
        predicted = pred.reshape(-1, self.nb_classes)
        # (bs, ts, ) -> (bs*ts, )
        gold = gold.reshape(-1)
        return self._loss(predicted, gold)

    def predict_probas(self, batch):
        log_probs = self.forward(batch)
        probs = torch.exp(log_probs)  # assume log softmax in the output
        return probs

    def predict_classes(self, batch):
        classes = torch.argmax(self.predict_probas(batch), -1)
        return classes
