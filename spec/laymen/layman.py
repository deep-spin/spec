import logging

import torch
from torch import nn

from spec import constants

logger = logging.getLogger(__name__)


class Layman(nn.Module):

    def __init__(self, fields_tuples, message_size):
        super().__init__()
        # Default fields and embeddings
        self.fields_dict = dict(fields_tuples)
        self.target_field = self.fields_dict['target']
        self.message_size = message_size
        self._loss = None

    @property
    def nb_classes(self):
        pad_shift = int(constants.PAD in self.target_field.vocab.stoi)
        return len(self.target_field.vocab) - pad_shift  # remove pad index

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, path):
        logger.info("Loading layman weights from {}".format(path))
        state = torch.load(str(path), map_location=lambda storage, loc: storage)
        if ('linear_out.weight' in state.keys()
                and 'linear_message.weight' not in state.keys()):
            state['linear_message.weight'] = state['linear_out.weight']
            state['linear_message.bias'] = state['linear_out.bias']

        self.load_state_dict(state, strict=False)

    def save(self, path):
        logger.info("Saving layman weights to {}".format(path))
        torch.save(self.state_dict(), str(path))

    def build_loss(self, loss_weights):
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

    def predict_probas(self, batch, message):
        log_probs = self.forward(batch, message)
        probs = torch.exp(log_probs)  # assume log softmax in the output
        return probs

    def predict_classes(self, batch, message):
        classes = torch.argmax(self.predict_probas(batch, message), -1)
        return classes
