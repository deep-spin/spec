import logging

import torch
from torch import nn

from spec import constants
from spec.explainers.utils import make_bow_matrix

logger = logging.getLogger(__name__)


class Explainer(nn.Module):

    def __init__(self, fields_tuples):
        super().__init__()
        self.fields_dict = dict(fields_tuples)
        self.target_field = self.fields_dict['target']
        self._loss = None
        # hack: create a dumb parameter so something is passed to the optimizer
        # this parameters does not affect anything in our code. Leave it alone
        self.dumb_parameter = nn.Parameter(torch.zeros(0))

    @property
    def nb_classes(self):
        pad_shift = int(constants.PAD in self.target_field.vocab.stoi)
        return len(self.target_field.vocab) - pad_shift  # remove pad index

    def loss(self, pred, gold):
        if self._loss is None:
            # dummy loss so we can call loss.backward() without breaking
            # define a self._loss if you want to train a explainer exclusively
            # take a look at the self.build_loss() method
            return torch.zeros(1, requires_grad=True).sum()
        else:
            # (bs, ts, nb_classes) -> (bs*ts, nb_classes)
            predicted = pred.reshape(-1, self.nb_classes)
            # (bs, ts, ) -> (bs*ts, )
            gold = gold.reshape(-1)
            return self._loss(predicted, gold)

    def build_loss(self, loss_weights=None):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def get_output_size(self):
        if self.message_type == 'bow':
            return self.words_vocab_size
        else:
            return self.emb_size

    def load(self, path):
        logger.info("Loading explainer weights from {}".format(path))
        self.load_state_dict(
            torch.load(str(path), map_location=lambda storage, loc: storage)
        )

    def save(self, path):
        logger.info("Saving explainer weights to {}".format(path))
        torch.save(self.state_dict(), str(path))

    def make_message(self, top_word_ids, top_probas, emb_layer):
        """
        Create a message according to the selected self.message_type.

        Args:
            top_word_ids (list of lists): a list of lists that contains word ids
            top_probas (torch.Tensor): tensor with probabilities (or any other
                kind of measure of importance) for each word id.
                Shape of (batch_size, seq_len)
            emb_layer (torch.nn.Module): an embedding layer to recover vectors
                for each select word_id (used when self.message_type depends on
                the word embeddings).

        Returns:
            torch.Tensor of shape
                (batch_size, vocab_size) if self.message_type = 'bow'
                (batch_size, emb_layer.hidden_size) otherwise
        """
        if self.message_type == 'bow':
            message = make_bow_matrix(
                top_word_ids, self.words_vocab_size, device=top_probas.device
            )
        else:
            # the embedding layer is configured with padding_idx, so it will
            # return an embedding with all zeros for PAD_IDs
            batch_size = len(top_word_ids)
            seq_len = top_probas.shape[1]
            top_word_ids_tensor = torch.full(
                (batch_size, seq_len), constants.PAD_ID,
                dtype=torch.long,
                device=top_probas.device
            )
            for i, x in enumerate(top_word_ids):
                top_word_ids_tensor[i, :len(x)] = torch.tensor(x)

            if self.message_type == 'embs_sum':
                embs = emb_layer(top_word_ids_tensor)
                message = embs.sum(dim=1)
            elif self.message_type == 'embs_mean':
                embs = emb_layer(top_word_ids_tensor)
                message = embs.mean(dim=1)
            elif self.message_type == 'weighted_embs_sum':
                embs = emb_layer(top_word_ids_tensor)
                embs = embs * top_probas.unsqueeze(-1)
                message = embs.sum(dim=1)
            elif self.message_type == 'weighted_embs_mean':
                embs = emb_layer(top_word_ids_tensor)
                embs = embs * top_probas.unsqueeze(-1)
                message = embs.mean(dim=1)
            else:
                raise Exception('Message type {} not implemented'.format(
                    self.message_type))
        return message
