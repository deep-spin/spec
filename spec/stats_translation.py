from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score)

from spec import constants
from spec.models.utils import unroll, unmask


class BestValueEpoch:
    def __init__(self, value, epoch):
        self.value = value
        self.epoch = epoch


class TranslationStats(object):
    """
    Keep stats information during training and evaluation

    Args:
        trg_vocab (Vocabulary): target/hypothesis vocab object
        pos_label (int): index of the positive label in the target vocab.
            It will be used to calculate the F1 score with average='binary'
    """
    def __init__(self, trg_vocab, pos_label=None):
        self.trg_vocab = trg_vocab
        self.pos_label = pos_label if pos_label is not None else 1
        self.average = 'macro'

        # this attrs will be updated every time a new prediction is added
        self.pred_classes = []
        self.gold_classes = []
        self.pred_probas = []
        self.gold_probas = []
        self.loss = 0
        self.nb_batches = 0

        # this attrs will be set when get_ methods are called
        self.avg_loss = None
        self.prec_rec_f1 = None
        self.acc = None
        self.mcc = None

        # this attrs will be set when calc method is called
        self.best_prec_rec_f1 = BestValueEpoch(value=[0, 0, 0], epoch=1)
        self.best_acc = BestValueEpoch(value=0, epoch=1)
        self.best_mcc = BestValueEpoch(value=0, epoch=1)
        self.best_loss = BestValueEpoch(value=float('inf'), epoch=1)

        # this are for the communicator special case
        self.tvd = 0
        self.true_gold_classes = []
        self.true_acc_clf = 0
        self.true_acc_layman = 0
        self.best_tvd = BestValueEpoch(value=0, epoch=1)

    def reset(self):
        """Reset internal stats variables to their initial values."""
        self.pred_classes.clear()
        self.gold_classes.clear()
        self.pred_probas.clear()
        self.gold_probas.clear()
        self.true_gold_classes.clear()
        self.loss = 0
        self.nb_batches = 0
        self.prec_rec_f1 = None
        self.acc = None
        self.mcc = None
        self.tvd = 0
        self.true_acc_clf = 0
        self.true_acc_layman = 0

    def update(self, loss, pred_classes, gold_classes, pred_probas=None,
               gold_probas=None, true_gold_classes=None):
        """
        Update stats internally for each batch iteration.

        Args:
            loss (float): mean loss value for a batch (loss reduction='mean')
            pred_classes (torch.Tensor): tensor with predicted classes indexes.
                Shape (batch_size, seq_len)
            gold_classes (torch.Tensor): tensor with gold labels.
                Shape (batch_size, seq_len)
            pred_probas (torch.Tensor): tensor with predicted classes probas.
                If not None, it is going be used to calculate the TVD between
                itself ant gold_probas. Shape (batch_size, seq_len, nb_classes).
                Default is None
            gold_probas (torch.Tensor): tensor with gold probas (usually 1-hot).
                If not None, it is going be used to calculate the TVD between
                itself ant pred_probas. Shape (batch_size, seq_len, nb_classes).
                Default is None
            true_gold_classes (torch.Tensor): tensor with gold labels. This is
                considered the original gold labels that comes from the dataset.
                Shape (batch_size, seq_len)
        """
        self.loss += loss
        self.nb_batches += 1
        # unmask & flatten predictions and gold labels before storing them
        mask = gold_classes != constants.TARGET_PAD_ID
        self.pred_classes.extend(unmask(pred_classes, mask))
        self.gold_classes.extend(unmask(gold_classes, mask))
        if pred_probas is not None and gold_probas is not None:
            self.pred_probas.extend(unmask(pred_probas, mask))
            self.gold_probas.extend(unmask(gold_probas, mask))
        if true_gold_classes is not None:
            self.true_gold_classes.extend(unmask(true_gold_classes, mask))

    def classes_to_sentences(self, classes):
        return [' '.join([self.trg_vocab.itos[w] for w in wids])
                for wids in classes]

    def calc(self, current_epoch):
        """
        Calculate metrics for the current_epoch with gold and predicted values
        previously stored from `update()`.
        """
        # calc metrics
        self.avg_loss = self.loss / self.nb_batches
        unrolled_gold_classes = unroll(self.gold_classes)
        unrolled_pred_classes = unroll(self.pred_classes)
        self.acc = accuracy_score(unrolled_gold_classes,
                                  unrolled_pred_classes)
        # self.mcc = matthews_corrcoef(unrolled_gold_classes,
        #                              unrolled_pred_classes)
        self.mcc = 0

        *self.prec_rec_f1, _ = precision_recall_fscore_support(
            unrolled_gold_classes,
            unrolled_pred_classes,
            average=self.average,
            pos_label=self.pos_label
        )

        # ignore tvd for translation (we need seq2seq alignments)
        # if len(self.pred_probas) > 0:
        #     self.tvd = total_variance_distance(self.gold_probas,
        #                                        self.pred_probas,
        #                                        reduction='mean')
        self.tvd = 0

        # special case for communicator:
        # here, gold_classes will be the clf predictions
        # true_gold_classes will be the true gold classes from the dataset
        # pred classes will be the layman predictions
        if len(self.true_gold_classes) > 0:
            # self.true_acc_clf = sacrebleu.raw_corpus_bleu(
            #     sys_stream=self.classes_to_sentences(self.gold_classes),
            #     ref_streams=[self.classes_to_sentences(self.true_gold_classes)]
            # ).score
            # self.true_acc_layman = sacrebleu.raw_corpus_bleu(
            #     sys_stream=self.classes_to_sentences(self.pred_classes),
            #     ref_streams=[self.classes_to_sentences(self.true_gold_classes)]
            # ).score
            # unrolled_true_gold_classes = unroll(self.true_gold_classes)
            self.true_acc_clf = 0.0
            self.true_acc_layman = 0.0

        # keep track of the best stats
        if self.avg_loss < self.best_loss.value:
            self.best_loss.value = self.avg_loss
            self.best_loss.epoch = current_epoch

        if self.prec_rec_f1[2] > self.best_prec_rec_f1.value[2]:
            self.best_prec_rec_f1.value[0] = self.prec_rec_f1[0]
            self.best_prec_rec_f1.value[1] = self.prec_rec_f1[1]
            self.best_prec_rec_f1.value[2] = self.prec_rec_f1[2]
            self.best_prec_rec_f1.epoch = current_epoch

        if self.acc > self.best_acc.value:
            self.best_acc.value = self.acc
            self.best_acc.epoch = current_epoch

        if self.mcc > self.best_mcc.value:
            self.best_mcc.value = self.mcc
            self.best_mcc.epoch = current_epoch

        # if len(self.pred_probas) > 0 and len(self.gold_probas) > 0:
        #     if self.tvd < self.best_tvd.value:
        #         self.best_tvd.value = self.tvd
        #         self.best_tvd.epoch = current_epoch

    def to_dict(self):
        if self.nb_batches == 0:
            raise Exception('You should update stats for something before.')
        if self.avg_loss is None:
            raise Exception('You should calculate stats metrics before.')
        return {
            'loss': self.avg_loss,
            'prec_rec_f1': self.prec_rec_f1,
            'acc': self.acc,
            'mcc': self.mcc,
            'best_loss': self.best_loss,
            'best_prec_rec_f1': self.best_prec_rec_f1,
            'best_acc': self.best_acc,
            'best_mcc': self.best_mcc,
            # communication stats
            'tvd': self.tvd,
            'best_tvd': self.best_tvd,
            'acc_c': self.true_acc_clf,
            'acc_l': self.true_acc_layman
        }
