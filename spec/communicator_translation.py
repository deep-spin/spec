import logging
import shutil
import time
from pathlib import Path

import torch

from spec import explainers, laymen, constants
from spec import optimizer
from spec.reporter import Reporter
from spec.stats_translation import TranslationStats

logger = logging.getLogger(__name__)


class CommunicatorTranslation:

    def __init__(
        self,
        train_iter,
        explainer,
        layman,
        explainer_optimizer,
        layman_optimizer,
        options,
        dev_iter=None,
        test_iter=None
    ):
        self.explainer = explainer
        self.layman = layman
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter
        self.explainer_optimizer = explainer_optimizer
        self.layman_optimizer = layman_optimizer
        self.epochs = options.epochs
        self.output_dir = options.output_dir
        self.dev_checkpoint_epochs = options.dev_checkpoint_epochs
        self.save_checkpoint_epochs = options.save_checkpoint_epochs
        self.save_best_only = options.save_best_only
        self.early_stopping_patience = options.early_stopping_patience
        self.restore_best_model = options.restore_best_model
        self.train_explainer = options.train_explainer
        self.current_epoch = 1
        self.last_saved_epoch = 0
        self.train_stats_history = []
        self.dev_stats_history = []
        self.test_stats_history = []
        self.final_report = options.final_report
        trg_vocab = self.train_iter.dataset.fields['target'].vocab
        self.train_stats = TranslationStats(trg_vocab)
        self.dev_stats = TranslationStats(trg_vocab)
        self.test_stats = TranslationStats(trg_vocab)
        self.reporter = Reporter(
            options.output_dir, options.tensorboard, for_communication=True
        )

    def train(self):
        # Perform an evaluation on dev set if it is available
        if self.dev_iter is not None:
            logger.info('Evaluating before training...')
            self.reporter.set_epoch(0)
            self.dev_epoch()

        # Perform an evaluation on test set if it is available
        if self.test_iter is not None:
            logger.info('Testing before training...')
            self.reporter.set_epoch(0)
            self.test_epoch()

        start_time = time.time()
        for epoch in range(self.current_epoch, self.epochs + 1):
            logger.info('Epoch {} of {}'.format(epoch, self.epochs))

            self.reporter.set_epoch(epoch)
            self.current_epoch = epoch

            # Train a single epoch
            logger.info('Training...')
            self.train_epoch()

            # Perform an evaluation on dev set if it is available
            if self.dev_iter is not None:
                # Only perform if a checkpoint was reached
                if (self.dev_checkpoint_epochs > 0
                        and epoch % self.dev_checkpoint_epochs == 0):
                    logger.info('Evaluating...')
                    self.dev_epoch()

            # Perform an evaluation on test set if it is available
            if self.test_iter is not None:
                logger.info('Testing...')
                self.test_epoch()

            # Only save if an improvement has occurred
            if self.save_best_only and self.dev_iter is not None:
                if self.dev_stats.best_acc.epoch == epoch:
                    logger.info('Acc. improved on epoch {}'.format(epoch))
                    self.save(epoch)
            else:
                # Otherwise, save if a checkpoint was reached
                if (self.save_checkpoint_epochs > 0
                        and epoch % self.save_checkpoint_epochs == 0):
                    self.save(epoch)

            # Stop training before the total number of epochs
            if self.early_stopping_patience > 0 and self.dev_iter is not None:
                # Only stop if the desired patience epochs was reached
                passed_epochs = epoch - self.dev_stats.best_acc.epoch
                if passed_epochs == self.early_stopping_patience:
                    logger.info('Training stopped! No improvements on Acc. '
                                'after {} epochs'.format(passed_epochs))
                    if self.restore_best_model:
                        if self.dev_stats.best_acc.epoch < epoch:
                            self.restore_epoch(self.dev_stats.best_acc.epoch)
                    break

            # Restore best model if early stopping didnt occur for final epoch
            if epoch == self.epochs and self.dev_iter is not None:
                if self.restore_best_model:
                    if self.dev_stats.best_acc.epoch < epoch:
                        self.restore_epoch(self.dev_stats.best_acc.epoch)

        elapsed = time.time() - start_time
        hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
        logger.info('Training ended after {}'.format(hms))

        if self.final_report:
            logger.info('Training final report: ')
            self.reporter.report_stats_history(self.train_stats_history)
            if self.dev_iter:
                logger.info('Dev final report: ')
                self.reporter.report_stats_history(
                    self.dev_stats_history, start=0
                )
            if self.test_iter:
                logger.info('Test final report: ')
                self.reporter.report_stats_history(
                    self.test_stats_history, start=0
                )
        self.reporter.close()

    def train_epoch(self):
        self.reporter.set_mode('train')
        self.train_stats.reset()
        self._train()
        self.train_stats_history.append(self.train_stats.to_dict())
        self.reporter.report_stats(self.train_stats.to_dict())

    def dev_epoch(self):
        self.reporter.set_mode('dev')
        self.dev_stats.reset()
        self._eval(self.dev_iter, self.dev_stats)
        self.dev_stats_history.append(self.dev_stats.to_dict())
        self.reporter.report_stats(self.dev_stats.to_dict())

    def test_epoch(self):
        self.reporter.set_mode('test')
        self.test_stats.reset()
        self._eval(self.test_iter, self.test_stats)
        self.test_stats_history.append(self.test_stats.to_dict())
        self.reporter.report_stats(self.test_stats.to_dict())

    def _train(self):
        if self.train_explainer:
            self.explainer.train()
        self.layman.train()
        for i, batch in enumerate(self.train_iter, start=1):
            # classifier basic training steps:
            clf_pred_target = batch.words_hyp

            # explainer basic training steps:
            self.explainer.zero_grad()
            message = self.explainer(batch, classifier=None)
            explainer_loss = self.explainer.loss(message, None)
            if not self.train_explainer:
                message = message.detach()

            # layman basic training steps:
            self.layman.zero_grad()
            layman_pred = self.layman(batch, message)
            layman_loss = self.layman.loss(layman_pred, clf_pred_target)

            # get gradients for the layman (+ explainer optionally) and update
            # the weights
            layman_loss.backward()
            self.layman_optimizer.step()

            # in case train explainer is True, get the gradients from its loss
            # (it can be a dummy loss) and update its gradients (which could
            # be computed using the layman_loss.backward())
            if self.train_explainer:
                explainer_loss.backward()
                self.explainer_optimizer.step()

            layman_pred_probas = self.layman.predict_probas(batch, message)
            layman_pred_target = torch.argmax(layman_pred_probas, dim=-1)

            # keep stats object updated:
            self.train_stats.update(
                layman_loss.item(),
                layman_pred_target,  # y_pred
                clf_pred_target,  # y_gold
                pred_probas=None,  # y_pred
                gold_probas=None,  # y_gold  (dummy)
                true_gold_classes=batch.target
            )

            # report current loss to the user:
            acum_loss = self.train_stats.loss / self.train_stats.nb_batches
            self.reporter.report_progress(i, len(self.train_iter), acum_loss)

        self.train_stats.calc(self.current_epoch)

    def _eval(self, ds_iterator, stats):
        self.explainer.eval()
        self.layman.eval()
        with torch.no_grad():
            for i, batch in enumerate(ds_iterator, start=1):
                # basic prediction steps:
                clf_pred_target = batch.words_hyp
                message = self.explainer(batch, classifier=None)

                layman_pred = self.layman(batch, message)
                layman_pred_probas = torch.exp(layman_pred)
                layman_pred_target = torch.argmax(layman_pred_probas, dim=-1)
                layman_loss = self.layman.loss(layman_pred, clf_pred_target)

                # keep stats object updated:
                stats.update(
                    layman_loss.item(),
                    layman_pred_target,  # y_pred
                    clf_pred_target,  # y_gold
                    pred_probas=None,  # y_pred
                    gold_probas=None,  # y_gold  (dummy)
                    true_gold_classes=batch.target
                )

                # report current loss to the user:
                acum_loss = stats.loss / stats.nb_batches
                self.reporter.report_progress(i, len(ds_iterator), acum_loss)

        stats.calc(self.current_epoch)
        # if ds_iterator == self.test_iter:
        #     pt = '/home/mtreviso/spec/data/debug-explanations/tmp.txt'
        #     self.save_explanations(pt, ds_iterator)

    def save(self, current_epoch):
        old_dir = 'epoch_{}'.format(self.last_saved_epoch)
        old_path = Path(self.output_dir, old_dir)
        if old_path.exists() and old_path.is_dir():
            logger.info('Removing old state {}'.format(old_path))
            shutil.rmtree(str(old_path))
        epoch_dir = 'epoch_{}'.format(current_epoch)
        output_path = Path(self.output_dir, epoch_dir)
        output_path.mkdir(exist_ok=True)
        logger.info('Saving training state to {}'.format(output_path))
        explainers.save(output_path, self.explainer)
        laymen.save(output_path, self.layman)
        optimizer.save(output_path, self.explainer_optimizer,
                       name=constants.EXPLAINER_OPTIMIZER)
        optimizer.save(output_path, self.layman_optimizer,
                       name=constants.LAYMAN_OPTIMIZER)
        self.last_saved_epoch = current_epoch

    def load(self, directory):
        logger.info('Loading training state from {}'.format(directory))
        explainers.load_state(directory, self.explainer)
        laymen.load_state(directory, self.layman)
        optimizer.load_state(directory, self.explainer_optimizer,
                             name=constants.EXPLAINER_OPTIMIZER)
        optimizer.load_state(directory, self.layman_optimizer,
                             name=constants.LAYMAN_OPTIMIZER)

    def restore_epoch(self, epoch):
        epoch_dir = 'epoch_{}'.format(epoch)
        self.load(str(Path(self.output_dir, epoch_dir)))

    def resume(self, epoch):
        self.restore_epoch(epoch)
        self.current_epoch = epoch
        self.train()

    def save_explanations(self, path, dataset_iterator, max_explanations=None):
        if max_explanations is None:
            max_explanations = float('inf')
        words_vocab = dataset_iterator.dataset.fields['words'].vocab
        target_vocab = dataset_iterator.dataset.fields['target'].vocab
        self.explainer.eval()
        file_path = Path(path)
        with file_path.open('w', encoding='utf8') as f:
            with torch.no_grad():
                nb_explanations = 0
                for i, batch in enumerate(dataset_iterator, start=1):
                    if nb_explanations >= max_explanations:
                        # consume the entire iterator in case we have a lazy
                        # iterator this is important
                        continue
                    # run the explainer and get the valid_top_word_ids which
                    # are the ids of the words that are considered explanations
                    _ = self.explainer(batch, classifier=None)
                    word_ids = self.explainer.valid_top_word_ids
                    hyp_ids = batch.words_hyp.tolist()
                    for wids, hids in zip(word_ids, hyp_ids):
                        if nb_explanations >= max_explanations:
                            continue
                        for ws, hyp_id in zip(wids, hids):
                            if hyp_id != constants.TARGET_PAD_ID:
                                src = ' '.join([words_vocab.itos[w] for w in ws])
                                hyp = target_vocab.itos[hyp_id]
                                f.write('{}\t{}\n'.format(hyp, src))
                        nb_explanations += 1
