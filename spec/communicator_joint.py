import logging
from pathlib import Path

import torch

from spec.communicator import Communicator

logger = logging.getLogger(__name__)


class CommunicatorJoint(Communicator):

    def __init__(self, train_iter, classifier, explainer, layman,
                 classifier_optimizer, classifier_scheduler,
                 explainer_optimizer, layman_optimizer, options, dev_iter=None,
                 test_iter=None):
        super().__init__(train_iter, classifier, explainer, layman,
                         classifier_optimizer, classifier_scheduler,
                         explainer_optimizer, layman_optimizer, options,
                         dev_iter=dev_iter, test_iter=test_iter)
        self.comm_lambda = options.explainer_lambda
        self.comm_sec_loss = options.explainer_second_loss

    def _train(self):
        if self.train_classifier:
            self.classifier.train()
        if self.train_explainer:
            self.explainer.train()
        self.layman.train()
        try:
            epoch_iters = len(self.train_iter)
        except ZeroDivisionError:
            epoch_iters = 50742  # lazy mode (only used for yelp)
        train_iters = self.epochs * epoch_iters

        for i, batch in enumerate(self.train_iter, start=1):

            # classifier basic training steps:
            self.classifier.zero_grad()
            clf_pred = self.classifier(batch)
            clf_loss = self.classifier.loss(clf_pred, batch.target)
            if self.train_classifier:
                clf_loss.backward()
                self.classifier_optimizer.step()
            clf_pred_probas = self.classifier.predict_probas(batch)
            clf_pred_target = torch.argmax(clf_pred_probas, dim=-1)

            if not self.train_classifier:
                clf_pred = clf_pred.detach()
                clf_pred_probas = clf_pred_probas.detach()
                clf_pred_target = clf_pred_target.detach()

            # explainer basic training steps:
            self.explainer.zero_grad()

            # p = ((self.current_epoch - 1) * epoch_iters + i) / train_iters
            p = i / epoch_iters
            message, message_emb = self.explainer(batch, self.classifier, p=p)

            explainer_loss = self.explainer.loss(message, clf_pred)
            if not self.train_explainer:
                message = message.detach()

            # layman basic training steps:
            self.layman.zero_grad()
            layman_pred = self.layman(batch, message)
            layman_loss = self.layman.loss(layman_pred, clf_pred_target)

            if self.comm_sec_loss != 'none':
                other_loss = self.explainer.get_second_loss(
                    batch, message, message_emb, self.classifier,
                    kind=self.comm_sec_loss
                )
                layman_loss = layman_loss + self.comm_lambda * other_loss

            # get gradients for the layman (+ explainer optionally) and update
            # the weights
            layman_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.layman.parameters(), 0.5)
            self.layman_optimizer.step()

            # in case train explainer is True, get the gradients from its loss
            # (it can be a dummy loss) and update its gradients (which could
            # be computed using the layman_loss.backward())
            if self.train_explainer:
                explainer_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.explainer.parameters(), 0.5)
                self.explainer_optimizer.step()

            layman_pred_probas = self.layman.predict_probas(batch, message)
            layman_pred_target = torch.argmax(layman_pred_probas, dim=-1)

            # keep stats object updated:
            self.train_stats.update(
                layman_loss.item(),
                layman_pred_target,  # y_pred
                clf_pred_target,  # y_gold
                pred_probas=layman_pred_probas,  # y_pred
                gold_probas=clf_pred_probas,  # y_gold
                true_gold_classes=batch.target
            )

            # report current loss to the user:
            acum_loss = self.train_stats.loss / self.train_stats.nb_batches
            self.reporter.report_progress(i, len(self.train_iter), acum_loss)

        self.train_stats.calc(self.current_epoch)

        # scheduler.step() after training
        self.classifier_scheduler.step()

    def _eval(self, ds_iterator, stats):
        self.classifier.eval()
        self.explainer.eval()
        self.layman.eval()
        with torch.no_grad():
            losses = []
            for i, batch in enumerate(ds_iterator, start=1):
                # basic prediction steps:
                clf_pred_probas = self.classifier.predict_probas(batch)
                clf_pred_target = torch.argmax(clf_pred_probas, dim=-1)

                message, message_emb = self.explainer(batch, self.classifier)

                layman_pred = self.layman(batch, message)
                layman_pred_probas = self.layman.predict_probas(batch, message)
                layman_pred_target = torch.argmax(layman_pred_probas, dim=-1)
                layman_loss = self.layman.loss(layman_pred, clf_pred_target)

                other_loss = self.explainer.get_second_loss(
                    batch, message, message_emb, self.classifier,
                    kind=self.comm_sec_loss
                )
                losses.append([layman_loss.item(), other_loss.item()])

                # keep stats object updated:
                stats.update(
                    layman_loss.item(),
                    layman_pred_target,  # y_pred
                    clf_pred_target,  # y_gold
                    pred_probas=layman_pred_probas,  # y_pred
                    gold_probas=clf_pred_probas,  # y_gold
                    true_gold_classes=batch.target
                )

                # report current loss to the user:
                acum_loss = stats.loss / stats.nb_batches
                self.reporter.report_progress(i, len(ds_iterator), acum_loss)

            avg_losses = torch.tensor(losses).mean(0)
            logger.info(str(avg_losses))
            logger.info((avg_losses.max() / avg_losses.min()).item())
        stats.calc(self.current_epoch)

    def save_explanations(self, path, dataset_iterator, max_explanations=None):
        if max_explanations is None:
            max_explanations = float('inf')
        words_vocab = dataset_iterator.dataset.fields['words'].vocab
        target_vocab = dataset_iterator.dataset.fields['target'].vocab
        self.classifier.eval()
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
                    # for posthoc we need clf hidden
                    # _ = self.classifier(batch)
                    # _ = self.explainer(batch, self.classifier)
                    p_c = self.classifier.predict_classes(batch)
                    message, _ = self.explainer(batch, self.classifier)
                    p_l = self.layman.predict_classes(batch, message)
                    word_ids = self.explainer.valid_top_word_ids
                    if not isinstance(word_ids, list):
                        word_ids = word_ids.tolist()
                    target_ids = batch.target.tolist()
                    lay_ids = p_l.tolist()
                    clf_ids = p_c.tolist()
                    for wids, tids, lids, cids in zip(word_ids, target_ids,
                                                      lay_ids, clf_ids):
                        if nb_explanations >= max_explanations:
                            continue
                        words = ' '.join([words_vocab.itos[w] for w in wids])
                        ts = ' '.join([target_vocab.itos[t] for t in tids])
                        ls = ' '.join([target_vocab.itos[t] for t in lids])
                        cs = ' '.join([target_vocab.itos[t] for t in cids])
                        f.write('{}\t{}\t{}\t{}\n'.format(ts, ls, cs, words))
                        nb_explanations += 1
