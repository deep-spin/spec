import argparse

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             matthews_corrcoef)

from spec.dataset.corpora import available_corpora


def calc_acc(pred, gold):
    acc = accuracy_score(gold, pred)
    print('Acc: {:.4f}'.format(acc))


def calc_f1(pred, gold, average='micro', pos_label=1):
    prec, rec, f1, _ = precision_recall_fscore_support(gold, pred,
                                                       average=average,
                                                       pos_label=pos_label)
    print('Prec: {:.4f}'.format(prec))
    print('Rec: {:.4f}'.format(rec))
    print('F1: {:.4f}'.format(f1))


def calc_mcc(pred, gold):
    mcc = matthews_corrcoef(pred, gold)
    print('MCC: {:.4f}'.format(mcc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--corpus",
                        type=str,
                        choices=list(available_corpora.keys()),
                        default='sst',
                        help="corpus type")
    parser.add_argument("--predictions-path",
                        type=str,
                        default=None,
                        help="path to the predicitons file",
                        required=True)
    parser.add_argument("--corpus-path",
                        type=str,
                        default=None,
                        help="path to the corpus",
                        required=True)
    parser.add_argument("--average",
                        type=str,
                        default=None,
                        help="average for F1 calculation. "
                             "Default: macro if the corpus is for binary "
                             "classification, micro otherwise")
    args = parser.parse_args()

    corpus_cls = available_corpora[args.corpus]
    fields_tuples = corpus_cls.create_fields_tuples()

    print('Reading corpus...')
    kwargs_corpus = {}
    if args.corpus == 'sst':
        kwargs_corpus = {
            'subtrees': False,
            'granularity': '2',
            'return_invalid_targets': True
        }
    corpus = corpus_cls(fields_tuples, lazy=True, **kwargs_corpus)
    corpus_targets = []
    invalid_positions = []
    for i, ex in enumerate(corpus.read(args.corpus_path)):
        if ex.target is None:
            invalid_positions.append(i)
        else:
            corpus_targets.extend(ex.target)
    corpus.close()

    print('Reading predictions...')
    predictions_targets = []
    with open(args.predictions_path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip():
                pred_target = line.strip()
                if isinstance(corpus_targets[0], int):
                    pred_target = int(pred_target)
                predictions_targets.append(pred_target)

    invalid_positions = set(invalid_positions)
    predictions_targets = [p for i, p in enumerate(predictions_targets) if
                           i not in invalid_positions]

    print('Calculating metrics...')
    pos_label = 1
    average = args.average
    if args.average is None:
        bin_corpus = ['agnews', 'imdb', 'sst', 'ttsbr']
        average = 'macro' if args.corpus in bin_corpus else 'micro'

    print('Nb preds x golds:', len(predictions_targets), len(corpus_targets))
    calc_acc(predictions_targets, corpus_targets)
    calc_f1(predictions_targets, corpus_targets, average=average)
    calc_mcc(predictions_targets, corpus_targets)
