# example:
# --------
# python3 get_corpus_examples_for_human_annotation.py \
# --corpus sst \
# --corpus-path "../data/corpus/sst/test.txt" \
# --output-path "../data/human-corpus/sst.txt" \
# --nb-examples 2000

# or to select all data:
# python3 scripts/get_corpus_examples_for_human_annotation.py \
#         --corpus imdb \
#         --corpus-path "data/corpus/imdb/test/" \
#         --output-path "data/human-corpus-all/imdb.txt" \
#         --nb-examples 99999999

# or to select dev data:
# python3 scripts/get_corpus_examples_for_human_annotation.py \
#         --corpus imdb \
#         --corpus-path "data/corpus/imdb/dev/" \
#         --output-path "data/human-corpus-dev/imdb.txt" \
#         --nb-examples 99999999

import argparse
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from spec.dataset.corpora import available_corpora


def select_indexes_stratified(words, targets, sel_nb_examples):
    x_placeholder = np.zeros(len(words))
    y_placeholder = np.zeros(len(words))

    if sel_nb_examples >= len(words):
        print('Corpus smaller than the selected nb of examples...')
        print('Selecting {} instances instead.'.format(len(words)))
        test_index = list(range(len(words)))
        random.shuffle(test_index)
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=sel_nb_examples, random_state=42
        )
        test_index = list(sss.split(x_placeholder, y_placeholder))[0][1]
    new_words = [words[i] for i in test_index]
    new_targets = [targets[i] for i in test_index]
    return new_words, new_targets


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get examples for humans")
    parser.add_argument("--corpus",
                        type=str,
                        choices=list(available_corpora.keys()),
                        default='sst',
                        help="corpus type")
    parser.add_argument("--corpus-path",
                        type=str,
                        default=None,
                        help="path to the corpus",
                        required=True)
    parser.add_argument("--output-path",
                        type=str,
                        default=None,
                        help="path to the output file",
                        required=True)
    parser.add_argument("--nb-examples",
                        type=int,
                        default=2000,
                        help="Number of examples to humman annotation")
    args = parser.parse_args()
    random.seed(42)

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
    corpus_words = []
    for i, ex in enumerate(corpus.read(args.corpus_path)):
        if ex.target is not None:
            corpus_targets.extend(ex.target)
            words_str = ' '.join(ex.words)
            if hasattr(ex, 'words_hyp'):
                words_str += ' ||| '
                words_str += ' '.join(ex.words_hyp)
            corpus_words.append(words_str)
    corpus.close()

    print('Selecting {} instances...'.format(args.nb_examples))
    sel_corpus_words, sel_corpus_targets = select_indexes_stratified(
        corpus_words, corpus_targets, args.nb_examples
    )

    print('Saving corpus...')
    output_file = Path(args.output_path)
    with output_file.open('w', encoding='utf8') as f:
        for words, target in zip(sel_corpus_words, sel_corpus_targets):
            line = '%s\t%s' % (target, words)
            f.write(line + '\n')
