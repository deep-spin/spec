# example:
# --------
# python3 get_esnli_examples_without_neutrals.py \
# --corpus esnli \
# --corpus-path "../data/corpus/esnli/esnli_test.csv" \
# --output-path "../data/human-corpus/esnli_without_neutrals_h0.txt" \
# --hid 0

import argparse
import random
from pathlib import Path

from spec.dataset.corpora import available_corpora


def select_indexes_stratified(words, targets, sel_nb_examples):
    test_index = list(range(len(words)))
    test_index = test_index[:sel_nb_examples]
    new_words = [words[i] for i in test_index]
    new_targets = [targets[i] for i in test_index]
    return new_words, new_targets


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get examples for humans")
    parser.add_argument("--corpus",
                        type=str,
                        default='snli',
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
    parser.add_argument("--hid",
                        type=int,
                        default=0,
                        help="Human id explanations")
    parser.add_argument("--nb-examples",
                        type=int,
                        default=999999999,
                        help="Number of examples to humman annotation")
    args = parser.parse_args()
    random.seed(42)

    corpus_cls = available_corpora[args.corpus]
    fields_tuples = corpus_cls.create_fields_tuples()

    print('Reading corpus...')
    kwargs_corpus = {}

    corpus = corpus_cls(fields_tuples, lazy=True, **kwargs_corpus)
    corpus_targets = []
    corpus_words = []
    for i, ex in enumerate(corpus.read(args.corpus_path)):
        if ex.target is not None:
            corpus_targets.extend(ex.target)
            words_str = ' '.join(ex.words)
            words_str += ' ||| '
            words_str += ' '.join(ex.words_hyp)
            words_str += ' ||| '
            words_str += ' '.join(list(map(str, ex.marks[args.hid])))
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
            if target == 'neutral':
                continue
            line = '%s\t%s' % (target, words)
            f.write(line + '\n')
