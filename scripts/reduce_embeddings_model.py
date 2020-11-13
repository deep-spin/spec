import argparse
import pickle
import re
from pathlib import Path

import numpy as np

from spec.dataset.corpora import available_corpora
from gensim.models import KeyedVectors, FastText


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Reduce embeddings model")
    parser.add_argument("--corpus",
                        type=str,
                        choices=list(available_corpora.keys()),
                        default='sst',
                        help="corpus type")
    parser.add_argument("--data-paths",
                        type=str,
                        nargs='+',
                        help="paths to the dataset files")
    parser.add_argument("--emb-path",
                        type=str,
                        default='../data/embs/word2vec/pt_word2vec_sg_600.kv.emb',  # NOQA
                        help="path to keyed vector embedding model")
    parser.add_argument("--output-path",
                        type=str,
                        default='../data/embs/word2vec/pt_word2vec_sg_600.small.raw.emb',  # NOQA
                        help="path to the new embeddings")
    parser.add_argument('--binary',
                        action='store_true',
                        help='Whether to save the embeddings are in binary'
                             'format or not.')
    parser.add_argument("-f", "--format",
                        type=str,
                        default="word2vec",
                        choices=['word2vec', 'fasttext', 'glove'],
                        help="embeddings format")
    args = parser.parse_args()

    if args.format == 'word2vec':
        embeddings = KeyedVectors.load_word2vec_format(
            args.emb_path, unicode_errors='ignore', binary=True
        )
    elif args.format == 'fasttext':
        embeddings = FastText.load_fasttext_format(args.emb_path)
    else:
        embeddings = {}
        with open(args.emb_path, 'r') as f:
            for line in f:
                try:
                    values = line.rstrip().split()
                    name, vector = values[0], list(map(float, values[1:]))
                    embeddings[name] = np.array(vector)
                except ValueError as e:
                    # some entries have something like:
                    # by name@domain.com 0.6882 -0.36436 ...
                    # thus, values[1] is not a float at all
                    print(e, line[:10])
                    continue

    vocab = set()
    corpus_cls = available_corpora[args.corpus]
    fields_tuples = corpus_cls.create_fields_tuples()
    for data_path in args.data_paths:
        corpus = corpus_cls(fields_tuples, lazy=True)
        for ex in corpus.read(data_path):
            if hasattr(ex, 'words_hyp') and args.corpus != 'iwslt':
                vocab.update(set(ex.words_hyp))
            vocab.update(set(ex.words))
        corpus.close()

    word_vectors = {}
    oov_words = []
    nb_oov = 0
    for word in vocab:
        if word in embeddings:
            word_vectors[word] = embeddings[word]
        else:
            nb_oov += 1
            oov_words.append(word)

    print('The following words were not found in model vocab. They will be '
          'replaced later by an unknown vector.')
    print(' '.join(oov_words))
    print('Vocab size: {}'.format(len(vocab)))
    print('Nb oov: {}'.format(nb_oov))

    if args.binary:
        with open(args.output_path, 'wb') as handle:
            pickle.dump(word_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        f = open(args.output_path, 'w', encoding='utf8')
        for word, vector in word_vectors.items():
            s = ' '.join([word] + [str(d) for d in vector])
            f.write(s + '\n')
        f.close()
