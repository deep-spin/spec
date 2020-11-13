# example:
# --------
# python3 calculate_nonzeros_for_sparse_attentions.py \
# --corpus sst \
# --corpus-path "../data/corpus/sst/test.txt"  \
# --load-model-path "../data/saved-models/test-sst-softmax/" \
# --load-explainer-path "../data/saved-models/communicate-sst-softmax/"  \
# --gpu-id 0

import argparse
import random

import numpy as np
import torch

from spec import iterator, models, explainers, laymen, constants
from spec.dataset import dataset, fields
from spec.dataset.corpora import available_corpora


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get nb of nonzero outputs of"
                                                 "sparse attentions.")
    parser.add_argument("--corpus",
                        type=str,
                        choices=list(available_corpora.keys()),
                        default='sst',
                        help="corpus type")
    parser.add_argument("--corpus-path",
                        type=str,
                        default=None,
                        help="path to the texts",
                        required=True)
    parser.add_argument("--load-model-path",
                        type=str,
                        default=None,
                        help="path to the saved model",
                        required=True)
    parser.add_argument("--load-explainer-path",
                        type=str,
                        default=None,
                        help="path to the saved explainer",
                        required=True)
    parser.add_argument("--lazy-loading",
                        type=bool,
                        default=True)
    parser.add_argument('--max-length',
                        type=int,
                        default=10 ** 12,
                        help='Maximum sequence length')
    parser.add_argument('--min-length',
                        type=int,
                        default=0,
                        help='Minimum sequence length.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='Maximum batch size for evaluating.')
    parser.add_argument('--gpu-id',
                        default=None,
                        type=int,
                        help='Use CUDA on the listed devices')
    args = parser.parse_args()
    args.lazy_loading = True
    set_seed(42)

    corpus_cls = available_corpora[args.corpus]
    fields_tuples = corpus_cls.create_fields_tuples()

    print('Building dataset...')
    text_dataset = dataset.build(args.corpus_path, fields_tuples, args)

    print('Building iterator...')
    dataset_iter = iterator.build(
        text_dataset, args.gpu_id, args.batch_size, is_train=False, lazy=True
    )

    print('Loading vocabularies...')
    fields.load_vocabs(args.load_model_path, fields_tuples)

    print('Loading model...')
    classifier = models.load(args.load_model_path, fields_tuples, args.gpu_id)

    print('Loading explainer...')
    explainer = explainers.load(
        args.load_explainer_path, fields_tuples, args.gpu_id
    )

    print('Calculating nonzeros...')
    classifier.eval()
    explainer.eval()

    tags = list(dataset_iter.dataset.fields['target'].vocab.stoi.keys())
    means_nonzero = dict(zip(tags, [[] for _ in range(len(tags))]))

    with torch.no_grad():
        for i, batch in enumerate(dataset_iter, start=1):
            _ = classifier.predict_classes(batch)
            attn_nonzero = classifier.attn_weights.squeeze() > 0
            nonpad = torch.ne(batch.words, constants.PAD_ID)
            attn_nonzero = (attn_nonzero & nonpad).float().sum(-1)
            # ratio:
            # attn_nonzero = attn_nonzero / mask.float().sum(-1)
            for m, c in zip(attn_nonzero.squeeze().tolist(),
                            batch.target.squeeze().tolist()):
                k = dataset_iter.dataset.fields['target'].vocab.itos[c]
                means_nonzero[k].append(m)

    all_means = []
    for c, m in means_nonzero.items():
        if len(m) > 0:
            all_means.extend(m)
            means = np.array(m)
            avg = means.mean()
            std = means.std()
            print('{}: {:.2f} ({:.2f})'.format(c, avg, std), end=' | ')
    print('')
    means = np.array(all_means)
    avg = means.mean()
    std = means.std()
    print('all: {:.2f} ({:.2f})'.format(avg, std))
