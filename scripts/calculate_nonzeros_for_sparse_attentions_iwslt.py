# example:
# --------
# python3 calculate_nonzeros_for_sparse_attentions_iwslt.py \
# --corpus iwslt \
# --corpus-path "../data/saved-translation-models/iwslt-ende-bahdanau-softmax-new/attn-test/test"  \
# --gpu-id 1

import argparse
import random

import numpy as np
import torch

from spec import iterator, constants
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
    args.embeddings_format = None
    args.vocab_size = 99999999
    args.vocab_min_frequency = 0
    set_seed(42)

    corpus_cls = available_corpora[args.corpus]
    fields_tuples = corpus_cls.create_fields_tuples()

    print('Building dataset...')
    text_dataset = dataset.build(args.corpus_path, fields_tuples, args)

    print('Building iterator...')
    dataset_iter = iterator.build(
        text_dataset, args.gpu_id, args.batch_size, is_train=False, lazy=True
    )

    print('Building vocabularies...')
    fields.build_vocabs(fields_tuples, text_dataset, [text_dataset], args)

    print('Calculating nonzeros...')
    lens_src = []
    lens_hyp = []
    lens_trg = []
    lens_att = []

    with torch.no_grad():
        for i, batch in enumerate(dataset_iter, start=1):

            bs = batch.words.shape[0]
            src_len = batch.words.shape[-1]
            hyp_len = batch.words_hyp.shape[-1]
            clf_attn_weights = batch.attn[:, :src_len, :hyp_len]

            mask = torch.ne(batch.words, constants.PAD_ID)
            lens_src.extend(mask.int().sum(-1).tolist())

            mask = torch.ne(batch.words_hyp, constants.TARGET_PAD_ID)
            lens_hyp.extend(mask.int().sum(-1).tolist())

            mask = torch.ne(batch.target, constants.TARGET_PAD_ID)
            lens_trg.extend(mask.int().sum(-1).tolist())

            # (bs, source, target) -> (bs, target, source)
            clf_attn_weights = clf_attn_weights.transpose(1, 2)
            attn_nonzero = clf_attn_weights.squeeze() > 0
            nonpad = torch.ne(batch.words, constants.PAD_ID)
            nonpad = nonpad.unsqueeze(1).expand(-1, hyp_len, -1)
            mask = (attn_nonzero & nonpad)
            # import ipdb; ipdb.set_trace()
            lens_att.extend(mask.float().sum(-1).mean(-1).tolist())

    def calc_and_print_stats(name, lens):
        arr = np.array(lens)
        nb = int(arr.sum())
        avg = arr.mean()
        std = arr.std()
        print('{}: {} words | {:.2f} ({:.2f}) sents'.format(name, nb, avg, std))

    calc_and_print_stats('src', lens_src)
    calc_and_print_stats('hyp', lens_hyp)
    calc_and_print_stats('trg', lens_trg)
    calc_and_print_stats('att', lens_att)
