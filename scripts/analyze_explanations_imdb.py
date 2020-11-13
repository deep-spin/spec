import sys
import random
from collections import Counter
from spec.dataset.corpora.imdb import IMDBCorpus
from termcolor import colored

def read_labels_and_expls(expl_path):
    labels = []
    expls = []
    expls_vocab = Counter()
    with open(expl_path, 'r', encoding='utf8') as f:
        for line in f:
            tmp = line.strip().split('\t')
            label = tmp[:-1]
            expl = tmp[-1]
            labels.append(label)
            expls.append(expl)
            for x in expl.split():
                expls_vocab[x] += 1
    return labels, expls, expls_vocab


def get_words(path):
    words = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            label, txt = line.split('\t')
            words.append(txt.strip().split())
    return words


def label_color(ww):
    # template = '\x1b[0;37;41{}\x1b[0m'
    # return '**{}**'.format(ww)
    c = 'green' if ww == 'POS' else 'red'
    return colored(ww, c, attrs=['bold'])


if __name__ == '__main__':

    is_random = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mismatch_n_gold_clf = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    mismatch_n_clf_lp = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    expl_paths = [
        "explanations-dev/imdb_eraser_softmax_with_preds.txt",
        "explanations-dev/imdb_grad_new_softmax_with_preds.txt",
        "explanations-dev/imdb_encoded_attn_softmax_with_preds.txt",
        "explanations-dev/imdb_encoded_attn_entmax15_with_preds.txt",
        "explanations-dev/imdb_encoded_attn_sparsemax_with_preds.txt",
        "explanations-dev/imdb_embedded_entmax15_with_preds.txt",
        "explanations-dev/imdb_embedded_sparsemax_with_preds.txt",
        "explanations-dev/imdb_bernoulli_sparsity001_fixl_dev.txt",
        "explanations-dev/imdb_latent_10pct_fixl_dev.txt",
        "explanations-dev/imdb_posthoc_idf_scores_stop_ph_lb1_02_sparsemax_with_preds.txt",
    ]
    expl_names = [
        "eraser", "grad", "top-k softmax", "top-k entmax15", "top-k sparsemax",
        "embedded entmax15", "embedded sparsemax", "bernoulli", "hardkuma",
        "joint"
    ]

    corpus_path = 'human-corpus-dev/imdb.txt'
    words = get_words(corpus_path)

    e_expls = []
    e_labels = []
    idxs = None
    for expl_path in expl_paths:
        labels, expls, _ = read_labels_and_expls(expl_path)
        if idxs is None:
            idxs = list(range(len(labels)))
            if is_random:
                random.shuffle(idxs)
        labels = [labels[i] for i in idxs]
        expls = [expls[i] for i in idxs]
        e_expls.append(expls)
        e_labels.append(labels)

    words = [words[i] for i in idxs]
    j = 0
    for i, w in enumerate(words):

        gold_clf_mismatches = sum([int(l[i][0] != l[i][2]) for l in e_labels])
        clf_lp_mismatches = sum([int(l[i][2] != l[i][1]) for l in e_labels])
        if gold_clf_mismatches > mismatch_n_gold_clf:
            continue
        if clf_lp_mismatches < mismatch_n_clf_lp:
            continue

        print('-'*120)
        print("### Example {}".format(i))
        all_expls = []
        for name, label, expl in zip (expl_names, e_labels, e_expls):
            label = label[i]
            expl = expl[i]
            all_expls.extend(expl.split())
            label = ['POS' if int(l) == 1 else 'NEG' for l in label]
            # if is_mismatch and label[1] == label[2]:
            #     continue

            print('Y: {} | C: {} | L: {}   ({})'.format(
                label_color(label[0]),
                label_color(label[2]),
                label_color(label[1]),
                name.upper())
            )
            print('{}'.format(expl))
            print('')
        all_expls = set(all_expls)
        # print(all_expls)
        print(' '.join([
            colored(ww, 'magenta', attrs=['bold']) if ww in all_expls else ww
            for ww in w
        ]))
        print('')
        j += 1
        if j % top_k == 0:
            input("press [ENTER] to see more or [CTRL C] to quit\n")
