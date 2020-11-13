import sys
import random
from collections import Counter
from spec.dataset.corpora.snli import SNLICorpus
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
            prem, hypo = txt.split('|||')
            words.append((prem.strip().split(), hypo.strip().split()))
    return words


def label_color(ww):
    # template = '\x1b[0;37;41{}\x1b[0m'
    # return '**{}**'.format(ww)
    m = {'entailment': 'ENT', 'contradiction': 'CONT', 'neutral': 'NEU'}
    c = {'entailment': 'green', 'contradiction': 'red', 'neutral': 'cyan'}
    word = m[ww]
    color = c[ww]
    return colored(word, color, attrs=['bold'])


if __name__ == '__main__':

    is_random = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mismatch_n_gold_clf = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    mismatch_n_clf_lp = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    expl_paths = [
        "explanations-dev/snli_eraser_softmax_with_preds.txt",
        "explanations-dev/snli_grad_new_softmax_with_preds.txt",
        "explanations-dev/snli_encoded_attn_softmax_with_preds.txt",
        "explanations-dev/snli_encoded_attn_entmax15_with_preds.txt",
        "explanations-dev/snli_encoded_attn_sparsemax_with_preds.txt",
        "explanations-dev/snli_embedded_entmax15_with_preds.txt",
        "explanations-dev/snli_embedded_sparsemax_with_preds.txt",
        "explanations-dev/snli_bernoulli_sparsity0003_fixl_dev.txt",
        "explanations-dev/snli_latent_10pct_fixl_dev.txt",
        "explanations-dev/snli_posthoc_idf_scores_stop_ph_lb1_02_sparsemax_with_preds.txt",
    ]
    expl_names = [
        "eraser", "grad", "top-k softmax", "top-k entmax15", "top-k sparsemax",
        "embedded entmax15", "embedded sparsemax", "bernoulli", "hardkuma",
        "joint"
    ]

    corpus_path = 'human-corpus-dev/snli.txt'
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

        if any([len(l[i]) < 3 for l in e_labels]):
            continue

        gold_clf_mismatches = sum([int(l[i][0] != l[i][2]) for l in e_labels])
        clf_lp_mismatches = sum([int(l[i][2] != l[i][1]) for l in e_labels])

        if gold_clf_mismatches > mismatch_n_gold_clf:
            continue
        if clf_lp_mismatches < mismatch_n_clf_lp:
            continue

        print('-'*120)
        print("### Example {}".format(idxs[i]))
        all_expls = []
        for name, label, expl in zip(expl_names, e_labels, e_expls):
            label = label[i]
            expl = expl[i]
            if name not in ['bernoulli', 'embedded entmax15']:
                all_expls.extend(expl.split())
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
        prem, hypo = w

        print('Prem: ', ' '.join([
            colored(ww, 'magenta', attrs=['bold']) if ww in all_expls else ww
            for ww in prem
        ]))
        print('Hypo: ', ' '.join(hypo))
        print('')
        j += 1
        if j % top_k == 0:
            input("press [ENTER] to see more or [CTRL C] to quit\n")
