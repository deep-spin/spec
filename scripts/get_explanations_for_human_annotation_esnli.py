# example:
# --------
# python3 get_explanations_for_human_annotation.py \
# --corpus sst \
# --text-path "../data/human-corpus/sst.txt"  \
# --load-model-path "../data/saved-models/test-sst-softmax/" \
# --load-explainer-path "../data/saved-models/communicate-sst-softmax/"  \
# --output-path "../data/explanations/sst_tmp.txt"  \
# --gpu-id 0

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from spec import iterator, models, explainers, laymen
from spec.dataset import dataset, fields
from spec.dataset.corpora import available_corpora


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get examples for humans")
    parser.add_argument("--corpus",
                        type=str,
                        choices=list(available_corpora.keys()),
                        default='sst',
                        help="corpus type")
    parser.add_argument("--text-path",
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
    parser.add_argument("--output-path",
                        type=str,
                        default=None,
                        help="path to the output file",
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
    parser.add_argument('--nb-explanations',
                        type=int,
                        default=10,
                        help='The number of explanations (will override the'
                             'explainer-attn-top-k in communication config)')
    parser.add_argument('--random-type',
                        type=str,
                        default='shuffle',
                        help='The type of random expliner. Only useful if the '
                             'load explainer is random_attn (will override the'
                             'explainer-attn-top-k in communication config)')
    args = parser.parse_args()
    args.lazy_loading = True

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    corpus_cls = available_corpora[args.corpus]
    fields_tuples = corpus_cls.create_fields_tuples()
    text_tuples = list(filter(lambda x: x[0] != 'target', fields_tuples))

    print('Reading text file...')
    texts = []
    texts_b = []
    texts_c = []
    targets = []
    text_file = Path(args.text_path)
    with text_file.open('r', encoding='utf8') as f:
        for i, line in enumerate(f):
            target, words = line.strip().split('\t')
            targets.append(target)

            words_a, words_b, words_c = words.split('|||')
            texts.append(words_a.strip())
            texts_b.append(words_b.strip())
            if words_c.strip() != '':
                idxs = list(map(int, words_c.strip().split()))
                texts_c.append([idxs])
            else:
                texts_c.append([[]])

            # if i + 1 == 200:
            #     break

    print('Building dataset...')
    texts_abc = [texts, texts_b, texts_c]
    text_dataset = dataset.build_pair_texts_with_marks(texts_abc,
                                                       text_tuples,
                                                       args)

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
    explainer.explainer_attn_top_k = args.nb_explanations

    print('Loading layman...')
    layman = laymen.load(
        args.load_explainer_path, fields_tuples, explainer.get_output_size(),
        args.gpu_id
    )

    print('Getting explanations...')
    explanations = []
    predictions_classifier = []
    predictions_layman = []
    classifier.eval()
    explainer.eval()

    def to_labels(preds):
        return [classifier.fields_dict['target'].vocab.itos[p] for p in preds]

    with torch.no_grad():
        for i, batch in enumerate(dataset_iter, start=1):
            pred_classes_clf = classifier.predict_classes(batch)
            message = explainer(batch, classifier)
            pred_classes_layman = layman.predict_classes(batch, message)
            for pred_clf, pred_layman in zip(pred_classes_clf,
                                             pred_classes_layman):
                if not isinstance(pred_clf, list):
                    pred_clf = [pred_clf]
                if not isinstance(pred_layman, list):
                    pred_layman = [pred_layman]
                predictions_classifier.extend(to_labels(pred_clf))
                predictions_layman.extend(to_labels(pred_layman))
            for wids in explainer.valid_top_word_ids:
                ws = [text_dataset.fields['words'].vocab.itos[w] for w in wids]
                explanations.append(ws)

    print('Saving explanations...')
    output_file = Path(args.output_path)
    with output_file.open('w', encoding='utf8') as f:
        for true_label, words, pred_l, pred_c in zip(targets,
                                                     explanations,
                                                     predictions_layman,
                                                     predictions_classifier):
            line = '{}\t{}\t{}\t{}\n'.format(
                true_label, pred_l, pred_c, ' '.join(words)
            )
            f.write(line)

    print('Communication acc for these explanations:')
    print('{:.4f}'.format(accuracy_score(predictions_classifier,
                                         predictions_layman)))

    print('Layman acc for these explanations:')
    print('{:.4f}'.format(accuracy_score(targets, predictions_layman)))

    print('Classifier acc for these explanations:')
    print('{:.4f}'.format(accuracy_score(targets, predictions_classifier)))
