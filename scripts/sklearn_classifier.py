import argparse

import numpy as np
from sklearn.feature_extraction.text import (TfidfVectorizer, CountVectorizer,
                                             HashingVectorizer)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, \
    precision_recall_fscore_support

from spec.dataset.corpora import available_corpora

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sklearn classifier")
    parser.add_argument("--corpus",
                        type=str,
                        choices=list(available_corpora.keys()),
                        default='sst',
                        help="corpus type",
                        required=True)
    parser.add_argument("--train-path",
                        type=str,
                        default=None,
                        help="path to the train corpus",
                        required=True)
    parser.add_argument("--test-path",
                        type=str,
                        default=None,
                        help="path to the test corpus",
                        required=True)
    parser.add_argument("--feature",
                        type=str,
                        default="bow",
                        choices=['bow', 'tfidf', 'hash'],
                        help="features format")
    args = parser.parse_args()

    seed = 42
    np.random.seed(42)

    print('Reading train data...')
    corpus_cls = available_corpora[args.corpus]
    fields_tuples = corpus_cls.create_fields_tuples()
    fields_dict = dict(fields_tuples)
    corpus = corpus_cls(fields_tuples, lazy=True)
    examples = corpus.read(args.train_path)
    x_train, y_train = [], []
    for ex in examples:
        y_train.extend(ex.target)
        text = ' '.join(ex.words)
        if args.corpus == 'snli':
            text = text + ' ' + ' '.join(ex.words_hyp)
        x_train.append(text)
    corpus.close()
    y_train = np.array(y_train)

    print('Vectorizing train data...')
    if args.feature == 'bow':
        vectorizer = CountVectorizer(lowercase=False)
        features_train = vectorizer.fit_transform(x_train)
    elif args.feature == 'bow':
        vectorizer = TfidfVectorizer(lowercase=False)
        features_train = vectorizer.fit_transform(x_train)
    else:
        vectorizer = HashingVectorizer(lowercase=False, n_features=2000)
        features_train = vectorizer.fit_transform(x_train)

    print('Training...')
    # classifier_linear = LogisticRegression(
    #     C=1000,
    #     max_iter=1000,
    #     solver='lbfgs',
    #     multi_class='multinomial',
    #     penalty='l2',
    #     random_state=seed,
    #     n_jobs=2
    # )
    classifier_linear = SGDClassifier(
        max_iter=50,
        alpha=0.00001,  # 0.0001
        eta0=0.001,  # not used for learning_rate=`optimal`
        learning_rate='constant',
        loss='hinge',
        penalty='l2',
        shuffle=True,
        random_state=seed,
        n_jobs=8,
        verbose=1
    )
    classifier_linear.fit(features_train, y_train)

    print('Reading test data...')
    corpus = corpus_cls(fields_tuples, lazy=True)
    examples = corpus.read(args.test_path)
    x_test, y_test = [], []
    for ex in examples:
        y_test.extend(ex.target)
        text = ' '.join(ex.words)
        if args.corpus == 'snli':
            text = text + ' ' + ' '.join(ex.words_hyp)
        x_test.append(text)
    corpus.close()
    y_test = np.array(y_test)

    print('Vectorizing test data...')
    features_test = vectorizer.transform(x_test)

    print('Predicting...')
    y_train_pred = classifier_linear.predict(features_train)
    y_test_pred = classifier_linear.predict(features_test)

    print('Train')
    print('-----')
    acc = accuracy_score(y_train, y_train_pred)
    mcc = matthews_corrcoef(y_train, y_train_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_train, y_train_pred,
                                                       average='macro')
    print('Acc: {:.4f}'.format(acc))
    print('Prec: {:.4f}'.format(prec))
    print('Rec: {:.4f}'.format(rec))
    print('F1: {:.4f}'.format(f1))
    print('MCC: {:.4f}'.format(mcc))

    print('Test')
    print('-----')
    acc = accuracy_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_test_pred,
                                                       average='macro')
    print('Acc: {:.4f}'.format(acc))
    print('Prec: {:.4f}'.format(prec))
    print('Rec: {:.4f}'.format(rec))
    print('F1: {:.4f}'.format(f1))
    print('MCC: {:.4f}'.format(mcc))

