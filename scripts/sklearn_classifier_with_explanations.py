import argparse

import numpy as np
from sklearn.feature_extraction.text import (TfidfVectorizer, CountVectorizer,
                                             HashingVectorizer)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, \
    precision_recall_fscore_support


def read_data(path):
    x, y = [], []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            tks = line.split()
            y.append(tks[0])
            x.append(' '.join(tks[1:]))
    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sklearn classifier")
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

    x_train, y_train = read_data(args.train_path)
    y_train = np.array(y_train)

    if args.feature == 'bow':
        vectorizer = CountVectorizer(lowercase=False)
        features_train = vectorizer.fit_transform(x_train)
    elif args.feature == 'bow':
        vectorizer = TfidfVectorizer(lowercase=False)
        features_train = vectorizer.fit_transform(x_train)
    else:
        vectorizer = HashingVectorizer(lowercase=False, n_features=2000)
        features_train = vectorizer.fit_transform(x_train)

    # classifier_linear = LogisticRegression(
    #     C=1000,
    #     max_iter=10000,
    #     solver='lbfgs',
    #     multi_class='multinomial',
    #     penalty='l2',
    #     random_state=seed,
    #     n_jobs=2
    # )
    classifier_linear = SGDClassifier(
        max_iter=50,
        alpha=0.0001,
        eta0=0.01,  # not used for learning_rate=`optimal`
        learning_rate='optimal',
        loss='hinge',
        penalty='l2',
        shuffle=True,
        random_state=seed,
        n_jobs=2
    )
    classifier_linear.fit(features_train, y_train)

    x_test, y_test = read_data(args.test_path)
    y_test = np.array(y_test)

    features_test = vectorizer.transform(x_test)

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

