""""
Script to split yelp corpus into train, test and dev.
"""
import random
import sys


if __name__ == '__main__':
    seed = 42
    random.seed(seed)

    nb_train = 500000
    nb_test = 50000
    nb_dev = 50000
    shuffle = True
    file_input_path = sys.argv[1]
    train_output_path = sys.argv[2]
    test_output_path = sys.argv[3]
    dev_output_path = sys.argv[4]

    nb_lines = 0
    labels = ['1', '2', '3', '4', '5']
    idxs = {lab: [] for lab in labels}
    pattern = '"stars":'
    with open(file_input_path, 'r', encoding='utf8') as f:
        for line in f:
            pos = line.find(pattern)
            i, j = pos + len(pattern), pos + len(pattern) + 3
            stars = int(float(line[i:j]))
            idxs[str(stars)].append(nb_lines)
            nb_lines += 1

    train_indexes = []
    test_indexes = []
    dev_indexes = []
    for lab in labels:
        random.shuffle(idxs[lab])
        i, j = 0, nb_train // len(labels)
        train_indexes.extend(idxs[lab][i:j])
        i, j = j, j + nb_test // len(labels)
        test_indexes.extend(idxs[lab][i:j])
        i, j = j, j + nb_dev // len(labels)
        dev_indexes.extend(idxs[lab][i:j])

    train_indexes = set(train_indexes)
    test_indexes = set(test_indexes)
    dev_indexes = set(dev_indexes)

    print('Nb examples')
    print(nb_lines)
    print('Nb examples to train | test | dev')
    print(len(train_indexes), len(test_indexes), len(dev_indexes))

    f_train = open(train_output_path, 'w', encoding='utf8')
    f_test = open(test_output_path, 'w', encoding='utf8')
    f_dev = open(dev_output_path, 'w', encoding='utf8')
    idx = 0
    with open(file_input_path, 'r', encoding='utf8') as f:
        for line in f:
            print('{}/{}'.format(idx, nb_lines), end='\r')
            # ele = json.loads(line.strip())
            if idx in test_indexes:
                f_test.write(line)
            elif idx in dev_indexes:
                f_dev.write(line)
            elif idx in train_indexes:
                f_train.write(line)
            idx += 1
    f_train.close()
    f_test.close()
    f_dev.close()
