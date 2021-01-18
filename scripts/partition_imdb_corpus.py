""""
Script to split imdb corpus into train and dev.
test_split_ratio = 0.1
"""
import random
import sys
from itertools import chain
from pathlib import Path

if __name__ == '__main__':
    seed = 42
    random.seed(seed)

    test_split_ratio = 0.1
    shuffle = True
    file_input_path = sys.argv[1]
    train_output_path = sys.argv[2]
    test_output_path = sys.argv[3]

    nb_lines = 0
    neg_files = sorted(Path(file_input_path, 'neg').glob('*.txt'))
    pos_files = sorted(Path(file_input_path, 'pos').glob('*.txt'))
    paths = chain(neg_files, pos_files)
    new_file_path = Path(file_input_path, 'data.txt')
    new_file = new_file_path.open('w', encoding='utf8')
    for file_path in paths:
        content = file_path.read_text().strip()
        content = content.replace('<br>', ' <br> ')
        content = content.replace('<br >', ' <br> ')
        content = content.replace('<br />', ' <br> ')
        content = content.replace('<br/>', ' <br> ')
        label = '1' if '/pos/' in str(file_path) else '0'
        new_file.write(label + ' ' + content + '\n')
        nb_lines += 1
    new_file.seek(0)
    new_file.close()

    data_indexes = list(range(nb_lines))
    random.shuffle(data_indexes)
    split_idx = int((1 - test_split_ratio) * len(data_indexes))
    train_indexes = set(data_indexes[:split_idx])
    test_indexes = set(data_indexes[split_idx:])

    print('Nb examples')
    print(len(data_indexes))
    print('Nb examples to train | test')
    print(len(train_indexes), len(test_indexes))

    f_train = open(train_output_path, 'w', encoding='utf8')
    f_test = open(test_output_path, 'w', encoding='utf8')
    idx = 0
    with open(new_file_path, 'r', encoding='utf8') as f:
        for line in f:
            print('{}/{}'.format(idx, nb_lines), end='\r')
            # ele = json.loads(line.strip())
            if idx in test_indexes:
                f_test.write(line)
            else:
                f_train.write(line)
            idx += 1
    f_train.close()
    f_test.close()
