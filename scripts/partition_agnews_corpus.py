""""
Script to split AgNews corpus into train and test.
test_split_ratio = 0.15 based on:
https://github.com/successar/AttentionExplanation/tree/master/preprocess
"""
import random
import sys
from xml.etree import ElementTree


if __name__ == '__main__':
    seed = 42
    random.seed(seed)

    test_split_ratio = 0.15
    shuffle = True
    valid_categories = ['Business', 'World']
    file_input_path = sys.argv[1]
    train_output_path = sys.argv[2]
    test_output_path = sys.argv[3]

    root = ElementTree.parse(file_input_path).getroot()
    if len(root) % 8 != 0:
        raise Exception('AgNews corpus should have 8 tags per line.')
    assert len(root) // 8 == 496835

    data = []
    all_categories = set()
    for i in range(len(root) // 8):
        instance = []
        is_valid = True
        for j in range(8):
            ele = root[i*8 + j]
            if ele.tag == 'category':
                all_categories.add(ele.text)
                if ele.text not in valid_categories:
                    is_valid = False
                    break
            elif ele.tag == 'description':
                if ele.text == '' or ele.text is None:
                    is_valid = False
                    break
            instance.append(ele)
        if is_valid:
            data.append(instance)

    for j in range(8):
        base = data[0][j].tag
        for instance in data:
            assert instance[j].tag == base

    print('All categories: ', all_categories)
    print('Selected categories: ', valid_categories)

    print('Nb examples | nb examples filtered by categories and null text')
    print(len(root), len(data))

    # shuffle inplace
    random.shuffle(data)

    # portion to the training set
    split_idx = int((1 - test_split_ratio) * len(data))
    print('Nb examples to train | test')
    print(split_idx, len(data)-split_idx)

    # save train and test sets
    train_file = open(train_output_path, 'w', encoding='utf8')
    train_file.write('<{}>\n'.format(root.tag))
    for instance in data[:split_idx]:
        s = ''.join(
            [ElementTree.tostring(ele, encoding='unicode') for ele in instance])
        train_file.write(s.strip() + '\n')
    train_file.write('</{}>\n'.format(root.tag))
    train_file.close()

    test_file = open(test_output_path, 'w', encoding='utf8')
    test_file.write('<{}>\n'.format(root.tag))
    for instance in data[split_idx:]:
        s = ''.join(
            [ElementTree.tostring(ele, encoding='unicode') for ele in instance])
        test_file.write(s.strip() + '\n')
    test_file.write('</{}>\n'.format(root.tag))
    test_file.close()

