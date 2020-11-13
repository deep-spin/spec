import argparse

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument("-p", "--data-path", type=str, help="path to the csv.")
    parser.add_argument("-o", "--output-path", type=str, help="path to a txt.")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, sep=',')
    corpus = args.data_path.lower().split()[-1].split('.')[0]
    print(corpus)

    accs_comm = df['ACC']
    accs_layman = df['ACC Layman']
    accs_clf = df['ACC Classifier']
    accs_bow = df['ACC BOW Classifier']

    if corpus == 'sst':
        ks = [3, 3, 6]
    elif corpus == 'snli':
        ks = [3, 3, 6]
    elif corpus in ['agnews', 'imdb']:
        ks = [4, 4, 6]
    elif corpus in ['yelp']:
        ks = [3, 3, 3]
    else:
        raise Exception('corpus not found')

    acc_rnn_softmax = accs_clf[0]
    acc_rnn_sparsemax = accs_clf[3]
    acc_rnn_entmax = accs_clf[6]
    acc_bow = accs_bow[0]

    def get_accs(for_layman=False):
        # which acc will be plotted
        if for_layman:
            accs = accs_layman
        else:
            accs = accs_comm

        # get accs for each corpus (manually for now)
        if corpus in ['sst', 'snli']:
            accs_encoded_attn_softmax = [accs[0], accs[1], accs[2]]
            accs_encoded_attn_sparsemax = [accs[3], accs[4], accs[5]]
            accs_encoded_attn_entmax = [accs[6], accs[7], accs[8]]

            accs_gradient_softmax = [accs[9], accs[10], accs[11]]
            accs_gradient_sparsemax = [accs[12], accs[13], accs[14]]
            accs_gradient_entmax = [accs[15], accs[16], accs[17]]

            accs_random_softmax = [accs[18], accs[19], accs[20],
                                   accs[21], accs[22], accs[23]]
            accs_random_sparsemax = [accs[24], accs[25], accs[26],
                                     accs[27], accs[28], accs[29]]
            accs_random_entmax = [accs[30], accs[31], accs[32],
                                  accs[33], accs[34], accs[35]]

        elif corpus in ['agnews', 'imdb']:
            accs_encoded_attn_softmax = [accs[0], accs[1], accs[2], accs[3]]
            accs_encoded_attn_sparsemax = [accs[4], accs[5], accs[6], accs[7]]
            accs_encoded_attn_entmax = [accs[8], accs[9], accs[10], accs[11]]

            accs_gradient_softmax = [accs[12], accs[13], accs[14], accs[15]]
            accs_gradient_sparsemax = [accs[16], accs[17], accs[18], accs[19]]
            accs_gradient_entmax = [accs[20], accs[21], accs[22], accs[23]]

            accs_random_softmax = [accs[24], accs[25], accs[26],
                                   accs[27], accs[28], accs[29]]
            accs_random_sparsemax = [accs[30], accs[31], accs[32],
                                     accs[33], accs[34], accs[35]]
            accs_random_entmax = [accs[36], accs[37], accs[38],
                                  accs[39], accs[40], accs[41]]

        elif corpus in ['yelp']:
            accs_encoded_attn_softmax = [accs[0], accs[1], accs[2]]
            accs_encoded_attn_sparsemax = [accs[3], accs[4], accs[5]]
            accs_encoded_attn_entmax = [accs[6], accs[7], accs[8]]

            accs_gradient_softmax = [accs[9], accs[10], accs[11]]
            accs_gradient_sparsemax = [accs[12], accs[13], accs[14]]
            accs_gradient_entmax = [accs[15], accs[16], accs[17]]

            accs_random_softmax = [accs[18], accs[19], accs[20]]
            accs_random_sparsemax = [accs[21], accs[22], accs[23]]
            accs_random_entmax = [accs[24], accs[25], accs[26]]

        else:
            raise Exception('corpus not available: {}'.format(corpus))

        return (accs_encoded_attn_softmax,
                accs_encoded_attn_sparsemax,
                accs_encoded_attn_entmax,
                accs_gradient_softmax,
                accs_gradient_sparsemax,
                accs_gradient_entmax,
                accs_random_softmax,
                accs_random_sparsemax,
                accs_random_entmax)

    accs = get_accs(for_layman=False)
    accs_encoded_attn_softmax_c = accs[0]
    accs_encoded_attn_sparsemax_c = accs[1]
    accs_encoded_attn_entmax_c = accs[2]
    accs_gradient_softmax_c = accs[3]
    accs_gradient_sparsemax_c = accs[4]
    accs_gradient_entmax_c = accs[5]
    accs_random_softmax_c = accs[6]
    accs_random_sparsemax_c = accs[7]
    accs_random_entmax_c = accs[8]

    accs = get_accs(for_layman=True)
    accs_encoded_attn_softmax_l = accs[0]
    accs_encoded_attn_sparsemax_l = accs[1]
    accs_encoded_attn_entmax_l = accs[2]
    accs_gradient_softmax_l = accs[3]
    accs_gradient_sparsemax_l = accs[4]
    accs_gradient_entmax_l = accs[5]
    accs_random_softmax_l = accs[6]
    accs_random_sparsemax_l = accs[7]
    accs_random_entmax_l = accs[8]

    x = 'RNN baseline \t& $n$ \t& & - \t& - \t& - \t& & {:.4f} \t& {:.4f} \t& {:.4f} \\\\'
    print(x.format(acc_rnn_softmax, acc_rnn_sparsemax, acc_rnn_entmax))
    x = 'BoW baseline \t& $n$ \t& & - \t& - \t& - \t& & {:.4f} \t& {:.4f} \t& {:.4f} \\\\'
    print(x.format(acc_bow, acc_bow, acc_bow))

    print('\\midrule')

    def to_str(v):
        return ['%.4f' % x for x in v]

    for i in range(ks[0]):
        c1 = ' \t& '.join(to_str([
            accs_encoded_attn_softmax_c[i],
            accs_encoded_attn_sparsemax_c[i],
            accs_encoded_attn_entmax_c[i]
        ]))
        c2 = ' \t& '.join(to_str([
            accs_encoded_attn_softmax_l[i],
            accs_encoded_attn_sparsemax_l[i],
            accs_encoded_attn_entmax_l[i]
        ]))
        print('Encoded attention \t& {} \t& & {} \t& & {} \\\\'.format(i + 1, c1, c2))
    print('\\midrule')

    for i in range(ks[1]):
        c1 = ' \t& '.join(to_str([
            accs_gradient_softmax_c[i],
            accs_gradient_sparsemax_c[i],
            accs_gradient_entmax_c[i]
        ]))
        c2 = ' \t& '.join(to_str([
            accs_gradient_softmax_l[i],
            accs_gradient_sparsemax_l[i],
            accs_gradient_entmax_l[i]
        ]))
        print('Gradient based \t& {} \t& & {} \t& & {} \\\\'.format(i + 1, c1, c2))
    print('\\midrule')

    random_names = ['Uniform', 'Beta', 'Shuffle',
                    'Zero-max-out', 'First states', 'Last states']
    for i in range(ks[2]):
        c1 = ' \t& '.join(to_str([
            accs_random_softmax_c[i],
            accs_random_sparsemax_c[i],
            accs_random_entmax_c[i]
        ]))
        c2 = ' \t& '.join(to_str([
            accs_random_softmax_l[i],
            accs_random_sparsemax_l[i],
            accs_random_entmax_l[i]
        ]))
        name = random_names[i]
        print('{} \t& {} \t& & {} \t& & {} \\\\'.format(name, i + 1, c1, c2))
    print('\\midrule')






