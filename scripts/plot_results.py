import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# plt.style.use('seaborn-whitegrid')
# plt.style.use('fivethirtyeight')
plt.style.use('seaborn-deep')

large = 18
med = 12
small = 10
params = {
    'legend.fontsize': small,
    'figure.figsize': (16, 10),
    'axes.labelsize': med,
    'axes.titlesize': med,
    'xtick.labelsize': small,
    'ytick.labelsize': small,
    'figure.titlesize': large,
}
plt.rcParams.update(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument("-p", "--data-path", type=str, help="path to the csv.")
    parser.add_argument("-o", "--output-path", type=str, help="save fig path.")
    parser.add_argument("--plot-layman", action='store_true')
    parser.add_argument("--plot-clfs", action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, sep=',')
    corpus = args.data_path.lower().split()[-1].split('.')[0]

    fig, ax = plt.subplots()
    # ax.grid(b=True, which='major')
    ax.set_axisbelow(True)

    # Show the major grid lines with dark grey lines
    ax.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.2)

    # Show the minor grid lines with very faint and almost transparent lines
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    # n = len(df)  # number of bars
    width = 0.25  # the width of the bars
    all_colors = list(plt.cm.colors.cnames.keys())  # list of colors

    accs_comm = df['ACC']
    accs_layman = df['ACC Layman']
    accs_clf = df['ACC Classifier']
    accs_bow = df['ACC BOW Classifier']

    # which acc will be plotted
    if args.plot_layman:
        accs = accs_layman
        ax.set_title('Layman Accuracy')
    else:
        accs = accs_comm
        ax.set_title('Communication Accuracy')

    print(corpus)

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

    # plot accs
    n = 0
    n += len(accs_encoded_attn_entmax)
    n += 1 + len(accs_gradient_softmax)
    n += 1 + len(accs_random_softmax)
    x = np.arange(n)  # the label locations

    rects1 = ax.bar(x+0*width,
                    accs_encoded_attn_softmax
                    + [0] + accs_gradient_softmax
                    + [0] + accs_random_softmax,
                    width=width,
                    # hatch='x',
                    linewidth=1,
                    edgecolor='#333333',
                    label='Softmax')

    rects2 = ax.bar(x+1*width,
                    accs_encoded_attn_sparsemax
                    + [0] + accs_gradient_sparsemax
                    + [0] + accs_random_sparsemax,
                    width=width,
                    # hatch='+',
                    linewidth=1,
                    edgecolor='#333333',
                    label='Sparsemax')

    rects3 = ax.bar(x+2*width,
                    accs_encoded_attn_entmax
                    + [0] + accs_gradient_entmax
                    + [0] + accs_random_entmax,
                    width=width,
                    # hatch='o',
                    linewidth=1,
                    edgecolor='#333333',
                    label='Entmax 1.5')

    if args.plot_clfs:
        ax.axhline(y=accs_clf[0],
                   xmin=0,
                   xmax=max(x) + 6 * width,
                   linestyle='solid',
                   color='#1a2d4a',
                   label='RNN classifier')
        ax.axhline(y=accs_clf[3],
                   xmin=0,
                   xmax=max(x) + 6 * width,
                   linestyle='solid',
                   color='#255228',
                   label='RNN classifier')
        ax.axhline(y=accs_clf[6],
                   xmin=0,
                   xmax=max(x) + 6 * width,
                   linestyle='solid',
                   color='#631e1e',
                   label='RNN classifier')
        ax.axhline(y=accs_bow[0],
                   xmin=0,
                   xmax=max(x) + 6 * width,
                   linestyle='dotted',
                   color='black',
                   label='BoW classifier')

    # manually for now
    if corpus == 'sst':
        ax.set_xticklabels(('$k=1$',
                            '$k=3$\nEncoded Attention',
                            '$k=5$',
                            '',
                            '$k=1$',
                            '$k=3$\nGradient * Emb',
                            '$k=5$',
                            '',
                            '$k=5$\nUniform',
                            '$k=5$\nBeta',
                            '$k=5$\nShuffle',
                            '$k=5$\n0-max',
                            '$k=5$\nFirst pos',
                            '$k=5$\nLast pos'))
    elif corpus == 'snli':
        ax.set_xticklabels(('$k=1$',
                            '$k=2$\nEncoded Attention',
                            '$k=4$',
                            '',
                            '$k=1$',
                            '$k=2$\nGradient * Emb',
                            '$k=4$',
                            '',
                            '$k=4$\nUniform',
                            '$k=4$\nBeta',
                            '$k=4$\nShuffle',
                            '$k=4$\n0-max',
                            '$k=4$\nFirst pos',
                            '$k=4$\nLast pos'))
    elif corpus in ['agnews', 'imdb']:
        ax.set_xticklabels(('$k=1$',
                            '$k=3$\nEncoded Attention',
                            '$k=5$',
                            '$k=10$',
                            '',
                            '$k=1$',
                            '$k=3$\nGradient * Emb',
                            '$k=5$',
                            '$k=10$',
                            '',
                            '$k=10$\nUniform',
                            '$k=10$\nBeta',
                            '$k=10$\nShuffle',
                            '$k=10$\n0-max',
                            '$k=10$\nFirst pos',
                            '$k=10$\nLast pos'))
    elif corpus in ['yelp']:
        ax.set_xticklabels(('$k=1$',
                            '$k=5$\nEncoded Attention',
                            '$k=10$',
                            '',
                            '$k=1$',
                            '$k=5$\nGradient * Emb',
                            '$k=10$',
                            '',
                            '$k=10$\nUniform',
                            '$k=10$\nShuffle',
                            '$k=10$\n0-max'))

    ax.set_xticks(x + width / 2)
    ax.set_xlabel('Top k\nExplainer')
    ax.set_ylabel('Acc.')
    ax.set_ylim(0.0, 1.0)
    # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(args.output_path)
    # plt.show()
