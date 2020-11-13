from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

if __name__ == '__main__':

    max_seq_len = 2950
    nb_basis = 64
    b = 10000

    plt.subplot(1, 2, 1)
    x = np.linspace(0, 1, max_seq_len)
    Z = np.zeros((max_seq_len, nb_basis))
    for i in range(0, nb_basis, 2):
        angles = max_seq_len * x * 1. / (b ** (i / nb_basis))
        # angles = x * 1. / (freq ** (i / nb_basis))
        Z[:, i] = np.sin(angles)
        Z[:, i+1] = np.cos(angles)
    # plt.pcolormesh(Z.T[::-1], cmap='RdBu')
    plt.pcolormesh(Z.T, cmap='RdBu')
    plt.colorbar()
    plt.title('ours')

    plt.subplot(1, 2, 2)
    x = np.arange(max_seq_len)
    Z2 = np.zeros((max_seq_len, nb_basis))
    for i in range(0, nb_basis, 2):
        angles = x * 1. / (b ** (i / nb_basis))
        Z2[:, i] = np.sin(angles)
        Z2[:, i + 1] = np.cos(angles)
    plt.pcolormesh(Z2.T, cmap='RdBu')
    plt.colorbar()
    plt.title('transformer')

    # all_colors = list(mcolors.TABLEAU_COLORS)  # list of colors
    # for i in range(0, 4):
    #     c = all_colors[(i - 1) % len(all_colors)]
    #     angles = x * 1. / (freq ** (2 * i / nb_basis))
    #     plt.plot(x, np.sin(angles),
    #              '-',
    #              label='sin-{},{},{}'.format(freq, i, nb_basis),
    #              color=c)
    #     plt.plot(x, np.cos(angles),
    #              ':',
    #              label='cos-{},{},{}'.format(freq, i, nb_basis),
    #              color=c)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()
