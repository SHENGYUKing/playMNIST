# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """load_mnist
    Load MNIST data from path and save as numpy.ndarray(label, data...)
    Args:
        path (filepath): where the dataset file is
        kind (string): "%s" means the file with "%s"
    """
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8).reshape(-1, )

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        # one picture(28x28) per line(28x28=784)

    return images, labels


def sample_show(x_train, y_train, mode=0, n=0):

    if mode == 0:
        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(10):
            img = x_train[y_train == i][0].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    else:
        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(25):
            img = x_train[y_train == n][i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
