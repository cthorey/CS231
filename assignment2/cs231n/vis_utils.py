from math import sqrt, ceil
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in xrange(grid_size):
        x0, x1 = 0, W
        for x in xrange(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1)
                  * W + x, :] = Xs[n, :, :, :]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1)
              * W + x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def inspect_checkpoint(checks):
    ''' solvers is a dict with key:solver.
    Return a plot with the different loss on of top of another '''

    fig = plt.figure(figsize=(10, 30))
    height = len(checks) + 1
    ax = plt.subplot(height, 1, 1)
    axs = [plt.subplot(height, 1, i + 2) for i in range(len(checks))]

    i = 0
    for key, check in checks.iteritems():
        ax.plot(check['loss_history'], label=key)
        ax.legend(fontsize=24)
        ax.tick_params(labelsize=24)
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')

        axs[i].plot(check['train_acc_history'])
        axs[i].plot(check['val_acc_history'])
        axs[i].legend(['train', 'val'], loc='upper left')
        axs[i].set_xlabel('epoch')
        axs[i].set_ylabel('accuracy')
        axs[i].set_title(key, fontsize=24)
        i += 1
