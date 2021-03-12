import numpy as np
import matplotlib.pyplot as plt
#
import torch
from torch import nn


def plot_batch(x):
    if type(x) == torch.Tensor:
        x = x.cpu().numpy()

    n_cols = x.shape[0]
    fig, axs = plt.subplots(1, n_cols)
    for idx in range(x.shape[0]):
        ax = axs[idx]
        img = x[idx]
        ax.imshow(img, cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def get_latent_img(fn_decode, n=30, d_z=2, random_samples=False, device=None):
    x_limit = np.linspace(-2, 2, n)
    y_limit = np.linspace(-2, 2, n)
    #
    img = np.empty((28 * n, 28 * n))
    #
    for i, zi in enumerate(x_limit):
        for j, pi in enumerate(y_limit):

            if random_samples:
                latent_layer = np.random.normal(0, 1, size=[1, d_z])
            else:
                latent_layer = np.array([[zi, pi]])

            latent_layer = torch.Tensor(latent_layer)
            latent_layer = latent_layer.to(device)
            with torch.no_grad():
                x_gen = fn_decode(latent_layer)
            x_gen = x_gen.cpu().numpy()
            img[(n - i - 1) * 28:(n - i) * 28,
                j * 28:(j + 1) * 28] = x_gen[0].reshape((28, 28))
    return img


def scatter_with_legend(Z, Y):
    #
    assert Z.shape[1] == 2
    #
    x = Z[:, 0]
    y = Z[:, 1]
    classes = Y

    unique = np.unique(classes)
    colors = [plt.cm.jet(i / float(len(unique) - 1))
              for i in range(len(unique))]
    for i, u in enumerate(unique):
        xi = [x[j] for j in range(len(x)) if classes[j] == u]
        yi = [y[j] for j in range(len(x)) if classes[j] == u]
        plt.scatter(xi, yi, color=colors[i], label=u, alpha=0.5)
    plt.legend()

    plt.show()


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
