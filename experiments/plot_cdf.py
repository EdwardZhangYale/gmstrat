import numpy as np
import matplotlib.pyplot as plt

def plot_empirical_cdf(samples, ax=None, **plot_kwargs):
    """
    Plot the empirical CDF of a 1D array of float samples.

    Parameters
    ----------
    samples     : array-like of floats
    ax          : optional matplotlib Axes to plot onto
    **plot_kwargs: passed through to ax.step() (e.g. color, label, linewidth)
    """
    samples = np.sort(samples)
    n = len(samples)
    cdf = np.arange(1, n + 1) / n   # i/n for i = 1..n

    if ax is None:
        _, ax = plt.subplots()

    ax.step(samples, cdf, where="post", **plot_kwargs)

    return ax

def plot_empirical_cdfs(sampleses, ax=None, labels=None, **plot_kwargs):
    trials = len(sampleses)
    for i in range(trials):
        plot_empirical_cdf(sampleses[i], ax=ax, label=labels[i], **plot_kwargs)

    ax.set_xlabel("Value")
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')


folder_name = 'results/'
paramses = [(2000, 1000), (2500, 1000), (3000, 1000), (4000, 1000)]

fig, ax = plt.subplots()

epsilonses = []
labels = []

for params in paramses:
    epsilons = np.loadtxt(f'{folder_name}epsilons{params[0]}x{params[1]}-1000.txt')
    epsilonses.append(epsilons)
    labels.append(f'{params[0]}x{params[1]}')


plot_empirical_cdfs(epsilonses, ax=ax, labels=labels)
ax.legend(loc='lower right')
plt.show()
