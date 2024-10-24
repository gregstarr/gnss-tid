import numpy as np
from matplotlib import pyplot as plt


def plot_radial(ax, center, wavelength, offset):
    for radius in np.arange(1, 1200, wavelength):
        circle = plt.Circle(center, radius + offset, facecolor='none', edgecolor=(0, 0, 0), linewidth=1, alpha=0.5)
        ax.add_patch(circle)


def plot_radial2(ax, center, wavelength, wavelength2, offset):
    for i in range(4):
        radius = np.sqrt(wavelength) * np.tan(i * np.sqrt(wavelength * wavelength2)) / wavelength2
        circle = plt.Circle(center, radius + offset, facecolor='none', edgecolor=(0, 0, 0), linewidth=1, alpha=0.5)
        ax.add_patch(circle)


def plot_points(ax, data, clim=(-.3, .3)):
    x = np.nanmean(data.x, axis=0)
    y = np.nanmean(data.y, axis=0)
    tid = np.nanmean(data.tid, axis=0)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(tid))
    x = x[mask]
    y = y[mask]
    tid = tid[mask]
    ax.scatter(x, y, c=tid, s=6, vmin=clim[0], vmax=clim[1], cmap='bwr')
    ax.grid(True)
    ax.set_xlim(np.nanmin(data.x), np.nanmax(data.x))
    ax.set_ylim(np.nanmin(data.y), np.nanmax(data.y))
