import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_circles(center, wavelength, offset, ax=None, data=None):
    if ax is None:
        ax = plt.gca()
    if data is not None:
        data.plot.scatter(ax=ax, x="x", y="y", hue="tec", vmax=.3, add_colorbar=False)
        maxr = min(np.max(data.x.values - center[0]), np.max(data.y.values - center[1])) * .9
    else:
        maxr = 1200
    ax.plot(center[0], center[1], 'k+')
    for radius in np.arange((-1 * offset) % wavelength, maxr, wavelength):
        circle = plt.Circle(center, radius, facecolor='none', edgecolor=(0, 0, 0), linewidth=1, alpha=0.5)
        ax.add_patch(circle)


def plot_patches(data, img=True, ax=None, scale_base=5, width=.006):
    if ax is None:
        ax = plt.gca()
    kmag = np.hypot(data.Fx, data.Fy)
    kscale = data.F / kmag.where(lambda x: x > 0, 1)
    data = data.assign(K=kmag, vx=data.Fx*kscale, vy=data.Fy*kscale)
    if img:
        data.image.plot(ax=ax, vmax=.3)
    # directions
    scale = scale_base * data.kx.shape[0] / 32
    data.plot.quiver(
        ax=ax, x="px", y="py", u="vx", v="vy", hue="K", cmap="bone",
        headwidth=0, headlength=0, headaxislength=0, add_guide=False, 
        scale=scale, width=width, vmin=0, vmax=.01, pivot="mid", scale_units="xy",
        angles="xy",
    )
