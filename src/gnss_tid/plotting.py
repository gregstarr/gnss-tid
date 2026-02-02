import numpy as np
import xarray
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.animation as animation
from scipy.ndimage import maximum_filter1d
from tqdm import autonotebook

from .utils import find_center


def plot_circles(center, wavelength, offset, ax=None, data=None, maxr=1200):
    if np.isnan(wavelength) or np.isnan(offset) or np.any(np.isnan(center)):
        return []
    if ax is None:
        ax = plt.gca()
    artists = []
    if data is not None and maxr is None:
        artists.append(
            data.plot.scatter(ax=ax, x="x", y="y", hue="tec", vmax=.3, add_colorbar=False)
        )
        maxr = min(np.max(data.x.values - center[0]), np.max(data.y.values - center[1])) * .9
    elif maxr is None:
        raise ValueError("specify maxr")
    artists += ax.plot(center[0], center[1], 'k+')
    for radius in np.arange((-1 * offset) % wavelength, maxr, wavelength):
        circle = plt.Circle(center, radius, facecolor='none', edgecolor=(0, 0, 0), linewidth=1, alpha=0.5)
        artists.append(ax.add_patch(circle))
    return artists


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

def plot_center_finder(data, scale=5):
    X, Y = np.meshgrid(data.px.values, data.py.values)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    weights = np.log10(data.F.values.ravel() + 1)
    vectors = np.column_stack((data.Fx.values.ravel(), data.Fy.values.ravel()))
    cw = find_center(pts, vectors, weights)
    cu = find_center(pts, vectors, np.ones_like(weights))
    XD, YD = np.meshgrid(data.x.values, data.y.values)
    tp = np.column_stack([XD.ravel(), YD.ravel()])

    vec_norm = np.linalg.norm(vectors, axis=1)
    mask = vec_norm > 0
    x = tp[:, 0][:, None]
    y = tp[:, 1][:, None]
    x1 = pts[mask, 0][None, :]
    y1 = pts[mask, 1][None, :]
    vx = vectors[mask, 0][None, :]
    vy = vectors[mask, 1][None, :]
    # Numerator: |vx * (y - y1) - vy * (x - x1)|
    numerator = np.abs(vx * (y - y1) - vy * (x - x1))
    # Denominator: sqrt(vx^2 + vy^2)
    denominator = np.sqrt(vx**2 + vy**2)
    # Distance matrix: D[i, j]
    distances = numerator / denominator
    dw = np.sum(weights[None, mask] * np.exp(-(distances/100)**2), axis=1).reshape(XD.shape)
    
    cg = tp[np.argmax(dw)]

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))
    plot_patches(data, ax=ax[0], img=True, scale_base=scale)
    ax[0].plot(cw[0], cw[1], 'x')
    ax[0].plot(cu[0], cu[1], 'x')
    ax[0].plot(cg[0], cg[1], 'x')
    
    ax[1].pcolormesh(XD, YD, dw)
    ax[1].plot(cw[0], cw[1], 'x')
    ax[1].plot(cu[0], cu[1], 'x')
    ax[1].plot(cg[0], cg[1], 'x')
    plot_patches(data, ax=ax[1], img=False, scale_base=scale)
    print(f"lsq unweighted: {cu}, lsq weighted {cw}, grid search: {cg}")
    return fig, ax


def plot_center_finder_fit(result_list, s=4, c0=None, cbox=None):
    fig, ax = plt.subplots(2, 4, figsize=(4*s, 2*s), tight_layout=True, sharex="col")

    best_iteration = np.argmax([r["metric"] for r in result_list])
    for ii, result in enumerate(result_list):
        ax[0, 0].plot(result["history"]["metric"])
        z = np.array(result["history"]["center"])
        ax[0, 2].plot(z[:, 0], z[:, 1], '-')
        if ii == best_iteration:
            ax[0, 2].plot(z[-1, 0], z[-1, 1], 'r.')
        else:
            ax[0, 2].plot(z[-1, 0], z[-1, 1], 'k.')
        ax[1, 0].semilogy(maximum_filter1d(np.diff(result["history"]["metric"]), 9))
    
    if c0 is not None:
        ax[0, 2].plot(c0[0], c0[1], 'kx')
    if cbox is not None:
        pts = np.array([
            [cbox[0], cbox[1]],
            [cbox[0], cbox[1] + cbox[3]],
            [cbox[0] + cbox[2], cbox[1] + cbox[3]],
            [cbox[0] + cbox[2], cbox[1]],
            [cbox[0], cbox[1]],
        ])
        ax[0, 2].plot(pts[:, 0], pts[:, 1], 'k--')

    result = result_list[best_iteration]
    z = np.array(result["history"]["wavelength"])
    ax[0, 1].plot(z, "-")
    z = np.unwrap(np.array(result["history"]["phase"]), axis=0)
    ax[1, 1].plot(z, "-")
    ax[1, 3].plot(result["offset"])
    ax[0, 3].plot(result["wavelength"])

    ax[0, 0].set_title("metric")
    ax[1, 0].set_xlabel("step")
    ax[0, 1].set_title("lambda, phase")
    ax[1, 1].set_xlabel("step")
    ax[0, 2].set_title(f"center={result['history']['center'][-1]}")
    ax[1, 2].set_xlabel("x")
    ax[0, 3].set_title("lambda, phase")
    ax[1, 3].set_xlabel("time")

    for row in ax:
        for a in row:
            a.grid(True)

    return fig, ax


def make_animation(data, save_fn, limit=-1, writer="pillow", maxr=2400):
    with plt.style.context("bmh"):
        layout = [
            ["A", "B"],
            ["C", "D"],
            ["E", "E"],
        ]
        fig, ax = plt.subplot_mosaic(
            layout,
            figsize=(10, 10),
            height_ratios=[10, 10, 22],
            tight_layout=True,
        )

        animation_artists = []
        if limit <= 0:
            limit = data.time.shape[0]
        for ii in autonotebook.tqdm(range(limit), "creating frames"):
            frame_artists = []
            frame_artists += data.height.plot(ax=ax["A"], c="blue")
            frame_artists += data.wavelength.plot(ax=ax["B"], c="blue")
            frame_artists += data.objective.plot(ax=ax["C"], c="blue")
            frame_artists += data.offset.plot(ax=ax["D"], c="blue")
            frame_artists += [
                ax["A"].axvline(data.time.values[ii], color="k", linestyle='--'),
                ax["B"].axvline(data.time.values[ii], color="k", linestyle='--'),
                ax["C"].axvline(data.time.values[ii], color="k", linestyle='--'),
                ax["D"].axvline(data.time.values[ii], color="k", linestyle='--'),
                data.isel(time=ii).image.drop_vars("time").plot(ax=ax["E"], vmax=.3, add_colorbar=False)
            ]

            frame_artists += plot_circles(
                data.center,
                data.wavelength.isel(time=ii),
                data.offset.isel(time=ii),
                ax=ax["E"], maxr=maxr,
            )

            animation_artists.append(frame_artists)

        ani = animation.ArtistAnimation(fig=fig, artists=animation_artists, interval=200)
        with autonotebook.tqdm(total=len(animation_artists), desc='Saving video') as progress_bar:
            ani.save(
                filename=save_fn, 
                writer=writer,
                progress_callback=lambda _i, _n: progress_bar.update()
            )


def plot_param_lines(data, lines_tec, time_idx=15):
    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))

    data.image.isel(time=time_idx).plot(ax=ax[0], x="x", y="y")
    ax[0].plot(data.center.values[0], data.center.values[1], '.r')
    lines_tec.isel(time=time_idx).y.plot.line(ax=ax[0], x="x", hue="az", add_legend=False)
    lines_tec.isel(time=time_idx).plot.line(ax=ax[1], x="r", hue="az")
    return fig, ax


def plot_param_fft_lines(data, tec, time_idx=15, az_idx=None, r_idx=None):
    if az_idx is None:
        az_idx = tec.az.shape[0] // 2
    if r_idx is None:
        r_idx = tec.r.shape[0] // 2
        dr = tec.r.shape[0] // 10
        r_idx = [r_idx - dr, r_idx, r_idx + dr]
    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))
    data.image.isel(time=time_idx).plot(ax=ax[0], x="x", y="y")
    ax[0].plot(data.center.values[0], data.center.values[1], '.r')

    if tec.isel(time=time_idx, az=az_idx).y.ndim == 2:
        t = tec.isel(time=time_idx, az=az_idx)
        ax[0].plot(t.x, t.sel(f=0, method="nearest").y)
        da = t.isel(r=r_idx)
        x_b, _ = xarray.broadcast(da.x, da.y)
        da = da.assign_coords(x=x_b)
        da.y.plot.line(ax=ax[0], x="x", hue="r")
        t.isel(r=r_idx).plot(ax=ax[1], x="f", hue="r")
    elif tec.isel(time=time_idx, az=az_idx).x.ndim == 2:
        t = tec.isel(time=time_idx, az=az_idx)
        ax[0].plot(t.sel(f=0, method="nearest").x, t.y)
        da = t.isel(r=r_idx)
        _, y_b = xarray.broadcast(da.x, da.y)
        da = da.assign_coords(y=y_b)
        da.y.plot.line(ax=ax[0], x="x", hue="r")
        t.isel(r=r_idx).plot(ax=ax[1], x="f", hue="r")
    else:
        raise Exception("neither x or y has 2 dimensions")
    return fig, ax


def plot_param_spatial_fft(
        S: xarray.DataArray,
        time_idx: int = 15,
        az_idx: int = None,
        ax: Axes = None
    ):
    if az_idx is None:
        az_idx = S.az.shape[0] // 2
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 5))
    else:
        fig = ax.figure
    S.isel(time=time_idx, az=az_idx).plot(ax=ax)
    if "frequency" in S.coords:
        Sf = S.isel(frequency=S.argmax("frequency"))
        Sf.isel(time=time_idx, az=az_idx).frequency.plot(ax=ax, y="r", c="w")
    else:
        Sf = S.isel(f=S.argmax("f"))
        Sf.isel(time=time_idx, az=az_idx).f.plot(ax=ax, y="r", c="w")
    return fig, ax


def plot_param_time_fft(
        S: xarray.DataArray,
        az_idx: int = None,
        r_idx: int = None,
        ax: Axes = None
    ):
    if az_idx is None:
        az_idx = S.az.shape[0] // 2
    if r_idx is None:
        r_idx = S.r.shape[0] // 2
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 5))
    else:
        fig = ax.figure
    S.isel(r=r_idx, az=az_idx).plot(ax=ax, y="time", x="frequency")
    Sf = S.isel(frequency=S.argmax("frequency"))
    Sf.isel(r=r_idx, az=az_idx).frequency.plot(ax=ax, y="time", c="w")
    return fig, ax
