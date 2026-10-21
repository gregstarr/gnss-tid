import matplotlib.animation as animation
import numpy as np
import xarray
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.ndimage import maximum_filter1d
from tqdm import autonotebook

from .center_finding import find_center, find_center_grid_search, find_center_huber


def _get_masked_image(data, density_thresh):
    """Return data.image, optionally masked where density <= density_thresh."""
    if density_thresh is not None and "density" in data:
        return data.image.where(data.density > density_thresh)
    return data.image


def _mark_max_patch(ax, data, label=None):
    """Draw a red ring at the patch with the highest finite peak power.

    Gated patches (``F`` NaN from the patch density gate) are excluded by the
    ``skipna`` argmax, so the ring marks the highest-power data-supported
    patch.

    Args:
        ax: Matplotlib axes to draw on.
        data: Dataset holding variable ``F`` on the patch-centre dims ``px``
            and ``py`` (either dim order; pass a 2-D slice of a time-varying
            field).
        label: Optional legend label for the marker.

    Returns:
        List of artists added; empty when ``F`` holds no finite value.
    """
    Fv = np.asarray(data.F.values)
    if not (Fv.size and np.any(np.isfinite(Fv))):
        return []
    peak = data.F.argmax(dim=["px", "py"], skipna=True)
    return ax.plot(
        [float(data.px.isel(px=peak["px"]).item())],
        [float(data.py.isel(py=peak["py"]).item())],
        "o",
        mfc="none",
        mec="red",
        ms=14,
        mew=2,
        label=label,
    )


def plot_circles(center, wavelength, offset, ax=None, data=None, maxr=1200):
    if np.isnan(wavelength) or np.isnan(offset) or np.any(np.isnan(center)):
        return []
    if ax is None:
        ax = plt.gca()
    artists = []
    if data is not None and maxr is None:
        artists.append(
            data.plot.scatter(
                ax=ax, x="x", y="y", hue="tec", vmax=0.3, add_colorbar=False
            )
        )
        maxr = (
            min(np.max(data.x.values - center[0]), np.max(data.y.values - center[1]))
            * 0.9
        )
    elif maxr is None:
        raise ValueError("specify maxr")
    artists += ax.plot(center[0], center[1], "k+")
    for radius in np.arange((-1 * offset) % wavelength, maxr, wavelength):
        circle = plt.Circle(
            center, radius, facecolor="none", edgecolor=(0, 0, 0), linewidth=1, alpha=0.5
        )
        artists.append(ax.add_patch(circle))
    return artists


def plot_patches(
    data, img=True, ax=None, length_fraction=0.3, width=0.006, density_thresh=None
):
    if ax is None:
        ax = plt.gca()
    if img:
        img_da = data.image
        if density_thresh is not None and "density" in data:
            img_da = img_da.where(data.density > density_thresh)
        img_da.plot(ax=ax, vmax=0.3)

    # Flatten the patch grid via xarray so coords stay aligned with values
    # regardless of (px, py) dim order.
    stacked = data[["F", "Fx", "Fy"]].stack(p=("px", "py"))
    px_vals = stacked.px.values
    py_vals = stacked.py.values
    Fx = stacked.Fx.values
    Fy = stacked.Fy.values
    F = stacked.F.values

    kmag = np.hypot(Fx, Fy)
    mask = (kmag > 0) & np.isfinite(kmag) & np.isfinite(F)

    # Constant arrow length in data units so size carries no information.
    dpx = float(np.abs(np.mean(np.diff(data.px.values))))
    length = dpx * length_fraction
    u = (Fx[mask] / kmag[mask]) * length
    v = (Fy[mask] / kmag[mask]) * length

    ax.quiver(
        px_vals[mask],
        py_vals[mask],
        u,
        v,
        F[mask],
        cmap="bone",
        scale=1,
        scale_units="xy",
        angles="xy",
        pivot="mid",
        headwidth=0,
        headlength=0,
        headaxislength=0,
        width=width,
    )


def plot_pre_center_finder(data, density_thresh=None):
    """Diagnostic plot of the raw inputs to ``find_center``.

    Image + per-patch wavevector quiver, with **no** center estimates
    overlaid.  The highest-objective patch (maximum ``F``) is marked with a
    red ring.  Use this when ``find_center`` itself fails (returns NaN or
    nonsense) to inspect what went into it.

    Args:
        data: Single time-slice dataset with dims ``(px, py)`` for the patch
            grid (``F``, ``Fx``, ``Fy``; coords ``px``, ``py`` in km) and
            dims ``(x, y)`` for ``image`` / ``density``.
        density_thresh: Minimum image density (data points per pixel)
            required for a pixel to be shown; ``None`` shows the full image.

    Returns:
        A ``(fig, ax)`` tuple for the caller to save or annotate further.
    """
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    plot_patches(data, ax=ax, img=True, density_thresh=density_thresh)
    if _mark_max_patch(ax, data, label="max objective patch"):
        ax.legend()
    return fig, ax


def plot_objective_lines(data, logscale=False):
    """Plot the height-selected and smoothed objectives over time.

    The first line is the unsmoothed objective at each time's selected focus
    height, plotted as ``log10(objective)`` when ``logscale`` is set so it
    shares units with the smoothed line.  The second line — only drawn when a
    smoothed objective is present (two-pass mode) — is the time-smoothed
    objective surface at the ``logscale_objective``-selected height, i.e. the
    quantity that drives height selection and center finding.

    Args:
        data: Result dataset with dim ``(time,)``, variable ``objective``
            ``(time,)`` (unsmoothed objective at the selected height, linear
            power), and one of ``objective_smoothed`` (legacy) or
            ``objective_smoothed_linear`` / ``objective_smoothed_log``
            ``(time,)`` (smoothed objectives at their respective selected
            heights; linear power for the linear track, log10 units for the
            log track).
        logscale: Whether ``logscale_objective`` was set for the run.

    Returns:
        A ``(fig, ax)`` tuple for the caller to save or annotate further.
    """
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    first = np.log10(data.objective) if logscale else data.objective
    first.plot.line(ax=ax, label="height-selected objective")
    if "objective_smoothed" in data:
        smoothed = data.objective_smoothed
    elif logscale and "objective_smoothed_log" in data:
        smoothed = data.objective_smoothed_log
    elif "objective_smoothed_linear" in data:
        smoothed = data.objective_smoothed_linear
    else:
        smoothed = None
    if smoothed is not None:
        smoothed.plot.line(ax=ax, label="smoothed objective")
    ax.set_ylabel("log10 objective" if logscale else "objective")
    ax.grid(True)
    ax.legend()
    return fig, ax


def plot_density_map(data):
    """Density (data points per pixel) at the focus slice."""
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    data.density.plot(ax=ax)
    return fig, ax


def plot_peak_patch_spectrum(data):
    """2-D power spectrum of the patch with the largest peak power.

    Picks the ``(px, py)`` location with maximum ``F`` and shows its full
    ``(kx, ky)`` spectrum, to confirm the peak is a real lobe rather than a
    single noisy bin.  When ``F`` holds no finite value (every patch gated by
    the density cut), an empty figure titled accordingly is returned instead
    of raising on the all-NaN argmax.
    """
    Fv = np.asarray(data.F.values)
    if not (Fv.size and np.any(np.isfinite(Fv))):
        fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
        ax.set_title("all patches gated: no finite peak power")
        return fig, ax
    peak = data.F.argmax(dim=["px", "py"])
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    data.patch.isel(peak).plot(ax=ax)
    ax.set_title(
        f"peak patch @ px={float(data.px.isel(px=peak['px']).item()):.0f}, "
        f"py={float(data.py.isel(py=peak['py']).item()):.0f}"
    )
    return fig, ax


def plot_center_finder(data, density_thresh=None):
    """Diagnostic plot pooling all time slices for the seed estimators.

    LSQ and grid-search objectives are summed over the full ``(time, py, px)``
    patch field; the quiver overlay uses only the highest-objective time
    slice to keep the figure legible.
    """
    px = data.px.values
    py = data.py.values
    Fx_all = data.Fx.values
    Fy_all = data.Fy.values
    F_all = data.F.values
    T = F_all.shape[0]

    X, Y = np.meshgrid(px, py)
    pts_single = np.column_stack([X.ravel(), Y.ravel()])
    pts = np.tile(pts_single, (T, 1))
    vectors = np.column_stack([Fx_all.ravel(), Fy_all.ravel()])
    weights = F_all.ravel()
    cw = find_center(pts, vectors, weights)
    cu = find_center(pts, vectors, np.ones_like(weights))
    ch = find_center_huber(pts, vectors, weights)
    cg, dw = find_center_grid_search(pts, vectors, weights, data.x.values, data.y.values)
    XD, YD = np.meshgrid(data.x.values, data.y.values)

    init_slice = data.isel(time=data.objective.argmax())

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))
    plot_patches(init_slice, ax=ax[0], img=True, density_thresh=density_thresh)
    ax[0].plot(cw[0], cw[1], "x", label="lsq weighted")
    ax[0].plot(cu[0], cu[1], "x", label="lsq unweighted")
    ax[0].plot(ch[0], ch[1], "x", label="huber")
    ax[0].plot(cg[0], cg[1], "x", label="grid search")
    ax[0].legend()

    ax[1].pcolormesh(XD, YD, dw)
    ax[1].plot(cw[0], cw[1], "x", label="lsq weighted")
    ax[1].plot(cu[0], cu[1], "x", label="lsq unweighted")
    ax[1].plot(ch[0], ch[1], "x", label="huber")
    ax[1].plot(cg[0], cg[1], "x", label="grid search")
    ax[1].legend()
    plot_patches(init_slice, ax=ax[1], img=False)
    print(f"lsq unweighted: {cu}, lsq weighted: {cw}, huber: {ch}, grid search: {cg}")
    return fig, ax


def plot_center_finder_fit(result_list, s=4, c0=None, cbox=None):
    fig, ax = plt.subplots(2, 4, figsize=(4 * s, 2 * s), tight_layout=True, sharex="col")

    best_iteration = np.argmax([r["metric"] for r in result_list])
    # Per-restart colors shared across columns 0, 1, 2 so a given restart is
    # visually traceable between metric / lambda-phase / center panels.
    colors = plt.cm.tab20(np.linspace(0, 1, len(result_list)))
    for ii, result in enumerate(result_list):
        color = colors[ii]
        ax[0, 0].plot(result["history"]["metric"], color=color)
        ax[1, 0].semilogy(
            maximum_filter1d(np.diff(result["history"]["metric"]), 9), color=color
        )

        z = np.array(result["history"]["center"])
        ax[0, 2].plot(z[:, 0], z[:, 1], "-", color=color)
        terminal_color = "r" if ii == best_iteration else "k"
        ax[0, 2].plot(z[-1, 0], z[-1, 1], ".", color=terminal_color)

        # Per-time-step parameter histories collapsed to median, one line per
        # restart, so the panel parallels the per-restart metric view.
        wl = np.array(result["history"]["wavelength"])
        ax[0, 1].plot(np.median(wl, axis=1), "-", color=color)
        ph = np.unwrap(np.array(result["history"]["phase"]), axis=0)
        ax[1, 1].plot(np.median(ph, axis=1), "-", color=color)

        # Per-restart final wavelength(t) / offset(t) as grey low-alpha
        # background so column 3 shows the spread across restarts.
        if ii != best_iteration:
            ax[0, 3].plot(result["wavelength"], color="grey", alpha=0.3, lw=0.5)
            ax[1, 3].plot(result["offset"], color="grey", alpha=0.3, lw=0.5)

    if c0 is not None:
        ax[0, 2].plot(c0[0], c0[1], "kx")
    if cbox is not None:
        pts = np.array(
            [
                [cbox[0], cbox[1]],
                [cbox[0], cbox[1] + cbox[3]],
                [cbox[0] + cbox[2], cbox[1] + cbox[3]],
                [cbox[0] + cbox[2], cbox[1]],
                [cbox[0], cbox[1]],
            ]
        )
        ax[0, 2].plot(pts[:, 0], pts[:, 1], "k--")

    result = result_list[best_iteration]
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


def make_animation(
    data, save_fn, limit=-1, writer="pillow", maxr=2400, density_thresh=None
):
    """Build a GIF/MP4 animation of the focusing sequence.

    Three rows: two line panels (focus height and objective) with a vertical
    cursor at the current time, a per-patch peak power map (``F`` over the
    patch-centre coordinates ``px``/``py``, shared color scale across frames),
    and a lower map panel showing the masked TEC image, fitted ring circles
    (only when center finding ran), and a red ring marking the highest-power
    *data-supported* FFT patch at each time step (patches gated to NaN by the
    density cut are excluded).

    Args:
        data: Result dataset with dims ``(time,)`` for ``height`` and
            ``objective``; ``(time, x, y)`` for ``image`` / ``density``;
            ``(time, px, py)`` for ``F`` (patch peak powers); and optionally
            ``(time,)`` for ``wavelength``/``offset`` and ``(ci,)`` for
            ``center`` — the ring circles are drawn only when those three
            variables are present.
        save_fn: Output file path (e.g. ``"event.gif"``).
        limit: Number of frames to render; ``-1`` renders every time step.
        writer: Matplotlib animation writer name.
        maxr: Maximum ring radius (km) drawn in the image panel.
        density_thresh: Minimum image density (data points per pixel)
            required for a pixel to be shown; ``None`` shows the full image.

    Returns:
        None; writes the animation to ``save_fn``.
    """
    with plt.style.context("bmh"):
        layout = [
            ["height", "objective"],
            ["power", "power"],
            ["map", "map"],
        ]
        fig, ax = plt.subplot_mosaic(
            layout,
            figsize=(10, 10),
            height_ratios=[5, 10, 10],
            tight_layout=True,
        )

        f_vmax = float(data.F.max())
        if not np.isfinite(f_vmax):
            f_vmax = None

        animation_artists = []
        if limit <= 0:
            limit = data.time.shape[0]
        for ii in autonotebook.tqdm(range(limit), "creating frames"):
            frame_artists = []
            frame_artists += data.height.plot(ax=ax["height"], c="blue")
            frame_artists += data.objective.plot(ax=ax["objective"], c="blue")
            frame_artists += [
                ax["height"].axvline(data.time.values[ii], color="k", linestyle="--"),
                ax["objective"].axvline(data.time.values[ii], color="k", linestyle="--"),
                data.F.isel(time=ii).plot(
                    ax=ax["power"], vmax=f_vmax, add_colorbar=False
                ),
                _get_masked_image(data.isel(time=ii), density_thresh)
                .drop_vars("time")
                .plot(ax=ax["map"], vmax=0.3, add_colorbar=False),
            ]

            if "center" in data and "wavelength" in data and "offset" in data:
                frame_artists += plot_circles(
                    data.center,
                    data.wavelength.isel(time=ii),
                    data.offset.isel(time=ii),
                    ax=ax["map"],
                    maxr=maxr,
                )

            frame_artists += _mark_max_patch(ax["map"], data.isel(time=ii))

            animation_artists.append(frame_artists)

        ani = animation.ArtistAnimation(fig=fig, artists=animation_artists, interval=300)
        with autonotebook.tqdm(
            total=len(animation_artists), desc="Saving video"
        ) as progress_bar:
            ani.save(
                filename=save_fn,
                writer=writer,
                progress_callback=lambda _i, _n: progress_bar.update(),
            )


def plot_param_lines(data, lines_tec, time_idx=15):
    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(12, 5))

    data.image.isel(time=time_idx).plot(ax=ax[0], x="x", y="y")
    ax[0].plot(data.center.values[0], data.center.values[1], ".r")
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
    ax[0].plot(data.center.values[0], data.center.values[1], ".r")

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
    az_idx: int | None = None,
    ax: Axes | None = None,
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
    az_idx: int | None = None,
    r_idx: int | None = None,
    ax: Axes | None = None,
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
