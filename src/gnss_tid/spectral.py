import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.fft import fft2

from .center_finding import run_smoothed_center_finder
from .coords import Local2D
from .fft import make_patches, make_wavenum_grid
from .image import ImageMakerBase, generate_image
from .parallel_logging import log_queue_listener, worker_logger
from .plotting import plot_center_finder
from .pointdata import TimeWindow, get_data, make_time_windows

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectralConfig:
    """Static inputs shared across spectral-focusing workers.

    Bundling these into one object keeps worker signatures short and makes
    joblib pickling explicit.  Fields are immutable so the same instance can
    be safely shared across processes.
    """

    obs: pd.DataFrame
    rx: pd.DataFrame
    image_maker: ImageMakerBase
    tec_name: str
    block_size: int
    block_step: int
    window: np.ndarray
    logscale_objective: bool
    lat_limits: tuple[float, float]
    lon_limits: tuple[float, float]
    proj: Local2D


def compute_patch_spectra(
    img: xarray.DataArray,
    block_size: int,
    hres: float,
    block_step: int,
    window: xarray.DataArray,
    logscale_objective: bool,
) -> xarray.DataArray:
    """Compute the FFT patches for a given image.

    Constructs overlapping blocks via a rolling window, applies the FFT
    windowing function, computes the 2-D power spectrum of each block, and
    optionally log-transforms the result.

    Args:
        img: Input image ``DataArray`` with dims ``(x, y)``.
        block_size: Edge length (pixels) of each square FFT block.
        hres: Horizontal resolution of the image (km per pixel); used to set
            wavenumber coordinates.
        block_step: Stride (in pixels) between consecutive block centres.
        window: FFT windowing array of shape ``(1, 1, block_size, block_size)``.
        logscale_objective: If ``True``, patch power is log10-transformed.

    Returns:
        A ``DataArray`` of 2-D power spectra with dims
        ``(px, py, kx, ky)`` where ``px``, ``py`` are spatial patch centres
        and ``kx``, ``ky`` are wavenumbers (cycles km⁻¹).
    """
    wavenum = make_wavenum_grid(block_size, hres)
    patches = make_patches(img, block_size, block_step).assign_coords(
        kx=wavenum, ky=wavenum
    )
    patches.values = abs(fft2(patches * window)) ** 2
    if logscale_objective:
        patches = np.log10(patches)

    return patches


def extract_patch_peaks(patch: xarray.DataArray) -> xarray.Dataset:
    """Summarise an FFT patch array by its spectral peak.

    Selects the wavenumber ``(kx, ky)`` of the maximum power at each spatial
    location and computes the total objective (sum over ``(px, py)``).

    Args:
        patch: ``DataArray`` with dims ``(..., px, py, kx, ky)``.

    Returns:
        A ``Dataset`` with variables:

        - ``F`` ``(..., px, py)``: power at the spectral peak.
        - ``Fx``, ``Fy`` ``(..., px, py)``: wavenumbers at the peak.
        - ``objective`` ``(...,)``: sum of ``F`` over ``(px, py)``.
    """
    return (
        patch.isel(patch.argmax(dim=["kx", "ky"]))
        .to_dataset(name="F")
        .reset_coords()
        .rename_vars({"kx": "Fx", "ky": "Fy"})
        .assign(objective=lambda x: x.F.sum(dim=["px", "py"]))
    )


def build_patch_stack(
    ts: TimeWindow,
    heights: np.ndarray,
    cfg: SpectralConfig,
    wlog: logging.Logger | None = None,
) -> xarray.DataArray | None:
    """Compute FFT patches at each height without keeping the gridded images.

    Used by the first-pass worker in two-pass mode, where only the objective
    surface is needed.  Skipping the per-height image concat saves a noticeable
    chunk of memory when ``len(heights)`` is large.

    Args:
        ts: Time window for this slice.
        heights: Array of IPP heights (km) to evaluate.
        cfg: Shared spectral configuration.
        wlog: Optional logger; falls back to module logger.

    Returns:
        A ``DataArray`` of patches with dims ``(height, px, py, kx, ky)``, or
        ``None`` if data retrieval failed for any height.
    """
    if wlog is None:
        wlog = logger
    patches = []
    for height in heights:
        wlog.info("[%s-%s]: height = %.1f", ts.start_time, ts.end_time, height)
        data = get_data(
            cfg.obs, cfg.rx, ts, height, cfg.lat_limits, cfg.lon_limits,
            proj=cfg.proj,
        )
        if data is None:
            wlog.warning("[%s-%s]: FAIL", ts.start_time, ts.end_time)
            return None
        img = generate_image(data, cfg.image_maker, cfg.tec_name)
        patches.append(
            compute_patch_spectra(
                img.image,
                cfg.block_size,
                cfg.image_maker.hres,
                cfg.block_step,
                cfg.window,
                cfg.logscale_objective,
            )
        )
    return xarray.concat(patches, "height").assign_coords(height=heights)


def build_image_patch_stack(
    ts: TimeWindow,
    heights: np.ndarray,
    cfg: SpectralConfig,
    wlog: logging.Logger | None = None,
) -> xarray.Dataset | None:
    """Compute images and FFT patches at every requested height.

    Used by the single-pass worker, which needs both image data and patches
    so it can return the best height's full slice in one go.

    Args:
        ts: Time window for this slice.
        heights: Array of IPP heights (km) to process.
        cfg: Shared spectral configuration.
        wlog: Optional logger; falls back to module logger.

    Returns:
        A ``Dataset`` with dims ``(height, x, y)`` containing variables
        ``image``, ``patch``, and ``n`` (data-point count per height), or
        ``None`` if data retrieval failed for any height.
    """
    if wlog is None:
        wlog = logger
    images = []
    patches = []
    npts = []
    for height in heights:
        wlog.info("[%s-%s]: height = %.1f", ts.start_time, ts.end_time, height)
        data = get_data(
            cfg.obs, cfg.rx, ts, height, cfg.lat_limits, cfg.lon_limits,
            proj=cfg.proj,
        )
        if data is None:
            wlog.warning("[%s-%s]: FAIL", ts.start_time, ts.end_time)
            return None
        npts.append(len(data))
        img = generate_image(data, cfg.image_maker, cfg.tec_name)
        patches.append(
            compute_patch_spectra(
                img.image,
                cfg.block_size,
                cfg.image_maker.hres,
                cfg.block_step,
                cfg.window,
                cfg.logscale_objective,
            )
        )
        images.append(img)

    return (
        xarray.concat(images, "height")
        .assign_coords(height=heights)
        .assign(n=(["height"], npts), patch=xarray.concat(patches, "height"))
    )


def compute_height_objectives_for_slice(
    ts: TimeWindow,
    time: np.datetime64,
    heights: np.ndarray,
    cfg: SpectralConfig,
    log_queue: Any | None = None,
) -> xarray.DataArray | None:
    """First-pass worker: return per-height FFT objectives for one time slice.

    Args:
        ts: Time window for this slice.
        time: Start timestamp; used as the ``time`` coordinate.
        heights: Array of heights (km) to evaluate.
        cfg: Shared spectral configuration.
        log_queue: Optional multiprocessing queue for worker logging.

    Returns:
        A ``DataArray`` of objective values with dims ``(time, height)``,
        or ``None`` if data retrieval failed for any height.
    """
    with worker_logger(log_queue) as wlog:
        wlog.info("[%s-%s]: processing heights", ts.start_time, ts.end_time)
        patches = build_patch_stack(ts, heights, cfg, wlog=wlog)
        if patches is None:
            return None
        return extract_patch_peaks(patches).objective.expand_dims(time=[time])


def _select_best_height_for_slice(
    ts: TimeWindow,
    time: np.datetime64,
    heights: np.ndarray,
    cfg: SpectralConfig,
    log_queue: Any | None = None,
) -> tuple[xarray.Dataset, xarray.DataArray] | None:
    """Single-pass worker: compute, select best height, and return full slice.

    Computes images and FFT patches at every candidate height, picks the
    height with the maximum objective, and returns the data at that height
    together with the per-height objective surface.  The returned slice
    already contains ``F``, ``Fx``, ``Fy``, and ``objective``, so no further
    ``extract_patch_peaks`` call is needed downstream.

    Args:
        ts: Time window for this slice.
        time: Start timestamp; used as the ``time`` coordinate.
        heights: Array of IPP heights (km) to evaluate.
        cfg: Shared spectral configuration.
        log_queue: Optional multiprocessing queue for worker logging.

    Returns:
        A 2-tuple ``(best_slice, objectives)``:

        - ``best_slice``: ``Dataset`` with ``image``, ``density``, ``patch``,
          ``height``, ``F``, ``Fx``, ``Fy``, ``objective``, expanded along
          ``time``.
        - ``objectives``: ``DataArray`` with dims ``(time, height)``.

        Returns ``None`` if data retrieval failed for any height.
    """
    with worker_logger(log_queue) as wlog:
        wlog.info("[%s-%s]: single-pass slice", ts.start_time, ts.end_time)
        data_all = build_image_patch_stack(ts, heights, cfg, wlog=wlog)
        if data_all is None:
            return None
        summary = extract_patch_peaks(data_all.patch)
        best_idx = int(summary.objective.argmax())
        best_height = float(heights[best_idx])
        best_slice = (
            data_all.isel(height=best_idx)
            .drop_vars("height")
            .merge(summary.isel(height=best_idx).drop_vars("height"))
            .expand_dims(time=[time])
            .assign(height=("time", [best_height]))
        )
        obj_da = summary.objective.expand_dims(time=[time])
        return best_slice, obj_da


def _rebuild_slice_at_height(
    ts: TimeWindow,
    time: np.datetime64,
    height: float,
    cfg: SpectralConfig,
    log_queue: Any | None = None,
) -> xarray.Dataset | None:
    """Second-pass worker: rebuild image and patch at a fixed focus height.

    Args:
        ts: Time window for this slice.
        time: Start timestamp; used as the ``time`` coordinate.
        height: IPP height (km) at which to evaluate this time slice.
        cfg: Shared spectral configuration.
        log_queue: Optional multiprocessing queue for worker logging.

    Returns:
        A ``Dataset`` with variables ``image``, ``density``, and ``patch``,
        expanded along ``time``, or ``None`` if data retrieval failed.
    """
    with worker_logger(log_queue) as wlog:
        wlog.info("focused slice time=%s height=%.1f", time, height)
        data = get_data(
            cfg.obs, cfg.rx, ts, height, cfg.lat_limits, cfg.lon_limits,
            proj=cfg.proj,
        )
        if data is None:
            return None
        img = generate_image(data, cfg.image_maker, cfg.tec_name)
        p = compute_patch_spectra(
            img.image,
            cfg.block_size,
            cfg.image_maker.hres,
            cfg.block_step,
            cfg.window,
            cfg.logscale_objective,
        )
        return img.assign(patch=p).expand_dims(time=[time])


def _plot_objective_surface(
    surface: xarray.DataArray,
    height_track: xarray.DataArray,
    path: str,
) -> None:
    """Save a height-objective heatmap with the selected height overlaid."""
    fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
    surface.plot(ax=ax)
    ax.plot(height_track, height_track.time, "r.-")
    fig.savefig(path)
    plt.close(fig)


def _build_sparse_image(
    data_focused: xarray.Dataset, density_thresh: int
) -> xarray.Dataset:
    """Flatten the focus-height image stack into a sparse ``(time, row)`` table.

    Pixels with density below ``density_thresh`` are masked out, and rows/times
    that end up all-NaN are dropped.
    """
    return (
        data_focused.image.where(data_focused.density > density_thresh)
        .stack(row=("x", "y"))
        .reset_index("row")
        .dropna(dim="time", how="all")
        .dropna(dim="row", how="all")
        .reset_coords()
    )


def _focus_per_slice(
    time_windows: list,
    times: list,
    heights: np.ndarray,
    cfg: SpectralConfig,
    n_jobs: int,
) -> xarray.Dataset:
    """Run the single-pass focusing branch and return the per-time best slice.

    Each worker selects the best height for its window and returns the
    corresponding image plus patch summary; no further ``extract_patch_peaks``
    call is needed downstream.
    """
    logger.info(
        "run_spectral_focusing: single pass (no smoothing), n_jobs=%d", n_jobs
    )
    with log_queue_listener() as q, Parallel(n_jobs=n_jobs) as parallel:
        single_slices = parallel(
            delayed(_select_best_height_for_slice)(ts, time, heights, cfg, q)
            for ts, time in zip(time_windows, times, strict=True)
        )

    valid_results = [r for r in single_slices if r is not None]
    if not valid_results:
        raise RuntimeError("single pass produced no valid slices")

    valid_slices, per_height_objs = zip(*valid_results, strict=True)
    data_focused = xarray.concat(valid_slices, "time").reindex(time=times)

    obj_surface = xarray.concat(per_height_objs, dim="time").reindex(time=times)
    _plot_objective_surface(obj_surface, data_focused.height, "plots/objective.png")
    return data_focused


def _focus_with_time_smoothing(
    time_windows: list,
    times: list,
    heights: np.ndarray,
    cfg: SpectralConfig,
    n_jobs: int,
    time_window: int,
) -> xarray.Dataset:
    """Run the two-pass focusing branch: objectives → smoothing → reconstruction."""
    # First pass — objectives only (images discarded to save memory)
    logger.info(
        "run_spectral_focusing: first pass (objectives), n_jobs=%d", n_jobs
    )
    with log_queue_listener() as q, Parallel(n_jobs=n_jobs) as parallel:
        objectives = parallel(
            delayed(compute_height_objectives_for_slice)(ts, time, heights, cfg, q)
            for ts, time in zip(time_windows, times, strict=True)
        )

    obj_data = xarray.concat(
        filter(lambda x: x is not None, objectives), dim="time"
    ).reindex(time=times)

    # Height smoothing — arithmetic mean on the objective surface (log or
    # linear scale is controlled by logscale_objective at the FFT stage)
    smoothed = obj_data.rolling(time=time_window, center=True, min_periods=1).mean()

    smooth_nonull = smoothed.dropna(dim="time")
    focus_height = smooth_nonull.isel(
        height=smooth_nonull.argmax(dim="height")
    ).reindex(time=smoothed.time)

    _plot_objective_surface(smoothed, focus_height.height, "plots/objective.png")

    # Second pass — reconstruct images at the smoothed focus heights
    second_pass_inputs = [
        (ts, time, focus_height.isel(time=ii).height.item())
        for ii, (ts, time) in enumerate(zip(time_windows, times, strict=True))
        if not focus_height.isel(time=ii).height.isnull().item()
    ]
    if not second_pass_inputs:
        raise RuntimeError("no valid focus heights after first pass; cannot continue")

    logger.info(
        "run_spectral_focusing: second pass (%d / %d windows have valid focus height)",
        len(second_pass_inputs),
        len(time_windows),
    )
    with log_queue_listener() as q, Parallel(n_jobs=n_jobs) as parallel:
        focused_slices = parallel(
            delayed(_rebuild_slice_at_height)(ts, time, height, cfg, q)
            for ts, time, height in second_pass_inputs
        )

    valid_slices = [s for s in focused_slices if s is not None]
    if not valid_slices:
        raise RuntimeError("second pass produced no valid slices")

    data_focused = xarray.concat(valid_slices, "time")
    return (
        data_focused.merge(extract_patch_peaks(data_focused.patch))
        .reindex(time=focus_height.time)
        .assign(height=focus_height.height)
    )


def run_spectral_focusing(
    cfg: SpectralConfig,
    window_size: int,
    step: int,
    center_finder: Callable,
    heights: np.ndarray,
    n_jobs: int,
    time_window: int = 1,
    density_thresh: int = 20,
) -> xarray.Dataset:
    """Unified spectral focusing pipeline.

    Orchestrates spectral focusing over a sequence of time windows in two modes
    depending on ``time_window``:

    **Single-pass** (``time_window=1``):

    1. **Init**: find the time window with the most data and initialize the
       image maker's interpolation grid.
    2. **Single pass**: for each time window, compute images and FFT-patch
       objectives at all candidate heights, then immediately select the height
       with the highest objective.  No smoothing is applied.

    **Two-pass** (``time_window>1``):

    1. **Init**: same as above.
    2. **First pass**: for each time window compute only the per-height FFT
       objectives (images are discarded to save memory).
    3. **Height smoothing**: apply a rolling geometric mean over time to the
       full objective surface before selecting heights.
    4. **Second pass**: for each time window reconstruct the image and FFT
       patches at the smoothed focus height.

    Both modes finish with a global center-finding step that fits source
    parameters jointly across all times.

    Args:
        cfg: Shared spectral configuration (observations, receiver table,
            image maker, FFT-block geometry, windowing, region bounds, and
            TEC variable name).  See :class:`SpectralConfig` for fields.
        window_size: Number of unique time steps per sliding window.
        step: Stride (in time steps) between consecutive windows.
        center_finder: Callable ``(c0, w0, x, y, image) -> dict`` used for
            global source-parameter estimation.
        heights: Array of IPP heights (km) to search.
        n_jobs: Number of parallel worker processes.
        time_window: Rolling-mean window length (in time steps) applied to the
            height-objective surface.  ``1`` disables smoothing (single-pass
            mode).  Default ``1``.
        density_thresh: Minimum image density (data points per pixel) required
            for a pixel to be included in the sparse image used for center
            finding.  Default ``20``.

    Returns:
        An ``xarray.Dataset`` with dimensions ``(time,)`` and variables:

        - ``image`` ``(time, x, y)``: gridded TEC image at the focus height.
        - ``patch`` ``(time, px, py, kx, ky)``: FFT patch at the spectral peak.
        - ``F``, ``Fx``, ``Fy`` ``(time, px, py)``: peak power and wavenumbers.
        - ``objective`` ``(time,)``: summed peak power used for height selection.
        - ``height`` ``(time,)``: focus height (km).
        - ``center`` ``(ci,)``: estimated source center ``[x, y]`` (km, global).
        - ``wavelength`` ``(time,)``: estimated horizontal wavelength (km).
        - ``offset`` ``(time,)``: estimated DC offset.
        - ``phase`` ``(time,)``: estimated wave phase (rad).

        Attributes include ``coord_center`` as ``(mean_lat, mean_lon)``.
    """
    cfg.image_maker.initialize_from_bounds(
        proj=cfg.proj,
        lat_limits=cfg.lat_limits,
        lon_limits=cfg.lon_limits,
    )

    time_windows = make_time_windows(cfg.obs["time"], window_size, step)
    times = [w.start_time for w in time_windows]
    Path("plots").mkdir(exist_ok=True)

    if time_window == 1:
        data_focused = _focus_per_slice(
            time_windows, times, heights, cfg, n_jobs
        )
    else:
        data_focused = _focus_with_time_smoothing(
            time_windows, times, heights, cfg, n_jobs, time_window
        )

    sparse_img = _build_sparse_image(data_focused, density_thresh)

    init_slice = data_focused.isel(time=data_focused.objective.argmax())
    fig, _ = plot_center_finder(init_slice)
    fig.savefig("plots/center_init.png")
    plt.close(fig)

    params = run_smoothed_center_finder(
        px=init_slice.px.values,
        py=init_slice.py.values,
        F=init_slice.F.values,
        Fx=init_slice.Fx.values,
        Fy=init_slice.Fy.values,
        sparse_x=sparse_img.x.values,
        sparse_y=sparse_img.y.values,
        sparse_image=sparse_img.image.values.T,
        center_finder=center_finder,
    )
    logger.info("params fit in %d iterations", len(params["history"]["metric"]))

    coord_center = (np.mean(cfg.lat_limits), np.mean(cfg.lon_limits))
    return data_focused.assign(
        center=("ci", params["center"]),
        wavelength=xarray.DataArray(
            params["wavelength"], coords={"time": sparse_img.time}
        ),
        offset=xarray.DataArray(params["offset"], coords={"time": sparse_img.time}),
        phase=xarray.DataArray(params["phase"], coords={"time": sparse_img.time}),
    ).assign_attrs(coord_center=coord_center)


