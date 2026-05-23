import logging
from collections.abc import Callable
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager
from os import getpid
from pathlib import Path
from typing import Any

import numpy as np
import xarray
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.fft import fft2, fftfreq
from tqdm_joblib import tqdm_joblib

from .center_finding import find_center
from .plotting import plot_center_finder
from .pointdata import get_data, make_time_windows

logger = logging.getLogger(__name__)


def configure_worker_logger(log_queue=None, log_level=logging.INFO):
    if log_queue is None:
        return logger, None
    worker_logger = logging.getLogger(f"worker {getpid()}")
    handler = QueueHandler(log_queue)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(log_level)
    return worker_logger, handler


def cleanup_worker_logger(worker_logger, handler):
    if handler is None:
        return
    worker_logger.removeHandler(handler)


def initialize_image_maker(
    obs: Any,
    rx: Any,
    window_size: int,
    step: int,
    image_maker: Any,
    heights: np.ndarray,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    n_jobs: int,
) -> None:
    """Find the time window with the most data points and call image_maker.initialize().

    Iterates over all time windows and measures how many data points are available
    at the median height.  The window with the most points is used to initialize
    the image maker's interpolation grid.

    Args:
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        window_size: Number of time steps per window.
        step: Stride between consecutive windows.
        image_maker: Image maker object whose ``initialize`` method will be called.
        heights: Array of heights to consider; the median height is used for sizing.
        lat_limits: ``(min_lat, max_lat)`` bounds for data retrieval.
        lon_limits: ``(min_lon, max_lon)`` bounds for data retrieval.
        n_jobs: Number of parallel jobs to use when counting window sizes.

    Returns:
        None.  ``image_maker.initialize`` is called as a side effect.
    """
    time_windows = make_time_windows(obs["time"], window_size, step)
    mid_height = heights[len(heights) // 2]
    logger.info("running initializer")

    if n_jobs > 1:

        @delayed
        def fn(ts):
            data = get_data(obs, rx, ts, mid_height, lat_limits, lon_limits)
            return len(data) if data is not None else 0

        with (
            tqdm_joblib(desc="initializing", total=len(time_windows)),
            Parallel(n_jobs=n_jobs) as parallel,
        ):
            sizes = parallel(fn(ts) for ts in time_windows)
    else:
        sizes = []
        for ts in time_windows:
            data = get_data(obs, rx, ts, mid_height, lat_limits, lon_limits)
            sizes.append(len(data) if data is not None else 0)

    ii = np.argmax(sizes)
    logger.info("initializer finished, best slice -> %d: %d", ii, sizes[ii])
    init_data = get_data(obs, rx, time_windows[ii], mid_height, lat_limits, lon_limits)
    image_maker.initialize(init_data["x"].values, init_data["y"].values)


def run_smoothed_center_finder(
    data: xarray.Dataset,
    sparse_img: xarray.Dataset,
    center_finder: Callable,
) -> Any:
    """Perform global center finding over a focused spectral dataset.

    Uses the time step with the highest objective to construct an initial guess
    for the source center and wavenumber, then calls ``center_finder`` on the
    full sparse image stack.

    Args:
        data: Focused spectral dataset with variables ``objective``, ``px``,
            ``py``, ``F``, ``Fx``, ``Fy``.
        sparse_img: Sparse representation of the image stack for optimization,
            with variables ``x``, ``y``, ``image``.
        center_finder: Callable that accepts
            ``(c0, w0, x, y, image)`` and returns a parameter dict with keys
            ``center``, ``wavelength``, ``offset``, ``phase``, ``history``.

    Returns:
        The result dict returned by ``center_finder``.
    """
    logger.info("finding center")
    d = data.isel(time=data.objective.argmax())

    fig, _ = plot_center_finder(d)
    fig.savefig("plots/center_init.png")
    plt.close(fig)

    X, Y = np.meshgrid(d.px.values, d.py.values)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    weights = d.F.values.ravel()
    vectors = np.column_stack((d.Fx.values.ravel(), d.Fy.values.ravel()))
    k = np.hypot(vectors[:, 0], vectors[:, 1])
    c0 = find_center(pts, vectors, weights)
    w0 = 1 / k.max()

    result = center_finder(
        c0, w0, sparse_img.x.values, sparse_img.y.values, sparse_img.image.values.T
    )
    return result


def run_spectral_time_slice(
    obs: Any,
    rx: Any,
    ts: Any,
    time: Any,
    log_queue: Any | None = None,
    heights: np.ndarray | None = None,
    lat_limits: tuple[float, float] | None = None,
    lon_limits: tuple[float, float] | None = None,
    image_maker: Any | None = None,
    tec_name: str | None = None,
    block_shape: tuple[int, int] | None = None,
    block_step: int | None = None,
    window: xarray.DataArray | None = None,
    logscale_objective: bool | None = None,
) -> xarray.DataArray | None:
    """Process a single time slice and return per-height FFT objectives.

    Retrieves data for each height, builds images, computes FFT patches, and
    returns the objective value at each height.  Used as the first-pass worker
    in :func:`run_spectral_focusing`.

    Args:
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        ts: Time window (``TimeWindow`` or ``(start, end)`` tuple).
        time: Start timestamp of this window; used as the ``time`` coordinate.
        log_queue: Optional multiprocessing queue for worker logging.
        heights: Array of heights (km) to evaluate.
        lat_limits: ``(min_lat, max_lat)`` bounds.
        lon_limits: ``(min_lon, max_lon)`` bounds.
        image_maker: Image maker object.
        tec_name: Name of the TEC variable in the observation data.
        block_shape: ``(ny, nx)`` shape of each FFT block.
        block_step: Stride used when extracting rolling blocks.
        window: FFT windowing array of shape ``(1, 1, ny, nx)``.
        logscale_objective: If ``True``, the patch power is log10-transformed
            before selecting the peak.

    Returns:
        A ``DataArray`` of objective values with dims ``(time, height)``,
        or ``None`` if data retrieval failed for any height.
    """
    wlog, handler = configure_worker_logger(log_queue)
    try:
        wlog.info("[%03d-%03d]: processing heights", ts.start, ts.stop)
        data = process_heights(
            obs,
            rx,
            ts,
            heights,
            lat_limits,
            lon_limits,
            image_maker,
            tec_name,
            block_shape,
            image_maker.hres,
            block_step,
            window,
            logscale_objective,
            wlog=wlog,
        )
        if data is None:
            return None
        patches = process_patches(data).expand_dims(time=[time])
        return patches.objective
    finally:
        cleanup_worker_logger(wlog, handler)


def _process_single_slice(
    obs: Any,
    rx: Any,
    ts: Any,
    time: Any,
    heights: np.ndarray,
    image_maker: Any,
    tec_name: str,
    block_shape: tuple[int, int],
    block_step: int,
    window: np.ndarray,
    logscale_objective: bool,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    log_queue: Any | None = None,
) -> tuple[xarray.Dataset, xarray.DataArray] | None:
    """Process a single time slice across all heights and return data at the best height.

    Computes images and FFT patches at every candidate height in a single pass,
    selects the height with the maximum objective, and returns both the full
    image/patch data at that height and the per-height objective surface.  Used
    as the single-pass worker in :func:`run_spectral_focusing` when
    ``time_window=1``.

    Args:
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        ts: Time window (``TimeWindow`` or ``(start, end)`` tuple).
        time: Start timestamp of this window; used as the ``time`` coordinate.
        heights: Array of IPP heights (km) to evaluate.
        image_maker: Image maker object.
        tec_name: Name of the TEC variable in the observation data.
        block_shape: ``(ny, nx)`` shape of each FFT block.
        block_step: Stride used when extracting rolling blocks.
        window: FFT windowing array of shape ``(1, 1, ny, nx)``.
        logscale_objective: If ``True``, the patch power is log10-transformed.
        lat_limits: ``(min_lat, max_lat)`` bounds.
        lon_limits: ``(min_lon, max_lon)`` bounds.
        log_queue: Optional multiprocessing queue for worker logging.

    Returns:
        A 2-tuple ``(best_slice, objectives)`` where:

        - ``best_slice`` is a ``Dataset`` with variables ``image``, ``density``,
          ``patch``, and ``height``, expanded along a ``time`` dimension.
        - ``objectives`` is a ``DataArray`` of per-height objective values with
          dims ``(time, height)``.

        Returns ``None`` if data retrieval failed for any height.
    """
    wlog, handler = configure_worker_logger(log_queue)
    try:
        wlog.info("[%03d-%03d]: single-pass slice", ts.start, ts.stop)
        data_all = process_heights(
            obs,
            rx,
            ts,
            heights,
            lat_limits,
            lon_limits,
            image_maker,
            tec_name,
            block_shape,
            image_maker.hres,
            block_step,
            window,
            logscale_objective,
            wlog=wlog,
        )
        if data_all is None:
            return None
        objectives = process_patches(data_all).objective
        best_idx = int(objectives.argmax())
        best_height = float(heights[best_idx])
        result = data_all.isel(height=best_idx).drop_vars("height")
        best_slice = result.expand_dims(time=[time]).assign(
            height=("time", [best_height])
        )
        obj_da = objectives.expand_dims(time=[time])
        return best_slice, obj_da
    finally:
        cleanup_worker_logger(wlog, handler)


def _process_focused_slice(
    obs: Any,
    rx: Any,
    ts: Any,
    time: Any,
    height: float,
    image_maker: Any,
    tec_name: str,
    block_shape: tuple[int, int],
    block_step: int,
    window: np.ndarray,
    logscale_objective: bool,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    log_queue: Any | None = None,
) -> xarray.Dataset | None:
    """Process a single time slice at a fixed height for the focused second pass.

    Retrieves data at the specified height, builds an image, and computes the
    FFT patches.  Used as the second-pass worker in :func:`run_spectral_focusing`.

    Args:
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        ts: Time window (``TimeWindow`` or ``(start, end)`` tuple).
        time: Start timestamp of this window; used as the ``time`` coordinate.
        height: IPP height (km) at which to evaluate this time slice.
        image_maker: Image maker object.
        tec_name: Name of the TEC variable in the observation data.
        block_shape: ``(ny, nx)`` shape of each FFT block.
        block_step: Stride used when extracting rolling blocks.
        window: FFT windowing array of shape ``(1, 1, ny, nx)``.
        logscale_objective: If ``True``, the patch power is log10-transformed.
        lat_limits: ``(min_lat, max_lat)`` bounds.
        lon_limits: ``(min_lon, max_lon)`` bounds.
        log_queue: Optional multiprocessing queue for worker logging.

    Returns:
        A ``Dataset`` with variables ``image`` and ``patch``, expanded along
        a ``time`` dimension, or ``None`` if data retrieval failed.
    """
    wlog, handler = configure_worker_logger(log_queue)
    try:
        wlog.info("focused slice time=%s height=%.1f", time, height)
        data = get_data(obs, rx, ts, height, lat_limits, lon_limits)
        if data is None:
            return None
        img = image_maker(data["x"].values, data["y"].values, data[tec_name].values)
        p = get_fft_patches(
            img.image,
            block_shape,
            image_maker.hres,
            block_step,
            window,
            logscale_objective,
        )
        return img.assign(patch=p).expand_dims(time=[time])
    finally:
        cleanup_worker_logger(wlog, handler)


def run_spectral_focusing(
    obs: Any,
    rx: Any,
    window_size: int,
    step: int,
    image_maker: Any,
    center_finder: Callable,
    heights: np.ndarray,
    block_shape: tuple[int, int],
    block_step: int,
    window: np.ndarray,
    logscale_objective: bool,
    n_jobs: int,
    tec_name: str,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
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
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        window_size: Number of unique time steps per sliding window.
        step: Stride (in time steps) between consecutive windows.
        image_maker: Image maker object; ``initialize`` will be called before
            processing begins.
        center_finder: Callable ``(c0, w0, x, y, image) -> dict`` used for
            global source-parameter estimation.
        heights: Array of IPP heights (km) to search.
        block_shape: ``(ny, nx)`` shape of each FFT block.
        block_step: Stride used when extracting rolling blocks.
        window: FFT windowing array of shape ``(1, 1, ny, nx)``.
        logscale_objective: If ``True``, patch power is log10-transformed
            before selecting the spectral peak.  When ``False`` and
            ``time_window>1``, the geometric mean is computed as
            ``exp(mean(log(L)))``.
        n_jobs: Number of parallel worker processes.
        tec_name: Name of the TEC variable in the observation data.
        lat_limits: ``(min_lat, max_lat)`` bounds for data retrieval.
        lon_limits: ``(min_lon, max_lon)`` bounds for data retrieval.
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
    # --- Step 1: initialise image maker ---------------------------------------
    initialize_image_maker(
        obs, rx, window_size, step, image_maker, heights, lat_limits, lon_limits, n_jobs
    )

    time_windows = make_time_windows(obs["time"], window_size, step)
    times = [w.start_time for w in time_windows]
    Path("plots").mkdir(exist_ok=True)
    root_logger = logging.getLogger()

    if time_window == 1:
        # --- Single-pass: process all heights and select best per time slice --
        logger.info(
            "run_spectral_focusing: single pass (no smoothing), n_jobs=%d", n_jobs
        )
        q = Manager().Queue()
        listener = QueueListener(q, *root_logger.handlers)
        try:
            listener.start()
            with Parallel(n_jobs=n_jobs) as parallel:
                single_slices = parallel(
                    delayed(_process_single_slice)(
                        obs,
                        rx,
                        ts,
                        time,
                        heights,
                        image_maker,
                        tec_name,
                        block_shape,
                        block_step,
                        window,
                        logscale_objective,
                        lat_limits,
                        lon_limits,
                        q,
                    )
                    for ts, time in zip(time_windows, times, strict=True)
                )
        finally:
            listener.stop()

        valid_results = [r for r in single_slices if r is not None]
        if not valid_results:
            raise RuntimeError("single pass produced no valid slices")

        valid_slices, per_height_objs = zip(*valid_results, strict=True)
        data_focused = xarray.concat(valid_slices, "time")
        data_focused = data_focused.merge(process_patches(data_focused)).reindex(
            time=times
        )

        # Plot the height-objective surface with the selected height overlaid,
        # matching the two-pass diagnostic plot layout
        obj_surface = xarray.concat(per_height_objs, dim="time").reindex(time=times)
        fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
        obj_surface.plot(ax=ax)
        ax.plot(data_focused.height, data_focused.time, "r.-")
        fig.savefig("plots/objective.png")
        plt.close(fig)

    else:
        # --- Two-pass: first collect all objectives, smooth, then reconstruct -
        # First pass — objectives only (images discarded to save memory)
        logger.info("run_spectral_focusing: first pass (objectives), n_jobs=%d", n_jobs)
        q = Manager().Queue()
        listener = QueueListener(q, *root_logger.handlers)
        try:
            listener.start()
            with Parallel(n_jobs=n_jobs) as parallel:
                objectives = parallel(
                    delayed(run_spectral_time_slice)(
                        obs,
                        rx,
                        ts,
                        time,
                        q,
                        heights,
                        lat_limits,
                        lon_limits,
                        image_maker,
                        tec_name,
                        block_shape,
                        block_step,
                        window,
                        logscale_objective,
                    )
                    for ts, time in zip(time_windows, times, strict=True)
                )
        finally:
            listener.stop()

        obj_data = xarray.concat(
            filter(lambda x: x is not None, objectives), dim="time"
        ).reindex(time=times)

        # Height smoothing — arithmetic mean on the objective surface (log or linear
        # scale is controlled by logscale_objective at the FFT stage)
        smoothed = obj_data.rolling(time=time_window, center=True, min_periods=1).mean()

        smooth_nonull = smoothed.dropna(dim="time")
        focus_height = smooth_nonull.isel(
            height=smooth_nonull.argmax(dim="height")
        ).reindex(time=smoothed.time)

        fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
        smoothed.plot(ax=ax)
        ax.plot(focus_height.height, focus_height.time, "r.-")
        fig.savefig("plots/objective.png")
        plt.close(fig)

        # Second pass — reconstruct images at the smoothed focus heights
        second_pass_inputs = [
            (ii, ts, time, focus_height.isel(time=ii).height.item())
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
        q2 = Manager().Queue()
        listener2 = QueueListener(q2, *root_logger.handlers)
        try:
            listener2.start()
            with Parallel(n_jobs=n_jobs) as parallel:
                focused_slices = parallel(
                    delayed(_process_focused_slice)(
                        obs,
                        rx,
                        ts,
                        time,
                        height,
                        image_maker,
                        tec_name,
                        block_shape,
                        block_step,
                        window,
                        logscale_objective,
                        lat_limits,
                        lon_limits,
                        q2,
                    )
                    for _ii, ts, time, height in second_pass_inputs
                )
        finally:
            listener2.stop()

        valid_slices = [s for s in focused_slices if s is not None]
        if not valid_slices:
            raise RuntimeError("second pass produced no valid slices")

        data_focused = xarray.concat(valid_slices, "time")
        data_focused = (
            data_focused.merge(process_patches(data_focused))
            .reindex(time=focus_height.time)
            .assign(height=focus_height.height)
        )

    # --- Sparse image for global center finding -------------------------------
    sparse_img = (
        data_focused.image.where(data_focused.density > density_thresh)
        .stack(row=("x", "y"))
        .reset_index("row")
        .dropna(dim="time", how="all")
        .dropna(dim="row", how="all")
        .reset_coords()
    )

    # --- Global center finding ------------------------------------------------
    params = run_smoothed_center_finder(data_focused, sparse_img, center_finder)
    logger.info("params fit in %d iterations", len(params["history"]["metric"]))

    coord_center = (np.mean(lat_limits), np.mean(lon_limits))
    data_focused = data_focused.assign(
        center=("ci", params["center"]),
        wavelength=xarray.DataArray(
            params["wavelength"], coords={"time": sparse_img.time}
        ),
        offset=xarray.DataArray(params["offset"], coords={"time": sparse_img.time}),
        phase=xarray.DataArray(params["phase"], coords={"time": sparse_img.time}),
    ).assign_attrs(coord_center=coord_center)

    return data_focused


def get_fft_patches(
    img: xarray.DataArray,
    block_shape: tuple[int, int],
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
        block_shape: ``(ny, nx)`` size of each FFT block.
        hres: Horizontal resolution of the image (km per pixel); used to set
            wavenumber coordinates.
        block_step: Stride (in pixels) between consecutive block centres.
        window: FFT windowing array of shape ``(1, 1, ny, nx)``.
        logscale_objective: If ``True``, patch power is log10-transformed.

    Returns:
        A ``DataArray`` of 2-D power spectra with dims
        ``(px, py, kx, ky)`` where ``px``, ``py`` are spatial patch centres
        and ``kx``, ``ky`` are wavenumbers (cycles km⁻¹).
    """
    wavenum = fftfreq(block_shape[0], hres)
    edges = block_shape[0] // (2 * block_step)
    patches = (
        img.rolling(y=block_shape[0], x=block_shape[1], center=True)
        .construct(x="kx", y="ky", stride=block_step)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .assign_coords(kx=wavenum, ky=wavenum)
        .rename({"x": "px", "y": "py"})
    )
    patches.values = abs(fft2(patches * window)) ** 2
    if logscale_objective:
        patches = np.log10(patches)

    return patches


def process_patches(data: xarray.Dataset) -> xarray.Dataset:
    """Identify the best FFT patch for each height.

    Selects the spatial location ``(px, py)`` and wavenumber ``(kx, ky)`` of
    the maximum power in the patch array, and computes the total objective
    (sum over all patch locations).

    Args:
        data: Dataset containing ``patch`` with dims
            ``(..., px, py, kx, ky)``.

    Returns:
        A ``Dataset`` with variables:

        - ``F`` ``(..., px, py)``: power at the spectral peak.
        - ``Fx``, ``Fy`` ``(..., px, py)``: wavenumbers at the peak.
        - ``objective`` ``(...,)``: sum of ``F`` over ``(px, py)``.
    """
    result = (
        data.patch.isel(data.patch.argmax(dim=["kx", "ky"]))
        .to_dataset(name="F")
        .reset_coords()
        .rename_vars({"kx": "Fx", "ky": "Fy"})
        .assign(objective=lambda x: x.F.sum(dim=["px", "py"]))
    )
    return result


def process_heights(
    obs: Any,
    rx: Any,
    ts: Any,
    heights: np.ndarray,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    image_maker: Any,
    tec_name: str,
    block_shape: tuple[int, int],
    hres: float,
    block_step: int,
    window: xarray.DataArray,
    logscale_objective: bool,
    wlog: logging.Logger | None = None,
) -> xarray.Dataset | None:
    """Process all requested heights for a given time slice.

    For each height, retrieves data, builds an image, and computes FFT patches.
    Returns ``None`` immediately if data retrieval fails for any height.

    Args:
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        ts: Time window (``TimeWindow`` or ``(start, end)`` tuple).
        heights: Array of IPP heights (km) to process.
        lat_limits: ``(min_lat, max_lat)`` bounds.
        lon_limits: ``(min_lon, max_lon)`` bounds.
        image_maker: Image maker object.
        tec_name: Name of the TEC variable in the observation data.
        block_shape: ``(ny, nx)`` shape of each FFT block.
        hres: Horizontal resolution of the image (km per pixel).
        block_step: Stride used when extracting rolling blocks.
        window: FFT windowing array of shape ``(1, 1, ny, nx)``.
        logscale_objective: If ``True``, patch power is log10-transformed.
        wlog: Optional logger for worker processes; falls back to module logger.

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
        wlog.info("[%03d-%03d]: height = %.1f", ts.start, ts.stop, height)
        data = get_data(obs, rx, ts, height, lat_limits, lon_limits)
        if data is None:
            wlog.warning("[%03d-%03d]: FAIL", ts.start, ts.stop)
            return None
        npts.append(len(data))
        img = image_maker(data["x"].values, data["y"].values, data[tec_name].values)
        patches.append(
            get_fft_patches(
                img.image,
                block_shape,
                image_maker.hres,
                block_step,
                window,
                logscale_objective,
            )
        )
        images.append(img)

    data = (
        xarray.concat(images, "height")
        .assign_coords(height=heights)
        .assign(n=(["height"], npts), patch=xarray.concat(patches, "height"))
    )
    return data


def run_image_generation(
    obs: Any,
    rx: Any,
    ts: Any,
    time: Any,
    image_maker: Any,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    log_queue: Any | None = None,
) -> xarray.DataArray | None:
    """Generate an image for a specific time slice at a fixed height.

    Retrieves data at 350 km IPP height and calls the image maker to produce
    a gridded TEC image.

    Args:
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        ts: Time window (``TimeWindow`` or ``(start, end)`` tuple).
        time: Start timestamp of this window; used as the ``time`` coordinate.
        image_maker: Image maker object.
        lat_limits: ``(min_lat, max_lat)`` bounds.
        lon_limits: ``(min_lon, max_lon)`` bounds.
        log_queue: Optional multiprocessing queue for worker logging.

    Returns:
        A ``DataArray`` with dims ``(time, x, y)``, or ``None`` if data
        retrieval failed.
    """
    wlog, handler = configure_worker_logger(log_queue)
    try:
        wlog.info("time %03d-%03d", ts.start, ts.stop)
        pd_ = get_data(obs, rx, ts, 350, lat_limits, lon_limits)
        if pd_ is None:
            return None
        img = image_maker(pd_["x"].values, pd_["y"].values, pd_["tec"].values)
        img = img.expand_dims(time=[time])
        return img
    finally:
        cleanup_worker_logger(wlog, handler)


def run_image_maker_orchestrator(
    obs: Any,
    rx: Any,
    window_size: int,
    step: int,
    image_maker: Any,
    n_jobs: int,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
) -> xarray.Dataset:
    """Orchestrate image generation across multiple time slices.

    Args:
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        window_size: Number of unique time steps per sliding window.
        step: Stride between consecutive windows.
        image_maker: Image maker object.
        n_jobs: Number of parallel worker processes.
        lat_limits: ``(min_lat, max_lat)`` bounds.
        lon_limits: ``(min_lon, max_lon)`` bounds.

    Returns:
        A ``Dataset`` produced by concatenating all per-slice images along
        the ``time`` dimension, with ``None`` slices dropped.
    """
    slices = make_time_windows(obs["time"], window_size, step)
    times = [w.start_time for w in slices]

    q = Manager().Queue()
    root_logger = logging.getLogger()
    listener = QueueListener(q, *root_logger.handlers)

    try:
        listener.start()
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                delayed(run_image_generation)(
                    obs,
                    rx,
                    ts,
                    time,
                    image_maker,
                    lat_limits,
                    lon_limits,
                    q,
                )
                for ts, time in zip(slices, times, strict=True)
            )
    finally:
        listener.stop()

    return xarray.concat(filter(lambda x: x is not None, results), dim="time")
