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
from scipy.signal.windows import kaiser
from tqdm_joblib import tqdm_joblib

from .image import ImageMaker as ImageMakerBase
from .plotting import plot_center_finder
from .pointdata import get_data, make_time_windows
from .utils import find_center

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


def run_center_finder(
    obs: Any,
    rx: Any,
    F: xarray.Dataset,
    ts: slice,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    center_finder: Callable,
    tec_name: str,
) -> Any:
    """
    Perform center finding for a specific spectral patch.

    Args:
        obs: Observation data.
        rx: Receiver data.
        F: Spectral dataset containing the patch to analyze.
        ts: Time slice.
        lat_limits: Latitude limits for data retrieval.
        lon_limits: Longitude limits for data retrieval.
        center_finder: Callable to perform the center finding optimization.
        tec_name: Name of the TEC variable in the data.

    Returns:
        The result of the center finder optimization.
    """
    X, Y = np.meshgrid(F.px.values, F.py.values)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    weights = F.F.values.ravel()
    vectors = np.column_stack((F.Fx.values.ravel(), F.Fy.values.ravel()))
    k = np.hypot(vectors[:, 0], vectors[:, 1])
    c0 = find_center(pts, vectors, weights)
    w0 = 1 / k.max()

    data = get_data(obs, rx, ts, F.height.values, lat_limits, lon_limits)
    result = center_finder(
        c0, w0, data["x"].values, data["y"].values, data[tec_name].values
    )
    return result


def run_block_spectral_focusing(
    obs: Any,
    rx: Any,
    window: int,
    step: int,
    image_maker: Any,
    center_finder: Callable,
    heights: np.ndarray,
    block_shape: tuple[int, int],
    block_step: int,
    window_func: xarray.DataArray,
    logscale_objective: bool,
    n_jobs: int,
    tec_name: str,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
) -> xarray.Dataset:
    """
    Orchestrate the block spectral focusing pipeline.

    Args:
        obs: Observation data.
        rx: Receiver data.
        window: Time window size.
        step: Time window step.
        image_maker: Image maker object.
        center_finder: Center finder callable.
        heights: Array of heights to process.
        block_shape: Shape of the FFT blocks.
        block_step: Stride for block construction.
        window_func: Windowing function.
        logscale_objective: Whether to use log scale for the objective.
        n_jobs: Number of parallel jobs.
        tec_name: Name of the TEC variable.
        lat_limits: Latitude limits.
        lon_limits: Longitude limits.

    Returns:
        The final dataset containing focused spectral parameters.
    """
    # Initialize image maker
    slices = make_time_windows(obs["time"], window, step)
    mid_height = heights[len(heights) // 2]
    logger.info("running initializer")

    if n_jobs > 1:

        @delayed
        def fn(ts):
            data = get_data(obs, rx, ts, mid_height, lat_limits, lon_limits)
            return len(data) if data is not None else 0

        with (
            tqdm_joblib(desc="initializing", total=len(slices)),
            Parallel(n_jobs=n_jobs) as parallel,
        ):
            sizes = parallel(fn(ts) for ts in slices)
    else:
        sizes = []
        for ts in slices:
            data = get_data(obs, rx, ts, mid_height, lat_limits, lon_limits)
            sizes.append(len(data) if data is not None else 0)

    ii = np.argmax(sizes)
    logger.info("initializer finished, best slice -> %d: %d", ii, sizes[ii])
    init_data = get_data(obs, rx, slices[ii], mid_height, lat_limits, lon_limits)
    image_maker.initialize(init_data["x"].values, init_data["y"].values)

    logger.info("running BlockSpectralFocusing n_jobs=%d", n_jobs)
    Path("plots").mkdir(exist_ok=True)
    times = [w.start_time for w in slices]

    q = Manager().Queue()
    root_logger = logging.getLogger()
    listener = QueueListener(q, *root_logger.handlers)

    try:
        listener.start()
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
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
                    window_func,
                    logscale_objective,
                    center_finder,
                )
                for ts, time in zip(slices, times, strict=True)
            )
    finally:
        listener.stop()

    logger.info("time steps finished")
    coord_center = (np.mean(lat_limits), np.mean(lon_limits))
    data = (
        xarray.concat(filter(lambda x: x is not None, results), dim="time")
        .reindex(time=times)
        .assign_attrs(coord_center=coord_center)
    )
    return data


def run_smoothed_patch_spectral(
    obs,
    rx,
    window,
    step,
    image_maker,
    center_finder,
    heights,
    block_shape,
    block_step,
    window_func,
    logscale_objective,
    n_jobs,
    tec_name,
    lat_limits,
    lon_limits,
    time_window,
    density_thresh,
):
    slices = make_time_windows(obs["time"], window, step)
    times = [w.start_time for w in slices]

    q = Manager().Queue()
    root_logger = logging.getLogger()
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
                    window_func,
                    logscale_objective,
                    None,  # center_finder = None to get objective
                )
                for ts, time in zip(slices, times, strict=True)
            )
    finally:
        listener.stop()

    # objectives is a list of DataArrays (each with dim 'height')
    obj_data = xarray.concat(objectives, dim="time").reindex(time=times)

    # Smoothing logic
    if logscale_objective:
        smoothed = obj_data.rolling(time=time_window, center=True, min_periods=1).mean()
    else:
        smoothed = np.exp(
            np.log(obj_data).rolling(time=time_window, center=True, min_periods=1).mean()
        )

    smooth_nonull = smoothed.dropna(dim="time")
    focus_height = smooth_nonull.isel(height=smooth_nonull.argmax(dim="height")).reindex(
        time=smoothed.time
    )

    # Plots
    Path("plots").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
    smoothed.plot(ax=ax)
    ax.plot(focus_height.height, focus_height.time, "r.-")
    fig.savefig("plots/objective.png")
    plt.close(fig)

    # Second pass: collect focused data
    images = []
    for ii, (ts, time) in enumerate(zip(slices, times, strict=True)):
        logger.info("collecting focused data %d / %d", ii + 1, len(slices))
        height = focus_height.isel(time=ii).height.item()
        data = get_data(obs, rx, ts, height, lat_limits, lon_limits)
        if data is None:
            continue
        img = image_maker(data["x"].values, data["y"].values, data[tec_name].values)
        p = get_fft_patches(
            img.image,
            block_shape,
            image_maker.hres,
            block_step,
            window_func,
            logscale_objective,
        )
        img = img.assign(patch=p).expand_dims(time=[time])
        images.append(img)

    data_focused = xarray.concat(images, "time")
    data_focused = (
        data_focused.merge(process_patches(data_focused))
        .reindex(time=focus_height.time)
        .assign(height=focus_height.height)
    )

    # Sparse image for center finder
    sparse_img = (
        data_focused.image.where(data_focused.density > density_thresh)
        .stack(row=("x", "y"))
        .reset_index("row")
        .dropna(dim="time", how="all")
        .dropna(dim="row", how="all")
        .reset_coords()
    )

    # Find center
    # We need a standalone version of SmoothedPatchSpectral.run_center_finder
    # Let's define it below or above.
    params = run_smoothed_center_finder(data_focused, sparse_img, center_finder)

    logger.info("params fit in %d iterations", len(params["history"]["metric"]))
    data_focused = data_focused.assign(
        center=("ci", params["center"]),
        wavelength=xarray.DataArray(
            params["wavelength"], coords={"time": sparse_img.time}
        ),
        offset=xarray.DataArray(params["offset"], coords={"time": sparse_img.time}),
        phase=xarray.DataArray(params["phase"], coords={"time": sparse_img.time}),
    ).assign_attrs(coord_center=(np.mean(lat_limits), np.mean(lon_limits)))

    return data_focused


def run_smoothed_center_finder(
    data: xarray.Dataset,
    sparse_img: xarray.Dataset,
    center_finder: Callable,
) -> Any:
    """
    Perform center finding for a smoothed spectral dataset.

    Args:
        data: The focused spectral dataset.
        sparse_img: A sparse representation of the image for optimization.
        center_finder: Callable to perform the center finding optimization.

    Returns:
        The result of the center finder optimization.
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
    ts: slice,
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
    center_finder: Callable | None = None,
    return_data: bool = False,
) -> xarray.Dataset | xarray.DataArray | None:
    """
    Process a single time slice through the spectral focusing pipeline.

    Args:
        obs: Observation data.
        rx: Receiver data.
        ts: Time slice.
        time: Start time of the slice.
        log_queue: Queue for worker logging.
        heights: Array of heights to process.
        lat_limits: Latitude limits.
        lon_limits: Longitude limits.
        image_maker: Image maker object.
        tec_name: Name of the TEC variable.
        block_shape: Shape of the FFT blocks.
        block_step: Stride for FFT block construction.
        window: Windowing function.
        logscale_objective: Whether to use log scale for the objective.
        center_finder: Center finder callable.
        return_data: If True, returns the full dataset instead of just the objective.

    Returns:
        The processed data for the time slice, or None if processing failed.
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
        wlog.info("[%03d-%03d]: processing patches", ts.start, ts.stop)

        patches = process_patches(data).expand_dims(time=[time])

        if return_data:
            return data.merge(patches)

        if center_finder is None:
            return patches.objective

        # BlockSpectralFocusing logic
        full_data = data.merge(patches.squeeze("time"))
        best_height_idx = full_data.objective.argmax()
        output = (
            full_data.isel(height=best_height_idx).expand_dims(time=[time]).reset_coords()
        )

        wlog.info("[%03d-%03d]: finding params", ts.start, ts.stop)
        params = run_center_finder(
            obs, rx, output, ts, lat_limits, lon_limits, center_finder, tec_name
        )

        output = output.assign(
            cx=(["time"], [params["center"][0]]),
            cy=(["time"], [params["center"][1]]),
            wavelength=(["time"], [params["wavelength"]]),
            offset=(["time"], [params["offset"]]),
        )

        wlog.info(
            "[%03d-%03d]: params fit in %d iterations",
            ts.start,
            ts.stop,
            len(params["history"]["metric"]),
        )
        wlog.info("[%03d-%03d]: SUCCESS", ts.start, ts.stop)
        return output
    finally:
        cleanup_worker_logger(wlog, handler)


def get_fft_patches(
    img: xarray.DataArray,
    block_shape: tuple[int, int],
    hres: float,
    block_step: int,
    window: xarray.DataArray,
    logscale_objective: bool,
) -> xarray.DataArray:
    """
    Compute the FFT patches for a given image.

    Args:
        img: The input image DataArray.
        block_shape: Shape of the FFT blocks.
        hres: Horizontal resolution of the image.
        block_step: Stride for block construction.
        window: Windowing function.
        logscale_objective: Whether to use log scale for the objective.

    Returns:
        The computed FFT patches as a DataArray.
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
    """
    Identify the best FFT patch for each height.

    Args:
        data: Dataset containing the FFT patches.

    Returns:
        A dataset with the best patch and the associated objective value.
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
    ts: slice,
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
    """
    Process all requested heights for a given time slice.

    Args:
        obs: Observation data.
        rx: Receiver data.
        ts: Time slice.
        heights: Array of heights to process.
        lat_limits: Latitude limits.
        lon_limits: Longitude limits.
        image_maker: Image maker object.
        tec_name: Name of the TEC variable.
        block_shape: Shape of the FFT blocks.
        hres: Horizontal resolution.
        block_step: Stride for block construction.
        window: Windowing function.
        logscale_objective: Whether to use log scale for the objective.
        wlog: Optional logger for workers.

    Returns:
        The dataset containing images and patches for all heights, or None if processing failed.
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


class BlockSpectralFocusing:
    def __init__(
        self,
        image_maker: ImageMakerBase,
        center_finder,
        height_min: int,
        height_max: int,
        height_step: int,
        block_size: int,
        block_step: int,
        kaiser_beta: int,
        logscale_objective: bool,
        n_jobs: int,
        tec_name: str,
        lat_limits=None,
        lon_limits=None,
    ):
        self.n_jobs = n_jobs
        self.tec_name = tec_name
        self.image_maker = image_maker
        self.center_finder = center_finder
        self.heights = np.arange(height_min, height_max, height_step)
        self.block_shape = (block_size, block_size)
        self.block_step = block_step
        k = kaiser(block_size, kaiser_beta)
        self.window = np.outer(k, k).reshape(1, 1, block_size, block_size)
        self.logscale_objective = logscale_objective
        self.lat_limits = lat_limits
        self.lon_limits = lon_limits
        if n_jobs > 1:
            self.run_time = delayed(self.run_time)

    def initialize_image_maker(self, obs, rx, window: int, step: int):
        slices = make_time_windows(obs["time"], window, step)
        height = self.heights[len(self.heights) // 2]
        logger.info("running initializer")

        if self.n_jobs > 1:

            @delayed
            def fn(ts):
                data = get_data(obs, rx, ts, height, self.lat_limits, self.lon_limits)
                if data is None:
                    return 0
                s = len(data)
                return s

            with (
                tqdm_joblib(desc="initializing", total=len(slices)),
                Parallel(n_jobs=self.n_jobs) as parallel,
            ):
                sizes = parallel(fn(ts) for ts in slices)
        else:
            sizes = []
            for ts in slices:
                data = get_data(obs, rx, ts, height, self.lat_limits, self.lon_limits)
                if data is None:
                    return 0
                sizes.append(len(data))

        ii = np.argmax(sizes)
        logger.info("initializer finished, best slice -> %d: %d", ii, sizes[ii])
        data = get_data(obs, rx, slices[ii], height, self.lat_limits, self.lon_limits)
        self.image_maker.initialize(data["x"].values, data["y"].values)

    def run(self, obs, rx, window: int, step: int):
        return run_block_spectral_focusing(
            obs,
            rx,
            window,
            step,
            self.image_maker,
            self.center_finder,
            self.heights,
            self.block_shape,
            self.block_step,
            self.window,
            self.logscale_objective,
            self.n_jobs,
            self.tec_name,
            self.lat_limits,
            self.lon_limits,
        )

    def _sequential_get_results(self, obs, rx, slices, times):
        results = []
        for ts, time in zip(slices, times, strict=True):
            r = run_spectral_time_slice(
                obs,
                rx,
                ts,
                time,
                None,
                self.heights,
                self.lat_limits,
                self.lon_limits,
                self.image_maker,
                self.tec_name,
                self.block_shape,
                self.block_step,
                self.window,
                self.logscale_objective,
                self.center_finder,
            )
            results.append(r)
        return results

    def _parallel_get_results(self, obs, rx, slices, times):
        q = Manager().Queue()
        root_logger = logging.getLogger()
        listener = QueueListener(q, *root_logger.handlers)

        try:
            listener.start()
            with Parallel(n_jobs=self.n_jobs) as parallel:
                results = parallel(
                    delayed(run_spectral_time_slice)(
                        obs,
                        rx,
                        ts,
                        time,
                        q,
                        self.heights,
                        self.lat_limits,
                        self.lon_limits,
                        self.image_maker,
                        self.tec_name,
                        self.block_shape,
                        self.block_step,
                        self.window,
                        self.logscale_objective,
                        self.center_finder,
                    )
                    for ts, time in zip(slices, times, strict=True)
                )
        finally:
            listener.stop()
        return results

    def run_time(self, obs, rx, ts: slice, time, log_queue=None):
        # This method is kept for compatibility with some internal mechanisms
        # but now simply delegates to the functional version.
        return run_spectral_time_slice(
            obs,
            rx,
            ts,
            time,
            log_queue,
            self.heights,
            self.lat_limits,
            self.lon_limits,
            self.image_maker,
            self.tec_name,
            self.block_shape,
            self.block_step,
            self.window,
            self.logscale_objective,
            self.center_finder,
        )

    def run_center_finder(self, obs, rx, F, ts):
        X, Y = np.meshgrid(F.px.values, F.py.values)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        weights = F.F.values.ravel()
        vectors = np.column_stack((F.Fx.values.ravel(), F.Fy.values.ravel()))
        k = np.hypot(vectors[:, 0], vectors[:, 1])
        c0 = find_center(pts, vectors, weights)
        w0 = 1 / k.max()

        data = get_data(obs, rx, ts, F.height.values, self.lat_limits, self.lon_limits)
        result = self.center_finder(
            c0, w0, data["x"].values, data["y"].values, data[self.tec_name].values
        )
        return result

    def process_heights(self, obs, rx, ts, wlog=None):
        return process_heights(
            obs,
            rx,
            ts,
            self.heights,
            self.lat_limits,
            self.lon_limits,
            self.image_maker,
            self.tec_name,
            self.block_shape,
            self.image_maker.hres,
            self.block_step,
            self.window,
            self.logscale_objective,
            wlog=wlog,
        )

    def get_fft_patches(self, img: xarray.DataArray) -> xarray.DataArray:
        return get_fft_patches(
            img,
            self.block_shape,
            self.image_maker.hres,
            self.block_step,
            self.window,
            self.logscale_objective,
        )

    def process_patches(self, data) -> xarray.Dataset:
        return process_patches(data)


class SmoothedPatchSpectral(BlockSpectralFocusing):
    def __init__(self, *args, time_window=15, density_thresh=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_window = time_window
        self.density_thresh = density_thresh

    def run_time(self, obs, rx, ts: slice, time, log_queue=None, return_data=False):
        return run_spectral_time_slice(
            obs,
            rx,
            ts,
            time,
            log_queue,
            self.heights,
            self.lat_limits,
            self.lon_limits,
            self.image_maker,
            self.tec_name,
            self.block_shape,
            self.block_step,
            self.window,
            self.logscale_objective,
            None,  # For SmoothedPatchSpectral, we don't use center_finder during run_time
            return_data=return_data,
        )

    def run(self, obs, rx, window: int, step: int):
        return run_smoothed_patch_spectral(
            obs,
            rx,
            window,
            step,
            self.image_maker,
            self.center_finder,
            self.heights,
            self.block_shape,
            self.block_step,
            self.window,
            self.logscale_objective,
            self.n_jobs,
            self.tec_name,
            self.lat_limits,
            self.lon_limits,
            self.time_window,
            self.density_thresh,
        )

    def run_center_finder(self, data, sparse_img):
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

        result = self.center_finder(
            c0, w0, sparse_img.x.values, sparse_img.y.values, sparse_img.image.values.T
        )
        return result


def run_image_generation(
    obs: Any,
    rx: Any,
    ts: slice,
    time: Any,
    image_maker: Any,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    log_queue: Any | None = None,
) -> xarray.DataArray | None:
    """
    Generate an image for a specific time slice at a fixed height.

    Args:
        obs: Observation data.
        rx: Receiver data.
        ts: Time slice.
        time: Start time of the slice.
        image_maker: Image maker object.
        lat_limits: Latitude limits.
        lon_limits: Longitude limits.
        log_queue: Queue for worker logging.

    Returns:
        The generated image as a DataArray, or None if processing failed.
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
    window: int,
    step: int,
    image_maker: Any,
    n_jobs: int,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
) -> xarray.Dataset:
    """
    Orchestrate the image generation pipeline across multiple time slices.

    Args:
        obs: Observation data.
        rx: Receiver data.
        window: Time window size.
        step: Time window step.
        image_maker: Image maker object.
        n_jobs: Number of parallel jobs.
        lat_limits: Latitude limits.
        lon_limits: Longitude limits.

    Returns:
        The concatenated dataset of generated images.
    """
    slices = make_time_windows(obs["time"], window, step)
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


class ImageMaker:
    def __init__(
        self,
        image_maker: ImageMakerBase,
        n_jobs: int,
        lat_limits=None,
        lon_limits=None,
        **kwargs,
    ):
        self.image_maker = image_maker
        self.n_jobs = n_jobs
        self.lat_limits = lat_limits
        self.lon_limits = lon_limits

    def run(self, obs, rx, window: int, step: int):
        return run_image_maker_orchestrator(
            obs,
            rx,
            window,
            step,
            self.image_maker,
            self.n_jobs,
            self.lat_limits,
            self.lon_limits,
        )

    def run_time(self, obs, rx, ts: slice, time, log_queue=None):
        return run_image_generation(
            obs,
            rx,
            ts,
            time,
            self.image_maker,
            self.lat_limits,
            self.lon_limits,
            log_queue,
        )
