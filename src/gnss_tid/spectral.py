import logging
from logging.handlers import QueueHandler, QueueListener
from os import getpid
from pathlib import Path
from multiprocessing import Manager

import numpy as np
import xarray
from scipy.fft import fft2, fftfreq
from scipy.signal.windows import kaiser
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from .pointdata import PointData
from .image import ImageMaker

logger = logging.getLogger(__name__)


def configure_worker_logger(log_queue=None, log_level=logging.INFO):
    if log_queue is None:
        return logger, None
    worker_logger = logging.getLogger(f'worker {getpid()}')
    handler = QueueHandler(log_queue)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(log_level)
    return worker_logger, handler


def cleanup_worker_logger(worker_logger, handler):
    if handler is None:
        return
    worker_logger.removeHandler(handler)


def find_center(pts, vectors, weights):
    vec_norm = np.linalg.norm(vectors, axis=1)
    mask = vec_norm > 0
    w = np.sqrt(weights[mask]) / vec_norm[mask]
    A = np.column_stack([vectors[mask, 1], -vectors[mask, 0]]) * w[:, None]
    b = np.sum(A * pts[mask], axis=1)
    center, *_ = np.linalg.lstsq(A, b)
    return center


class BlockSpectralFocusing:

    def __init__(
            self,
            image_maker: ImageMaker,
            center_finder,
            height_min: int,
            height_max: int,
            height_step: int,
            block_size: int,
            block_step: int,
            kaiser_beta: int,
            quantile_thresh: float,
            logscale_objective: bool,
            n_jobs: int,
        ):
        self.n_jobs = n_jobs
        self.image_maker = image_maker
        self.center_finder = center_finder
        self.heights = np.arange(height_min, height_max, height_step)
        self.block_shape = (block_size, block_size)
        self.block_step = block_step
        k = kaiser(block_size, kaiser_beta)
        self.window = np.outer(k, k).reshape(1, 1, block_size, block_size)
        self.quantile_thresh = quantile_thresh
        self.logscale_objective = logscale_objective
        if n_jobs > 1:
            self.run_time = delayed(self.run_time)

    def initialize_image_maker(self, points: PointData, window: int, step: int):
        slices, times = points.get_time_slices(window, step)
        height = self.heights[len(self.heights)//2]
        logger.info("running initializer")

        @delayed
        def fn(ts):
            data = points.get_data(ts, height)
            if data is None:
                return 0
            s = data.x.shape[0]
            return s

        with Parallel(n_jobs=self.n_jobs) as parallel:
            sizes = parallel(fn(ts) for ts in slices)

        ii = np.argmax(sizes)
        logger.info("initializer finished, best slice -> %d: %d", ii, sizes[ii])
        data = points.get_data(slices[ii], height)
        self.image_maker.initialize(data.x.values, data.y.values)

    def run(self, points: PointData, window: int, step: int):
        self.initialize_image_maker(points, window, step)

        logger.info("running %s n_jobs=%d", self.__class__.__name__, self.n_jobs)
        Path("plots").mkdir(exist_ok=True)
        slices, times = points.get_time_slices(window, step)
        if self.n_jobs > 1:
            results = self._parallel_get_results(points, slices, times)
        else:
            results = self._sequential_get_results(points, slices, times)
        logger.info("time steps finished")
        data = (
            xarray.concat(filter(lambda x: x is not None, results), dim="time")
            .reindex(time=times)
            .assign_attrs(coord_center=points.get_coord_center())
        )
        return data
    
    def _sequential_get_results(self, points, slices, times):
        results = []
        for ts, time in zip(slices, times):
            r = self.run_time(points, ts, time, None)
            results.append(r)
        return results
    
    def _parallel_get_results(self, points, slices, times):
        q = Manager().Queue()
        root_logger = logging.getLogger()
        listener = QueueListener(q, *root_logger.handlers)
        
        try:
            listener.start()
            with Parallel(n_jobs=self.n_jobs) as parallel:
                results = parallel(
                    self.run_time(points, ts, time, q) for ts, time in zip(slices, times)
                )
        finally:
            listener.stop()
        return results
    
    def run_time(self, points, ts: slice, time, log_queue=None):
        wlog, handler = configure_worker_logger(log_queue)
        try:
            wlog.info("[%03d-%03d]: processing heights", ts.start, ts.stop)
            data = self.process_heights(points, ts, wlog)
            if data is None:
                return
            wlog.info("[%03d-%03d]: processing patches", ts.start, ts.stop)
            data = data.merge(self.process_patches(data))
            output = (
                data
                .isel(height=data.objective.argmax())
                .expand_dims(time=[time])
                .reset_coords()
            )
            wlog.info("[%03d-%03d]: finding params", ts.start, ts.stop)
            params = self.run_center_finder(points, output, ts)
            output = output.assign(
                cx=(["time"], [params["center"][0]]),
                cy=(["time"], [params["center"][1]]),
                wavelength=(["time"], [params["wavelength"]]),
                offset=(["time"], [params["offset"]]),
            )
            wlog.info(
                "[%03d-%03d]: params fit in %d iterations",
                ts.start, ts.stop, len(params["history"]["metric"])
            )
            wlog.info("[%03d-%03d]: SUCCESS", ts.start, ts.stop)
        finally:
            cleanup_worker_logger(wlog, handler)
        return output
    
    def run_center_finder(self, points: PointData, F, ts):
        X, Y = np.meshgrid(F.px.values, F.py.values)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        weights = F.F.values.ravel()
        vectors = np.column_stack((F.Fx.values.ravel(), F.Fy.values.ravel()))
        k = np.hypot(vectors[:, 0], vectors[:, 1])
        c0 = find_center(pts, vectors, weights)
        w0 = 1 / k.max()

        data = points.get_data(ts, F.height.values)
        result = self.center_finder(c0, w0, data.x.values, data.y.values, data.tec.values)
        return result
        
    def process_heights(self, points, ts, wlog=None):
        if wlog is None:
            wlog = logger
        images = []
        patches = []
        npts = []
        for height in self.heights:
            wlog.info("[%03d-%03d]: height = %.1f", ts.start, ts.stop, height)
            data = points.get_data(ts, height)
            if data is None:
                wlog.warning("[%03d-%03d]: FAIL", ts.start, ts.stop)
                return
            npts.append(data.x.shape[0])
            img = self.image_maker(data.x.values, data.y.values, data.tec.values)
            patches.append(self.get_fft_patches(img.image))
            images.append(img)
        
        data = (
            xarray.concat(images, "height")
            .assign_coords(height=self.heights)
            .assign(n=(["height"], npts), patch=xarray.concat(patches, "height"))
        )
        return data

    def get_fft_patches(self, img: xarray.DataArray) -> xarray.DataArray:
        wavenum = fftfreq(self.block_shape[0], self.image_maker.hres)
        edges = self.block_shape[0] // (2 * self.block_step)
        patches = (
            img
            .rolling(y=self.block_shape[0], x=self.block_shape[1], center=True)
            .construct(x="kx", y="ky", stride=self.block_step)
            .isel(x=slice(edges, -edges), y=slice(edges, -edges))
            .assign_coords(kx=wavenum, ky=wavenum)
            .rename({"x": "px", "y": "py"})
        )
        patches.values = abs(fft2(patches * self.window)) ** 2
        if self.logscale_objective:
            patches = np.log10(patches)
            
        return patches

    def process_patches(self, data) -> xarray.Dataset:
        result = (
            data.patch
            .isel(data.patch.argmax(dim=["kx", "ky"]))
            .to_dataset(name="F")
            .reset_coords()
            .rename_vars({"kx": "Fx", "ky": "Fy"})
            .assign(objective=lambda x: x.F.sum(dim=["px", "py"]))
        )
        return result


class SmoothedPatchSpectral(BlockSpectralFocusing):
    def __init__(self, *args, time_window=15, density_thresh=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_window = time_window
        self.density_thresh = density_thresh

    def run_time(self, points, ts: slice, time, log_queue=None, return_data=False):
        wlog, handler = configure_worker_logger(log_queue)
        try:
            wlog.info("[%03d-%03d]: processing heights", ts.start, ts.stop)
            data = self.process_heights(points, ts, wlog)
            if data is None:
                return
            wlog.info("[%03d-%03d]: processing patches", ts.start, ts.stop)
            
            patches = self.process_patches(data).expand_dims(time=[time])
        finally:
            cleanup_worker_logger(wlog, handler)
        if return_data:
            return data.merge(patches)
        return patches.objective
    
    def run(self, points: PointData, window: int, step: int):
        data = super().run(points, window, step)

        if self.logscale_objective:
            smoothed = (
                data
                .rolling(time=self.time_window, center=True, min_periods=1)
                .mean()
            )
        else:
            smoothed = np.exp(
                np.log(data)
                .rolling(time=self.time_window, center=True, min_periods=1)
                .mean()
            )
        smooth_nonull = smoothed.dropna(dim="time")
        focus_height = (
            smooth_nonull.isel(height=smooth_nonull.argmax(dim="height"))
            .reindex(time=smoothed.time)
        )

        fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
        smoothed.plot(ax=ax)
        ax.plot(focus_height.height, focus_height.time, 'r.-')
        fig.savefig("plots/objective.png")
        plt.close(fig)

        images = []
        slices, times = points.get_time_slices(window, step)
        for ii, (ts, time) in enumerate(zip(slices, times)):
            logger.info("collecting focused data %d / %d", ii + 1, len(slices))
            height = focus_height.isel(time=ii).height.item()
            data = points.get_data(ts, height)
            if data is None:
                continue
            img = self.image_maker(data.x.values, data.y.values, data.tec.values)
            p = self.get_fft_patches(img.image)
            img = img.assign(patch=p).expand_dims(time=[time])
            images.append(img)
        data = xarray.concat(images, "time")
        data = (
            data.merge(self.process_patches(data))
            .reindex(time=focus_height.time)
            .assign(height=focus_height.height)
        )

        # sparse_img = (
        #     data[["image", "density"]]
        #     .stack(row=("time", "x", "y"))
        #     .reset_index("row")
        # )
        # sparse_img = (
        #     sparse_img.image
        #     .where(sparse_img.density > self.density_thresh, drop=True)
        #     .reset_coords()
        # )
        sparse_img = (
            data.image
            .where(data.density > self.density_thresh)
            .stack(row=("x", "y"))
            .reset_index("row")
            .dropna(dim="time", how="all")
            .dropna(dim="row", how="all")
            .reset_coords()
        )

        params = self.run_center_finder(data, sparse_img)
        logger.info("params fit in %d iterations", len(params["history"]["metric"]))
        data = data.assign(
            center=("ci", params["center"]),
            wavelength=xarray.DataArray(params["wavelength"], coords={"time": sparse_img.time}),
            offset=xarray.DataArray(params["offset"], coords={"time": sparse_img.time}),
            phase=xarray.DataArray(params["phase"], coords={"time": sparse_img.time}),
        )

        return data

    def run_center_finder(self, data, sparse_img):
        logger.info("finding center")
        d = data.isel(time=data.objective.argmax())
        X, Y = np.meshgrid(d.px.values, d.py.values)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        weights = d.F.values.ravel()
        vectors = np.column_stack((d.Fx.values.ravel(), d.Fy.values.ravel()))
        k = np.hypot(vectors[:, 0], vectors[:, 1])
        c0 = find_center(pts, vectors, weights)
        w0 = 1 / k.max()
        
        result = self.center_finder(c0, w0, sparse_img.x.values, sparse_img.y.values, sparse_img.image.values.T)
        return result
