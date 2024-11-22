import logging
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from multiprocessing import Manager

import numpy as np
import xarray
from scipy.fft import fft2, fftfreq
from scipy.signal.windows import kaiser
from joblib import Parallel, delayed

from .pointdata import PointData
from .image import ImageMaker

logger = logging.getLogger(__name__)


def configure_worker_logger(log_queue=None, log_level=logging.INFO):
    if log_queue is None:
        return logger
    worker_logger = logging.getLogger('worker')
    if not worker_logger.hasHandlers():
        h = QueueHandler(log_queue)
        worker_logger.addHandler(h)
    worker_logger.setLevel(log_level)
    return worker_logger


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

        logger.info("running BlockSpectralProcessing")
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
        
        listener.start()
        with Parallel(n_jobs=self.n_jobs) as parallel:
            results = parallel(
                self.run_time(points, ts, time, q) for ts, time in zip(slices, times)
            )
        listener.stop()
        return results
    
    def run_time(self, points, ts: slice, time, log_queue=None):
        wlog = configure_worker_logger(log_queue)
        wlog.info("[%03d-%03d]: processing heights", ts.start, ts.stop)
        data = self.process_heights(points, ts, wlog)
        if data is None:
            return
        wlog.info("[%03d-%03d]: processing patches", ts.start, ts.stop)
        data = self.process_patches(data)
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
            patches.append(self.get_fft_patches(img))
            images.append(img)
        
        data = (
            xarray.concat(images, "height")
            .assign_coords(height=self.heights)
            .to_dataset(name="image")
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
        return data.merge(result)
