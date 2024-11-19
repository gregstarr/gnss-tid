import logging
from logging.handlers import QueueHandler, QueueListener
import os
from multiprocessing import Manager

import numpy as np
import xarray
from skimage.util import view_as_windows
from scipy.fft import fft2
from scipy.signal.windows import kaiser
from joblib import Parallel, delayed

from pointdata import PointData
from image import ImageMaker
from plotting import plot_objective_vs_height

logger = logging.getLogger(__name__)


def configure_worker_logger(log_queue, log_level=logging.INFO):
    worker_logger = logging.getLogger('worker')
    if not worker_logger.hasHandlers():
        h = QueueHandler(log_queue)
        worker_logger.addHandler(h)
    worker_logger.setLevel(log_level)
    return worker_logger


class BlockSpectralFocusing:

    def __init__(
            self,
            image_maker: ImageMaker,
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
        self.heights = np.arange(height_min, height_max, height_step)
        self.block_shape = (block_size, block_size)
        self.block_step = block_step
        k = kaiser(block_size, kaiser_beta)
        self.window = np.outer(k, k).reshape(1, 1, block_size, block_size)
        self.quantile_thresh = quantile_thresh
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
        logger.info("running BlockSpectralProcessing")
        q = Manager().Queue()
        root_logger = logging.getLogger()
        listener = QueueListener(q, *root_logger.handlers)
        
        logger.info("initalizing image_maker")
        self.initialize_image_maker(points, window, step)

        logger.info("running time steps")
        os.mkdir("plots")
        slices, times = points.get_time_slices(window, step)
        listener.start()
        with Parallel(n_jobs=self.n_jobs) as parallel:
            results = parallel(
                self.run_time(points, ts, time, q) for ts, time in zip(slices, times)
            )
        listener.stop()
        logger.info("time steps finished")
        return xarray.concat(filter(lambda x: x is not None, results), dim="time")
    
    def run_time(self, points, ts: slice, time, log_queue):
        wlog = configure_worker_logger(log_queue)
        images = []
        patches = []
        npts = []
        for height in self.heights:
            wlog.info("[%02d-%02d]: height = %.1f", ts.start, ts.stop, height)
            data = points.get_data(ts, height)
            if data is None:
                wlog.warning("[%02d-%02d]: FAIL", ts.start, ts.stop)
                return
            npts.append(data.x.shape[0])
            img = self.image_maker(data.x.values, data.y.values, data.tec.values)
            patches.append(self.get_fft_patches(img))
            images.append(img)
        # FFTs (height, patchy, patchx, ffty, fftx)
        patches = np.stack(patches, axis=0)
        objective = self.get_objective(patches)
        best_i = np.argmax(objective)
        output = xarray.Dataset(
            data_vars={
                "image": (["time", "y", "x"], [images[best_i]]),
                "focus_height": (["time"], [self.heights[best_i]]),
                "objective": (["time", "height"], [objective]),
                "npts": (["time"], [npts[best_i]]),
            },
            coords={
                "time": [time],
                "x": self.image_maker.xp,
                "y": self.image_maker.yp,
                "height": self.heights,
            },
        )

        plot_objective_vs_height(
            self.heights,
            objective,
            f"{ts.start}_{ts.stop}",
            f"plots/obj_{ts.start:03d}.png"
        )

        wlog.info("[%02d-%02d]: SUCCESS", ts.start, ts.stop)
        return output

    def get_fft_patches(self, img) -> np.ndarray:
        blocks = view_as_windows(img, self.block_shape, self.block_step)
        fft_blocks = abs(fft2(blocks * self.window)) ** 2
        return fft_blocks

    def get_objective(self, patches) -> float:
        # fft value of best frequency in each patch, at any height
        patch_max = patches.max(axis=(0, 3, 4), keepdims=True)
        # index in FFTs of those values (remove duplicates)
        idx = np.argwhere(patches == patch_max)
        uidx = np.unique(idx[:, [1,2]], axis=0, return_index=True)[1]
        # selects max from FFT in each patch (leaving out height indexer because we
        # want to select freq in each patch at all heights, same freq for all heights)
        patch_freq_index = idx[uidx, 1:]
        # indexing (using argwhere) flattens out the spatial dimensions so this array
        # is 2D (heights x spatial) and has fft values at the selected frequencies
        patch_mag = patches[:, *tuple(patch_freq_index.T)]

        # possibly only select patches with strong frequency component
        if self.quantile_thresh > 0:
            max_over_height = np.max(patch_mag, axis=0)
            mask = max_over_height > np.quantile(max_over_height, self.quantile_thresh)
            patch_mag = patch_mag[:, mask]

        # sum over patches
        spatial_sum = patch_mag.sum(axis=1)
        return spatial_sum
