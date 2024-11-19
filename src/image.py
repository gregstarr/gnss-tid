from typing import Protocol
import logging

import numpy as np
from metpy import interpolate as mtpi
from skimage import filters

logger = logging.getLogger(__name__)


class ImageMaker(Protocol):
    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> np.ndarray: ...
    def initialize(self, x: np.ndarray, y: np.ndarray): ...


class MetpyImageMaker:
    def __init__(self, hres, hp_freq=.05, **kwargs):
        self.kwargs = kwargs
        self.hp_freq = hp_freq
        self.hres = hres
        self.points = None
        self.shape = None
        self.xp = None
        self.yp = None

    def initialize(self, x: np.ndarray, y: np.ndarray):
        boundary_coords = mtpi.grid.get_boundary_coords(x, y)
        for k, v in boundary_coords.items():
            logger.info("image boundary %s: %.2f", k, v)
        x_grid, y_grid = mtpi.grid.generate_grid(self.hres, boundary_coords)
        self.points = mtpi.grid.generate_grid_coords(x_grid, y_grid)
        self.shape = x_grid.shape
        self.xp = x_grid[0]
        self.yp = y_grid[:, 0]
    
    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> np.ndarray:
        pts = np.column_stack((x, y))
        img = mtpi.interpolate_to_points(pts, tec, self.points, **self.kwargs)
        img = img.reshape(self.shape)

        img[np.isnan(img)] = 0
        img = filters.butterworth(img, self.hp_freq, high_pass=True)
        return img
