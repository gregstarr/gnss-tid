from typing import Protocol
import logging

import numpy as np
import xarray
from metpy import interpolate as mtpi
from scipy.interpolate import RBFInterpolator
from skimage import filters
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


class ImageMaker(Protocol):
    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> xarray.DataArray: ...
    def initialize(self, x: np.ndarray, y: np.ndarray): ...
    def get_data_density(self, x, y, threshold) -> xarray.DataArray: ...


class MetpyImageMaker:
    def __init__(self, hres, hp_freq=.05, neighbor_radius=100, **kwargs):
        self.kwargs = kwargs
        self.hp_freq = hp_freq
        self.hres = hres
        self.neighbor_radius = neighbor_radius
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
    
    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> xarray.DataArray:
        if self.points is None:
            logger.warning("ImageMaker not initialized. Initializing from first inputs.")
            self.initialize(x, y)
        
        pts = np.column_stack((x, y))
        img = mtpi.interpolate_to_points(pts, tec, self.points, **self.kwargs)
        img = img.reshape(self.shape)

        img[np.isnan(img)] = 0
        img = filters.butterworth(img, self.hp_freq, high_pass=True)
        img = xarray.DataArray(img, coords=[self.yp, self.xp], dims=["y", "x"])
        w = self.get_data_density(x, y, self.neighbor_radius)
        return xarray.Dataset({"image": img, "density": w})
    
    def get_data_density(self, x, y, threshold):
        pd = pairwise_distances(self.points, np.column_stack((x, y)))
        n = np.sum(pd < threshold, axis=1)
        w = n.reshape(self.shape)
        return xarray.DataArray(w, coords=[self.yp, self.xp], dims=["y", "x"])


class ScipyRbfImageMaker:
    def __init__(self, hres, hp_freq=.05, neighbor_radius=100, **kwargs):
        self.kwargs = kwargs
        self.hp_freq = hp_freq
        self.hres = hres
        self.neighbor_radius = neighbor_radius
        self.points = None
        self.shape = None
        self.xp = None
        self.yp = None

    def initialize(self, x: np.ndarray, y: np.ndarray, boundary_coords=None):
        if boundary_coords is None:
            boundary_coords = mtpi.grid.get_boundary_coords(x, y)
        for k, v in boundary_coords.items():
            logger.info("image boundary %s: %.2f", k, v)
        x_grid, y_grid = mtpi.grid.generate_grid(self.hres, boundary_coords)
        self.points = mtpi.grid.generate_grid_coords(x_grid, y_grid)
        self.shape = x_grid.shape
        self.xp = x_grid[0]
        self.yp = y_grid[:, 0]
        # matches legacy rbf by default
        if "epsilon" not in self.kwargs:
            edges = np.array([
                boundary_coords["east"] - boundary_coords["west"],
                boundary_coords["north"] - boundary_coords["south"]
            ])
            self.kwargs["epsilon"] = 1 / np.power(np.prod(edges)/len(x), .5)
            logger.info("computed epsilon: %.2f", self.kwargs["epsilon"])
    
    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> xarray.DataArray:
        if self.points is None:
            logger.warning("ImageMaker not initialized. Initializing from first inputs.")
            self.initialize(x, y)
        
        pts = np.column_stack((x, y))
        rbf = RBFInterpolator(pts, tec, **self.kwargs)

        img = rbf(self.points)
        img = img.reshape(self.shape)

        img[np.isnan(img)] = 0
        img = filters.butterworth(img, self.hp_freq, high_pass=True)
        img = xarray.DataArray(img, coords=[self.yp, self.xp], dims=["y", "x"])
        w = self.get_data_density(x, y, self.neighbor_radius)
        return xarray.Dataset({"image": img, "density": w})
    
    def get_data_density(self, x, y, threshold):
        pd = pairwise_distances(self.points, np.column_stack((x, y)))
        n = np.sum(pd < threshold, axis=1)
        w = n.reshape(self.shape)
        return xarray.DataArray(w, coords=[self.yp, self.xp], dims=["y", "x"])