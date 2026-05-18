from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol
from collections.abc import Mapping

import numpy as np
import xarray as xr
from metpy import interpolate as mtpi
from scipy.interpolate import RBFInterpolator
from skimage import filters
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageGrid:
    """Regular 2-D grid used for scattered-point image interpolation."""

    x: np.ndarray
    y: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y.size, self.x.size)

    @property
    def points(self) -> np.ndarray:
        x_grid, y_grid = np.meshgrid(self.x, self.y)
        return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def boundary_from_points(
    x: np.ndarray,
    y: np.ndarray,
    *,
    pad: float = 0.0,
) -> dict[str, float]:
    """Return MetPy-style grid boundaries from finite scattered points."""

    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError("cannot build an image boundary from empty/non-finite points")
    return {
        "west": float(np.nanmin(x[finite]) - pad),
        "east": float(np.nanmax(x[finite]) + pad),
        "south": float(np.nanmin(y[finite]) - pad),
        "north": float(np.nanmax(y[finite]) + pad),
    }


def make_grid(
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    hres: float,
    boundary: Mapping[str, float] | None = None,
    pad: float = 0.0,
) -> ImageGrid:
    """Create a regular image grid from explicit boundaries or point extents."""

    if boundary is None:
        if x is None or y is None:
            raise ValueError("x and y are required when boundary is not provided")
        boundary = boundary_from_points(x, y, pad=pad)

    x_grid, y_grid = mtpi.grid.generate_grid(float(hres), dict(boundary))
    return ImageGrid(x=x_grid[0].astype(float), y=y_grid[:, 0].astype(float))


def grid_points(grid: ImageGrid) -> np.ndarray:
    """Return flattened ``(x, y)`` coordinates for an image grid."""

    return grid.points


def data_density(
    x: np.ndarray,
    y: np.ndarray,
    grid: ImageGrid,
    *,
    radius: float,
) -> np.ndarray:
    """Count scattered points within ``radius`` of each grid pixel."""

    points = np.column_stack((np.asarray(x), np.asarray(y)))
    distances = pairwise_distances(grid.points, points)
    return np.sum(distances < radius, axis=1).reshape(grid.shape)


def interpolate_points_to_array(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    method: str = "rbf",
    hres: float = 20.0,
    grid: ImageGrid | None = None,
    boundary: Mapping[str, float] | None = None,
    hp_freq: float | None = 0.02,
    fill_value: float = 0.0,
    **kwargs,
) -> tuple[np.ndarray, ImageGrid]:
    """Interpolate scattered points to a regular image array."""

    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values)
    if grid is None:
        grid = make_grid(x, y, hres=hres, boundary=boundary)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if not finite.any():
        raise ValueError("cannot interpolate an image from empty/non-finite values")

    pts = np.column_stack((x[finite], y[finite]))
    target = grid.points
    method = method.lower()

    if method == "rbf":
        rbf_kwargs = dict(kwargs)
        if "epsilon" not in rbf_kwargs:
            edge = np.array([np.ptp(grid.x), np.ptp(grid.y)], dtype=float)
            rbf_kwargs["epsilon"] = 1 / np.sqrt(np.prod(edge) / finite.sum())
            logger.info("computed rbf epsilon: %.2f", rbf_kwargs["epsilon"])
        interpolator = RBFInterpolator(pts, values[finite], **rbf_kwargs)
        image = interpolator(target)
    elif method in {"metpy", "natural_neighbor", "barnes", "cressman"}:
        interp_type = kwargs.pop("interp_type", method)
        if interp_type == "metpy":
            raise ValueError("metpy interpolation requires an interp_type")
        image = mtpi.interpolate_to_points(pts, values[finite], target, interp_type=interp_type, **kwargs)
    else:
        raise ValueError(f"unknown interpolation method: {method}")

    image = np.asarray(image, dtype=float).reshape(grid.shape)
    image[np.isnan(image)] = fill_value
    if hp_freq is not None:
        image = filters.butterworth(image, hp_freq, high_pass=True)
    return image, grid


def interpolate_points_to_image(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    method: str = "rbf",
    hres: float = 20.0,
    grid: ImageGrid | None = None,
    boundary: Mapping[str, float] | None = None,
    density_radius: float | None = 100.0,
    hp_freq: float | None = 0.02,
    **kwargs,
) -> xr.Dataset:
    """Interpolate scattered points to an xarray image dataset."""

    image, grid = interpolate_points_to_array(
        x,
        y,
        values,
        method=method,
        hres=hres,
        grid=grid,
        boundary=boundary,
        hp_freq=hp_freq,
        **kwargs,
    )
    img = xr.DataArray(image, coords={"y": grid.y, "x": grid.x}, dims=("y", "x"))
    data_vars = {"image": img}
    if density_radius is not None:
        density = data_density(x, y, grid, radius=density_radius)
        data_vars["density"] = xr.DataArray(density, coords=img.coords, dims=img.dims)
    return xr.Dataset(data_vars)


def make_interpolator(
    *,
    method: str = "rbf",
    hres: float = 20.0,
    grid: ImageGrid | None = None,
    boundary: Mapping[str, float] | None = None,
    density_radius: float | None = 100.0,
    hp_freq: float | None = 0.02,
    **kwargs,
):
    """Return a small closure suitable for algorithm functions."""

    state = {"grid": grid}

    def _interpolate(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> xr.Dataset:
        ds = interpolate_points_to_image(
            x,
            y,
            values,
            method=method,
            hres=hres,
            grid=state["grid"],
            boundary=boundary,
            density_radius=density_radius,
            hp_freq=hp_freq,
            **kwargs,
        )
        if state["grid"] is None:
            state["grid"] = ImageGrid(x=ds.x.values, y=ds.y.values)
        return ds

    return _interpolate


class ImageMaker(Protocol):
    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> xr.DataArray: ...

    def initialize(self, x: np.ndarray, y: np.ndarray): ...

    def get_data_density(self, x, y, threshold) -> xr.DataArray: ...


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
        grid = make_grid(x, y, hres=self.hres)
        self.points = grid.points
        self.shape = grid.shape
        self.xp = grid.x
        self.yp = grid.y

    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> xr.DataArray:
        if self.points is None:
            logger.warning("ImageMaker not initialized. Initializing from first inputs.")
            self.initialize(x, y)

        pts = np.column_stack((x, y))
        img = mtpi.interpolate_to_points(pts, tec, self.points, **self.kwargs)
        img = img.reshape(self.shape)

        img[np.isnan(img)] = 0
        img = filters.butterworth(img, self.hp_freq, high_pass=True)
        img = xr.DataArray(img, coords=[self.yp, self.xp], dims=["y", "x"])
        w = self.get_data_density(x, y, self.neighbor_radius)
        return xr.Dataset({"image": img, "density": w})

    def get_data_density(self, x, y, threshold):
        pd = pairwise_distances(self.points, np.column_stack((x, y)))
        n = np.sum(pd < threshold, axis=1)
        w = n.reshape(self.shape)
        return xr.DataArray(w, coords=[self.yp, self.xp], dims=["y", "x"])


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
            boundary_coords = boundary_from_points(x, y)
        grid = make_grid(hres=self.hres, boundary=boundary_coords)
        self.points = grid.points
        self.shape = grid.shape
        self.xp = grid.x
        self.yp = grid.y
        if "epsilon" not in self.kwargs:
            edges = np.array([
                boundary_coords["east"] - boundary_coords["west"],
                boundary_coords["north"] - boundary_coords["south"]
            ])
            self.kwargs["epsilon"] = 1 / np.power(np.prod(edges)/len(x), .5)
            logger.info("computed epsilon: %.2f", self.kwargs["epsilon"])

    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> xr.DataArray:
        if self.points is None:
            logger.warning("ImageMaker not initialized. Initializing from first inputs.")
            self.initialize(x, y)

        fin = np.isfinite(tec)
        pts = np.column_stack((x, y))[fin]
        rbf = RBFInterpolator(pts, tec[fin], **self.kwargs)

        img = rbf(self.points)
        img = img.reshape(self.shape)

        img[np.isnan(img)] = 0
        img = filters.butterworth(img, self.hp_freq, high_pass=True)
        img = xr.DataArray(img, coords=[self.yp, self.xp], dims=["y", "x"])
        w = self.get_data_density(x, y, self.neighbor_radius)
        return xr.Dataset({"image": img, "density": w})

    def get_data_density(self, x, y, threshold):
        pd = pairwise_distances(self.points, np.column_stack((x, y)))
        n = np.sum(pd < threshold, axis=1)
        w = n.reshape(self.shape)
        return xr.DataArray(w, coords=[self.yp, self.xp], dims=["y", "x"])
