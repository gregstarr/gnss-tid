import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray
from joblib import Parallel, delayed
from metpy import interpolate as mtpi
from scipy.interpolate import RBFInterpolator
from skimage import filters
from sklearn.metrics import pairwise_distances

from .coords import Local2D
from .parallel_logging import WorkerChannel, log_queue_listener
from .pointdata import TimeWindow, get_data

logger = logging.getLogger(__name__)


class ImageMakerBase(ABC):
    def __init__(self, hres, hp_freq=0.05, neighbor_radius=100, **kwargs):
        self.kwargs = kwargs
        self.hp_freq = hp_freq
        self.hres = hres
        self.neighbor_radius = neighbor_radius
        self.points = None
        self.shape = None
        self.xp = None
        self.yp = None
        self.area: float | None = None

    def initialize_from_bounds(
        self,
        proj: Local2D,
        lat_limits: tuple[float, float],
        lon_limits: tuple[float, float],
        n_perim: int = 21,
    ) -> None:
        """Initialize the interpolation grid from geodetic bounds.

        Projects a dense perimeter of the lat/lon box through ``proj`` (the
        same local-cartesian frame the data will be projected into) and takes
        the cartesian bounding box of the projected perimeter as the grid
        extent.

        Args:
            proj: Shared local-cartesian projection; pass the same instance
                used by :func:`gnss_tid.pointdata.get_data` so the grid and
                the data are guaranteed to live in the same frame.
            lat_limits: ``(min_lat, max_lat)`` of the region of interest.
            lon_limits: ``(min_lon, max_lon)`` of the region of interest.
            n_perim: Number of samples per lat/lon edge.  Larger values give a
                tighter bounding box when the projection curves noticeably.
        """
        lat_edge = np.linspace(lat_limits[0], lat_limits[1], n_perim)
        lon_edge = np.linspace(lon_limits[0], lon_limits[1], n_perim)
        perim_lat = np.concatenate(
            [
                np.full(n_perim, lat_limits[0]),
                lat_edge,
                np.full(n_perim, lat_limits[1]),
                lat_edge,
            ]
        )
        perim_lon = np.concatenate(
            [
                lon_edge,
                np.full(n_perim, lon_limits[1]),
                lon_edge,
                np.full(n_perim, lon_limits[0]),
            ]
        )
        x, y = proj.convert_from_spherical(perim_lat, perim_lon)
        boundary_coords = {
            "west": float(x.min()),
            "east": float(x.max()),
            "south": float(y.min()),
            "north": float(y.max()),
        }
        for k, v in boundary_coords.items():
            logger.info("image boundary %s: %.2f", k, v)
        x_grid, y_grid = mtpi.grid.generate_grid(self.hres, boundary_coords)
        self.points = mtpi.grid.generate_grid_coords(x_grid, y_grid)
        self.shape = x_grid.shape
        self.xp = x_grid[0]
        self.yp = y_grid[:, 0]
        self.area = float(
            (boundary_coords["east"] - boundary_coords["west"])
            * (boundary_coords["north"] - boundary_coords["south"])
        )

    @abstractmethod
    def _interpolate(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> np.ndarray:
        """Return interpolated values at ``self.points`` as a flat array."""

    def __call__(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> xarray.Dataset:
        if self.points is None:
            raise RuntimeError(
                "ImageMaker not initialized; call initialize_from_bounds() first"
            )

        img = self._interpolate(x, y, tec).reshape(self.shape)
        img[np.isnan(img)] = 0
        img = filters.butterworth(img, self.hp_freq, high_pass=True)
        img = xarray.DataArray(img, coords=[self.yp, self.xp], dims=["y", "x"])
        w = self.get_data_density(x, y, self.neighbor_radius)
        return xarray.Dataset({"image": img, "density": w})

    def get_data_density(
        self, x: np.ndarray, y: np.ndarray, threshold: float
    ) -> xarray.DataArray:
        pd = pairwise_distances(self.points, np.column_stack((x, y)))
        n = np.sum(pd < threshold, axis=1)
        w = n.reshape(self.shape)
        return xarray.DataArray(w, coords=[self.yp, self.xp], dims=["y", "x"])


class MetpyImageMaker(ImageMakerBase):
    def _interpolate(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> np.ndarray:
        pts = np.column_stack((x, y))
        return mtpi.interpolate_to_points(pts, tec, self.points, **self.kwargs)


class ScipyRbfImageMaker(ImageMakerBase):
    def _interpolate(self, x: np.ndarray, y: np.ndarray, tec: np.ndarray) -> np.ndarray:
        fin = np.isfinite(tec)
        pts = np.column_stack((x, y))[fin]
        kwargs = dict(self.kwargs)
        if "epsilon" not in kwargs:
            if self.area is None:
                raise RuntimeError(
                    "ScipyRbfImageMaker.area is unset; call initialize() or "
                    "initialize_from_bounds() before __call__, or set "
                    "`epsilon` in the constructor kwargs"
                )
            kwargs["epsilon"] = 1 / np.sqrt(self.area / len(pts))
        rbf = RBFInterpolator(pts, tec[fin], **kwargs)
        return rbf(self.points)


def generate_image(
    data: pd.DataFrame,
    image_maker: ImageMakerBase,
    tec_name: str,
) -> xarray.Dataset:
    """Interpolate a pre-fetched slice onto the image-maker's grid.

    Args:
        data: DataFrame already projected into the image_maker's frame; must
            contain ``x``, ``y``, and ``tec_name`` columns.
        image_maker: Initialized image maker.
        tec_name: Name of the TEC column in ``data``.

    Returns:
        A ``Dataset`` with variables ``image (y, x)`` and ``density (y, x)``.
    """
    return image_maker(
        data["x"].values, data["y"].values, data[tec_name].values
    )


def _build_image_for_slice(
    obs: pd.DataFrame,
    rx: pd.DataFrame,
    ts: TimeWindow,
    height: float,
    image_maker: ImageMakerBase,
    tec_name: str,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    proj: Local2D,
    ch: WorkerChannel | None = None,
) -> xarray.Dataset | None:
    """Fetch one time window and build its image at the given height.

    Returns a ``Dataset`` expanded along ``time`` (= ``ts.start_time``) with a
    ``height`` coord, or ``None`` if data retrieval failed.
    """
    with (ch or WorkerChannel()) as ch:
        ch.info("[%s-%s]: height = %.1f", ts.start_time, ts.end_time, height)
        data = get_data(obs, rx, ts, height, lat_limits, lon_limits, proj=proj)
        if data is None:
            ch.warning("[%s-%s]: FAIL", ts.start_time, ts.end_time)
            return None
        img = generate_image(data, image_maker, tec_name)
        return img.expand_dims(time=[ts.start_time]).assign_coords(
            height=("time", [height])
        )


def generate_image_stack(
    obs: pd.DataFrame,
    rx: pd.DataFrame,
    image_maker: ImageMakerBase,
    time_windows: list[TimeWindow],
    heights: float | np.ndarray,
    tec_name: str,
    lat_limits: tuple[float, float],
    lon_limits: tuple[float, float],
    proj: Local2D,
    n_jobs: int,
) -> xarray.Dataset:
    """Build an image for each time window in parallel.

    Args:
        obs: Observation DataFrame.
        rx: Receiver lookup DataFrame.
        image_maker: Initialized image maker (see
            :meth:`ImageMakerBase.initialize_from_bounds`).
        time_windows: Sequence of time windows produced by
            :func:`gnss_tid.pointdata.make_time_windows`.
        heights: Either a scalar IPP height (km, applied to every window) or
            a 1-D array of length ``len(time_windows)`` giving a per-window
            height.
        tec_name: Name of the TEC variable in the observation data.
        lat_limits: ``(min_lat, max_lat)`` bounds for data retrieval.
        lon_limits: ``(min_lon, max_lon)`` bounds for data retrieval.
        proj: Shared local-cartesian projection; must match the one used to
            initialize ``image_maker``.
        n_jobs: Number of parallel worker processes.

    Returns:
        A ``Dataset`` concatenated along ``time`` with variables ``image``,
        ``density``, and ``height(time)``.  Failed slices are dropped.
    """
    height_arr = np.broadcast_to(np.asarray(heights), (len(time_windows),))
    with log_queue_listener() as q, Parallel(n_jobs=n_jobs) as parallel:
        ch = WorkerChannel(log_queue=q)
        results = parallel(
            delayed(_build_image_for_slice)(
                obs, rx, ts, float(h), image_maker, tec_name,
                lat_limits, lon_limits, proj, ch,
            )
            for ts, h in zip(time_windows, height_arr, strict=True)
        )
    valid = [r for r in results if r is not None]
    if not valid:
        raise RuntimeError("image generation produced no valid slices")
    return xarray.concat(valid, dim="time")
