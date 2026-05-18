from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from .center import (
    fit_radial_center,
    fit_stationary_center,
    initial_center_from_spectral_vectors,
)
from .image import make_grid, make_interpolator
from .plotting import plot_center_initialization, plot_objective_with_focus
from .spectra import patch_power_xarray, spectral_peak

logger = logging.getLogger(__name__)


ImageFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], xr.Dataset]
CenterFunction = Callable[[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray], dict]


def time_windows(times: Iterable, window: int, step: int):
    """Return autofocus time slices and representative times."""

    times = np.asarray(list(times))
    slices = [slice(i, i + window) for i in range(0, times.shape[0] - window, step)]
    labels = [times[i] for i in range(0, times.shape[0] - window, step)]
    return slices, labels


def best_initialization_slice(points, slices, height):
    """Select the time slice with the most finite point samples."""

    sizes = []
    for ts in slices:
        data = points.get_data(ts, height)
        sizes.append(0 if data is None else data.x.shape[0])
    if not sizes or max(sizes) == 0:
        raise ValueError("could not find a non-empty slice for image initialization")
    return slices[int(np.argmax(sizes))]


def build_image_function(
    points,
    slices: list[slice],
    heights: Iterable[float],
    *,
    tec_name: str,
    method: str = "rbf",
    hres: float = 20.0,
    hp_freq: float | None = 0.02,
    density_radius: float | None = 100.0,
    **kwargs,
) -> ImageFunction:
    """Build a fixed-grid scattered-point image function for autofocus."""

    heights = np.asarray(list(heights))
    init_height = heights[len(heights) // 2]
    init_slice = best_initialization_slice(points, slices, init_height)
    init_data = points.get_data(init_slice, init_height)
    if init_data is None:
        raise ValueError("could not initialize image grid from point data")
    grid = make_grid(init_data.x.values, init_data.y.values, hres=hres)
    logger.info("initialized image grid: shape=%s", grid.shape)
    return make_interpolator(
        method=method,
        hres=hres,
        grid=grid,
        hp_freq=hp_freq,
        density_radius=density_radius,
        **kwargs,
    )


def spectral_patch_power(
    image: xr.DataArray,
    *,
    hres: float,
    block_size: int = 32,
    block_step: int = 16,
    kaiser_beta: float = 1.0,
    logscale_objective: bool = False,
) -> xr.DataArray:
    """Compute the autofocus patch-power representation for an image."""

    return patch_power_xarray(
        image,
        hres=hres,
        block_size=block_size,
        step_size=block_step,
        nfft=block_size,
        kaiser_beta=kaiser_beta,
        logscale=logscale_objective,
        shifted=False,
    )


def make_height_images(
    points,
    time_slice: slice,
    heights: Iterable[float],
    *,
    image_fn: ImageFunction,
    tec_name: str = "dtec1",
    hres: float = 20.0,
    block_size: int = 32,
    block_step: int = 16,
    kaiser_beta: float = 1.0,
    logscale_objective: bool = False,
) -> xr.Dataset | None:
    """Interpolate one time window at many heights and attach spectral patches."""

    images = []
    patches = []
    npts = []
    used_heights = []
    for height in heights:
        data = points.get_data(time_slice, height)
        if data is None:
            logger.warning("[%03d-%03d]: empty height %.1f", time_slice.start, time_slice.stop, height)
            continue
        img = image_fn(data.x.values, data.y.values, data[tec_name].values)
        p = spectral_patch_power(
            img.image,
            hres=hres,
            block_size=block_size,
            block_step=block_step,
            kaiser_beta=kaiser_beta,
            logscale_objective=logscale_objective,
        )
        images.append(img)
        patches.append(p)
        npts.append(data.x.shape[0])
        used_heights.append(height)

    if not images:
        return None
    return (
        xr.concat(images, "height")
        .assign_coords(height=np.asarray(used_heights))
        .assign(n=("height", npts), patch=xr.concat(patches, "height"))
    )


def score_spectral_patches(data: xr.Dataset) -> xr.Dataset:
    """Select the strongest spectral component per patch and objective value."""

    return spectral_peak(data.patch)


def _default_center_fn(kind: str, kwargs: dict) -> CenterFunction:
    if kind == "stationary":
        return lambda c0, w0, x, y, tec: fit_stationary_center(c0, w0, x, y, tec, **kwargs)
    if kind == "radial":
        return lambda c0, w0, x, y, tec: fit_radial_center(c0, w0, x, y, tec, **kwargs)
    raise ValueError(f"unknown center algorithm: {kind}")


def _fit_center_for_focused_slice(
    points,
    focused: xr.Dataset,
    time_slice: slice,
    *,
    tec_name: str,
    center_fn: CenterFunction,
) -> dict:
    c0, w0 = initial_center_from_spectral_vectors(
        focused.px.values,
        focused.py.values,
        focused.Fx.values,
        focused.Fy.values,
        focused.F.values,
    )
    point_data = points.get_data(time_slice, float(focused.height.values), use_local_cs=True)
    if point_data is None:
        raise ValueError("focused height produced no point data for center fitting")
    return center_fn(c0, w0, point_data.x.values, point_data.y.values, point_data[tec_name].values)


def autofocus_spectral_blocks(
    points,
    *,
    heights: Iterable[float],
    window: int,
    step: int,
    tec_name: str = "dtec1",
    image_fn: ImageFunction | None = None,
    image_method: str = "rbf",
    image_kwargs: dict | None = None,
    center_fn: CenterFunction | None = None,
    center_algorithm: str = "radial",
    center_kwargs: dict | None = None,
    hres: float = 20.0,
    block_size: int = 32,
    block_step: int = 16,
    kaiser_beta: float = 1.0,
    logscale_objective: bool = False,
    n_jobs: int = 1,
) -> xr.Dataset:
    """Run the legacy spectral-block autofocus as a plain function."""

    heights = np.asarray(list(heights))
    slices, times = time_windows(points.times, window, step)
    if image_kwargs is None:
        image_kwargs = {}
    if center_kwargs is None:
        center_kwargs = {}
    if image_fn is None:
        image_fn = build_image_function(
            points,
            slices,
            heights,
            tec_name=tec_name,
            method=image_method,
            hres=hres,
            **image_kwargs,
        )
    if center_fn is None:
        center_fn = _default_center_fn(center_algorithm, center_kwargs)
    Path("plots").mkdir(exist_ok=True)

    def run_one(time_slice, time):
        data = make_height_images(
            points,
            time_slice,
            heights,
            image_fn=image_fn,
            tec_name=tec_name,
            hres=hres,
            block_size=block_size,
            block_step=block_step,
            kaiser_beta=kaiser_beta,
            logscale_objective=logscale_objective,
        )
        if data is None:
            return None
        scored = data.merge(score_spectral_patches(data))
        focused = (
            scored.isel(height=scored.objective.argmax())
            .expand_dims(time=[time])
            .reset_coords()
        )
        params = _fit_center_for_focused_slice(
            points,
            focused.isel(time=0),
            time_slice,
            tec_name=tec_name,
            center_fn=center_fn,
        )
        return focused.assign(
            cx=("time", [params["center"][0]]),
            cy=("time", [params["center"][1]]),
            wavelength=("time", [params["wavelength"]]),
            offset=("time", [params["offset"]]),
        )

    if n_jobs > 1:
        with tqdm_joblib(desc="autofocus", total=len(slices)):
            results = Parallel(n_jobs=n_jobs)(delayed(run_one)(ts, t) for ts, t in zip(slices, times))
    else:
        results = [run_one(ts, t) for ts, t in zip(slices, times)]

    valid = [r for r in results if r is not None]
    if not valid:
        raise ValueError("autofocus produced no valid time slices")
    return (
        xr.concat(valid, dim="time")
        .reindex(time=times)
        .assign_attrs(coord_center=points.get_coord_center())
    )


def autofocus_smoothed_spectral(
    points,
    *,
    heights: Iterable[float],
    window: int,
    step: int,
    tec_name: str = "dtec1",
    image_fn: ImageFunction | None = None,
    image_method: str = "rbf",
    image_kwargs: dict | None = None,
    center_fn: CenterFunction | None = None,
    center_algorithm: str = "stationary",
    center_kwargs: dict | None = None,
    hres: float = 20.0,
    block_size: int = 32,
    block_step: int = 16,
    kaiser_beta: float = 1.0,
    density_thresh: int = 10,
    time_window: int = 15,
    logscale_objective: bool = False,
    n_jobs: int = 1,
    diagnostics_dir: str | Path | None = None,
) -> xr.Dataset:
    """Run time-smoothed spectral autofocus as a plain function."""

    heights = np.asarray(list(heights))
    slices, times = time_windows(points.times, window, step)
    image_kwargs = image_kwargs or {}
    center_kwargs = center_kwargs or {}
    if image_fn is None:
        image_fn = build_image_function(
            points,
            slices,
            heights,
            tec_name=tec_name,
            method=image_method,
            hres=hres,
            **image_kwargs,
        )
    if center_fn is None:
        center_fn = _default_center_fn(center_algorithm, center_kwargs)
    Path("plots").mkdir(exist_ok=True)

    def score_one(time_slice, time):
        data = make_height_images(
            points,
            time_slice,
            heights,
            image_fn=image_fn,
            tec_name=tec_name,
            hres=hres,
            block_size=block_size,
            block_step=block_step,
            kaiser_beta=kaiser_beta,
            logscale_objective=logscale_objective,
        )
        if data is None:
            return None
        return score_spectral_patches(data).objective.expand_dims(time=[time])

    scores = [score_one(ts, t) for ts, t in zip(slices, times)]
    scores = xr.concat([s for s in scores if s is not None], "time").reindex(time=times)
    if logscale_objective:
        smoothed = scores.rolling(time=time_window, center=True, min_periods=1).mean()
    else:
        positive_scores = scores.where(scores > 0, np.finfo(float).tiny)
        smoothed = np.exp(
            np.log(positive_scores).rolling(time=time_window, center=True, min_periods=1).mean()
        )
    focus_height = (
        smoothed.dropna(dim="time")
        .isel(height=smoothed.dropna(dim="time").argmax(dim="height"))
        .reindex(time=smoothed.time)
    )
    if diagnostics_dir is not None:
        diagnostics_dir = Path(diagnostics_dir)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        fig, _ = plot_objective_with_focus(smoothed, focus_height)
        fig.savefig(diagnostics_dir / "objective.png")
        plt.close(fig)

    images = []
    for ii, (time_slice, time) in enumerate(zip(slices, times)):
        height = focus_height.isel(time=ii).height.item()
        point_data = points.get_data(time_slice, height)
        if point_data is None:
            continue
        img = image_fn(point_data.x.values, point_data.y.values, point_data[tec_name].values)
        patch = spectral_patch_power(
            img.image,
            hres=hres,
            block_size=block_size,
            block_step=block_step,
            kaiser_beta=kaiser_beta,
            logscale_objective=logscale_objective,
        )
        images.append(img.assign(patch=patch).expand_dims(time=[time]))

    if not images:
        raise ValueError("autofocus produced no focused images")
    data = xr.concat(images, "time")
    data = (
        data.merge(score_spectral_patches(data))
        .reindex(time=focus_height.time)
        .assign(height=focus_height.height)
    )
    sparse_img = (
        data.image.where(data.density > density_thresh)
        .stack(row=("x", "y"))
        .reset_index("row")
        .dropna(dim="time", how="all")
        .dropna(dim="row", how="all")
        .reset_coords()
    )
    init = data.isel(time=data.objective.argmax(dim="time"))
    if diagnostics_dir is not None:
        fig, _ = plot_center_initialization(init)
        fig.savefig(Path(diagnostics_dir) / "center_init.png")
        plt.close(fig)
    c0, w0 = initial_center_from_spectral_vectors(
        init.px.values,
        init.py.values,
        init.Fx.values,
        init.Fy.values,
        init.F.values,
    )
    params = center_fn(c0, w0, sparse_img.x.values, sparse_img.y.values, sparse_img.image.values.T)
    return data.assign(
        center=("ci", params["center"]),
        wavelength=xr.DataArray(params["wavelength"], coords={"time": sparse_img.time}),
        offset=xr.DataArray(params["offset"], coords={"time": sparse_img.time}),
        phase=xr.DataArray(params.get("phase", np.nan), coords={"time": sparse_img.time}),
    ).assign_attrs(coord_center=points.get_coord_center())
