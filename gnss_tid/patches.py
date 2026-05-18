from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal.windows import kaiser


PATCH_DIMS = ("ky", "kx")


def kaiser2d(size: int, beta: float = 5.0, *, normalize: bool = True) -> np.ndarray:
    """Create a square 2-D Kaiser window."""

    win = kaiser(size, beta).astype("float32")
    window = np.outer(win, win)
    if normalize:
        window = window / np.sum(window)
    return window


def patch_centers(values: np.ndarray, block_size: int, step_size: int) -> np.ndarray:
    """Return coordinate values for centered rolling patches."""

    values = np.asarray(values)
    return values[block_size // 2 : -block_size // 2 + 1 : step_size]


def image_patches_array(
    image: np.ndarray,
    *,
    block_size: int,
    step_size: int,
) -> np.ndarray:
    """Extract centered spatial patches from a 2-D image."""

    image = np.asarray(image)
    patches = np.lib.stride_tricks.sliding_window_view(image, (block_size, block_size))
    return patches[::step_size, ::step_size]


def image_patches_xarray(
    image: xr.DataArray,
    *,
    block_size: int,
    step_size: int,
    patch_dims: tuple[str, str] = PATCH_DIMS,
) -> xr.DataArray:
    """Extract centered spatial patches from an xarray image stack."""

    edges = block_size // (2 * step_size)
    patches = (
        image.rolling(y=block_size, x=block_size, center=True)
        .construct(x=patch_dims[1], y=patch_dims[0], stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )
    return patches


def normalize_patches(patches: xr.DataArray | np.ndarray):
    """Normalize each patch independently, guarding zero-variance patches."""

    if isinstance(patches, xr.DataArray):
        dims = list(PATCH_DIMS)
        centered = patches - patches.mean(dims)
        scale = patches.std(dims)
        return centered.where(scale > 0, 0) / scale.where(scale > 0, 1)
    centered = patches - np.nanmean(patches, axis=(-2, -1), keepdims=True)
    scale = np.nanstd(patches, axis=(-2, -1), keepdims=True)
    return np.where(scale > 0, centered / scale, 0)
