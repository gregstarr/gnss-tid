from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.fft import fft2, fftfreq, fftshift

from .patches import PATCH_DIMS, image_patches_xarray, kaiser2d

KDIMS = ["kx", "ky"]


def wavenumbers(nfft: int, hres: float, *, shifted: bool = True) -> np.ndarray:
    """Return FFT wavenumber coordinates in cycles/km."""

    k = fftfreq(nfft, hres).astype("float32")
    if shifted:
        k = fftshift(k)
    return k


def fft_patches_array(
    patches: np.ndarray,
    *,
    nfft: int | None = None,
    window: np.ndarray | None = None,
    shifted: bool = True,
) -> np.ndarray:
    """FFT an array whose final two dimensions are patch pixels."""

    if nfft is None:
        nfft = patches.shape[-1]
    x = patches if window is None else patches * window
    F = fft2(x, s=(nfft, nfft), axes=(-2, -1))
    if shifted:
        F = fftshift(F, axes=(-2, -1))
    return F


def spectral_power(F):
    """Return squared spectral magnitude."""

    return abs(F) ** 2


def patch_power_xarray(
    image: xr.DataArray,
    *,
    hres: float,
    block_size: int,
    step_size: int,
    nfft: int | None = None,
    kaiser_beta: float = 5.0,
    logscale: bool = False,
    shifted: bool = False,
) -> xr.DataArray:
    """Compute patch FFT power for an image stack."""

    if nfft is None:
        nfft = block_size
    patches = image_patches_xarray(image, block_size=block_size, step_size=step_size)
    window = kaiser2d(block_size, kaiser_beta, normalize=False)

    F = xr.apply_ufunc(
        lambda x: fft_patches_array(x, nfft=nfft, window=window, shifted=shifted),
        patches,
        input_core_dims=[list(PATCH_DIMS)],
        output_core_dims=[list(PATCH_DIMS)],
        output_dtypes=[np.complex64],
        dask_gufunc_kwargs={"output_sizes": {"ky": nfft, "kx": nfft}},
        dask="parallelized",
        exclude_dims=set(PATCH_DIMS),
    )
    k = wavenumbers(nfft, hres, shifted=shifted)
    power = spectral_power(F).assign_coords(kx=k, ky=k)
    if logscale:
        power = np.log10(power)
    return power


def spectral_peak(power: xr.DataArray) -> xr.Dataset:
    """Select the strongest k-space component for each patch."""

    return (
        power.isel(power.argmax(dim=KDIMS))
        .to_dataset(name="F")
        .reset_coords()
        .rename_vars({"kx": "Fx", "ky": "Fy"})
        .assign(objective=lambda x: x.F.sum(dim=["px", "py"]))
    )


def spectral_weights(power, q_threshold: float):
    """Threshold spectral power and normalize weights across k-space."""

    if isinstance(power, xr.DataArray):
        threshold = power.quantile(q_threshold, KDIMS).drop_vars("quantile")
        W = power.where(power > threshold, 0)
        return W / W.sum(KDIMS), threshold

    threshold = np.quantile(power, q_threshold, axis=(-2, -1), keepdims=True)
    W = np.where(power > threshold, power, 0)
    return W / np.sum(W, axis=(-2, -1), keepdims=True), threshold[..., 0, 0]
