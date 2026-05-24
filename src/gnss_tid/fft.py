"""Shared FFT helpers used by :mod:`gnss_tid.spectral` and :mod:`gnss_tid.parameter`.

These are intentionally small, pure functions; they exist to remove the
copy-paste of Kaiser-window construction, rolling-patch building, and
wavenumber-grid generation between the two callers.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr
from scipy.fft import fftfreq, fftshift
from scipy.signal.windows import kaiser


def make_kaiser_2d(
    block_size: int, beta: float, *, normalize: bool = True
) -> np.ndarray:
    """Return the 2-D outer-product Kaiser window.

    When ``normalize=True`` the window is divided by its sum, so that an FFT of
    a uniform image returns a value of one at DC.  This is the convention used
    by :func:`gnss_tid.parameter.estimate_parameters_block` and (since the
    Part B refactor) :mod:`apps.autofocus`.

    Args:
        block_size: Side length of the (square) window.
        beta: Kaiser shape parameter.
        normalize: Divide by ``sum(window)`` so the window has unit total mass.

    Returns:
        Array of shape ``(block_size, block_size)``.
    """
    k = kaiser(block_size, beta)
    w = np.outer(k, k)
    if normalize:
        w = w / w.sum()
    return w


def make_patches(
    img: xr.DataArray, block_size: int, step_size: int
) -> xr.DataArray:
    """Build rolling FFT patches from a 2-D image.

    Returns a DataArray with patch-centre dims ``(py, px)`` and per-patch
    spatial-offset dims ``(ky, kx)`` (the offsets are pre-FFT pixel indices;
    callers reassign them as wavenumber coordinates after taking the FFT).
    Trims ``block_size // (2 * step_size)`` patches from each spatial side so
    that no patch overlaps the image boundary.

    Each patch has shape ``(block_size, block_size)``.  Zero-padding for the
    FFT is **not** done here — the caller controls it via the ``s=`` argument
    of ``fft2``; this matches both
    :func:`gnss_tid.parameter.estimate_parameters_block`'s ``Nfft`` parameter
    and :func:`gnss_tid.spectral.get_fft_patches`'s implicit
    ``Nfft = block_size``.

    Dask: works transparently with dask-backed DataArrays because
    ``DataArray.rolling().construct()`` preserves dask chunking.  Not used by
    :func:`gnss_tid.parameter.estimate_parameters_dask`, which operates on raw
    dask arrays via ``da.overlap.sliding_window_view`` for explicit chunk
    control.

    Args:
        img: 2-D image with dims including ``y`` and ``x``; may carry extra
            leading dims (e.g. ``time``).
        block_size: Patch side length in pixels.
        step_size: Stride between consecutive patch centres in pixels.
    """
    edges = block_size // (2 * step_size)
    return (
        img.rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )


def make_wavenum_grid(
    Nfft: int,
    hres: float,
    *,
    shift: bool = False,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Return the 1-D wavenumber axis (cycles per km) for an ``Nfft``-point FFT.

    Args:
        Nfft: FFT length (post zero-padding).
        hres: Pixel spacing in km.
        shift: If True, return ``fftshift``-ed axis (DC at the centre).
        dtype: Result dtype.

    Returns:
        Array of shape ``(Nfft,)``.
    """
    wavenum = fftfreq(Nfft, hres)
    if shift:
        wavenum = fftshift(wavenum)
    return wavenum.astype(dtype)


__all__: Sequence[str] = ["make_kaiser_2d", "make_patches", "make_wavenum_grid"]
