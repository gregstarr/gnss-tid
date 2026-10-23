"""Shared FFT helpers used by :mod:`gnss_tid.spectral` and :mod:`gnss_tid.parameter`.

These are intentionally small, pure functions; they exist to remove the
copy-paste of Kaiser-window construction, rolling-patch building, and
wavenumber-grid generation between the two callers.
"""

from collections.abc import Sequence
from typing import Literal

import dask.array as da
import numpy as np
import xarray as xr
from scipy.fft import fft2, fftfreq, fftshift
from scipy.signal.windows import kaiser


def make_kaiser_2d(block_size: int, beta: float, *, normalize: bool = True) -> np.ndarray:
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


def make_patches(img: xr.DataArray, block_size: int, step_size: int) -> xr.DataArray:
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
    and :func:`gnss_tid.spectral.compute_patch_spectra`'s implicit
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


def zscore(data: xr.DataArray, dims: str | Sequence[str]) -> xr.DataArray:
    """Z-score a DataArray along dims, mapping zero standard deviation to zero.

    Constant regions (standard deviation exactly zero — e.g. fully-void
    patches after the density cut) are mapped to zero instead of dividing by
    zero, so the result stays finite.

    Args:
        data: Input DataArray.
        dims: Dimension or sequence of dimensions over which the mean and
            standard deviation are computed.

    Returns:
        A DataArray with the same dims and coordinates, holding
        ``(data - mean) / std`` where ``std > 0`` and ``0.0`` where
        ``std == 0``.
    """
    std = data.std(dims)
    valid = std > 0
    mu = data.mean(dims)
    return xr.where(valid, (data - mu) / std.where(valid), 0.0)


def prepare_patch_image(
    image: xr.DataArray,
    density: xr.DataArray | None = None,
    density_thresh: float | None = None,
    normalization: Literal["image", "patch"] | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Apply the density cut and image normalization preceding patch FFTs.

    Single source of truth for the preprocessing that
    :func:`gnss_tid.spectral.compute_patch_spectra` applies before extracting
    and FFT-ing patches; diagnostic plots that must display the exact
    pre-FFT patch reuse this function instead of reimplementing it.

    Args:
        image: 2-D DataArray with dims ``x`` and ``y``.
        density: DataArray on the same dims as ``image``, or ``None``.
        density_thresh: Minimum density for a pixel to enter the patch FFT;
            pixels at or below the threshold are cut. ``None`` (default)
            applies no cut.
        normalization: ``"image"`` z-scores the cut, zero-filled image;
            ``None`` or ``"patch"`` leaves amplitudes unchanged here
            (``"patch"`` z-scoring happens per extracted block, after
            :func:`make_patches`).

    Returns:
        A 2-tuple ``(prepared, cut)``: ``prepared`` is the zero-filled (and,
        when requested, z-scored) image ready for :func:`make_patches`, and
        ``cut`` is the un-filled cut image (NaN where cut) when a cut was
        applied, else ``None``.
    """
    cut = None
    if density_thresh is not None and density is not None:
        cut = image.where(density > density_thresh)
        image = cut.fillna(0)
    if normalization == "image":
        image = zscore(image, ("x", "y"))
    return image, cut


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


def fft_patches[T: (xr.DataArray, da.Array, np.ndarray)](
    patches: T,
    window: np.ndarray,
    *,
    Nfft: int,
    shift: bool = False,
) -> T:
    """2-D FFT of windowed patches with optional zero-padding and fftshift.

    Dispatches on input type:

    - ``xr.DataArray``: uses :func:`xarray.apply_ufunc` with
      ``input_core_dims=[["kx", "ky"]]``, so dim ordering of the input is
      irrelevant.  Works transparently with dask-backed DataArrays.
    - ``dask.array.Array`` / ``np.ndarray``: operates on the last two axes;
      the caller owns axis ordering.

    Output dtype is ``complex64``.

    Args:
        patches: Patch array with trailing ``(kx, ky)`` axes of length
            ``block_size``.
        window: FFT window broadcastable against the trailing two axes
            of ``patches``.
        Nfft: FFT length per axis; pads when ``Nfft > block_size``.
        shift: If True, apply ``fftshift`` along the last two axes
            (DC at the centre).
    """
    if isinstance(patches, xr.DataArray):

        def _fft(x: np.ndarray) -> np.ndarray:
            F = fft2(x * window, s=(Nfft, Nfft))
            if shift:
                F = fftshift(F, axes=(-2, -1))
            return F

        return xr.apply_ufunc(
            _fft,
            patches,
            input_core_dims=[["kx", "ky"]],
            output_core_dims=[["kx", "ky"]],
            output_dtypes=[np.complex64],
            dask_gufunc_kwargs={"output_sizes": {"kx": Nfft, "ky": Nfft}},
            dask="parallelized",
            exclude_dims={"kx", "ky"},
        )
    if isinstance(patches, da.Array):
        F = da.fft.fft2(patches * window, s=(Nfft, Nfft))
        if shift:
            F = da.fft.fftshift(F, axes=(-2, -1))
        return F
    F = fft2(patches * window, s=(Nfft, Nfft))
    if shift:
        F = fftshift(F, axes=(-2, -1))
    return F


__all__: Sequence[str] = [
    "fft_patches",
    "make_kaiser_2d",
    "make_patches",
    "make_wavenum_grid",
    "prepare_patch_image",
    "zscore",
]
