"""Unit tests for the patch density gate in spectral focusing.

Covers the coverage computation and density cut in
:func:`gnss_tid.spectral.compute_patch_spectra` and the coverage gating in
:func:`gnss_tid.spectral.extract_patch_peaks`, on synthetic images with a
deliberate data hole.
"""

import numpy as np
import xarray as xr

from gnss_tid.fft import (
    fft_patches,
    make_kaiser_2d,
    make_patches,
    make_wavenum_grid,
)
from gnss_tid.spectral import compute_patch_spectra, extract_patch_peaks

BLOCK_SIZE = 32
BLOCK_STEP = 16
HRES = 20.0
# Data hole inside the central patch window (32 rows x 24 cols): the central
# patch keeps only 8 data columns, so its coverage is 0.25 (gated at 0.5) but
# its ungated peak power is still nonzero, and patches straddling the hole
# keep coverage >= 0.5.
HOLE_ROWS = slice(48, 80)
HOLE_COLS = slice(48, 72)


def _make_window() -> np.ndarray:
    """Build the FFT windowing array for the test block geometry."""
    return make_kaiser_2d(BLOCK_SIZE, 1.0).reshape(1, 1, BLOCK_SIZE, BLOCK_SIZE)


def _make_image_dataset(hole: bool = False) -> xr.Dataset:
    """Build a synthetic image Dataset with an optional rectangular data hole.

    The image is a pure sinusoid in ``x`` (amplitude 0.1, wavelength 200 km)
    on a 128x128 grid at 20 km/px.  ``density`` is 20 everywhere except, when
    ``hole`` is set, a 32x24-pixel rectangle inside the central patch window,
    where it is 0.
    """
    nx = ny = 128
    x = np.arange(nx) * HRES
    y = np.arange(ny) * HRES
    xx, _ = np.meshgrid(x, y)
    image = 0.1 * np.sin(2 * np.pi * xx / 200.0)
    density = np.full((ny, nx), 20.0)
    if hole:
        density[HOLE_ROWS, HOLE_COLS] = 0.0
    return xr.Dataset(
        {
            "image": (("y", "x"), image),
            "density": (("y", "x"), density),
        },
        coords={"x": x, "y": y},
    )


def _legacy_patch_spectra(ds: xr.Dataset) -> xr.DataArray:
    """Reproduce the pre-refactor patch spectra computation on ``ds.image``."""
    wavenum = make_wavenum_grid(BLOCK_SIZE, HRES, shift=True)
    F = fft_patches(
        make_patches(ds.image, BLOCK_SIZE, BLOCK_STEP),
        _make_window(),
        Nfft=BLOCK_SIZE,
        shift=True,
    )
    return (abs(F) ** 2).assign_coords(kx=wavenum, ky=wavenum)


def _expected_coverage(ds: xr.Dataset, py_vals, px_vals) -> xr.DataArray:
    """Compute the expected per-patch coverage from the ``density`` field.

    For each patch centre in ``py_vals``/``px_vals`` (32x32 pixel window,
    centred), the fraction of pixels with ``density > 10``.
    """
    density = ds.density.values
    half = BLOCK_SIZE // 2
    rows = []
    for py in py_vals:
        iy = round(float(py) / HRES)
        cols = []
        for px in px_vals:
            ix = round(float(px) / HRES)
            block = density[iy - half : iy + half, ix - half : ix + half]
            cols.append((block > 10).mean())
        rows.append(cols)
    return xr.DataArray(
        np.asarray(rows, dtype="float32"),
        coords={"py": np.asarray(py_vals), "px": np.asarray(px_vals)},
        dims=("py", "px"),
    )


def test_patch_coverage_and_gate():
    """Coverage matches the density field; the gate masks low-coverage patches."""
    ds = _make_image_dataset(hole=True)
    out = compute_patch_spectra(
        ds, BLOCK_SIZE, BLOCK_STEP, _make_window(), density_thresh=10.0
    )

    assert "patch_coverage" in out
    cov = out.patch_coverage
    assert set(cov.dims) == {"py", "px"}
    expected = _expected_coverage(ds, cov.py.values, cov.px.values)
    np.testing.assert_allclose(cov.values, expected.values)
    # The central patch keeps only 8 data columns; straddling patches keep >= 0.5.
    assert float(cov.sel(py=1280.0, px=1280.0).item()) == 0.25

    summary = extract_patch_peaks(out, min_coverage=0.5)
    gated = (cov < 0.5).values
    assert gated.sum() == 1
    for var in ("F", "Fx", "Fy"):
        vals = summary[var].values
        assert np.isnan(vals[gated]).all(), f"{var} not gated on the hole patch"
        assert np.isfinite(vals[~gated]).all(), f"{var} NaN on a supported patch"

    # The objective excludes gated patches (NaN skipped by the xarray sum), and
    # it is strictly smaller than the ungated objective: the gated patch's
    # window still holds data, so its ungated peak power is nonzero.
    np.testing.assert_allclose(summary.objective.values, np.nansum(summary.F.values))
    ungated = extract_patch_peaks(out, min_coverage=0.0)
    assert np.isfinite(ungated.F.values).all()
    assert float(summary.objective) < float(ungated.objective)

    # Stored patch spectra stay ungated (finite), but are computed on the
    # zero-filled cut image, so they differ from the legacy spectra in the hole.
    assert np.isfinite(out.patch.values).all()
    assert not np.allclose(out.patch.values, _legacy_patch_spectra(ds).values)


def test_patch_coverage_disabled_reproduces_legacy():
    """``density_thresh=None`` applies no cut and matches the legacy spectra."""
    ds = _make_image_dataset(hole=True)
    out = compute_patch_spectra(ds, BLOCK_SIZE, BLOCK_STEP, _make_window())

    assert "patch_coverage" not in out
    np.testing.assert_allclose(out.patch.values, _legacy_patch_spectra(ds).values)


def test_image_normalization_zscores_full_image():
    """Image normalization z-scores before extracting FFT patches."""
    ds = _make_image_dataset()
    out = compute_patch_spectra(
        ds,
        BLOCK_SIZE,
        BLOCK_STEP,
        _make_window(),
        normalization="image",
    )

    normalized = (ds.image - ds.image.mean(("x", "y"))) / ds.image.std(("x", "y"))
    expected = fft_patches(
        make_patches(normalized, BLOCK_SIZE, BLOCK_STEP),
        _make_window(),
        Nfft=BLOCK_SIZE,
        shift=True,
    )
    np.testing.assert_allclose(out.patch.values, (abs(expected) ** 2).values)
    np.testing.assert_allclose(ds.image.values, _make_image_dataset().image.values)


def test_patch_normalization_zscores_each_patch():
    """Patch normalization z-scores each extracted FFT block independently."""
    ds = _make_image_dataset()
    out = compute_patch_spectra(
        ds,
        BLOCK_SIZE,
        BLOCK_STEP,
        _make_window(),
        normalization="patch",
    )

    patches = make_patches(ds.image, BLOCK_SIZE, BLOCK_STEP)
    normalized = (patches - patches.mean(("kx", "ky"))) / patches.std(("kx", "ky"))
    expected = fft_patches(normalized, _make_window(), Nfft=BLOCK_SIZE, shift=True)
    np.testing.assert_allclose(out.patch.values, (abs(expected) ** 2).values)


def test_invalid_normalization_is_rejected():
    """Patch spectra reject normalization modes outside the public choices."""
    with np.testing.assert_raises_regex(ValueError, "normalization must be"):
        compute_patch_spectra(
            _make_image_dataset(),
            BLOCK_SIZE,
            BLOCK_STEP,
            _make_window(),
            normalization="invalid",
        )


def test_extract_patch_peaks_gate():
    """Coverage + ``min_coverage`` gates the peaks; ``min_coverage=0`` does not."""
    ds = _make_image_dataset(hole=False)
    legacy = compute_patch_spectra(ds, BLOCK_SIZE, BLOCK_STEP, _make_window())
    assert "patch_coverage" not in legacy

    patch2d = legacy.patch.isel(kx=0, ky=0)
    coverage = patch2d.copy(data=np.ones(patch2d.shape, dtype="float32"))
    coverage.values[2, 2] = 0.1
    gated_ds = legacy.assign(patch_coverage=coverage)

    gated = extract_patch_peaks(gated_ds, min_coverage=0.5)
    keep = np.ones(gated.F.shape, dtype=bool)
    keep[2, 2] = False
    for var in ("F", "Fx", "Fy"):
        vals = gated[var].values
        assert np.isnan(vals[2, 2]), f"{var} not gated on the low-coverage patch"
        assert np.isfinite(vals[keep]).all(), f"{var} NaN on a supported patch"
    np.testing.assert_allclose(gated.objective.values, np.nansum(gated.F.values))

    ungated = extract_patch_peaks(gated_ds, min_coverage=0.0)
    for var in ("F", "Fx", "Fy"):
        assert np.isfinite(ungated[var].values).all()
    assert float(gated.objective) < float(ungated.objective)

    # DataArray input (no coverage field) reproduces the ungated behaviour.
    from_da = extract_patch_peaks(legacy)
    np.testing.assert_allclose(from_da.F.values, ungated.F.values)
    np.testing.assert_allclose(from_da.objective.values, ungated.objective.values)
