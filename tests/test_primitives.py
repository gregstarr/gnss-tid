import numpy as np
import xarray as xr

from gnss_tid.center import initial_center_from_spectral_vectors
from gnss_tid.image import boundary_from_points, make_grid, interpolate_points_to_image
from gnss_tid.patches import image_patches_array, patch_centers
from gnss_tid.spectra import patch_power_xarray, spectral_peak


def test_grid_creation_and_patch_centers():
    boundary = boundary_from_points(np.array([0, 10]), np.array([-5, 5]))
    grid = make_grid(hres=5, boundary=boundary)

    assert grid.shape == (3, 3)
    np.testing.assert_allclose(patch_centers(np.arange(10), 4, 2), [2, 4, 6, 8])


def test_numpy_patch_extraction_shape():
    image = np.arange(25).reshape(5, 5)
    patches = image_patches_array(image, block_size=3, step_size=2)

    assert patches.shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(patches[0, 0], image[:3, :3])


def test_patch_fft_peak_on_plane_wave():
    x = np.arange(0, 64)
    y = np.arange(0, 64)
    X, _ = np.meshgrid(x, y)
    image = xr.DataArray(np.cos(2 * np.pi * X / 8), coords={"y": y, "x": x}, dims=("y", "x"))

    power = patch_power_xarray(image, hres=1, block_size=32, step_size=16, nfft=32)
    peak = spectral_peak(power)

    assert power.dims == ("py", "px", "ky", "kx")
    assert np.nanmedian(np.abs(peak.Fx.values)) == 0.125
    assert np.nanmedian(np.abs(peak.Fy.values)) == 0.0


def test_image_interpolation_smoke_returns_finite_dataset():
    x = np.array([0, 0, 1, 1], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=float)
    values = np.array([0, 1, 1, 0], dtype=float)

    image = interpolate_points_to_image(
        x,
        y,
        values,
        method="rbf",
        hres=0.5,
        hp_freq=None,
        density_radius=1.0,
        neighbors=4,
        smoothing=0.0,
    )

    assert set(image.data_vars) == {"image", "density"}
    assert np.isfinite(image.image.values).all()


def test_center_initialization_from_radial_vectors():
    cx, cy = 1.0, -2.0
    px = np.array([-1.0, 1.0, 3.0])
    py = np.array([-4.0, -2.0, 0.0])
    X, Y = np.meshgrid(px, py)
    Fx = X - cx
    Fy = Y - cy
    weights = np.ones_like(Fx)

    center, wavelength = initial_center_from_spectral_vectors(px, py, Fx, Fy, weights)

    np.testing.assert_allclose(center, [cx, cy], atol=1e-10)
    assert wavelength > 0
