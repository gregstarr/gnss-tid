import numpy as np
import pandas as pd
import xarray as xr

from gnss_tid.autofocus import autofocus_smoothed_spectral
from gnss_tid.parameter import estimate_parameters_block, estimate_parameters_dask


def _tiny_plane_wave():
    time = pd.date_range("2025-01-01", periods=8, freq="60s")
    x = np.arange(0, 64, 4)
    y = np.arange(0, 64, 4)
    X, _ = np.meshgrid(x, y)
    images = []
    for ii in range(time.size):
        images.append(np.cos(2 * np.pi * (X / 32 - ii / 8)))
    return xr.Dataset(
        {"image": (("time", "y", "x"), np.asarray(images, dtype="float32"))},
        coords={"time": time, "y": y, "x": x},
    )


def test_estimate_parameters_dask_is_lazy_and_scientifically_matches_block():
    data = _tiny_plane_wave()
    dask_data = data.chunk({"time": -1, "x": 8, "y": 8})

    params = estimate_parameters_dask(
        dask_data,
        nfft=16,
        block_size=8,
        step_size=4,
        smooth_win=3,
        q_threshold=0.90,
    )
    block = estimate_parameters_block(
        data,
        nfft=16,
        block_size=8,
        step_size=4,
        smooth_win=3,
        q_threshold=0.90,
    )

    assert hasattr(params.phase_velocity.data, "dask")
    assert params.phase_velocity.dims == block.phase_velocity.dims
    dask_wavelength = params.wavelength.compute()
    assert np.isfinite(dask_wavelength).any()
    assert np.isfinite(block.wavelength).any()
    np.testing.assert_allclose(
        float(dask_wavelength.median(skipna=True)),
        float(block.wavelength.median(skipna=True)),
        rtol=0.25,
    )


class FakePointData:
    def __init__(self):
        self.times = pd.date_range("2025-01-01", periods=6, freq="60s").values

    def get_data(self, time_slice, h, use_local_cs=True):
        x = np.linspace(-10, 10, 9)
        y = np.linspace(-10, 10, 9)
        return xr.Dataset(
            {"dtec1": ("los_id", np.full(x.shape, h, dtype=float))},
            coords={"los_id": np.arange(x.size), "x": ("los_id", x), "y": ("los_id", y)},
        )

    def get_coord_center(self):
        return (0.0, 0.0)


def test_autofocus_smoothed_spectral_selects_known_height():
    x = np.arange(0, 32)
    y = np.arange(0, 32)
    X, _ = np.meshgrid(x, y)

    def image_fn(_x, _y, values):
        height = float(np.nanmean(values))
        amplitude = 10.0 - abs(height - 200.0)
        image = amplitude * np.cos(2 * np.pi * X / 8)
        density = np.ones_like(image) * 20
        return xr.Dataset(
            {
                "image": (("y", "x"), image),
                "density": (("y", "x"), density),
            },
            coords={"y": y, "x": x},
        )

    def center_fn(c0, w0, x, y, tec):
        return {
            "center": np.asarray([0.0, 0.0]),
            "wavelength": np.full(5, 8.0),
            "offset": np.zeros(5),
            "phase": np.zeros(5),
            "history": {"metric": [1.0]},
        }

    result = autofocus_smoothed_spectral(
        FakePointData(),
        heights=[190.0, 200.0, 210.0],
        window=1,
        step=1,
        image_fn=image_fn,
        center_fn=center_fn,
        hres=1,
        block_size=16,
        block_step=8,
        density_thresh=1,
        time_window=1,
    )

    np.testing.assert_allclose(result.height.values, 200.0)
