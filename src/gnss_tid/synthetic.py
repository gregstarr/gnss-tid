from numpy.typing import ArrayLike
import numpy as np
import xarray
import pandas


def constant_model(
        time: pandas.DatetimeIndex | None = None,
        center: ArrayLike = (100, 100),
        wavelength: float = 300,
        phase_speed: float = 200,
        xlim: ArrayLike = (-1000, 1000),
        ylim: ArrayLike = (-1000, 1000),
        hres: float = 20,
        offset: float = 0,
        snr: float = 0,
        ) -> xarray.DataArray:
    """constant center, wavelength and phase speed model

    Args:
        time (pandas.DatetimeIndex | None, optional): time. Defaults to None.
        center (ArrayLike, optional): center in km. Defaults to (100, 100).
        wavelength (float, optional): wavelength in km. Defaults to 300.
        phase_speed (float, optional): phase speed in m/s. Defaults to 200.
        xlim (ArrayLike, optional): region x limits in km. Defaults to (-1000, 1000).
        ylim (ArrayLike, optional): region y limits in km. Defaults to (-1000, 1000).
        hres (float, optional): horizontal resolution in km. Defaults to 20.
        offset (float, optional): phase offset in km. Defaults to 0.
        snr (float, optional): signal to noise ratio in dB. Defaults to 0.

    Returns:
        xarray.DataArray: noisy tec
    """
    if time is None:
        time = xarray.date_range("2025-01-01 00:00:00", "2025-01-01 01:00:00", freq="60s")
    time = xarray.DataArray(time, dims=["time"], name="time")
    x = xarray.DataArray(np.arange(xlim[0], xlim[1] + hres, hres), dims=["x"], name="x")
    y = xarray.DataArray(np.arange(ylim[0], ylim[1] + hres, hres), dims=["y"], name="y")
    r = np.hypot(x - center[0], y - center[1]).rename("r").assign_coords(x=x, y=y)
    t = (time - time[0]).dt.total_seconds()
    cycle = (r - (phase_speed / 1000) * t - offset) / wavelength
    tec = np.cos(2 * np.pi * cycle)
    
    noise_factor = np.sqrt((10 ** (snr / (-10))) / 2)
    noise = np.random.randn(*tec.shape) * noise_factor
    noisy_tec = tec + noise

    return noisy_tec
