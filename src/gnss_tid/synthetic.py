from numpy.typing import ArrayLike
import numpy as np
import dask.array as da
import xarray as xr
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
        ) -> xr.DataArray:
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
        xr.DataArray: noisy tec image dataset (with center)
    """
    if time is None:
        time = xr.date_range("2025-01-01 00:00:00", "2025-01-01 02:00:00", freq="60s")
    time = xr.DataArray(time, dims=["time"], name="time")
    x = xr.DataArray(np.arange(xlim[0], xlim[1] + hres, hres), dims=["x"], name="x")
    y = xr.DataArray(np.arange(ylim[0], ylim[1] + hres, hres), dims=["y"], name="y")
    r = np.hypot(x - center[0], y - center[1]).rename("r").assign_coords(x=x, y=y)
    t = (time - time[0]).dt.total_seconds()
    cycle = (r - (phase_speed / 1000) * t - offset) / wavelength
    tec = np.cos(2 * np.pi * cycle)
    
    noise_factor = np.sqrt((10 ** (snr / (-10))) / 2)
    print(f"{snr = }, {noise_factor = }")
    noise = np.random.randn(*tec.shape) * noise_factor
    noisy_tec = tec + noise
    density = xr.ones_like(tec) * 20

    return xr.Dataset({"image": noisy_tec, "density": density, "center": center})


def constant_model2(
        time: pandas.DatetimeIndex | None = None,
        center: ArrayLike = (100, 100),
        snr = da.arange(-3, 2, 1),
        wavelength = da.arange(150, 201, 50),
        tau = da.arange(40, 51, 10),
        xlim: ArrayLike = (-1000, 1000),
        ylim: ArrayLike = (-1000, 1000),
        hres: float = 20,
        offset: float = 0,
        ) -> xr.DataArray:
    if time is None:
        time = xr.date_range("2025-01-01 00:00:00", "2025-01-01 02:00:00", freq="60s")
    coords = xr.Coordinates(dict(
        snr=snr,
        lam=wavelength,
        tau=tau,
        time=time,
        x=da.arange(xlim[0], xlim[1] + hres, hres, chunks=-1),
        y=da.arange(ylim[0], ylim[1] + hres, hres, chunks=-1),
    ))
    r = xr.ufuncs.hypot(coords["x"] - center[0], coords["y"] - center[1]).rename("r").chunk("auto").persist()
    t = (coords["time"] - coords["time"][0]).dt.total_seconds()
    phase_speed = coords["lam"] * 1000 / (coords["tau"] * 60)
    cycle = (r - (phase_speed / 1000) * t - offset) / coords["lam"]
    tec = xr.ufuncs.cos(2 * np.pi * cycle).chunk("auto").persist()
    
    noise_factor = xr.ufuncs.sqrt((10 ** (coords["snr"] / (-10))) / 2)
    noise = xr.DataArray(
        da.random.normal(size=tec.shape, chunks=tec.data.chunksize),
        dims=tec.dims,
        coords=tec.coords,
    ) * noise_factor
    noisy_tec = tec + noise
    density = xr.ones_like(noisy_tec, chunks=noisy_tec.chunks) * 20

    data = xr.Dataset({"image": noisy_tec, "density": density, "center": center})
    data["image"] = data["image"].rolling(x=3, y=3, center=True, min_periods=1).median()
    return data.chunk(x=-1, y=-1, lam=1, tau=1, time=-1, snr="auto")
