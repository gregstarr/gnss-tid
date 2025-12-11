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
        start = np.datetime64("2025-01-01T00:00:00")
        stop  = np.datetime64("2025-01-01T02:00:00")
        step  = np.timedelta64(60, "s")   # 60 seconds
        time = da.arange(start, stop + step, step, dtype="M8[ns]", chunks=1)
    
    coords = xr.Coordinates(dict(
        snr=snr,
        lam=wavelength,
        tau=tau,
        time=time,
        x=da.arange(xlim[0], xlim[1] + hres, hres, chunks=-1),
        y=da.arange(ylim[0], ylim[1] + hres, hres, chunks=-1),
    ))
    r = xr.ufuncs.hypot(coords["x"] - center[0], coords["y"] - center[1]).rename("r").chunk("auto")
    t = (coords["time"] - coords["time"][0]).dt.total_seconds().chunk("auto")
    phase_speed = (coords["lam"] * 1000 / (coords["tau"] * 60)).chunk({"lam": 1, "tau": 1})
    cycle = (r - (phase_speed / 1000) * t - offset) / coords["lam"]
    tec = xr.ufuncs.cos(2 * np.pi * cycle)
    
    noise_factor = xr.ufuncs.sqrt((10 ** (coords["snr"] / (-10))) / 2).chunk(5)
    noise = xr.DataArray(
        da.random.normal(size=tec.shape, chunks=tec.data.chunksize),
        dims=tec.dims,
        coords=tec.coords,
    ) * noise_factor
    noisy_tec = tec + noise
    density = xr.ones_like(noisy_tec, chunks=noisy_tec.chunks) * 20

    data = xr.Dataset({"image": noisy_tec, "density": density, "center": center})
    data["image"] = data["image"].rolling(x=3, y=3, center=True, min_periods=1).median()
    return data.chunk(x=-1, y=-1, lam=1, tau=1, time=-1, snr=1)


def constant_model3(
        time: pandas.DatetimeIndex | None = None,
        center: ArrayLike = (100, 100),
        snr_interval = [-6, 6],
        wavelength_interval = [150, 400],
        tau_interval = [10, 50],
        n_trials = 10,
        xlim: ArrayLike = (-1000, 1000),
        ylim: ArrayLike = (-1000, 1000),
        hres: float = 20,
        offset: float = 0,
        ) -> xr.DataArray:
    if time is None:
        start = np.datetime64("2025-01-01T00:00:00")
        stop  = np.datetime64("2025-01-01T01:00:00")
        step  = np.timedelta64(60, "s")   # 60 seconds
        time = da.arange(start, stop + step, step, dtype="M8[ns]", chunks=1)
    coords = xr.Coordinates(dict(
        trial=da.arange(0, n_trials, chunks=1),
        time=time,
        x=da.arange(xlim[0], xlim[1] + hres, hres, chunks=-1),
        y=da.arange(ylim[0], ylim[1] + hres, hres, chunks=-1),
    ))
    snr = np.random.random(n_trials) * (snr_interval[1] - snr_interval[0]) + snr_interval[0]
    wavelength = np.random.random(n_trials) * (wavelength_interval[1] - wavelength_interval[0]) + wavelength_interval[0]
    tau = np.random.random(n_trials) * (tau_interval[1] - tau_interval[0]) + tau_interval[0]
    coords = coords.assign(snr=("trial", snr), lam=("trial", wavelength), tau=("trial", tau))

    r = xr.ufuncs.hypot(coords["x"] - center[0], coords["y"] - center[1]).rename("r").chunk("auto")
    t = (coords["time"] - coords["time"][0]).dt.total_seconds().chunk("auto")
    phase_speed = (coords["lam"] * 1000 / (coords["tau"] * 60)).chunk(1)
    cycle = (r - (phase_speed / 1000) * t - offset) / coords["lam"]
    tec = xr.ufuncs.cos(2 * np.pi * cycle)
    noise_factor = xr.ufuncs.sqrt((10 ** (coords["snr"] / (-10))) / 2)
    noise = xr.DataArray(
        da.random.normal(size=tec.shape, chunks=tec.data.chunksize),
        dims=tec.dims,
        coords=tec.coords,
    ) * noise_factor
    noisy_tec = tec + noise
    density = xr.ones_like(noisy_tec, chunks=noisy_tec.chunks) * 20

    data = xr.Dataset({"image": noisy_tec, "density": density})
    data = data.assign_attrs(center=center)
    data["image"] = data["image"].rolling(x=3, y=3, center=True, min_periods=1).median()

    return data.chunk(x=-1, y=-1, time=-1, trial=1)
