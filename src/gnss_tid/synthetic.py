import logging

from numpy.typing import ArrayLike
import numpy as np
import dask.array as da
import xarray as xr
import pandas


def spherical_model(
        time: pandas.DatetimeIndex | None = None,
        center: ArrayLike = (100, 100),
        snr_lim = [-6, 6],
        lam_lim = [150, 400],
        tau_lim = [10, 50],
        n_trials = 10,
        xlim: ArrayLike = (-1000, 1000),
        ylim: ArrayLike = (-1000, 1000),
        hres: float = 20,
        batch_size: int = 20,
    ) -> xr.DataArray:
    if time is None:
        time = pandas.date_range("2025-01-01", "2025-01-01T01:00:00", freq="60s")
    
    # don't chunk trial=1 for "generation + write" workflows with many trials. trial=1 
    # creates thousands of tiny blocks, which inflates the task graph and slows scheduling.
    # Use a reasonable batch size (e.g., 20–100) to keep graph size and overhead down.
    batch_size = min(batch_size, n_trials)
    
    trials = np.arange(0, n_trials)
    
    # keep per-trial parameters as simple NumPy (then chunk once). Using 
    # da.random(..., chunks=1) for these made the graph larger because it adds RNG tasks
    # (and chunks=1 adds one chunk per trial). These arrays are small, so NumPy is fine.
    snr = xr.DataArray(
        np.random.rand(n_trials) * (snr_lim[1] - snr_lim[0]) + snr_lim[0],
        coords=[("trial", trials)],
    ).chunk(trial=batch_size)
    lam = xr.DataArray(
        np.random.rand(n_trials) * (lam_lim[1] - lam_lim[0]) + lam_lim[0],
        coords=[("trial", trials)],
    ).chunk(trial=batch_size)
    tau = xr.DataArray(
        np.random.rand(n_trials) * (tau_lim[1] - tau_lim[0]) + tau_lim[0],
        coords=[("trial", trials)],
    ).chunk(trial=batch_size)

    # Important: x/y/t are "broadcast drivers" for the big (trial,time,x,y) arrays.
    # Even if they're small 1-D vectors, making them dask-backed prevents eager broadcast
    # intermediates (like r) and avoids repeatedly embedding NumPy constants into many tasks,
    # which can bloat the serialized graph sent to the scheduler.
    xlab = np.arange(xlim[0], xlim[1] + hres, hres)
    ylab = np.arange(ylim[0], ylim[1] + hres, hres)
    x = xr.DataArray(da.from_array(xlab, chunks=-1), coords=[("x", xlab)])
    y = xr.DataArray(da.from_array(ylab, chunks=-1), coords=[("y", ylab)])
    r = da.hypot(x - center[0], y - center[1]).rename("r")
    # We keep t numeric (seconds) instead of leaning on heavy datetime ops in the graph.
    # (Datetime arithmetic/total_seconds can create extra graph overhead.)
    t = xr.DataArray(
        da.from_array((time - time[0]).total_seconds().values, chunks=-1),
        coords=[("time", time)],
    )

    phase_speed = lam * 1000 / (tau * 60)
    cycle = (r - (phase_speed / 1000) * t) / lam
    tec = da.cos(2 * np.pi * cycle)
    noise_factor = da.sqrt((10 ** (snr / (-10))) / 2)
    noise = xr.DataArray(
        da.random.normal(size=tec.shape, chunks=tec.data.chunksize),
        dims=tec.dims,
        coords=tec.coords,
    ) * noise_factor
    noisy_tec = tec + noise
    density = xr.ones_like(noisy_tec, chunks=noisy_tec.chunks) * 20

    # Rolling median was the big offender for graph/task explosion; rolling mean is much
    # friendlier (decomposable reduction, much more fuseable in dask). rechunk after the
    # rolling op because otherwise xarray or dask changes the chunk layout
    noisy_tec = (
        noisy_tec
        .rolling(x=3, y=3, center=True, min_periods=1)
        .mean()
        .chunk(x=-1, y=-1, time=-1, trial=batch_size)
    )
    data = (
        xr.Dataset({"image": noisy_tec, "density": density})
        .assign_attrs(center=center)
        .assign_coords(snr=snr, lam=lam, tau=tau)
    )

    return data


def planar_model(
        time: pandas.DatetimeIndex | None = None,
        snr_lim = [-6, 6],
        lam_lim = [150, 400],
        tau_lim = [10, 50],
        n_trials = 10,
        xlim: ArrayLike = (-1000, 1000),
        ylim: ArrayLike = (-1000, 1000),
        hres: float = 20,
        batch_size: int = 20,
    ) -> xr.DataArray:
    if time is None:
        time = pandas.date_range("2025-01-01", "2025-01-01T01:00:00", freq="60s")
    
    batch_size = min(batch_size, n_trials)
    
    trials = np.arange(0, n_trials)
    
    snr = xr.DataArray(
        np.random.rand(n_trials) * (snr_lim[1] - snr_lim[0]) + snr_lim[0],
        coords=[("trial", trials)],
    ).chunk(trial=batch_size)
    lam = xr.DataArray(
        np.random.rand(n_trials) * (lam_lim[1] - lam_lim[0]) + lam_lim[0],
        coords=[("trial", trials)],
    ).chunk(trial=batch_size)
    tau = xr.DataArray(
        np.random.rand(n_trials) * (tau_lim[1] - tau_lim[0]) + tau_lim[0],
        coords=[("trial", trials)],
    ).chunk(trial=batch_size)
    direction = xr.DataArray(
        np.random.random(n_trials) * 2 * np.pi,
        coords=[("trial", trials)],
    ).chunk(trial=batch_size)

    xlab = np.arange(xlim[0], xlim[1] + hres, hres)
    ylab = np.arange(ylim[0], ylim[1] + hres, hres)
    x = xr.DataArray(da.from_array(xlab, chunks=-1), coords=[("x", xlab)])
    y = xr.DataArray(da.from_array(ylab, chunks=-1), coords=[("y", ylab)])
    t = xr.DataArray(
        da.from_array((time - time[0]).total_seconds().values, chunks=-1),
        coords=[("time", time)],
    )

    k = da.exp(1j * direction) * 2 * np.pi / lam
    r = x + 1j * y
    w = 2 * np.pi / (tau * 60)
    arg = (k.conj() * r).real - w * t
    tec = da.cos(arg)
    noise_factor = da.sqrt((10 ** (snr / (-10))) / 2)
    noise = xr.DataArray(
        da.random.normal(size=tec.shape, chunks=tec.data.chunksize),
        dims=tec.dims,
        coords=tec.coords,
    ) * noise_factor
    noisy_tec = tec + noise
    density = xr.ones_like(noisy_tec, chunks=noisy_tec.chunks) * 20

    noisy_tec = (
        noisy_tec
        .rolling(x=3, y=3, center=True, min_periods=1)
        .mean()
        .chunk(x=-1, y=-1, time=-1, trial=batch_size)
    )
    data = (
        xr.Dataset({"image": noisy_tec, "density": density})
        .assign_coords(snr=snr, lam=lam, tau=tau, dir=direction)
    )

    return data
