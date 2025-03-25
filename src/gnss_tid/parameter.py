import numpy as np
import pandas as pd
import xarray as xr
from scipy.fft import fft2, fftfreq
from scipy.signal import get_window
from scipy.interpolate import make_splrep, make_smoothing_spline

import gnss_tid.plotting


def get_line_coordinates(start_x, start_y, azimuth, distance):
    """Calculates the end coordinates of a line given the start coordinates, azimuth, and distance."""
    # Convert azimuth to radians
    azimuth_rad = np.radians(azimuth)
    # Calculate end coordinates
    end_x = start_x + distance[:, None] * np.sin(azimuth_rad)[None, :]
    end_y = start_y + distance[:, None] * np.cos(azimuth_rad)[None, :]
    return end_x, end_y


def estimate_period(lines_tec: xr.DataArray, nperseg=32, noverlap=31, nfft=1024) -> xr.DataArray:
    minute_lines = lines_tec.assign_coords(time=np.arange(lines_tec.time.size))
    Stt = spectrogram(minute_lines, dim="time", nperseg=nperseg, nfft=nfft, noverlap=noverlap)
    gnss_tid.plotting.plot_param_time_fft(Stt)
    max_comp = Stt.isel(frequency=Stt.argmax("frequency"))
    max_comp = max_comp.where(max_comp.frequency > 0, drop=True)
    tau = (
        (1 / max_comp.frequency)
        .rename("tau")
        .reindex_like(minute_lines)
        .assign_coords(time=lines_tec.time)
    )
    return tau


def estimate_wavelength(data, lines_x, lines_y, L=800, threshold=.02, min_k=1.0e-4):
    dx = (data.x[1] - data.x[0]).item()
    v = np.arange(-(L / 2), (L / 2), dx)
    v = xr.DataArray(v, coords={"f": v}, dims=["f"])
    
    Nfft = 256
    tec_zonal = data.image.interp(x=lines_x + v, y=lines_y)
    Sx = spectrogram(tec_zonal, dim="f", noverlap=0, nfft=Nfft, nperseg=Nfft).isel(f=0)
    Sx = Sx.fillna(0)
    gnss_tid.plotting.plot_param_fft_lines(data, tec_zonal, az_idx=2)
    gnss_tid.plotting.plot_param_spatial_fft(Sx, az_idx=2)
    # gnss_tid.plotting.plot_param_fft_lines(data, tec_zonal, az_idx=0)
    # gnss_tid.plotting.plot_param_spatial_fft(Sx, az_idx=0)
    # gnss_tid.plotting.plot_param_fft_lines(data, tec_zonal, az_idx=-1)
    # gnss_tid.plotting.plot_param_spatial_fft(Sx, az_idx=-1)

    max_comp_x = Sx.isel(frequency=Sx.argmax("frequency"))
    params = (
        max_comp_x.where(max_comp_x.frequency > 0, drop=True)
        .where(max_comp_x > 200, drop=True)
        .reset_coords("frequency")
        .rename(frequency="kx", spectrogram_image="Sx")
    )

    tec_meridional = data.image.interp(x=lines_x, y=lines_y + v)
    Sy = spectrogram(tec_meridional, dim="f", noverlap=0, nfft=Nfft, nperseg=Nfft).isel(f=0)
    Sy = Sy.fillna(0)
    gnss_tid.plotting.plot_param_fft_lines(data, tec_meridional, az_idx=2)
    gnss_tid.plotting.plot_param_spatial_fft(Sy, az_idx=2)
    # gnss_tid.plotting.plot_param_fft_lines(data, tec_meridional, az_idx=0)
    # gnss_tid.plotting.plot_param_spatial_fft(Sy, az_idx=0)
    # gnss_tid.plotting.plot_param_fft_lines(data, tec_meridional, az_idx=-1)
    # gnss_tid.plotting.plot_param_spatial_fft(Sy, az_idx=-1)

    max_comp_y = Sy.isel(frequency=Sy.argmax("frequency"))
    max_comp_y = (
        max_comp_y.where(max_comp_y.frequency > 0, drop=True)
        .where(max_comp_y > 200, drop=True)
        .reset_coords("frequency")
        .rename(frequency="ky", spectrogram_image="Sy")
    )
    params = params.merge(max_comp_y)

    params = params.assign(k=np.hypot(params.kx, params.ky))
    params = params.assign(wavelength=1 / params.k)
    return params


def estimate_wavelength_v2(data, Nfft=256, block_size=32, step_size=8):
    dx = (data.x[1] - data.x[0]).item()
    wavenum = fftfreq(Nfft, dx)
    edges = block_size // (2 * step_size)
    win = get_window("hann", block_size)
    win = np.outer(win, win)
    patches = (
        data.image
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .pipe(lambda x: x * win)
        .pad(kx=(0, Nfft-block_size), ky=(0, Nfft-block_size), constant_values=0)
        .assign_coords(kx=wavenum, ky=wavenum)
        .rename({"x": "px", "y": "py"})
    )
    patches.values = abs(fft2(patches)) ** 2
    S = patches.isel(patches.argmax(dim=["kx", "ky"]))
    k = np.hypot(S.kx, S.ky)
    wavelength = 1 / k
    return S


def estimate_parameters(
        data: str | xr.Dataset,
        range_coords: np.ndarray,
        az_coords: np.ndarray,
    ) -> xr.Dataset:
    """estimate parameters of TIDs

    Args:
        data (str | xr.Dataset): requires "image" variable, "center" attribute and
            ["x", "y", "time"] coordinates

    Returns:
        xr.Dataset: ["tau", "wavelength", "phase_speed"]
    """
    if isinstance(data, str):
        data = xr.open_dataset(data)
    cx0, cy0 = data.center.values[0], data.center.values[1]  # TID center
    lines_x, lines_y = get_line_coordinates(cx0, cy0, az_coords, range_coords)
    lines_x = xr.DataArray(
        lines_x, coords={"r": range_coords, "az": az_coords}, dims=["r", "az"]
    )
    lines_y = xr.DataArray(
        lines_y, coords={"r": range_coords, "az": az_coords}, dims=["r", "az"]
    )
    lines_tec = data.image.interp(x=lines_x, y=lines_y)

    gnss_tid.plotting.plot_param_lines(data, lines_tec)

    params = estimate_wavelength(data, lines_x, lines_y)
    tau = estimate_period(lines_tec)
    params = params.assign(
        phase_speed=(params.wavelength * 1000 / (tau * 60)),
        tau=tau,
        tec=lines_tec,
    )
    return params


def estimate_parameters_v2(
        data: str | xr.Dataset,
        range_coords: np.ndarray,
        az_coords: np.ndarray,
    ) -> xr.Dataset:
    """estimate parameters of TIDs

    Args:
        data (str | xr.Dataset): requires "image" variable, "center" attribute and
            ["x", "y", "time"] coordinates

    Returns:
        xr.Dataset: ["tau", "wavelength", "phase_speed"]
    """
    if isinstance(data, str):
        data = xr.open_dataset(data)
    cx0, cy0 = data.center.values[0], data.center.values[1]  # TID center
    lines_x, lines_y = get_line_coordinates(cx0, cy0, az_coords, range_coords)
    lines_x = xr.DataArray(
        lines_x, coords={"r": range_coords, "az": az_coords}, dims=["r", "az"]
    )
    lines_y = xr.DataArray(
        lines_y, coords={"r": range_coords, "az": az_coords}, dims=["r", "az"]
    )
    lines_tec = data.image.interp(x=lines_x, y=lines_y)

    gnss_tid.plotting.plot_param_lines(data, lines_tec)

    params = estimate_wavelength_v2(data)
    tau = estimate_period(lines_tec)
    params = params.assign(
        phase_speed=(params.wavelength * 1000 / (tau * 60)),
        tau=tau,
        tec=lines_tec,
    )
    return params


def _spline_wrapper(v):
    x = np.arange(v.shape[0])
    s = make_smoothing_spline(x, v, lam=100 * x.shape[0])
    return s(x), s.derivative()(x)


def estimate_parameters_block(data: str | xr.Dataset):
    Nfft = 128
    block_size = 32
    step_size = 8
    k_smooth_window = 3
    phase_median_window = 3
    wavelength_median_window = 3
    tau_median_window = 3
    power_threshold = .25

    dx = (data.x[1] - data.x[0]).item()
    wavenum = fftfreq(Nfft, dx)
    edges = block_size // (2 * step_size)
    win = get_window("hann", block_size)
    win = np.outer(win, win)
    if "density" in data:
        img = data.image.where(data.density >= 10, 0)
    else:
        img = data.image
    img_patches = (
        (img / img.std(["x", "y"]))
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .pipe(lambda x: x * win)
        .pad(kx=(0, Nfft-block_size), ky=(0, Nfft-block_size), constant_values=0)
        .assign_coords(kx=wavenum, ky=wavenum)
        .rename({"x": "px", "y": "py"})
    )
    f = fft2(img_patches) / win.sum()
    patch_power = xr.DataArray(abs(f) ** 2, dims=img_patches.dims, coords=img_patches.coords)
    patch_phase = xr.DataArray(np.angle(f), dims=img_patches.dims, coords=img_patches.coords)
    patches = xr.Dataset(dict(
        image=img_patches,
        power=patch_power,
        phase=patch_phase,
    ))

    max_idx = patch_power.where(patch_power.kx >= 0).argmax(dim=["kx", "ky"])
    Sp = patch_power.isel(max_idx)
    k = (
        np.hypot(Sp.kx, Sp.ky)
        .where(Sp > power_threshold)
        .rolling(time=k_smooth_window, center=True, min_periods=1).median()
    )
    wavelength = (
        (1 / k)
        .where(Sp > power_threshold)
        .rolling(px=wavelength_median_window, py=wavelength_median_window, center=True, min_periods=1)
        .median()
    )

    phase = (
        patch_phase
        .isel(max_idx)
        # .rolling(time=phase_median_window, center=True, min_periods=1).median()
    )
    phase = xr.apply_ufunc(
        np.unwrap,
        phase,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )

    smooth_phase, smooth_freq = xr.apply_ufunc(
        _spline_wrapper,
        phase,
        input_core_dims=[["time"]],
        output_core_dims=[["time"], ["time"]],
        vectorize=True,
        output_dtypes=[np.float64, np.float64]
    )
    tau = (
        (2 * np.pi / abs(smooth_freq))
        .where(Sp > power_threshold)
        .rolling(px=tau_median_window, py=tau_median_window, center=True, min_periods=1)
        .median()
    )

    r = np.hypot(tau.px, tau.py)
    phase_speed = wavelength * 1000 / (tau * 60)
    params = xr.Dataset(dict(
        power=Sp,
        k=k,
        wavelength=wavelength,
        phase=phase,
        smooth_phase=smooth_phase,
        tau=tau,
        r=r,
        phase_speed=phase_speed,
    ))
    return params, patches
