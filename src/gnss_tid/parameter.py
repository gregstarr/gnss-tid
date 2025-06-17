import numpy as np
import pandas as pd
import xarray as xr
from scipy.fft import fft, fft2, fftfreq
from scipy.signal.windows import kaiser
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


def _spline_wrapper(v):
    x = np.arange(v.shape[0])
    s = make_smoothing_spline(x, v, lam=100 * x.shape[0])
    return s(x), s.derivative()(x)


def estimate_period_spline(patch_power, patch_phase, median_window=3, power_threshold=.25):
    max_idx = patch_power.where(patch_power.kx >= 0).argmax(dim=["kx", "ky"])
    Sp = patch_power.isel(max_idx)

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
        .rolling(px=median_window, py=median_window, center=True, min_periods=1)
        .median()
    )
    return xr.Dataset(dict(tau=tau, phase=phase, smooth_phase=smooth_phase))


def estimate_period(
        tec,
        window_size = 64,
        window_step = 1,
        Nfft = 512,
        power_threshold=.25,
        kaiser_beta=6,
    ):
    edges = window_size // (2 * window_step)
    win = kaiser(window_size, kaiser_beta)
    dt = 1
    freq = fftfreq(Nfft, dt)
    windows = (
        tec.rolling(time=window_size, center=True)
        .construct(time="f", stride=window_step)
        .isel(time=slice(edges, -edges))
        .pipe(lambda x: x * win)
        .pad(f=(0, Nfft-window_size), constant_values=0)
        .assign_coords(f=freq)
    )
    St = xr.DataArray(
        abs(fft(windows) / np.sum(win)) ** 2,
        dims=windows.dims,
        coords=windows.coords,
    )
    St = St.sel(f=freq>=0)
    max_comp = St.isel(f=St.argmax("f"))
    tau = (
        (1 / max_comp.f)
        .rename("tau")
        .where(max_comp > power_threshold)
    )
    return xr.Dataset(dict(tau=tau, max_power=max_comp, power=St, windows=windows))


def estimate_parameters_block(
        data: xr.Dataset,
        Nfft=256,
        block_size=32,
        step_size=8,
        wavelength_median_window=3,
        space_power_threshold=0.1,
        time_power_threshold=0.2,
        time_window=64,
        time_nfft=512,
        normalize=True,
        kaiser_beta=5,
    ):

    dx = (data.x[1] - data.x[0]).item()
    wavenum = fftfreq(Nfft, dx)
    edges = block_size // (2 * step_size)
    win = kaiser(block_size, kaiser_beta)
    win = np.outer(win, win)

    if "density" in data:
        img = data.image.where(data.density >= 5, 0)
    else:
        img = data.image
    
    if normalize:
        img = img / img.std(["x", "y"])
    
    img_patches = (
        img
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .pipe(lambda x: x * win)
        .pad(kx=(0, Nfft-block_size), ky=(0, Nfft-block_size), constant_values=0)
        .assign_coords(kx=wavenum, ky=wavenum)
        .rename({"x": "px", "y": "py"})
    )
    
    f = fft2(img_patches) / np.sum(win)
    patch_power = xr.DataArray(abs(f) ** 2, dims=img_patches.dims, coords=img_patches.coords)
    patch_phase = xr.DataArray(np.angle(f), dims=img_patches.dims, coords=img_patches.coords)
    patches = xr.Dataset(dict(
        image=img_patches,
        power=patch_power,
        phase=patch_phase,
    ))

    max_idx = patch_power.argmax(dim=["kx", "ky"])
    Sp = patch_power.isel(**max_idx)
    k = np.hypot(Sp.kx, Sp.ky)
    wavelength = (
        (1 / k)
        .where(Sp > space_power_threshold)
        .rolling(
            px=wavelength_median_window,
            py=wavelength_median_window,
            center=True,
            min_periods=1,
        ).median()
    )
    print(f"valid patches for wavelength FFT: {(Sp > space_power_threshold).sum(['px', 'py']).mean().item()}")

    tec = img.sel(x=patch_power.px, y=patch_power.py)
    tau = estimate_period(
        tec,
        window_size=time_window,
        Nfft=time_nfft,
        power_threshold=time_power_threshold,
    )
    print(f"valid patches for tau FFT: {(tau.max_power > time_power_threshold).sum(['px', 'py']).mean().item()}")

    r = np.hypot(img_patches.px, img_patches.py)
    phase_speed = wavelength * 1000 / (tau.tau * 60)
    print(f"valid phase speeds: {(~phase_speed.isnull()).sum(['px', 'py']).mean().item()}")
    params = xr.Dataset(dict(
        power=Sp,
        k=k,
        wavelength=wavelength,
        r=r,
    ))
    params = params.sel(time=phase_speed.time).assign(phase_speed=phase_speed)
    params = params.assign_attrs(
        Nfft=Nfft,
        block_size=block_size,
        step_size=step_size,
        wavelength_median_window=wavelength_median_window,
        space_power_threshold=space_power_threshold,
        time_power_threshold=time_power_threshold,
        time_window=time_window,
        time_nfft=time_nfft,
        normalize=normalize,
    )
    return params, tau, patches

def estimate_parameters_block_v2(
        data: xr.Dataset,
        Nfft=256,
        block_size=32,
        step_size=8,
        smooth_win=5,
        power_threshold=0.2,
        normalize=True,
        kaiser_beta=5,
    ):

    dx = (data.x[1] - data.x[0]).item()
    wavenum = fftfreq(Nfft, dx)
    edges = block_size // (2 * step_size)
    win = kaiser(block_size, kaiser_beta)
    win = np.outer(win, win)

    if "density" in data:
        img = data.image.where(data.density >= 5, 0)
    else:
        img = data.image
    
    if normalize:
        img = img / img.std(["x", "y"])
    
    img_patches = (
        img
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .pipe(lambda x: x * win)
        .pad(kx=(0, Nfft-block_size), ky=(0, Nfft-block_size), constant_values=0)
        .assign_coords(kx=wavenum, ky=wavenum)
        .rename({"x": "px", "y": "py"})
    )
    
    F = fft2(img_patches) / np.sum(win)
    patch_power = xr.DataArray(abs(F) ** 2, dims=img_patches.dims, coords=img_patches.coords)
    patch_phase = xr.DataArray(np.angle(F), dims=img_patches.dims, coords=img_patches.coords)
    patches = xr.Dataset(dict(
        image=img_patches,
        power=patch_power,
        phase=patch_phase,
    ))

    KDIMS = ["kx", "ky"]
    TAU = 2 * np.pi
    TIME_DIM = np.argmax(np.array(patch_phase.dims) == "time")

    phase = patch_phase.copy()  # radians
    phase.values = np.unwrap(phase, axis=TIME_DIM) / TAU  # cycles
    freq = (
        phase
        .differentiate("time", datetime_unit="s")
        .rolling(time=smooth_win, center=True).mean()
    )  # Hz

    # set up weighted average, only keep k bins with power exceeding threshold
    weight = patch_power.where(patch_power > power_threshold)
    Sp = weight.sum(KDIMS)
    weight = weight / Sp

    # weighted average phase velocity and phase speed
    k2 = phase.kx ** 2 + phase.ky ** 2
    vx = (weight * phase.kx * freq / k2).sum(KDIMS)  # km^-1
    vy = (weight * phase.ky * freq / k2).sum(KDIMS)  # km^-1
    phase_speed = (
        np.hypot(vx, vy)
        .where(~freq.isnull().all(KDIMS))
        .rolling(time=smooth_win, center=True).mean()
    ) * 1000  # m^-1

    wf = weight * freq * np.sign(freq.kx * vx + freq.ky * vy)
    period = 1 / (60 * wf.sum(KDIMS))  # minutes
    wavelength = (
        (phase_speed * period)
        .rolling(time=smooth_win, center=True).mean()
     ) * 60 / 1000  # km

    r = np.hypot(img_patches.px, img_patches.py)  # km
    params = xr.Dataset(dict(
        power=Sp,
        phase_speed=phase_speed,
        wavelength=wavelength,
        period=period,
        phase=phase,
        freq=freq,
        r=r,
        vx=vx,
        vy=vy,
    ))
    params = params.assign_attrs(
        Nfft=Nfft,
        block_size=block_size,
        step_size=step_size,
        smooth_win=smooth_win,
        power_threshold=power_threshold,
        normalize=normalize,
    )
    return params, patches
