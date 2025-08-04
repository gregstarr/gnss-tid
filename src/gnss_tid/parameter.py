import pickle

import dask
import numba
import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da
from dask.array.fft import fft2 as dafft2
from dask.utils import format_bytes
from scipy.fft import fft, fft2, fftfreq
from scipy.signal.windows import kaiser
from scipy.interpolate import make_splrep, make_smoothing_spline

import gnss_tid.plotting


def estimate_noise_hs74(spectrum, navg=1, nnoise_min=1):
    """
    Estimate noise parameters of a Doppler spectrum.

    (copied from https://arm-doe.github.io/pyart/_modules/pyart/util/hildebrand_sekhon.html)

    Use the method of estimating the noise level in Doppler spectra outlined
    by Hildebrand and Sehkon, 1974.

    Parameters
    ----------
    spectrum : array like
        Doppler spectrum in linear units.
    navg : int, optional
        The number of spectral bins over which a moving average has been
        taken. Corresponds to the **p** variable from equation 9 of the
        article. The default value of 1 is appropriate when no moving
        average has been applied to the spectrum.
    nnoise_min : int, optional
        Minimum number of noise samples to consider the estimation valid.

    Returns
    -------
    mean : float-like
        Mean of points in the spectrum identified as noise.
    threshold : float-like
        Threshold separating noise from signal. The point in the spectrum with
        this value or below should be considered as noise, above this value
        signal. It is possible that all points in the spectrum are identified
        as noise. If a peak is required for moment calculation then the point
        with this value should be considered as signal.
    var : float-like
        Variance of the points in the spectrum identified as noise.
    nnoise : int
        Number of noise points in the spectrum.

    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.

    """
    sorted_spectrum = np.sort(spectrum)
    nnoise = len(spectrum)  # default to all points in the spectrum as noise

    rtest = 1 + 1 / navg
    sum1 = 0.0
    sum2 = 0.0
    for i, pwr in enumerate(sorted_spectrum):
        npts = i + 1
        sum1 += pwr
        sum2 += pwr * pwr

        if npts < nnoise_min:
            continue

        if npts * sum2 < sum1 * sum1 * rtest:
            nnoise = npts
        else:
            # partial spectrum no longer has characteristics of white noise.
            sum1 -= pwr
            sum2 -= pwr * pwr
            break

    mean = sum1 / nnoise
    var = sum2 / nnoise - mean * mean
    threshold = sorted_spectrum[nnoise - 1]
    return mean, threshold, var, nnoise


def get_line_coordinates(start_x, start_y, azimuth, distance):
    """Calculates the end coordinates of a line given the start coordinates, azimuth, and distance."""
    # Convert azimuth to radians
    azimuth_rad = np.radians(azimuth)
    # Calculate end coordinates
    end_x = start_x + distance[:, None] * np.sin(azimuth_rad)[None, :]
    end_y = start_y + distance[:, None] * np.cos(azimuth_rad)[None, :]
    return end_x, end_y


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
    
    # img = img / img.std(["x", "y"])
    
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
    KXI = np.argmax(np.array(img_patches.dims) == "kx")
    KYI = np.argmax(np.array(img_patches.dims) == "ky")
    
    F = fft2(img_patches, axes=(KYI, KXI)) / np.sum(win)
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
    pmed = patch_power.median(KDIMS)
    psig = 1.4826 * abs(patch_power - pmed).median(KDIMS)
    threshold = pmed + 50 * psig
    weight = patch_power.where(patch_power > threshold)
    Sp = weight.sum(KDIMS)
    weight = weight / Sp
    nsig = (patch_power > threshold).sum(KDIMS)
    pnoise = patch_power.where(patch_power < threshold).mean(KDIMS) * nsig
    R = (Sp - pnoise) / pnoise

    # weighted average phase velocity and phase speed
    k2 = phase.kx ** 2 + phase.ky ** 2
    vx = (weight * phase.kx * freq / k2).sum(KDIMS)  # km^-1
    vy = (weight * phase.ky * freq / k2).sum(KDIMS)  # km^-1
    phase_speed = (
        np.hypot(vx, vy)
        .where(~freq.isnull().all(KDIMS))
        .rolling(time=smooth_win, center=True).mean()
    ) * 1000  # m^-1

    k = np.sqrt(k2)
    Rx = (weight * abs(phase.kx) / k).sum(KDIMS)
    Ry = (weight * abs(phase.ky) / k).sum(KDIMS)
    coherence = np.hypot(Rx, Ry)

    wf = weight * freq * np.sign(freq.kx * vx + freq.ky * vy)
    wfsum = wf.sum(KDIMS)
    period = 1 / (60 * wfsum)  # minutes
    period = period.where(wfsum > 0)
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
        R=R,
        coherence=coherence,
    ))
    params = params.assign_attrs(
        Nfft=Nfft,
        block_size=block_size,
        step_size=step_size,
        smooth_win=smooth_win,
    )
    return params, patches


def estimate_parameters_block_v3(
        patches: xr.Dataset,
        smooth_win=5,
    ) -> xr.Dataset:
    
    KDIMS = ["kx", "ky"]
    graph_size("patches", patches)

    kx = patches.kx.persist()
    ky = patches.ky.persist()

    # set up weighted average, only keep k bins with power exceeding threshold
    pmed = patches.power.median(KDIMS)
    psig = 1.4826 * abs(patches.power - pmed).median(KDIMS)
    threshold = pmed + 50 * psig
    weight = patches.power.where(patches.power > threshold)
    Sp = weight.sum(KDIMS)
    weight = (weight / Sp).persist()
    graph_size("weight", weight)
    # if use_threshold:
    #     nsig = (patches.power > threshold).sum(KDIMS)
    #     pnoise = patches.power.where(patches.power < threshold).mean(KDIMS) * nsig
    #     snr = (Sp - pnoise) / pnoise
    # else:
    #     snr = Sp

    freq = (
        patches.phase.differentiate("time", datetime_unit="s") # Hz
        .rolling(time=smooth_win, center=True, min_periods=1)
        .mean()
    )
    graph_size("freq", freq)

    # weighted average phase velocity and phase speed
    wfreq = (weight * freq).persist()
    k2 = kx ** 2 + ky ** 2
    wf_over_k2 = wfreq / k2
    # vx = (kx * f_over_k2).weighted(weight).mean(KDIMS)
    # vy = (ky * f_over_k2).weighted(weight).mean(KDIMS)
    vx = (kx * wf_over_k2).sum(KDIMS)  # km^-1
    graph_size("vx", vx)
    vy = (ky * wf_over_k2).sum(KDIMS)  # km^-1
    graph_size("vy", vy)
    phase_speed = (
        xr.ufuncs.hypot(vx, vy)
        .rolling(time=smooth_win, center=True, min_periods=1)
        .mean() * 1000
     ).persist()
    graph_size("phase_speed", phase_speed)

    # k = np.sqrt(k2)
    # Rx = (weight * abs(phase.kx) / k).sum(KDIMS)
    # Ry = (weight * abs(phase.ky) / k).sum(KDIMS)
    # coherence = np.hypot(Rx, Ry)

    wfreq = (wfreq * xr.ufuncs.sign(kx * vx + ky * vy)).sum(KDIMS)
    period = (1 / (60 * wfreq)).where(wfreq > 0)  # minutes
    wavelength = (phase_speed * period) * 60 / 1000  # km
    graph_size("wavelength", wavelength)

    # r = xr.ufuncs.hypot(img_patches.px, img_patches.py)  # km
    params = xr.Dataset(dict(
        # power=Sp,
        phase_speed=phase_speed,
        wavelength=wavelength,
        period=period,
        # phase=phase,
        # freq=freq,
        # r=r,
        vx=vx,
        vy=vy,
        # snr=snr,
        # coherence=coherence,
    )).chunk(time=-1, px=-1, py=-1, lam="auto", tau="auto", snr="auto")
    return params


def graph_size(name, coll):
    # Get the raw dict of tasks
    dsk = coll.__dask_graph__().to_dict()
    n_tasks = len(dsk)
    n_bytes = len(pickle.dumps(dsk))
    print(f"{name:8s} → {n_tasks:7d} tasks, graph is {format_bytes(n_bytes)}")


def get_patches(
        data: xr.Dataset,
        Nfft=256,
        block_size=32,
        step_size=8,
        kaiser_beta=5,
    ) -> xr.Dataset:
    graph_size("data", data)

    dx = (data.x[1] - data.x[0]).item()
    wavenum = da.fft.fftfreq(Nfft, dx, chunks=-1)
    edges = block_size // (2 * step_size)
    win = kaiser(block_size, kaiser_beta)
    win = da.from_array(win, chunks=-1)
    win = da.outer(win, win).persist()
    winsum = da.sum(win).persist()

    if "density" in data:
        img = data.image.where(data.density >= 5, 0)
    else:
        img = data.image
    
    graph_size("img", img)
    
    img_patches = (
        img
        .rolling(y=block_size, x=block_size, center=True)
        .construct(
            x="kx",
            y="ky",
            stride=step_size,
            sliding_window_view_kwargs={"automatic_rechunk": False}
        )
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
        .chunk(ky=-1, kx=-1, time=-1, py="auto", tau="auto", px="auto", lam="auto", snr="auto")
    )
    graph_size("img_patches", img_patches)
    
    KX = np.argmax(np.array(img_patches.dims) == "kx").item()
    KY = np.argmax(np.array(img_patches.dims) == "ky").item()
    TIME_DIM = np.argmax(np.array(img_patches.dims) == "time").item()
    TAU = 2 * np.pi

    F = dafft2(img_patches.data * win, [Nfft, Nfft], [KX, KY])
    power = (da.abs(F) / winsum) ** 2
    graph_size("power", power)
    phase = da.apply_gufunc(np.unwrap, "()->()", da.angle(F), axis=TIME_DIM) / TAU  # cycles
    graph_size("phase", phase)

    coords = img_patches.coords.assign(kx=wavenum, ky=wavenum)
    return xr.Dataset({
        "power": xr.DataArray(power, coords, coords.dims),
        "phase": xr.DataArray(phase, coords, coords.dims),
    })


KDIMS = ["kx", "ky"]
TAU = 2 * np.pi

@numba.njit
def _get_phase(x):
    ang = np.angle(x)
    phase = np.unwrap(ang)
    return phase / TAU


@numba.guvectorize(
    [(numba.float64[:, :], numba.float64[:, :], numba.float64[:])],
    '(kx,ky)->(kx,ky),()',
)
def _get_weight(pwr, weight, snr):
    # set up weighted average, only keep k bins with power exceeding threshold
    pmed = np.nanmedian(pwr)
    psig = np.nanmedian(1.4826 * np.abs(pwr - pmed))
    threshold = pmed + 50 * psig
    mask = pwr > threshold
    nsig = np.sum(mask)
    pnoise = np.nanmean(np.where(~mask, pwr, np.nan)) * nsig
    w = np.where(mask, pwr, np.nan)
    Sp = np.nansum(w)
    snr[0] = (Sp - pnoise) / pnoise
    weight[:, :] = w / Sp


def estimate_parameters_block_v4(
        data: xr.Dataset,
        Nfft=256,
        block_size=32,
        step_size=8,
        smooth_win=5,
        kaiser_beta=5,
    ):

    dx = (data.x[1] - data.x[0]).item()
    edges = block_size // (2 * step_size)
    win = kaiser(block_size, kaiser_beta)
    win = np.outer(win, win)

    if "density" in data:
        img = data.image.where(data.density >= 5, 0)
    else:
        img = data.image
    
    img_patches = (
        img
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )

    del data
    del img

    KXI = np.argmax(np.array(img_patches.dims) == "kx")
    KYI = np.argmax(np.array(img_patches.dims) == "ky")
    wavenum = fftfreq(Nfft, dx)
    
    coords = img_patches.coords.assign(kx=wavenum, ky=wavenum)
    F = xr.DataArray(
        fft2(img_patches.data * win, s=(Nfft, Nfft), axes=(KYI, KXI)) / np.sum(win),
        coords=coords,
        dims=img_patches.dims
    )

    del img_patches

    kx = F.kx
    ky = F.ky
    phase = xr.apply_ufunc(_get_phase, F, input_core_dims=[["time"]], output_core_dims=[["time"]], vectorize=False)
    W = abs(F) ** 2

    del F

    W, R = xr.apply_ufunc(_get_weight, W, input_core_dims=[KDIMS], output_core_dims=[KDIMS, []], vectorize=False)
    
    K = np.hypot(kx, ky)
    uvx = (W * np.abs(kx) / K).sum(KDIMS)
    uvy = (W * np.abs(ky) / K).sum(KDIMS)
    coherence = np.hypot(uvx, uvy)

    wfreq = (
        phase.differentiate("time", datetime_unit="s")
        .rolling(time=smooth_win, center=True, min_periods=1).mean()  # Hz
    ) * W

    del W
    del phase

    # weighted average phase velocity and phase speed
    K *= K
    vx = (kx * wfreq / K).sum(KDIMS)  # km^-1
    vy = (ky * wfreq / K).sum(KDIMS)  # km^-1
    phase_speed = (
        np.hypot(vx, vy)
        .where(~wfreq.isnull().all(KDIMS))
        .rolling(time=smooth_win, center=True, min_periods=1).mean()
    ) * 1000  # m^-1

    wfsum = (wfreq * np.sign(kx * vx + ky * vy)).sum(KDIMS)
    period = (1 / (60 * wfsum)).where(wfsum > 0)  # minutes
    wavelength = (
        (phase_speed * period)
        .rolling(time=smooth_win, center=True, min_periods=1).mean()
     ) * 60 / 1000  # km

    params = xr.Dataset(dict(
        phase_speed=phase_speed,
        wavelength=wavelength,
        period=period,
        vx=vx,
        vy=vy,
        R=R,
        coherence=coherence,
    ))
    return params
