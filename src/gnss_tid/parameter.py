import itertools
import time

import dask
import numba
import numpy as np
import xarray as xr
from scipy.fft import fft, fft2, fftfreq
from scipy.signal.windows import kaiser
from scipy.interpolate import make_splrep, make_smoothing_spline

import hvplot.xarray
import holoviews as hv
from holoviews import opts
import colorcet as cc
from bokeh.models import PrintfTickFormatter
import panel as pn


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
    ) * 1000  # m/s

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


def estimate_parameters_block_debug(
        data: xr.Dataset,
        Nfft=256,
        block_size=32,
        step_size=8,
        smooth_win=9,
        kaiser_beta=5,
    ):
    t0 = time.perf_counter()
    print("start")
    hres = (data.x[1] - data.x[0]).item()
    edges = block_size // (2 * step_size)
    window = kaiser(block_size, kaiser_beta)
    window = np.outer(window, window) / np.sum(window)

    if "density" in data:
        img = data.image.where(data.density >= 5, 0)
    else:
        img = data.image
        
    params = xr.Dataset()
    img_patches = (
        img
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )
    print("patches")

    wavenum = fftfreq(Nfft, hres)
    
    params["F"] = (
        xr.apply_ufunc(
            lambda x: fft2(x * window, s=(Nfft, Nfft)),
            img_patches,
            input_core_dims=[KDIMS],
            output_core_dims=[KDIMS],
            output_dtypes=[np.complex128],
            dask_gufunc_kwargs={"output_sizes": {"kx": Nfft, "ky": Nfft}},
            dask="parallelized",
            exclude_dims={"kx", "ky"}
        )
        .assign_coords(kx=wavenum, ky=wavenum)
        .sortby("kx").sortby("ky")
    )
    params["img_patches"] = img_patches.rename({"kx": "dx", "ky": "dy"})
    if dask.is_dask_collection(params):
        params = params.chunk({"time": -1})
    print("fft")

    params["power"] = abs(params["F"]) ** 2
    print("power")

    # set up weighted average, only keep k bins with power exceeding threshold
    threshold = params["power"].quantile(.95, KDIMS)
    params["power_threshold"] = threshold
    params["weight"] = params["power"].where(params["power"] > threshold)
    params["weight"] = params["weight"] / params["weight"].sum(KDIMS)

    print("weight")

    params["phase"] = xr.apply_ufunc(
        np.unwrap,
        xr.ufuncs.angle(params["F"]),
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
    ) / TAU
    print("phase")

    params["freq"] = params["phase"].differentiate("time", datetime_unit="s")
    freq_noise_power = params["freq"].rolling(time=smooth_win, center=True, min_periods=1).var()
    params["freq"] = params["freq"].rolling(time=smooth_win, center=True, min_periods=1).mean()
    params["freq_snr"] = (params["freq"]**2 / freq_noise_power)
    params["patch_freq_snr"] = (params["freq_snr"] * params["weight"]).sum(KDIMS)
    print("freq")
    
    # weighted average phase velocity and phase speed
    k = params.kx + params.ky * 1j
    k2 = k ** 2
    params["phase_velocity"] = (params["weight"] * k * params["freq"] / abs(k)**2).sum(KDIMS)
    params["phase_velocity_angle"] = xr.ufuncs.angle(params["phase_velocity"])
    params["phase_speed"] = abs(params["phase_velocity"]) * 1000  # m/s
    print("phase velocity")

    S0 = (params["weight"] * k * k.conj()).sum(KDIMS)
    S2 = (params["weight"] * k2).sum(KDIMS)

    params["K_rms"] = xr.ufuncs.sqrt(abs((params["weight"] * k2).sum(KDIMS)))
    unit_k2 = k2 / abs(k2)
    params["coherence"] = abs((params["weight"] * unit_k2).sum(KDIMS))
    print("coherence and K_rms")

    direction = xr.ufuncs.sign((k.conj() * params["phase_velocity"]).real)
    weighted_freq_mean = (params["weight"] * params["freq"] * direction).sum(KDIMS)
    params["period"] = (1 / (60 * weighted_freq_mean)).where(weighted_freq_mean > 0)  # minutes
    params["wavelength"] = params["phase_speed"] * params["period"] * 60 / 1000  # km
    print("period / wavelength")
    
    if dask.is_dask_collection(params):
        img.load()
        params.load()
    print("load")

    L = 500
    opts.defaults(
        opts.Points(marker="x", size=15),
        opts.VLine(line_width=2, color="k"),
        opts.Curve(show_grid=True, width=L, height=L-100),
        opts.Image(
            axiswise=True,
            framewise=True,
            cformatter=PrintfTickFormatter(format="%.2e"),
            width=L,
            height=L-100,
        ),
    )
    
    def plotter(time, x, y, kx, ky):
        txy = params.isel(time=time, px=x, py=y)
        kxy = params.isel(px=x, py=y, kx=kx, ky=ky)
        pxy = params.isel(px=x, py=y)
        
        tec_plot = (
            hv.Image(img.isel(time=time))
            .opts(cmap=cc.cm.diverging_bwr_55_98_c37, colorbar=True, clim=(-.3, .3)) *
            hv.Points((params.px.values[x], params.py.values[y]), kdims=["x", "y"])
            .opts(color="k") *
            hv.VectorField(
                params.isel(time=time),
                kdims=["px", "py"],
                vdims=["phase_velocity_angle", "phase_speed", "coherence", "power_threshold"],
            )
            .opts(magnitude="power_threshold", color="coherence", clim=(.2, .8), cmap=cc.cm.kg)
        )

        patch_plot = (
            hv.Image(txy["img_patches"].data[::-1], kdims=["xp", "yp"])
            .opts(cmap=cc.cm.diverging_bwr_55_98_c37, clim=(-.3, .3))
        )

        power_plot = (
            hv.Image(txy["power"])
            .opts(cmap=cc.cm.gouldian, colorbar=True)
        )
        levels = txy["power"].quantile([.8, .95], KDIMS).values
        power_plot = (
            hv.operation.contours(power_plot, levels=levels, overlaid=True)
            .opts(show_legend=False) *
            hv.Points((params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]).opts(color="r")
        )

        freq_img_plot = (
            hv.Image(txy["freq_snr"])
            .opts(cmap=cc.cm.gouldian, colorbar=True) *
            hv.Points((params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]).opts(color="r")
        )

        wf_plot = (
            hv.Image(txy["weight"] * txy["freq"])
            .opts(cmap=cc.cm.diverging_bwr_55_98_c37, colorbar=True) *
            hv.Points((params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]).opts(color="k")
        )

        freq_plot = hv.Curve(kxy["freq"]) * hv.VLine(params.time.values[time])
        freq_snr_plot = hv.Curve(kxy["freq_snr"]) * hv.Curve(pxy["patch_freq_snr"]) * hv.VLine(params.time.values[time])
        coherence_plot = hv.Curve(pxy["coherence"]) * hv.VLine(params.time.values[time])

        tec_plot.opts(width=2*L, height=L-100, title="TEC + Phase Velocity [time]")

        layout = pn.GridSpec()
        layout[0, 0] = coherence_plot.opts(title="Coherence [x, y]")
        layout[1, 0] = freq_snr_plot.opts(logy=True, title="Freq SNR [x, y]")
        layout[2, 0] = freq_plot.opts(title="Freq [x, y, kx, ky]")
        
        layout[0, 1:] = tec_plot
        
        layout[1, 1] = patch_plot.opts(title="Patch [x, y]")
        layout[1, 2] = wf_plot.opts(title="weight * freq [x, y]")
        layout[2, 1] = power_plot.opts(title="power [x, y]")
        layout[2, 2] = freq_img_plot.opts(title="Freq SNR")
        
        return layout
    
    time_slider = pn.widgets.IntSlider(start=0, end=img.time.shape[0]-1, step=1, name="time")
    x_slider = pn.widgets.IntSlider(start=0, end=params.px.shape[0]-1, step=1, name="x")
    y_slider = pn.widgets.IntSlider(start=0, end=params.py.shape[0]-1, step=1, name="y")
    kx_slider = pn.widgets.IntSlider(start=0, end=params.kx.shape[0]-1, step=1, name="kx")
    ky_slider = pn.widgets.IntSlider(start=0, end=params.ky.shape[0]-1, step=1, name="ky")

    box = pn.Row(
        time_slider,
        x_slider,
        y_slider,
        kx_slider,
        ky_slider,
    )
    
    plot = pn.bind(plotter, time=time_slider, x=x_slider, y=y_slider, kx=kx_slider, ky=ky_slider)

    layout = pn.Column(box, plot)
    
    print((time.perf_counter() - t0) / 60)
    return layout


def estimate_parameters_block_unopt(
        data: xr.Dataset,
        Nfft=256,
        block_size=32,
        step_size=8,
        smooth_win=9,
        kaiser_beta=5,
        normalize=None,
    ):
    hres = (data.x[1] - data.x[0]).item()
    edges = block_size // (2 * step_size)
    window = kaiser(block_size, kaiser_beta)
    window = np.outer(window, window) / np.sum(window)

    if "density" in data:
        img = data.image.where(data.density >= 5, 0)
    else:
        img = data.image
    
    if normalize == "image":
        img = (img - img.mean(["x", "y"])) / img.std(["x", "y"])
        
    params = xr.Dataset()
    img_patches = (
        img
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )
    if normalize == "patch":
        img_patches = (img_patches - img_patches.mean(["kx", "ky"])) / img_patches.std(["kx", "ky"])

    wavenum = fftfreq(Nfft, hres)
    
    F = (
        xr.apply_ufunc(
            lambda x: fft2(x * window, s=(Nfft, Nfft)),
            img_patches,
            input_core_dims=[KDIMS],
            output_core_dims=[KDIMS],
            output_dtypes=[np.complex128],
            dask_gufunc_kwargs={"output_sizes": {"kx": Nfft, "ky": Nfft}},
            dask="parallelized",
            exclude_dims={"kx", "ky"}
        )
        .assign_coords(kx=wavenum, ky=wavenum)
        .sortby("kx").sortby("ky")
    ).chunk({"time": -1})

    power = abs(F) ** 2

    # set up weighted average, only keep k bins with power exceeding threshold
    threshold = power.quantile(.95, KDIMS).drop("quantile")
    params["power_threshold"] = threshold
    W = power.where(power > threshold)
    W = W / W.sum(KDIMS)

    phase = xr.apply_ufunc(
        np.unwrap,
        xr.ufuncs.angle(F),
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
    ) / TAU

    freq = phase.differentiate("time", datetime_unit="s")
    freq_noise_power = freq.rolling(time=smooth_win, center=True, min_periods=1).var()
    freq = freq.rolling(time=smooth_win, center=True, min_periods=1).mean()
    freq_snr = (freq**2 / freq_noise_power)
    params["patch_freq_snr"] = (freq_snr * W).sum(KDIMS)
    
    # weighted average phase velocity and phase speed
    k = W.kx + W.ky * 1j
    k2 = k ** 2
    params["phase_velocity"] = (W * k * freq / abs(k)**2).sum(KDIMS)
    params["phase_velocity_angle"] = xr.ufuncs.angle(params["phase_velocity"])
    params["phase_speed"] = abs(params["phase_velocity"]) * 1000  # m/s

    params["S0"] = (W * k * k.conj()).sum(KDIMS).real
    params["S2"] = (W * k2).sum(KDIMS)
    params["anisotropy_mag"] = abs(params["S2"]) / params["S0"]

    K_rms = xr.ufuncs.sqrt(abs((W * k2).sum(KDIMS)))
    unit_k2 = k2 / abs(k2)
    params["coherence"] = abs((W * unit_k2).sum(KDIMS))

    direction = xr.ufuncs.sign((k.conj() * params["phase_velocity"]).real)
    params["weighted_freq_mean"] = (W * freq * direction).sum(KDIMS)
    # arbitrarily setting max period to 2x length of data interval
    max_period = 2 * (data.time[-1] - data.time[0]).dt.total_seconds() / 60  # minutes
    params["period"] = (1 / (60 * params["weighted_freq_mean"])).where(params["weighted_freq_mean"] > 0)  # minutes
    params["period"] = params["period"].where(params["period"] <= max_period)
    params["wavelength"] = params["phase_speed"] * params["period"] * 60 / 1000  # km

    params["alt_phase_speed"] = 1000 * params["weighted_freq_mean"] / K_rms
    
    return params
