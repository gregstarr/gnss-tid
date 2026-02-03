import time

import dask
import numpy as np
import xarray as xr
from scipy.fft import fft, fft2, fftfreq
from scipy.signal.windows import kaiser
from scipy.interpolate import make_splrep, make_smoothing_spline

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


KDIMS = ["kx", "ky"]
TAU = 2 * np.pi

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
        Nfft: int = 256,
        block_size: int = 32,
        step_size: int = 8,
        smooth_win: int = 9,
        kaiser_beta: float = 5,
        normalize: str | None = None,
        q_threshold: float = .95
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
        img_patches = (
            (img_patches - img_patches.mean(["kx", "ky"]))
            / img_patches.std(["kx", "ky"])
        )

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
    threshold = power.quantile(q_threshold, KDIMS).drop_vars("quantile")
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
    
    k = W.kx + W.ky * 1j
    k2 = k ** 2
    params["S0"] = (W * k * k.conj()).sum(KDIMS).real
    params["S2"] = (W * k2).sum(KDIMS)
    params["anisotropy_mag"] = abs(params["S2"]) / params["S0"]
    direction = np.exp(1j * xr.ufuncs.angle(params["S2"]) / 2)

    # positive / negative projection from phase velocity
    m = np.sign((direction * k.conj()).real)
    params["wmean_freq"] = (W * freq * m).sum(KDIMS)
    params["wmean_wavevector"] = (W * k * m).sum(KDIMS)

    kres = wavenum[1] - wavenum[0]
    params["group_velocity"] = 1000 * (
        (
            freq.sel(kx=params["wmean_wavevector"].real+kres, ky=params["wmean_wavevector"].imag, method="nearest") - 
            freq.sel(kx=params["wmean_wavevector"].real-kres, ky=params["wmean_wavevector"].imag, method="nearest")
        ) + 
        1j * (
            freq.sel(kx=params["wmean_wavevector"].real, ky=params["wmean_wavevector"].imag+kres, method="nearest") - 
            freq.sel(kx=params["wmean_wavevector"].real, ky=params["wmean_wavevector"].imag-kres, method="nearest")
        )
    ) / (2 * kres)  # m/s

    # weighted average phase velocity and phase speed
    params["phase_velocity"] = 1000 * params["wmean_freq"] / params["wmean_wavevector"]
    params["phase_velocity_angle"] = xr.ufuncs.angle(params["phase_velocity"])
    params["phase_speed"] = abs(params["phase_velocity"])  # m/s

    unit_k2 = k2 / abs(k2)
    params["coherence"] = abs((W * unit_k2).sum(KDIMS))

    # arbitrarily setting max period to 2x length of data interval
    max_period = 2 * (data.time[-1] - data.time[0]).dt.total_seconds() / 60  # minutes
    params["period"] = (
        (1 / (60 * params["wmean_freq"]))
        .where(params["wmean_freq"] > 0)
    )  # minutes
    params["period"] = params["period"].where(params["period"] <= max_period)
    params["wavelength"] = 1 / abs(params["wmean_wavevector"])  # km
    
    return params
