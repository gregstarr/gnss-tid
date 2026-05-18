from __future__ import annotations

import logging
import time
from math import ceil

import dask
import dask.array as da
import numpy as np
import xarray as xr
from scipy.fft import fft2, fftfreq, fftshift
from scipy.signal.windows import kaiser

KDIMS = ["kx", "ky"]
TAU = 2 * np.pi


def estimate_noise_hs74(spectrum, navg=1, nnoise_min=1):
    """Estimate noise parameters of a Doppler spectrum (Hildebrand-Sekhon 1974)."""

    sorted_spectrum = np.sort(spectrum)
    nnoise = len(spectrum)
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
            sum1 -= pwr
            sum2 -= pwr * pwr
            break

    mean = sum1 / nnoise
    var = sum2 / nnoise - mean * mean
    threshold = sorted_spectrum[nnoise - 1]
    return mean, threshold, var, nnoise


def get_chunk_size(shape, step, patch, nfft, mem, cx, cy=None):
    if cy is None:
        cy = cx
    if cx == -1:
        cx = shape[-1]
    if cy == -1:
        cy = shape[-2]

    npx = (shape[-1] - patch) // step + 1
    npy = (shape[-2] - patch) // step + 1
    px = min(npx, ceil(cx / step))
    py = min(npy, ceil(cy / step))
    sf = 2 * nfft**2 * px * py
    sr = cx * cy
    factor = sf / sr
    logging.debug("img chunk target: %.2fMB", (mem / factor) / (2**20))
    ct = ceil(((mem / 4) / factor) / sr)
    if (shape[1] // 2) < ct < shape[1]:
        logging.debug("balancing chunk size")
        ct = shape[1] // 2
    return (1, ct, cy, cx)


def estimate_parameters_block_debug(
    data: xr.Dataset,
    Nfft=256,
    block_size=32,
    step_size=8,
    smooth_win=9,
    kaiser_beta=5,
):
    import colorcet as cc
    import holoviews as hv
    import panel as pn
    from bokeh.models import PrintfTickFormatter
    from holoviews import opts

    t0 = time.perf_counter()
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
        img.rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )

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
            exclude_dims={"kx", "ky"},
        )
        .assign_coords(kx=wavenum, ky=wavenum)
        .sortby("kx")
        .sortby("ky")
    )
    params["img_patches"] = img_patches.rename({"kx": "dx", "ky": "dy"})
    if dask.is_dask_collection(params):
        params = params.chunk({"time": -1})

    params["power"] = abs(params["F"]) ** 2
    threshold = params["power"].quantile(0.95, KDIMS)
    params["power_threshold"] = threshold
    params["weight"] = params["power"].where(params["power"] > threshold)
    params["weight"] = params["weight"] / params["weight"].sum(KDIMS)

    params["phase"] = xr.apply_ufunc(
        np.unwrap,
        xr.ufuncs.angle(params["F"]),
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
    ) / TAU

    params["freq"] = params["phase"].differentiate("time", datetime_unit="s")
    freq_noise_power = params["freq"].rolling(time=smooth_win, center=True, min_periods=1).var()
    params["freq"] = params["freq"].rolling(time=smooth_win, center=True, min_periods=1).mean()
    params["freq_snr"] = params["freq"] ** 2 / freq_noise_power
    params["patch_freq_snr"] = (params["freq_snr"] * params["weight"]).sum(KDIMS)

    k = params.kx + params.ky * 1j
    k2 = k**2
    params["phase_velocity"] = (params["weight"] * k * params["freq"] / abs(k) ** 2).sum(KDIMS)
    params["phase_velocity_angle"] = xr.ufuncs.angle(params["phase_velocity"])
    params["phase_speed"] = abs(params["phase_velocity"]) * 1000

    params["K_rms"] = xr.ufuncs.sqrt(abs((params["weight"] * k2).sum(KDIMS)))
    unit_k2 = k2 / abs(k2)
    params["coherence"] = abs((params["weight"] * unit_k2).sum(KDIMS))

    direction = xr.ufuncs.sign((k.conj() * params["phase_velocity"]).real)
    weighted_freq_mean = (params["weight"] * params["freq"] * direction).sum(KDIMS)
    params["period"] = (1 / (60 * weighted_freq_mean)).where(weighted_freq_mean > 0)
    params["wavelength"] = params["phase_speed"] * params["period"] * 60 / 1000

    if dask.is_dask_collection(params):
        img.load()
        params.load()

    L = 500
    opts.defaults(
        opts.Points(marker="x", size=15),
        opts.VLine(line_width=2, color="k"),
        opts.Curve(show_grid=True, width=L, height=L - 100),
        opts.Image(
            axiswise=True,
            framewise=True,
            cformatter=PrintfTickFormatter(format="%.2e"),
            width=L,
            height=L - 100,
        ),
    )

    def plotter(time, x, y, kx, ky):
        txy = params.isel(time=time, px=x, py=y)
        kxy = params.isel(px=x, py=y, kx=kx, ky=ky)
        pxy = params.isel(px=x, py=y)
        tec_plot = (
            hv.Image(img.isel(time=time)).opts(cmap=cc.cm.diverging_bwr_55_98_c37, colorbar=True, clim=(-0.3, 0.3))
            * hv.Points((params.px.values[x], params.py.values[y]), kdims=["x", "y"]).opts(color="k")
            * hv.VectorField(
                params.isel(time=time),
                kdims=["px", "py"],
                vdims=["phase_velocity_angle", "phase_speed", "coherence", "power_threshold"],
            ).opts(magnitude="power_threshold", color="coherence", clim=(0.2, 0.8), cmap=cc.cm.kg)
        )
        patch_plot = hv.Image(txy["img_patches"].data[::-1], kdims=["xp", "yp"]).opts(
            cmap=cc.cm.diverging_bwr_55_98_c37, clim=(-0.3, 0.3)
        )
        power_plot = hv.Image(txy["power"]).opts(cmap=cc.cm.gouldian, colorbar=True)
        levels = txy["power"].quantile([0.8, 0.95], KDIMS).values
        power_plot = hv.operation.contours(power_plot, levels=levels, overlaid=True).opts(show_legend=False) * hv.Points(
            (params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]
        ).opts(color="r")
        freq_img_plot = hv.Image(txy["freq_snr"]).opts(cmap=cc.cm.gouldian, colorbar=True) * hv.Points(
            (params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]
        ).opts(color="r")
        wf_plot = hv.Image(txy["weight"] * txy["freq"]).opts(cmap=cc.cm.diverging_bwr_55_98_c37, colorbar=True) * hv.Points(
            (params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]
        ).opts(color="k")
        freq_plot = hv.Curve(kxy["freq"]) * hv.VLine(params.time.values[time])
        freq_snr_plot = hv.Curve(kxy["freq_snr"]) * hv.Curve(pxy["patch_freq_snr"]) * hv.VLine(params.time.values[time])
        coherence_plot = hv.Curve(pxy["coherence"]) * hv.VLine(params.time.values[time])

        tec_plot.opts(width=2 * L, height=L - 100, title="TEC + Phase Velocity [time]")
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

    time_slider = pn.widgets.IntSlider(start=0, end=img.time.shape[0] - 1, step=1, name="time")
    x_slider = pn.widgets.IntSlider(start=0, end=params.px.shape[0] - 1, step=1, name="x")
    y_slider = pn.widgets.IntSlider(start=0, end=params.py.shape[0] - 1, step=1, name="y")
    kx_slider = pn.widgets.IntSlider(start=0, end=params.kx.shape[0] - 1, step=1, name="kx")
    ky_slider = pn.widgets.IntSlider(start=0, end=params.ky.shape[0] - 1, step=1, name="ky")

    box = pn.Row(time_slider, x_slider, y_slider, kx_slider, ky_slider)
    plot = pn.bind(plotter, time=time_slider, x=x_slider, y=y_slider, kx=kx_slider, ky=ky_slider)
    layout = pn.Column(box, plot)
    logging.info("block debug setup took %.2f min", (time.perf_counter() - t0) / 60)
    return layout


def estimate_parameters_block_unopt(
    data: xr.Dataset,
    Nfft: int = 256,
    block_size: int = 32,
    step_size: int = 8,
    smooth_win: int = 9,
    kaiser_beta: float = 5,
    normalize: str | None = None,
    q_threshold: float = 0.98,
):
    hres = (data.x[1] - data.x[0]).item()
    edges = block_size // (2 * step_size)
    window = kaiser(block_size, kaiser_beta)
    window = np.outer(window, window) / np.sum(window)

    img = data.image.where(data.density >= 5, 0) if "density" in data else data.image
    if normalize == "image":
        img = (img - img.mean(["x", "y"])) / img.std(["x", "y"])

    params = xr.Dataset()
    img_patches = (
        img.rolling(y=block_size, x=block_size, center=True)
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
            exclude_dims={"kx", "ky"},
        )
        .assign_coords(kx=wavenum, ky=wavenum)
        .sortby("kx")
        .sortby("ky")
    ).chunk({"time": -1})

    power = abs(F) ** 2
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
    freq = freq.rolling(time=smooth_win, center=True, min_periods=1).mean()
    params["freq_snr"] = 1 / (freq.rolling(time=smooth_win, center=True, min_periods=1).std() * W).sum(KDIMS)

    k = W.kx + W.ky * 1j
    k2 = k**2
    S0 = (W * k * k.conj()).sum(KDIMS).real
    S2 = (W * k2).sum(KDIMS)
    params["e1"] = (S0 + abs(S2)) / 2
    params["e2"] = (S0 - abs(S2)) / 2
    params["wavelength_snr"] = 1 / np.sqrt(params["e2"])
    direction = np.exp(1j * xr.ufuncs.angle(S2) / 2)

    m = np.sign((direction * k.conj()).real)
    params["wmean_freq"] = abs((W * freq * m).sum(KDIMS))
    params["wmean_wavevector"] = (W * k * m).sum(KDIMS)

    kres = wavenum[1] - wavenum[0]
    params["group_velocity"] = 1000 * (
        (
            freq.sel(kx=params["wmean_wavevector"].real + kres, ky=params["wmean_wavevector"].imag, method="nearest")
            - freq.sel(kx=params["wmean_wavevector"].real - kres, ky=params["wmean_wavevector"].imag, method="nearest")
        )
        + 1j
        * (
            freq.sel(kx=params["wmean_wavevector"].real, ky=params["wmean_wavevector"].imag + kres, method="nearest")
            - freq.sel(kx=params["wmean_wavevector"].real, ky=params["wmean_wavevector"].imag - kres, method="nearest")
        )
    ) / (2 * kres)

    params["phase_velocity"] = 1000 * params["wmean_freq"] / params["wmean_wavevector"]
    params["phase_velocity_angle"] = xr.ufuncs.angle(params["phase_velocity"])
    params["phase_speed"] = abs(params["phase_velocity"])

    max_period = 2 * (data.time[-1] - data.time[0]).dt.total_seconds() / 60
    params["period"] = (1 / (60 * params["wmean_freq"])).where(params["wmean_freq"] > 0)
    params["period"] = params["period"].where(params["period"] <= max_period)
    params["wavelength"] = 1 / abs(params["wmean_wavevector"])
    return params


def estimate_parameters_block(
    data: xr.Dataset,
    *,
    hres: float | None = None,
    nfft: int = 256,
    Nfft: int | None = None,
    block_size: int = 32,
    step_size: int = 8,
    smooth_win: int = 9,
    kaiser_beta: float = 5,
    normalize: str | None = None,
    q_threshold: float = 0.98,
) -> xr.Dataset:
    """xarray spectral parameter estimator (legacy-compatible signature)."""

    if Nfft is not None:
        nfft = Nfft
    if hres is None:
        hres = (data.x[1] - data.x[0]).item()

    wavenum = fftshift(fftfreq(nfft, hres)).astype("float32")
    kres = wavenum[1] - wavenum[0]
    dt = (data.time[1] - data.time[0]).dt.total_seconds().astype("float32")

    img = data.image.where(data.density >= 5, 0) if "density" in data else data.image
    img = img.astype("float32")
    if normalize == "image":
        img = (img - img.mean(["x", "y"])) / img.std(["x", "y"])

    edges = block_size // (2 * step_size)
    window = kaiser(block_size, kaiser_beta).astype("float32")
    window = np.outer(window, window) / np.sum(window)
    img_patches = (
        img.rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )
    if normalize == "patch":
        img_patches = (img_patches - img_patches.mean(["kx", "ky"])) / img_patches.std(["kx", "ky"])

    F = xr.apply_ufunc(
        lambda x: fftshift(fft2(x * window, s=(nfft, nfft)), axes=(-2, -1)),
        img_patches,
        input_core_dims=[KDIMS],
        output_core_dims=[KDIMS],
        output_dtypes=[np.complex64],
        dask_gufunc_kwargs={"output_sizes": {"kx": nfft, "ky": nfft}},
        dask="parallelized",
        exclude_dims={"kx", "ky"},
    ).assign_coords(kx=wavenum, ky=wavenum)

    power = abs(F) ** 2
    freq = F.isel(time=slice(1, None))
    freq = xr.DataArray(
        np.angle(freq.data * F.isel(time=slice(None, -1)).data.conj()) / (TAU * dt.data),
        coords=freq.coords,
    )

    power_threshold = power.quantile(q_threshold, KDIMS).drop_vars("quantile")
    W = power.where(power > power_threshold)
    W = W / W.sum(KDIMS)

    freq = freq.reindex(time=data.time)
    freq_snr = freq.rolling(time=smooth_win, center=True, min_periods=1).std()
    freq = freq.rolling(time=smooth_win, center=True, min_periods=1).mean()
    freq_snr = 1 / ((freq_snr * W).sum(KDIMS))

    k = W.kx + W.ky * 1j
    k2 = k**2
    S0 = (W * k * k.conj()).sum(KDIMS).real
    S2 = (W * k2).sum(KDIMS)
    e2 = (S0 - abs(S2)) / 2
    wavelength_snr = 1 / np.sqrt(e2)
    direction = np.exp(1j * xr.ufuncs.angle(S2) / 2)

    m = np.sign((direction * k.conj()).real)
    wmean_freq = abs((W * freq * m).sum(KDIMS))
    wmean_wavevector = (W * k * m).sum(KDIMS)

    group_velocity = 1000 * (
        (
            freq.sel(kx=wmean_wavevector.real + kres, ky=wmean_wavevector.imag, method="nearest")
            - freq.sel(kx=wmean_wavevector.real - kres, ky=wmean_wavevector.imag, method="nearest")
        )
        + 1j
        * (
            freq.sel(kx=wmean_wavevector.real, ky=wmean_wavevector.imag + kres, method="nearest")
            - freq.sel(kx=wmean_wavevector.real, ky=wmean_wavevector.imag - kres, method="nearest")
        )
    ) / (2 * kres)

    d = wmean_wavevector / abs(wmean_wavevector)
    phase_velocity = 1000 * d * wmean_freq / abs(wmean_wavevector)

    max_period = 2 * (data.time[-1] - data.time[0]).dt.total_seconds() / 60
    period = 1 / (60 * wmean_freq)
    period = period.where((wmean_freq > 0) & (period <= max_period))
    wavelength = 1 / abs(wmean_wavevector)

    return xr.Dataset(
        {
            "period": period,
            "wavelength": wavelength,
            "phase_velocity": phase_velocity,
            "group_velocity": group_velocity,
            "freq_snr": freq_snr,
            "wavelength_snr": wavelength_snr,
            "power_threshold": power_threshold,
            "S0": S0,
            "S2": S2,
        }
    ).drop_vars(["kx", "ky"])


def estimate_parameters_dask(
    data: xr.Dataset,
    *,
    nfft: int = 128,
    Nfft: int | None = None,
    block_size: int = 32,
    step_size: int = 8,
    smooth_win: int = 9,
    kaiser_beta: float = 5,
    normalize: str | None = None,
    q_threshold: float = 0.95,
    density_threshold: int = 5,
    chunk_mem: float = 200 * 2**20,
) -> xr.Dataset:
    """High-throughput Dask spectral parameter estimator."""

    if Nfft is not None:
        nfft = Nfft

    img = data.image.astype("float32")
    if "density" in data:
        img = img.where(data.density >= density_threshold, 0)

    x_vals = img.x.values
    y_vals = img.y.values
    time_vals = img.time.values
    with_trials = img.ndim == 4
    trial_vals = img.trial.values if with_trials else None

    hres = x_vals[1] - x_vals[0]
    wavenum = da.fft.fftshift(da.fft.fftfreq(nfft, hres)).astype("float32")
    dt = (data.time[1] - data.time[0]).dt.total_seconds().data.astype("float32")
    max_period = (data.time[-1] - data.time[0]).dt.total_seconds().data / 60

    if with_trials:
        img = img.transpose("trial", "time", "y", "x").data
    else:
        img = da.expand_dims(img.transpose("time", "y", "x").data, 0)

    _log_ntasks("img", img)
    chunk_size = get_chunk_size(img.shape, step_size, block_size, nfft, chunk_mem, -1)
    img = da.rechunk(img, chunk_size)
    _log_ntasks("img rechunk", img)
    logging.debug("img chunk size: %.2fMB", np.prod(img.chunksize) * 4 / (2**20))

    if normalize == "image":
        img = img - da.nanmean(img, axis=(-2, -1), keepdims=True)
        s = da.nanstd(img, axis=(-2, -1), keepdims=True)
        img = da.where(s > 0, img / s, 0)

    x = da.overlap.sliding_window_view(img, (block_size, block_size), (-2, -1), False)[:, :, ::step_size, ::step_size]
    _log_ntasks("patchify", x)

    if normalize == "patch":
        x = x - da.nanmean(x, axis=(-2, -1), keepdims=True)
        s = da.nanstd(x, axis=(-2, -1), keepdims=True)
        x = da.where(s > 0, x / s, 0)

    window = _kaiser_window_for_dask(block_size, kaiser_beta)
    F = da.fft.fftshift(da.fft.fft2(x * window, s=(nfft, nfft)), axes=(-2, -1))
    _log_ntasks("FFT", F)
    logging.debug("FFT chunk size: %.2fMB", np.prod(F.chunksize) * 8 / (2**20))

    power = abs(F) ** 2
    power_threshold = da.quantile(power, q_threshold, (-2, -1), keepdims=True)
    W = da.where(power > power_threshold, power, 0)
    power_signal = da.sum(W, axis=(-2, -1))
    power_noise = da.sum((power < power_threshold) * power, axis=(-2, -1))
    power_total = da.sum(power, axis=(-2, -1))
    W = W / da.sum(W, axis=(-2, -1), keepdims=True)
    power_threshold = power_threshold[..., 0, 0]

    k = da.expand_dims(wavenum[None, :] + 1j * wavenum[:, None], axis=(0, 1, 2, 3))
    S0 = da.sum(W * k * da.conj(k), axis=(-2, -1)).real
    S2 = da.sum(W * k**2, axis=(-2, -1))
    direction = da.exp(1j * da.angle(S2) / 2)
    m = da.sign((da.expand_dims(direction, axis=(4, 5)) * k.conj()).real)
    wmean_wavevector = da.sum(W * k * m, axis=(-2, -1))

    freq = da.angle(F[:, 1:] * da.conj(F[:, :-1])) / (TAU * dt)
    freq = _center_pad_forward_difference(freq, smooth_win)
    f_windows = da.overlap.sliding_window_view(freq, smooth_win, 1, False)
    freq = da.nanmean(f_windows, -1)
    freq_snr = da.nanstd(f_windows, -1)
    freq_snr = 1 / da.sum(freq_snr * W, axis=(-2, -1))
    freq_snr = da.where(da.isfinite(freq_snr), freq_snr, da.nan)
    wmean_freq = da.sum(abs(W * freq * m), axis=(-2, -1))

    period = 1 / (60 * wmean_freq)
    period = da.where((wmean_freq > 0) & (period <= max_period), period, da.nan)
    wavelength = 1 / abs(wmean_wavevector)
    d = wmean_wavevector / abs(wmean_wavevector)
    phase_velocity = 1000 * d * wmean_freq / abs(wmean_wavevector)

    dims = ["time", "py", "px"]
    coords = {
        "px": x_vals[block_size // 2 : -block_size // 2 + 1 : step_size],
        "py": y_vals[block_size // 2 : -block_size // 2 + 1 : step_size],
        "time": time_vals,
    }
    data_vars = {
        "period": period,
        "wavelength": wavelength,
        "phase_velocity": phase_velocity,
        "freq_snr": freq_snr,
        "power_threshold": power_threshold,
        "S0": S0,
        "S2": S2,
        "power_signal": power_signal,
        "power_noise": power_noise,
        "power_total": power_total,
    }
    if trial_vals is not None:
        dims = ["trial"] + dims
        coords["trial"] = trial_vals
        ds = xr.Dataset({name: (dims, value) for name, value in data_vars.items()}, coords=coords)
        return ds.chunk(trial=10, time=-1, px=-1, py=-1)

    ds = xr.Dataset({name: (dims, value[0]) for name, value in data_vars.items()}, coords=coords)
    return ds.chunk(time=-1, px=-1, py=-1)


def _log_ntasks(name, data):
    logging.debug("%s: %d tasks", name, len(data.__dask_graph__()))


def _kaiser_window_for_dask(block_size: int, beta: float):
    window = kaiser(block_size, beta).astype("float32")
    window = np.outer(window, window) / np.sum(window)
    return np.expand_dims(window, (0, 1, 2, 3))


def _center_pad_forward_difference(freq, smooth_win: int):
    padding = 6 * [(0, 0)]
    padding[1] = ((smooth_win - 1) // 2, (smooth_win - 1) // 2 + 1)
    return da.pad(freq, padding, mode="constant", constant_values=da.nan)
