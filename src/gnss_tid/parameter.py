import logging
from math import ceil

import dask
import dask.array as da
import numpy as np
import xarray as xr

from .fft import fft_patches, make_kaiser_2d, make_patches, make_wavenum_grid


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


def _phase_diff_freq(F_next, F_prev, dt):
    """Forward-difference instantaneous frequency from time-adjacent spectra."""
    return np.angle(F_next * np.conj(F_prev)) / (TAU * dt)


def _spectral_moments(W, k):
    """Weighted 0th and 2nd wavenumber moments reduced over the last two axes."""
    S0 = (W * k * np.conj(k)).sum(axis=(-2, -1)).real
    S2 = (W * k**2).sum(axis=(-2, -1))
    return S0, S2


def _principal_direction(S2):
    """Unit vector along the wavenumber principal axis (half-angle of S2)."""
    return np.exp(1j * np.angle(S2) / 2)


def _sign_projection(direction, k):
    """+/-1 per k-bin: sign of the projection onto ``direction``.

    xarray broadcasts ``direction`` against ``k`` by dim name; raw numpy/dask
    arrays need explicit trailing (ky, kx) axes on ``direction``.
    """
    if isinstance(direction, xr.DataArray):
        return np.sign((direction * np.conj(k)).real)
    return np.sign((direction[..., None, None] * np.conj(k)).real)


def _period_from_freq(wmean_freq, max_period_min):
    """Period in minutes; masks ``freq <= 0`` and ``period > max_period_min``."""
    period = 1 / (60 * wmean_freq)
    mask = (wmean_freq > 0) & (period <= max_period_min)
    if isinstance(wmean_freq, xr.DataArray):
        return period.where(mask)
    xp = da if isinstance(wmean_freq, da.Array) else np
    return xp.where(mask, period, xp.nan)


def _phase_velocity(wmean_wavevector, wmean_freq):
    """Complex phase velocity (m/s): unit-direction * (f / |k|)."""
    d = wmean_wavevector / abs(wmean_wavevector)
    return 1000 * d * wmean_freq / abs(wmean_wavevector)


def _wmean_freq(W, freq, m):
    """Weighted-mean frequency magnitude with directional sign projection."""
    return abs((W * freq * m).sum(axis=(-2, -1)))


def _finite_or_nan(x):
    """Replace non-finite values with NaN, preserving array backend."""
    if isinstance(x, xr.DataArray):
        return x.where(np.isfinite(x))
    xp = da if isinstance(x, da.Array) else np
    return xp.where(xp.isfinite(x), x, xp.nan)


def _rolling_freq_stats(freq, win, axis_or_dim):
    """Centered rolling nan-mean and nan-std along time.

    Pass a dim name for xarray inputs (uses ``rolling`` with ``min_periods=1``);
    pass an integer axis for raw numpy/dask inputs (uses pad + sliding window
    so the output length matches forward-diff input + 1).
    """
    if isinstance(freq, xr.DataArray):
        rolling = freq.rolling({axis_or_dim: win}, center=True, min_periods=1)
        return rolling.mean(), rolling.std()
    pad = [(0, 0)] * freq.ndim
    pad[axis_or_dim] = ((win - 1) // 2, (win - 1) // 2 + 1)
    if isinstance(freq, da.Array):
        padded = da.pad(freq, pad, mode="constant", constant_values=da.nan)
        windows = da.overlap.sliding_window_view(padded, win, axis_or_dim, False)
        return da.nanmean(windows, -1), da.nanstd(windows, -1)
    padded = np.pad(freq, pad, mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, win, axis_or_dim)
    return np.nanmean(windows, -1), np.nanstd(windows, -1)


def estimate_parameters_block(
    data: xr.Dataset,
    *,
    Nfft: int = 256,
    block_size: int = 32,
    step_size: int = 8,
    smooth_win: int = 9,
    kaiser_beta: float = 5,
    normalize: str | None = None,
    q_threshold: float = 0.98,
) -> xr.Dataset:
    """Spectral parameter estimation over rolling FFT patches.

    Eager / xarray-native variant.  For large grids prefer
    :func:`estimate_parameters_dask`.

    Args:
        data: Dataset with ``image (time, y, x)`` and optionally
            ``density (y, x)``.  ``x`` must be uniformly spaced.
        Nfft: FFT length; pads patches when ``Nfft > block_size``.
        block_size: Spatial patch side length (pixels).
        step_size: Stride between patch centres (pixels).
        smooth_win: Rolling-window length for time-axis frequency smoothing.
        kaiser_beta: Kaiser window shape parameter.
        normalize: ``"image"`` z-scores each frame; ``"patch"`` z-scores each
            patch; ``None`` skips normalization.
        q_threshold: Quantile of patch power used to threshold the weighted
            k-space average (in ``[0, 1]``).

    Returns:
        Dataset with dims ``(time, py, px)`` and variables ``period``,
        ``wavelength``, ``phase_velocity``, ``group_velocity``, ``freq_snr``,
        ``wavelength_snr``, ``power_threshold``, ``S0``, ``S2``.
    """
    hres = (data.x[1] - data.x[0]).item()
    wavenum = make_wavenum_grid(Nfft, hres, shift=True, dtype="float32")
    kres = wavenum[1] - wavenum[0]
    dt = (data.time[1] - data.time[0]).dt.total_seconds().astype("float32")

    img = data.image.where(data.density >= 5, 0) if "density" in data else data.image
    img = img.astype("float32")

    if normalize == "image":
        img = (img - img.mean(["x", "y"])) / img.std(["x", "y"])

    window = make_kaiser_2d(block_size, kaiser_beta).astype("float32")
    img_patches = make_patches(img, block_size, step_size)
    if normalize == "patch":
        img_patches = (img_patches - img_patches.mean(["kx", "ky"])) / img_patches.std(
            ["kx", "ky"]
        )

    F = fft_patches(img_patches, window, Nfft=Nfft, shift=True).assign_coords(
        kx=wavenum, ky=wavenum
    )
    if dask.is_dask_collection(F):
        F = F.chunk({"time": -1})
    del img_patches

    power = abs(F) ** 2
    F_next = F.isel(time=slice(1, None))
    freq = xr.DataArray(
        _phase_diff_freq(F_next.data, F.isel(time=slice(None, -1)).data, dt.data),
        coords=F_next.coords,
    )
    del F, F_next

    # weighted average: keep only k bins with power exceeding the q_threshold
    power_threshold = power.quantile(q_threshold, KDIMS).drop_vars("quantile")
    W = power.where(power > power_threshold)
    W = W / W.sum(KDIMS)

    freq = freq.reindex(time=data.time)
    freq, freq_std = _rolling_freq_stats(freq, smooth_win, "time")
    freq_snr = _finite_or_nan(1 / (freq_std * W).sum(KDIMS))

    k = W.kx + W.ky * 1j
    S0, S2 = _spectral_moments(W, k)
    e2 = (S0 - abs(S2)) / 2
    wavelength_snr = 1 / np.sqrt(e2)
    direction = _principal_direction(S2)
    m = _sign_projection(direction, k)
    wmean_freq = _wmean_freq(W, freq, m)
    wmean_wavevector = (W * k * m).sum(KDIMS)

    group_velocity = (
        1000
        * (
            (
                freq.sel(
                    kx=wmean_wavevector.real + kres,
                    ky=wmean_wavevector.imag,
                    method="nearest",
                )
                - freq.sel(
                    kx=wmean_wavevector.real - kres,
                    ky=wmean_wavevector.imag,
                    method="nearest",
                )
            )
            + 1j
            * (
                freq.sel(
                    kx=wmean_wavevector.real,
                    ky=wmean_wavevector.imag + kres,
                    method="nearest",
                )
                - freq.sel(
                    kx=wmean_wavevector.real,
                    ky=wmean_wavevector.imag - kres,
                    method="nearest",
                )
            )
        )
        / (2 * kres)
    )  # m/s

    phase_velocity = _phase_velocity(wmean_wavevector, wmean_freq)  # m/s

    # arbitrarily setting max period to 2x length of data interval
    max_period = 2 * (data.time[-1] - data.time[0]).dt.total_seconds() / 60  # minutes
    period = _period_from_freq(wmean_freq, max_period)  # minutes
    wavelength = 1 / abs(wmean_wavevector)  # km

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


def log_ntasks(name, data):
    logging.debug(f"{name}: {len(data.__dask_graph__())} tasks")


def get_chunk_size(shape, step, patch, Nfft, mem, cx, cy=None):
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
    sf = 2 * Nfft**2 * px * py
    sr = cx * cy
    factor = sf / sr
    logging.debug(f"img chunk target: {(mem/factor)/(2**20)}MB")
    ct = ceil(((mem / 4) / factor) / sr)
    if (shape[1] // 2) < ct < shape[1]:
        logging.debug("balancing chunk size")
        ct = shape[1] // 2
    return (1, ct, cy, cx)


def estimate_parameters_dask(
    data: xr.Dataset,
    Nfft: int = 128,
    block_size: int = 32,
    step_size: int = 8,
    smooth_win: int = 9,
    kaiser_beta: float = 5,
    normalize: str | None = None,
    q_threshold: float = 0.95,
    density_threshold: int = 5,
    chunk_mem: float = 200 * 2**20,
):
    img = data.image.astype("float32")
    if "density" in data:
        img = img.where(data.density >= density_threshold, 0)

    x_vals = img.x.values
    y_vals = img.y.values
    time_vals = img.time.values
    with_trials = img.ndim == 4  # trial, time, y, x

    hres = x_vals[1] - x_vals[0]
    wavenum = make_wavenum_grid(Nfft, hres, shift=True, dtype="float32")
    dt = (data.time[1] - data.time[0]).dt.total_seconds().data.astype("float32")
    # setting max period to length of data interval
    max_period = (data.time[-1] - data.time[0]).dt.total_seconds().data / 60  # minutes

    window = make_kaiser_2d(block_size, kaiser_beta).astype("float32")
    window = np.expand_dims(window, (0, 1, 2, 3))

    # ensure dim order
    if with_trials:
        trial_vals = img.trial.values
        img = img.transpose("trial", "time", "y", "x")
        img = img.data
    else:
        img = img.transpose("time", "y", "x")
        img = da.expand_dims(img.data, 0)

    log_ntasks("img", img)

    # sliding window over spatial dims is best if spatial dims are in single chunk
    cx = -1
    chunk_size = get_chunk_size(img.shape, step_size, block_size, Nfft, chunk_mem, cx)
    img = da.rechunk(img, chunk_size)
    log_ntasks("img rechunk", img)
    logging.debug(img)
    logging.debug(f"img chunk size: {np.prod(img.chunksize)*4/(2**20)}MB")

    # IMG NOW DASK ARRAY
    if normalize == "image":
        img = img - da.nanmean(img, axis=(-2, -1), keepdims=True)
        s = da.nanstd(img, axis=(-2, -1), keepdims=True)
        img = da.where(s > 0, img / s, 0)

    # (trial, time, py, px, ky, kx)
    x = da.overlap.sliding_window_view(
        img,
        (block_size, block_size),
        (-2, -1),
        False,
    )[:, :, ::step_size, ::step_size]
    log_ntasks("patchify", x)

    if normalize == "patch":
        x = x - da.nanmean(x, axis=(-2, -1), keepdims=True)
        s = da.nanstd(x, axis=(-2, -1), keepdims=True)
        x = da.where(s > 0, x / s, 0)
        log_ntasks("normalize patches", x)

    F = fft_patches(x, window, Nfft=Nfft, shift=True)
    log_ntasks("FFT", F)
    logging.debug(F)
    logging.debug(f"FFT chunk size: {np.prod(F.chunksize)*8/(2**20)}MB")

    power = abs(F) ** 2
    log_ntasks("power", power)

    power_threshold = da.quantile(power, q_threshold, (-2, -1), keepdims=True)
    log_ntasks("power thresh", power_threshold)

    W = da.where(power > power_threshold, power, 0)
    log_ntasks("threshold", W)
    power_signal = da.sum(W, axis=(-2, -1))
    power_noise = da.sum((power < power_threshold) * power, axis=(-2, -1))
    power_total = da.sum(power, axis=(-2, -1))
    W = W / da.sum(W, axis=(-2, -1), keepdims=True)
    log_ntasks("normalize W", W)
    power_threshold = power_threshold[..., 0, 0]

    k = da.expand_dims(wavenum[None, :] + 1j * wavenum[:, None], axis=(0, 1, 2, 3))
    S0, S2 = _spectral_moments(W, k)
    log_ntasks("S2", S2)
    direction = _principal_direction(S2)
    log_ntasks("direction", direction)

    m = _sign_projection(direction, k)
    log_ntasks("m", m)
    wmean_wavevector = (W * k * m).sum(axis=(-2, -1))
    log_ntasks("weighted mean wavevector", wmean_wavevector)

    freq = _phase_diff_freq(F[:, 1:], F[:, :-1], dt)
    log_ntasks("freq", freq)
    freq, freq_std = _rolling_freq_stats(freq, smooth_win, 1)
    log_ntasks("freq smoothed", freq)
    freq_snr = _finite_or_nan(1 / (freq_std * W).sum(axis=(-2, -1)))
    log_ntasks("freq snr", freq_snr)
    wmean_freq = _wmean_freq(W, freq, m)
    log_ntasks("weighted mean freq", wmean_freq)
    period = _period_from_freq(wmean_freq, max_period)
    log_ntasks("period", period)
    wavelength = 1 / abs(wmean_wavevector)  # km
    log_ntasks("wavelength", wavelength)

    phase_velocity = _phase_velocity(wmean_wavevector, wmean_freq)  # m/s
    log_ntasks("phase velocity", phase_velocity)

    dims = ["time", "py", "px"]
    coords = {
        "px": x_vals[block_size // 2 : -block_size // 2 + 1 : step_size],
        "py": y_vals[block_size // 2 : -block_size // 2 + 1 : step_size],
        "time": time_vals,
    }
    if with_trials:
        dims = ["trial", *dims]
        coords["trial"] = trial_vals
        params = xr.Dataset(
            data_vars={
                "period": (dims, period),
                "wavelength": (dims, wavelength),
                "phase_velocity": (dims, phase_velocity),
                "power_threshold": (dims, power_threshold),
                "S0": (dims, S0),
                "S2": (dims, S2),
                "power_signal": (dims, power_signal),
                "power_noise": (dims, power_noise),
                "power_total": (dims, power_total),
            },
            coords=coords,
        )
        params = params.chunk(trial=10, time=-1, px=-1, py=-1)
    else:
        params = xr.Dataset(
            data_vars={
                "period": (dims, period[0]),
                "wavelength": (dims, wavelength[0]),
                "phase_velocity": (dims, phase_velocity[0]),
                "power_threshold": (dims, power_threshold[0]),
                "S0": (dims, S0[0]),
                "S2": (dims, S2[0]),
                "power_signal": (dims, power_signal[0]),
                "power_noise": (dims, power_noise[0]),
                "power_total": (dims, power_total[0]),
            },
            coords=coords,
        )
        params = params.chunk(time=-1, px=-1, py=-1)

    log_ntasks("phase velocity final chunk", params.phase_velocity.data)
    return params
