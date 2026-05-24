"""Interactive HoloViews/Panel dashboard for inspecting block parameter estimates.

This is a development-only sandbox.  It mirrors the early structure of
``gnss_tid.parameter.estimate_parameters_block`` but adds a Panel UI so you can
scrub through ``time / px / py / kx / ky`` and visualize per-patch quantities
(FFT, weight, frequency SNR, phase velocity, etc.).

Not imported by the library.  Run as a notebook cell or `python -i` script.
"""

import time

import dask
import numpy as np
import xarray as xr
from scipy.fft import fft2, fftfreq
from scipy.signal.windows import kaiser

from gnss_tid.parameter import KDIMS, TAU


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
        img.rolling(y=block_size, x=block_size, center=True)
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
            exclude_dims={"kx", "ky"},
        )
        .assign_coords(kx=wavenum, ky=wavenum)
        .sortby("kx")
        .sortby("ky")
    )
    params["img_patches"] = img_patches.rename({"kx": "dx", "ky": "dy"})
    if dask.is_dask_collection(params):
        params = params.chunk({"time": -1})
    print("fft")

    params["power"] = abs(params["F"]) ** 2
    print("power")

    threshold = params["power"].quantile(0.95, KDIMS)
    params["power_threshold"] = threshold
    params["weight"] = params["power"].where(params["power"] > threshold)
    params["weight"] = params["weight"] / params["weight"].sum(KDIMS)

    print("weight")

    params["phase"] = (
        xr.apply_ufunc(
            np.unwrap,
            xr.ufuncs.angle(params["F"]),
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
        )
        / TAU
    )
    print("phase")

    params["freq"] = params["phase"].differentiate("time", datetime_unit="s")
    freq_noise_power = (
        params["freq"].rolling(time=smooth_win, center=True, min_periods=1).var()
    )
    params["freq"] = (
        params["freq"].rolling(time=smooth_win, center=True, min_periods=1).mean()
    )
    params["freq_snr"] = params["freq"] ** 2 / freq_noise_power
    params["patch_freq_snr"] = (params["freq_snr"] * params["weight"]).sum(KDIMS)
    print("freq")

    k = params.kx + params.ky * 1j
    k2 = k**2
    params["phase_velocity"] = (
        params["weight"] * k * params["freq"] / abs(k) ** 2
    ).sum(KDIMS)
    params["phase_velocity_angle"] = xr.ufuncs.angle(params["phase_velocity"])
    params["phase_speed"] = abs(params["phase_velocity"]) * 1000  # m/s
    print("phase velocity")

    params["K_rms"] = xr.ufuncs.sqrt(abs((params["weight"] * k2).sum(KDIMS)))
    unit_k2 = k2 / abs(k2)
    params["coherence"] = abs((params["weight"] * unit_k2).sum(KDIMS))
    print("coherence and K_rms")

    direction = xr.ufuncs.sign((k.conj() * params["phase_velocity"]).real)
    weighted_freq_mean = (params["weight"] * params["freq"] * direction).sum(KDIMS)
    params["period"] = (1 / (60 * weighted_freq_mean)).where(
        weighted_freq_mean > 0
    )  # minutes
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
            hv.Image(img.isel(time=time)).opts(
                cmap=cc.cm.diverging_bwr_55_98_c37, colorbar=True, clim=(-0.3, 0.3)
            )
            * hv.Points(
                (params.px.values[x], params.py.values[y]), kdims=["x", "y"]
            ).opts(color="k")
            * hv.VectorField(
                params.isel(time=time),
                kdims=["px", "py"],
                vdims=[
                    "phase_velocity_angle",
                    "phase_speed",
                    "coherence",
                    "power_threshold",
                ],
            ).opts(
                magnitude="power_threshold",
                color="coherence",
                clim=(0.2, 0.8),
                cmap=cc.cm.kg,
            )
        )

        patch_plot = hv.Image(txy["img_patches"].data[::-1], kdims=["xp", "yp"]).opts(
            cmap=cc.cm.diverging_bwr_55_98_c37, clim=(-0.3, 0.3)
        )

        power_plot = hv.Image(txy["power"]).opts(cmap=cc.cm.gouldian, colorbar=True)
        levels = txy["power"].quantile([0.8, 0.95], KDIMS).values
        power_plot = hv.operation.contours(
            power_plot, levels=levels, overlaid=True
        ).opts(show_legend=False) * hv.Points(
            (params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]
        ).opts(color="r")

        freq_img_plot = hv.Image(txy["freq_snr"]).opts(
            cmap=cc.cm.gouldian, colorbar=True
        ) * hv.Points(
            (params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]
        ).opts(color="r")

        wf_plot = hv.Image(txy["weight"] * txy["freq"]).opts(
            cmap=cc.cm.diverging_bwr_55_98_c37, colorbar=True
        ) * hv.Points(
            (params.kx.values[kx], params.ky.values[ky]), kdims=["kx", "ky"]
        ).opts(color="k")

        freq_plot = hv.Curve(kxy["freq"]) * hv.VLine(params.time.values[time])
        freq_snr_plot = (
            hv.Curve(kxy["freq_snr"])
            * hv.Curve(pxy["patch_freq_snr"])
            * hv.VLine(params.time.values[time])
        )
        coherence_plot = hv.Curve(pxy["coherence"]) * hv.VLine(
            params.time.values[time]
        )

        tec_plot.opts(
            width=2 * L, height=L - 100, title="TEC + Phase Velocity [time]"
        )

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

    time_slider = pn.widgets.IntSlider(
        start=0, end=img.time.shape[0] - 1, step=1, name="time"
    )
    x_slider = pn.widgets.IntSlider(
        start=0, end=params.px.shape[0] - 1, step=1, name="x"
    )
    y_slider = pn.widgets.IntSlider(
        start=0, end=params.py.shape[0] - 1, step=1, name="y"
    )
    kx_slider = pn.widgets.IntSlider(
        start=0, end=params.kx.shape[0] - 1, step=1, name="kx"
    )
    ky_slider = pn.widgets.IntSlider(
        start=0, end=params.ky.shape[0] - 1, step=1, name="ky"
    )

    box = pn.Row(time_slider, x_slider, y_slider, kx_slider, ky_slider)

    plot = pn.bind(
        plotter, time=time_slider, x=x_slider, y=y_slider, kx=kx_slider, ky=ky_slider
    )

    layout = pn.Column(box, plot)

    print((time.perf_counter() - t0) / 60)
    return layout
