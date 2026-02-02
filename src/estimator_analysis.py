import itertools
import time
import pathlib
import shutil

from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter
from scipy.optimize import least_squares
import seaborn as sns
import tqdm
import numpy as np
import dask
from dask.distributed import Client
import xarray
from xarray.groupers import BinGrouper, UniqueGrouper
import matplotlib.pyplot as plt


ERR_VARS = {"period_error": "tau", "wavelength_error": "lam", "speed_error": "spd"}
DIST_VARS = ["r", "w", "logw"]
ERROR_TYPES = ["bias", "mae"]
NORM_TYPES = ["abs", "rel"]


def plot_snr_range_error(error_val, groupers, fn, **plot_kwargs):
    comp = error_val.groupby(groupers).median(...)
    artist = comp.plot(**plot_kwargs)
    if hasattr(artist, "fig"):
        fig = artist.fig
    else:
        fig = artist.get_figure()
    plt.savefig(fn)
    plt.close(fig)


def get_groupers(results):
    return {
        "r": BinGrouper(21),
        "w": BinGrouper(21),
        "logw": BinGrouper(21),
        "snr": UniqueGrouper(),
        "tau": UniqueGrouper(),
        "lam": UniqueGrouper(),
    }


def get_results(fn, center=None):
    results = xarray.open_zarr(fn)
    results = results.assign_coords(spd=results.lam * 1000 / (results.tau * 60))
    if center is not None:
        results = results.assign_coords(r=xarray.ufuncs.hypot(results.px-center[0], results.py-center[1]))
        results = results.assign_coords(w=results.r / results.lam)
        results = results.assign_coords(logw=xarray.ufuncs.log10(results.w))
    results = results.assign(
        period_error=results.period-results.tau,
        wavelength_error=results.wavelength-results.lam,
        speed_error=results.phase_speed-results.spd,
    )
    return results.sortby("snr")


def run_snr_range_task(results, groupers, varname, group_var, err_type, norm_type, facet=None, logscale=False):
    name = f"{varname}_{norm_type}_{err_type}__by__{group_var}"
    if facet:
        name = name + f"_and_{facet}"
    if logscale:
        name = "log_" + name
    
    data = results[varname]
    if norm_type == "rel":
        data = data / results[ERR_VARS[varname]]
    if err_type == "mae":
        data = abs(data)
    if logscale:
        data = xarray.ufuncs.log10(data)
    
    groups = {
        "snr": groupers["snr"],
        group_var: groupers[group_var],
    }
    if facet is not None:
        groups[facet] = groupers[facet]
    
    kwargs = {
        "fn": f"plots/{varname}_{err_type}/{name}.png",
        "center": 0 if err_type == "bias" else False,
        "robust": True,
    }
    if facet:
        kwargs["col"] = facet
        kwargs["col_wrap"] = 3
    
    plot_snr_range_error(data, groups, **kwargs)


def make_line_plots(results, varname, norm_type, logscale=False):
    name = f"{varname}_{norm_type}__by__snr__line"
    if logscale:
        name = "log_" + name
    
    data = results[varname]
    if norm_type == "rel":
        data = data / results[ERR_VARS[varname]]
    data = abs(data)
    df = data.to_dataframe(name=varname)
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df, x="snr", y=varname,
        errorbar="pi",
        estimator="median",
        ax=ax,
    )
    if logscale:
        ax.set_yscale("log")
        set_log_ticks_y(ax)

    fig.savefig(f"plots/{varname}_line/{name}.png")
    plt.close(fig)


def plot_coupling(results):
    speed_error = results["speed_error"] / results["spd"]
    period_error = results["period_error"] / results["tau"]
    wavelength_error = results["wavelength_error"] / results["lam"]

    ds = speed_error.values.ravel()
    dp = period_error.values.ravel()
    dw = wavelength_error.values.ravel()
    m = np.isfinite(ds) & np.isfinite(dp) & np.isfinite(dw)
    ds = ds[m]
    dp = dp[m]
    dw = dw[m]
    m = (dw < np.quantile(dw, .999)) & (dp < np.quantile(dp, .999))

    pe = abs(period_error).to_dataframe(name="period_error")
    fg = sns.relplot(pe, x="snr", y="period_error", hue="tau", kind="line", estimator="median", errorbar=None)
    fg.set(ylim=[.009, .6])
    fg.set(yscale="log")
    set_log_ticks_y(fg.ax)
    plt.savefig("plots/period.png")
    plt.close()

    we = abs(wavelength_error).to_dataframe(name="wavelength_error")
    fg = sns.relplot(we, x="snr", y="wavelength_error", hue="lam", kind="line", estimator="median", errorbar=None)
    # fg.set(ylim=[.009, .6])
    fg.set(yscale="log")
    set_log_ticks_y(fg.ax)
    plt.savefig("plots/wavelength.png")
    plt.close()

    def model(p, dp):
        return p[0]*dp / (1 + dp * p[1])

    def residuals(p, dp, ds):
        return ds - model(p, dp)

    init = [-1, 1]
    fit  = least_squares(residuals, init, args=(dp, ds), loss='soft_l1')
    print(fit)

    fig, ax = plt.subplots()
    ax.plot(dp, ds, 'k.', alpha=.1, ms=5)
    x = np.arange(-1/fit.x[1], 4, .1) + .01
    sns.kdeplot(x=dp[m], y=ds[m], levels=[.05, .2, .5], linewidths=2)
    ax.plot(x, model(fit.x, x), label=f"y=${fit.x[0]:.2f}x/(1+{fit.x[1]:.2f}x)$")
    ax.legend()
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    fig.savefig("plots/speed_vs_period.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(dw, dp, 'k.', alpha=.1, ms=4)
    sns.kdeplot(x=dw[m], y=dp[m], levels=[.05, .2, .5], linewidths=2)
    ax.set_xlabel("wavelength relative error")
    ax.set_ylabel("period relative error")
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    plt.savefig("plots/wavelength_vs_period.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(dw, dp, 'k.', alpha=.1, ms=4)
    sns.kdeplot(x=dw[m], y=dp[m], levels=[.05, .2, .5], linewidths=2)
    ax.set_xlabel("wavelength relative error")
    ax.set_ylabel("period relative error")
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    plt.savefig("plots/wavelength_vs_period.png")
    plt.close()


def set_log_ticks_y(ax, base=10, label_minor_if_no_major=True, numticks=15):
    # Major ticks: powers of base
    major = LogLocator(base=base, subs=(1.0,), numticks=numticks)
    ax.yaxis.set_major_locator(major)

    # Minor ticks: 2..(base-1) between powers
    subs = np.arange(2, int(base)) / base
    minor = LogLocator(base=base, subs=subs, numticks=numticks)
    ax.yaxis.set_minor_locator(minor)

    # Major formatter (label only powers of base)
    ax.yaxis.set_major_formatter(LogFormatter(base=base, labelOnlyBase=True))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # If no major ticks visible, optionally label minors
    def has_visible_major(axis, loc):
        lo, hi = axis.get_view_interval()
        return np.any((loc > min(lo, hi)) & (loc < max(lo, hi)))

    if label_minor_if_no_major:
        ymaj = major.tick_values(*ax.get_ylim())
        if not has_visible_major(ax.yaxis, ymaj):
            ax.yaxis.set_minor_formatter(LogFormatter(base=base, labelOnlyBase=False))

    # Ensure ticks are visible
    ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.5)



def make_dirs():
    for varname in ERR_VARS:
        pathlib.Path(f"plots/{varname}_line").mkdir(exist_ok=True)
        for err_type in ERROR_TYPES:
            pathlib.Path(f"plots/{varname}_{err_type}").mkdir(exist_ok=True)


def make_archive():
    print("making archive")
    src = pathlib.Path("plots")
    archive = shutil.make_archive("plots", "gztar", src.parent, src.name)
    print(archive)


def main():
    RESULTS_FILE = "/disk1/tid/users/starr/results/results3.zarr"
    sns.set_theme("notebook", "whitegrid")
    sns.color_palette("crest", as_cmap=True)
    
    make_dirs()
    
    dask.config.set({
        "distributed.worker.memory.spill": 0.85,  # default: 0.7
        "distributed.worker.memory.target": 0.75,  # default: 0.6
        "distributed.worker.memory.terminate": 0.98,  # default: 0.95
    })
    with Client(processes=True, n_workers=8, threads_per_worker=1) as client:
        print(client.dashboard_link)
        results = get_results(RESULTS_FILE).compute()
        print("groupers")
        groupers = get_groupers(results)

        r = results.median(["time"])
        
        plot_coupling(r)
    
        args = list(itertools.product(ERR_VARS, NORM_TYPES))
        for a in tqdm.tqdm(args, "line plots"):
            make_line_plots(r, *a, logscale=True)

        # BASIC PLOTS
        args = list(itertools.product(ERR_VARS, DIST_VARS, ERROR_TYPES, NORM_TYPES))
        for a in tqdm.tqdm(args, "basic plots"):
            run_snr_range_task(results, groupers, *a)

        args = list(itertools.product(ERR_VARS, DIST_VARS, ["mae"], NORM_TYPES))
        for a in tqdm.tqdm(args, "log basic plots"):
            run_snr_range_task(results, groupers, *a, logscale=True)
        
        # FACET PLOTS
        facet_vars = ["lam", "tau"]
        args = list(itertools.product(ERR_VARS, DIST_VARS, ERROR_TYPES, NORM_TYPES, facet_vars))
        for a in tqdm.tqdm(args, "facet plots"):
            run_snr_range_task(results, groupers, *a)
        
        args = list(itertools.product(ERR_VARS, DIST_VARS, ["mae"], NORM_TYPES, facet_vars))
        for a in tqdm.tqdm(args, "log facet plots"):
            run_snr_range_task(results, groupers, *a, logscale=True)
    
    print("CLUSTER CLOSED")

    make_archive()


if __name__ == "__main__":
    main()
