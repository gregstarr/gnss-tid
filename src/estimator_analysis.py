import itertools
import time
import pathlib
import shutil

from matplotlib.colors import LogNorm
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
    comp = error_val.groupby(groupers).mean(...)
    artist = comp.plot(**plot_kwargs)
    if hasattr(artist, "fig"):
        fig = artist.fig
    else:
        fig = artist.get_figure()
    plt.savefig(fn)
    plt.close(fig)


def get_groupers(results):
    return {
        "r": BinGrouper(np.concat([np.arange(0, 1550, 50)])),
        "w": BinGrouper(np.concat([np.arange(0, 8.25, .25)])),
        "logw": BinGrouper(np.arange(
            results.logw.min() - .01,
            results.logw.max() + .01,
            .1
        )),
        "snr": UniqueGrouper(results.snr.values),
        "tau": UniqueGrouper(results.tau.values),
        "lam": UniqueGrouper(results.lam.values),
    }


def get_results():
    results = xarray.open_zarr("results.zarr")
    results = results.assign_coords(spd=results.lam * 1000 / (results.tau * 60))
    results = results.assign_coords(r=xarray.ufuncs.hypot(results.px, results.py))
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
    fg = sns.relplot(
        data=df, x="snr", y=varname,
        # hue="lam",
        # height=10,
        kind="line",
        errorbar="pi",
        # err_style="bars",
        estimator="median",
    )
    if logscale:
        fg.ax.set_yscale("log")

    plt.savefig(f"plots/{varname}_line/{name}.png")


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
    sns.set_theme("paper", "whitegrid")
    
    make_dirs()
    
    dask.config.set({
        "distributed.worker.memory.spill": 0.85,  # default: 0.7
        "distributed.worker.memory.target": 0.75,  # default: 0.6
        "distributed.worker.memory.terminate": 0.98,  # default: 0.95
    })
    with Client(processes=True, n_workers=8, threads_per_worker=1) as client:
        print(client.dashboard_link)
        results = get_results().compute()
        print("groupers")
        groupers = get_groupers(results)

        r = results.sel(snr=slice(-6, 6)).median(["time", "px", "py"])
        print("df")
        r["speed_error"] = r["speed_error"] / r["spd"]
        r["period_error"] = r["period_error"] / r["tau"]
        r["wavelength_error"] = r["wavelength_error"] / r["lam"]
        df = r.to_dataframe()
        print("plot")
        sns.relplot(df, x="speed_error", y="period_error", hue="snr", height=10, zorder=1)
        sns.regplot(df, x="speed_error", y="period_error", scatter=False, truncate=True, color=".2")
        plt.savefig("test.png")
        return
    
        args = list(itertools.product(ERR_VARS, NORM_TYPES))
        for a in tqdm.tqdm(args, "line plots"):
            make_line_plots(r, *a)

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
