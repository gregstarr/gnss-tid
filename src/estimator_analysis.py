import itertools

import dask
from dask.delayed import delayed
from dask.distributed import Client, progress
import xarray
from xarray.groupers import BinGrouper, UniqueGrouper
import matplotlib.pyplot as plt


def plot_error(error_val, groupers, fn, **plot_kwargs):
    comp = error_val.groupby(groupers).mean(...)
    comp.plot(**plot_kwargs)
    plt.savefig(fn)
    plt.close()
    # print("DONE:", fn)


def basic_plots(results):
    BINS = 21
    pbar = itertools.product(ERR_VARS, DIST_VARS, ERROR_TYPES, NORM_TYPES)

    @delayed
    def run_task(varname, group_var, err_type, norm_type):
        name = f"{varname}_{norm_type}_{err_type}__by__{group_var}"
        data = results[varname]
        if norm_type == "rel":
            data = data / results[ERR_VARS[varname]]
        if err_type == "mae":
            data = abs(data)
        
        groups = {
            "snr": UniqueGrouper(),
            group_var: BinGrouper(BINS),
        }
        fn = f"plots/basic/{name}.png"
        center = 0 if err_type == "bias" else False
        plot_error(data, groups, fn, robust=True, center=center)
    
    return [run_task(*args) for args in pbar]


def log_rel_plots(results):
    BINS = 21
    pbar = itertools.product(ERR_VARS, DIST_VARS)

    @delayed
    def run_task(varname, group_var):
        name = f"{varname}_log_rel_mae__by__{group_var}"
        data = results[varname]
        data = data / results[ERR_VARS[varname]]
        data = xarray.ufuncs.log10(abs(data) + 1)
        
        groups = {
            "snr": UniqueGrouper(),
            group_var: BinGrouper(BINS),
        }
        fn = f"plots/log_rel/{name}.png"
        plot_error(data, groups, fn, robust=True, center=False)
    
    return [run_task(*args) for args in pbar]


def facet_plots(results):
    BINS = 21

    facet_vars = ["lam", "tau"]
    pbar = itertools.product(ERR_VARS, DIST_VARS, ERROR_TYPES, NORM_TYPES, facet_vars)
    
    @delayed
    def run_task(varname, group_var, err_type, norm_type, facet):
        name = f"{varname}_{norm_type}_{err_type}__by__{group_var}_and_{facet}"
        data = results[varname]
        if norm_type == "rel":
            data = data / results[ERR_VARS[varname]]
        if err_type == "mae":
            data = abs(data)
        
        groups = {
            "snr": UniqueGrouper(),
            group_var: BinGrouper(BINS),
            facet: UniqueGrouper(),
        }
        fn = f"plots/facet/{name}.png"
        center = 0 if err_type == "bias" else False
        plot_error(
            data,
            groups,
            fn,
            robust=True,
            center=center,
            col=facet,
            col_wrap=3,
        )
    
    return [run_task(*args) for args in pbar]


def log_rel_facet_plots(results):
    BINS = 21

    facet_vars = ["lam", "tau"]
    pbar = itertools.product(ERR_VARS, DIST_VARS, facet_vars)
    
    @delayed
    def run_task(varname, group_var, facet):
        name = f"{varname}_log_rel_mae__by__{group_var}_and_{facet}"
        data = results[varname]
        data = data / results[ERR_VARS[varname]]
        data = xarray.ufuncs.log10(abs(data) + 1)
        
        groups = {
            "snr": UniqueGrouper(),
            group_var: BinGrouper(BINS),
            facet: UniqueGrouper(),
        }
        fn = f"plots/log_rel_facet/{name}.png"
        plot_error(
            data,
            groups,
            fn,
            robust=True,
            center=False,
            col=facet,
            col_wrap=3,
        )
    
    return [run_task(*args) for args in pbar]


ERR_VARS = {"period_error": "tau", "wavelength_error": "lam", "speed_error": "spd"}
DIST_VARS = ["r", "w"]
ERROR_TYPES = ["bias", "mae"]
NORM_TYPES = ["abs", "rel"]

def main():
    dask.config.set({
        "distributed.worker.memory.spill": 0.85,  # default: 0.7
        "distributed.worker.memory.target": 0.75,  # default: 0.6
        "distributed.worker.memory.terminate": 0.98,  # default: 0.95
    })
    with Client(processes=True, n_workers=8, threads_per_worker=2) as client:
        results = xarray.open_zarr("results.zarr", chunks="auto")
        phase_speed = results.lam * 1000 / (results.tau * 60)
        results = results.assign_coords(r=xarray.ufuncs.hypot(results.px, results.py))
        results = results.assign_coords(w=results.r / results.lam)
        results = results.assign(
            period_error=results.period-results.tau,
            wavelength_error=results.wavelength-results.lam,
            speed_error=results.phase_speed-phase_speed,
            spd=phase_speed,
        )
        results = results.chunk("auto").persist()
        print(results)

        tasks = (
            facet_plots(results) +
            basic_plots(results) + 
            log_rel_plots(results) +
            log_rel_facet_plots(results)
        )
        futures = client.compute(tasks, sync=False)
        print("progress:")
        progress(futures, notebook=False)
    print("CLUSTER CLOSED")


if __name__ == "__main__":
    main()
