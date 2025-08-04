import itertools
import functools

import dask.delayed
import numpy as np
import dask
from dask.distributed import Client, LocalCluster
import xarray

import gnss_tid.synthetic
import gnss_tid.parameter


def initialize(estimator, snr_vals, wl_vals, tau_vals, output_fn):
    print("INITIALIZING")
    block = run_trial(estimator, snr_vals[0], wl_vals[0], tau_vals[0])
    allocate_space(block, snr_vals, wl_vals, tau_vals)
    block.chunk({"snr": 1, "wavelength": 1, "tau": 1}).to_zarr(
        output_fn,
        region={
            "snr": slice(0, 1),
            "wavelength": slice(0, 1),
            "tau": slice(0, 1),
            "time": slice(None),
            "p": slice(None),
        }
    )


def allocate_space(
        block: xarray.Dataset,
        snr_vals: np.ndarray,
        wavelength_vals: np.ndarray,
        tau_vals: np.ndarray,
        fn: str = "results.zarr"):
    coords = {
        "time": block.time,
        "p": block.p,
        "snr": snr_vals,
        "wavelength": wavelength_vals,
        "tau": tau_vals,
    }
    data_vars = {
        "wl_error": (
            ("snr", "wavelength", "tau", "time", "p"),
            np.empty(
                (
                    len(snr_vals),
                    len(wavelength_vals),
                    len(tau_vals),
                    len(block.time),
                    len(block.p)
                ),
                dtype="float32",
            ),
        ),
        "ps_error": (
            ("snr", "wavelength", "tau", "time", "p"),
            np.empty(
                (
                    len(snr_vals),
                    len(wavelength_vals),
                    len(tau_vals),
                    len(block.time),
                    len(block.p)
                ),
                dtype="float32",
            ),
        ),
        "phase_speed": (
            ("wavelength", "tau"),
            np.empty((len(wavelength_vals), len(tau_vals)), dtype="float32"),
        ),
        "r": (("p",), np.empty((len(block.p),), dtype="float32")),
    }
    allocation = xarray.Dataset(
        coords=coords, data_vars=data_vars, attrs=block.attrs
    ).chunk({"snr": 1, "wavelength": 1, "tau": 1})
    allocation.to_zarr(fn, mode="w", compute=True)


def run_trial(estimator: callable, snr: float, wl: float, tau: float):
    ps = wl * 1000 / (tau * 60)
    print(f"{snr = }, {wl = }, {ps = }, {tau = }")
    data = gnss_tid.synthetic.constant_model(
        center=xarray.DataArray([0, 0], dims=["ci"]),
        wavelength=wl,
        phase_speed=ps,
        xlim=(-1500, 1500),
        ylim=(-1500, 1500),
        snr=snr,
        hres=20,
    )
    data["image"] = data["image"].rolling(x=3, y=3, center=True, min_periods=1).median()
    params, *_ = estimator(data)

    errors = (
        xarray.Dataset({
            "wl_error": (params.wavelength - wl),
            "ps_error": (params.phase_speed - ps),
        })
        .drop_vars(["kx", "ky", "x", "y"], errors="ignore")
        .stack(p=("px", "py"))
        .assign_coords(r=lambda d: np.hypot(d.px, d.py))
        .reset_index("p", drop=True)
        .expand_dims(snr=[snr], wavelength=[wl], tau=[tau])
        .assign_coords(phase_speed=lambda d: d.wavelength * 1000 / (d.tau * 60))
    )
    return errors


def main():
    dask.config.set({
        # start spilling when a worker is using 60% of its limit
        "distributed.worker.memory.target": 0.5,
        # stop queuing new tasks when using 70%
        "distributed.worker.memory.spill": 0.6,
        # pause worker when using 80%
        "distributed.worker.memory.pause": 0.8,
        # kill worker at 95% to avoid total OOM
        "distributed.worker.memory.terminate": 0.95,
    })
    client = Client(
        processes=True, n_workers=1, threads_per_worker=4, memory_limit="116GiB"
    )
    print(client.dashboard_link)

    # estimator = functools.partial(
    #     gnss_tid.parameter.estimate_parameters_block,
    #     kaiser_beta=5,
    #     Nfft=256,
    #     space_power_threshold=0,
    #     time_power_threshold=0,
    # )
    estimator = functools.partial(
        gnss_tid.parameter.estimate_parameters_block_v2,
        Nfft=128,
        use_threshold=False,
    )
    
    # snr_vals = np.arange(12, -16, -1)
    # wl_vals = np.arange(150, 451, 50)
    # tau_vals = np.arange(10, 51, 10)
    snr_vals = np.arange(-3, 2, 1)
    wl_vals = np.arange(150, 201, 50)
    tau_vals = np.arange(40, 51, 10)
    output_fn = "results.zarr"

    # INITIALIZE
    initialize(estimator, snr_vals, wl_vals, tau_vals, output_fn)

    @dask.delayed
    def run_task(ii, jj, kk):
        block = run_trial(estimator, snr_vals[ii], wl_vals[jj], tau_vals[kk])
        block.chunk({"snr": 1, "wavelength": 1, "tau": 1}).to_zarr(
            output_fn,
            region={
                "snr": slice(ii, ii+1),
                "wavelength": slice(jj, jj+1),
                "tau": slice(kk, kk+1),
                "time": slice(None),
                "p": slice(None),
            },
            compute=False,
        )

    tasks = [
        run_task(ii, jj, kk) for ii, jj, kk in itertools.product(
            range(len(snr_vals)), range(len(wl_vals)), range(1, len(tau_vals))
        )
    ]
    print("RUNNING TASKS")
    dask.compute(*tasks)


if __name__ == "__main__":
    main()
