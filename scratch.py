import functools
import shutil

import zarr
import dask
from tqdm.std import trange
from tqdm.dask import TqdmCallback
import dask.array as da
import numpy as np
import xarray
from zarr.codecs import BloscCodec, BloscCname, BloscShuffle

import gnss_tid.synthetic
import gnss_tid.parameter


def save_data(data_fn):
    print("#" * 80)
    print("DATA")
    print("#" * 80)
    print()
    shutil.rmtree(data_fn, ignore_errors=True)
    data = gnss_tid.synthetic.constant_model3(
        snr_interval=[-6, 6],
        wavelength_interval=[100, 700],
        tau_interval=[10, 80],
        n_trials=1500,
        xlim=(-1500, 1500),
        ylim=(-1500, 1500),
        hres=20,
    )
    
    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {var: {"compressors": compressor} for var in data.data_vars}
    with TqdmCallback():
        data.to_zarr(data_fn, mode="w", encoding=encoding, consolidated=True)


def initialize(estimator, data, output_fn):
    print("INITIALIZING")
    shutil.rmtree(output_fn, ignore_errors=True)
    params = estimator(data.isel(trial=0))
    params = params.expand_dims(trial=data.trial)
    params = params.assign_coords(snr=data.snr, tau=data.tau, lam=data.lam)
    params = params.chunk(px=-1, py=-1, time=-1, trial=1)
    print(params["S0"])
    print(params["anisotropy_mag"])
    with TqdmCallback():
        params.to_zarr(output_fn, mode="w", consolidated=False)


def main():
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    dask.config.set(scheduler="threads", num_workers=24)

    BLOCK_SIZE = 32
    STEP_SIZE = 8
    NFFT = 128

    data_fn = "/disk1/tid/users/starr/results/data5.zarr"
    save_data(data_fn)
    data = xarray.open_zarr(
        data_fn,
        chunks=dict(px=-1, py=-1, time=-1, trial=1)
    )
    
    for NORM in ["patch", "image"]:
        estimator = functools.partial(
            gnss_tid.parameter.estimate_parameters_block_unopt,
            Nfft=NFFT,
            block_size=BLOCK_SIZE,
            step_size=STEP_SIZE,
            normalize=NORM,
        )
        output_fn = f"/disk1/tid/users/starr/results/results5_{NORM}.zarr"
        initialize(estimator, data, output_fn)

        print("RUNNING TRIALS")
        for ii in trange(data.sizes["trial"], desc=f"{NORM} TRIALS"):
            params = estimator(data.isel(trial=ii))
            params = params.expand_dims(trial=[data.trial[ii].item()])
            params = params.assign_coords(
                snr=("trial", [data.snr[ii].values.item()]),
                lam=("trial", [data.lam[ii].values.item()]),
                tau=("trial", [data.tau[ii].values.item()]),
            )
            params = params.chunk(px=-1, py=-1, time=-1, trial=1)
            with TqdmCallback(position=1, leave=False):
                params.to_zarr(
                    output_fn,
                    region={
                        "trial": slice(ii, ii+1),
                        "px": slice(None),
                        "py": slice(None),
                        "time": slice(None),
                    },
                    consolidated=False,
                )
        zarr.consolidate_metadata(output_fn)


if __name__ == "__main__":
    main()
