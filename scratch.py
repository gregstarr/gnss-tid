import functools
import shutil

import zarr
import dask
from tqdm.contrib.itertools import product
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
    data = gnss_tid.synthetic.constant_model2(
        center=xarray.DataArray([0, 0], dims=["ci"]),
        snr=da.linspace(-6, 6, 30, chunks=1),
        wavelength=da.linspace(150, 400, 6, chunks=1),
        tau=da.linspace(10, 50, 6, chunks=1),
        xlim=(-1500, 1500),
        ylim=(-1500, 1500),
        hres=20,
    )
    
    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {var: {"compressors": compressor} for var in data.data_vars}
    data.to_zarr(data_fn, mode="w", encoding=encoding, consolidated=True)


def initialize(estimator, data, output_fn):
    print("INITIALIZING")
    shutil.rmtree(output_fn, ignore_errors=True)
    params = estimator(data.isel(snr=0, lam=0, tau=0))
    params = params.expand_dims(lam=data.lam, tau=data.tau, snr=data.snr)
    params = params.chunk(px=-1, py=-1, lam=1, tau=1, time=-1, snr=1)
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

    estimator = functools.partial(
        gnss_tid.parameter.estimate_parameters_block_unopt,
        Nfft=NFFT,
        block_size=BLOCK_SIZE,
        step_size=STEP_SIZE,
        normalize="patch",
    )
    
    data_fn = "/disk1/tid/users/starr/results/data3.zarr"
    output_fn = "/disk1/tid/users/starr/results/results4.zarr"

    data = xarray.open_zarr(
        data_fn,
        chunks=dict(px=-1, py=-1, lam=1, tau=1, time=-1, snr=1)
    )
    # INITIALIZE
    initialize(estimator, data, output_fn)

    print("RUNNING TRIALS")
    for ii, jj, kk in product(range(data.sizes["snr"]), range(data.sizes["lam"]), range(data.sizes["tau"]), desc="TRIALS"):
        params = estimator(data.isel(snr=ii, lam=jj, tau=kk))
        params = params.expand_dims(snr=[data.snr[ii].item()], lam=[data.lam[jj].item()], tau=[data.tau[kk].item()])
        params = params.chunk(px=-1, py=-1, lam=1, tau=1, time=-1, snr=1)
        with TqdmCallback(position=1, leave=False):
            params.to_zarr(
                output_fn,
                region={
                    "snr": slice(ii, ii+1),
                    "lam": slice(jj, jj+1),
                    "tau": slice(kk, kk+1),
                    "px": slice(None),
                    "py": slice(None),
                    "time": slice(None),
                },
                consolidated=False,
            )
    zarr.consolidate_metadata(output_fn)


if __name__ == "__main__":
    main()
