import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import xarray
import numpy as np
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
    with ProgressBar():
        data.to_zarr(data_fn, mode="w", encoding=encoding)


def get_template(data, nfft, block_size, step_size):
    temp = gnss_tid.parameter.estimate_parameters_block_unopt(data.isel(lam=0, tau=0, snr=0), nfft, block_size, step_size)
    temp = temp.expand_dims(lam=data.lam, tau=data.tau, snr=data.snr)
    return temp.chunk(px=-1, py=-1, lam=1, tau=1, time=-1, snr=1)


if __name__ == "__main__":
    dask.config.set(scheduler="threads", num_workers=6)
    
    data_fn = "/disk1/tid/users/starr/results/data3.zarr"
    output_fn = "results3.zarr"
    # save_data(data_fn)
    data = xarray.open_zarr(
        data_fn,
        chunks=dict(px=-1, py=-1, lam=1, tau=1, time=-1, snr=1)
    )
    print(data)
    print()
    print("#" * 80)
    print("PARAMETERS")
    print("#" * 80)
    print()
    
    BLOCK_SIZE = 32
    STEP_SIZE = 8
    NFFT = 128
    
    template = get_template(data, NFFT, BLOCK_SIZE, STEP_SIZE)

    params = data.map_blocks(
        gnss_tid.parameter.estimate_parameters_block_unopt,
        kwargs={
            "Nfft": NFFT,
            "block_size": BLOCK_SIZE,
            "step_size": STEP_SIZE,
        },
        template=template,
    )
    print(params)
    with ProgressBar():
        params.to_zarr(output_fn, mode="w")
