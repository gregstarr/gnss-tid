import dask
from dask.distributed import Client
import dask.array as da
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
        snr=da.arange(12, -16, -1).persist(),
        wavelength=da.arange(150, 451, 50).persist(),
        tau=da.arange(10, 51, 10).persist(),
        xlim=(-1500, 1500),
        ylim=(-1500, 1500),
        hres=20,
    )
    
    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {var: {"compressors": compressor} for var in data.data_vars}
    data.to_zarr(data_fn, mode="w", encoding=encoding)


def get_template(data, block_size, step_size):
    edges = block_size // (2 * step_size)
    patches = (
        data.image.isel(time=[0], tau=[0], lam=[0], snr=[0])
        .rolling(y=block_size, x=block_size, center=True)
        .construct(x="kx", y="ky", stride=step_size)
        .isel(x=slice(edges, -edges), y=slice(edges, -edges))
        .rename({"x": "px", "y": "py"})
    )
    s = (
        patches.sizes["px"],
        patches.sizes["py"],
        data.sizes["lam"],
        data.sizes["tau"],
        data.sizes["time"],
        data.sizes["snr"],
    )
    coords={
        "px": patches.px,
        "py": patches.py,
        "time": data.time,
        "lam": data.lam,
        "tau": data.tau,
        "snr": data.snr,
    }
    temp_arr = xarray.DataArray(
        da.zeros(s),
        coords=coords,
        dims=("px", "py", "lam", "tau", "time", "snr"),
    )
    var_names = ["phase_speed", "wavelength", "period", "vx", "vy", "R", "coherence"]
    temp = xarray.Dataset({name: temp_arr.rename(name) for name in var_names})
    return temp.chunk(px=-1, py=-1, lam=1, tau=1, time=-1, snr=1)


if __name__ == "__main__":
    dask.config.set({
        "distributed.worker.memory.spill": 0.85,  # default: 0.7
        "distributed.worker.memory.target": 0.75,  # default: 0.6
        "distributed.worker.memory.terminate": 0.98,  # default: 0.95
    })
    client = Client(processes=True, n_workers=4, threads_per_worker=1)
    print(client.dashboard_link)
    
    data_fn = "/disk1/tid/users/starr/results/data.zarr"
    output_fn = "results.zarr"
    save_data(data_fn)
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
    template = get_template(data, BLOCK_SIZE, STEP_SIZE)

    params = data.map_blocks(
        gnss_tid.parameter.estimate_parameters_block_v4,
        kwargs={
            "Nfft": NFFT,
            "block_size": BLOCK_SIZE,
            "step_size": STEP_SIZE,
        },
        template=template,
    )
    print(params)
    params.to_zarr(output_fn, mode="w")
