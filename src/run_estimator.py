import pathlib

import dask.config
import numpy as np
import xarray
import dask
from dask.distributed import Client
from tqdm.dask import TqdmCallback


import gnss_tid.parameter


def main():
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # dask.config.set(scheduler="threads", num_workers=2)
    client = Client(processes=False, threads_per_worker=16, memory_limit="20GiB")
    print(client.dashboard_link)
    
    BLOCK_SIZE = 32
    STEP_SIZE = 8
    NFFT = 128
    FOLDER = pathlib.Path("nofocus").absolute()
    print(FOLDER)
    
    data = xarray.open_dataset(
        FOLDER / "autofocus.h5",
        chunks={"time": -1, "x": STEP_SIZE, "y": STEP_SIZE},
    )
    print(data)
    
    for NORM in ["patch", "image"]:
        params = gnss_tid.parameter.estimate_parameters_block_unopt(
            data,
            Nfft=NFFT,
            block_size=BLOCK_SIZE,
            step_size=STEP_SIZE,
            normalize=NORM,
        ).chunk("auto")
        with TqdmCallback(desc=f"estimating parameters: {NORM}"):
            params.to_zarr(FOLDER / f"params_{NORM}.zarr", mode="w")


if __name__ == "__main__":
    main()
