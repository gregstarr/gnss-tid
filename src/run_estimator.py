import hvplot
import hvplot.xarray
import numpy as np
import xarray
import dask
from dask.distributed import Client

import gnss_tid.parameter


if __name__ == "__main__":
    file = "outputs/2024-12-06/22-20-41/autofocus.h5"
    USE_DASK = True

    if USE_DASK:
        dask.config.set({
            "distributed.worker.memory.spill": 0.85,  # default: 0.7
            "distributed.worker.memory.target": 0.75,  # default: 0.6
            "distributed.worker.memory.terminate": 0.98,  # default: 0.95
        })
        client = Client(processes=True, n_workers=2, threads_per_worker=1)
        print(client.dashboard_link)
        data = xarray.open_dataset(file, chunks="auto")
    else:
        data = xarray.open_dataset(file)

    BLOCK_SIZE = 32
    STEP_SIZE = 8
    NFFT = 128

    plots = gnss_tid.parameter.estimate_parameters_block_debug(data.isel(time=slice(-20, None)), NFFT, BLOCK_SIZE, STEP_SIZE)
    hvplot.show(plots)
    