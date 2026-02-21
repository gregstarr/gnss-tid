import logging

import xarray
from tqdm.dask import TqdmCallback
import hydra
from dask.distributed import Client

import gnss_tid.parameter


@hydra.main(config_path="conf", config_name="param_config", version_base=None)
def main(cfg):
    client: Client = hydra.utils.instantiate(cfg.client)
    print(client.dashboard_link)
    
    data = xarray.open_dataset(
        "autofocus.h5",
        chunks={"time": -1, "x": cfg.block_size, "y": cfg.block_size},
    )
    logging.info(data)
    
    params = gnss_tid.parameter.estimate_parameters_dask(
        data,
        Nfft=cfg.nfft,
        block_size=cfg.block_size,
        step_size=cfg.step_size,
        normalize=cfg.norm,
    )
    params.to_zarr("params.zarr", mode="w")

    client.close()
    
    logging.info("SUCCESS")


if __name__ == "__main__":
    main()
