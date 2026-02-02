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
        chunks={"time": -1, "x": cfg.step_size, "y": cfg.step_size},
    )
    logging.info(data)
    
    params = gnss_tid.parameter.estimate_parameters_block_unopt(
        data,
        Nfft=cfg.nfft,
        block_size=cfg.block_size,
        step_size=cfg.step_size,
        normalize=cfg.norm,
    ).chunk("auto")
    with TqdmCallback(desc="estimating parameters"):
        params.to_zarr("params.zarr", mode="w")
    
    client.close()


if __name__ == "__main__":
    main()
