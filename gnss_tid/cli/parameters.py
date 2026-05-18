from __future__ import annotations

import logging

import hydra
import xarray as xr
from dask.distributed import Client
from omegaconf import DictConfig, OmegaConf

from gnss_tid.parameter import estimate_parameters_dask


def _plain(cfg) -> dict:
    return {} if cfg is None else OmegaConf.to_container(cfg, resolve=True)


def estimate_named_parameters(name: str, data: xr.Dataset, kwargs: dict) -> xr.Dataset:
    if name == "spectral_dask":
        return estimate_parameters_dask(data, **kwargs)
    raise ValueError(f"unknown parameter estimation algorithm: {name}")


@hydra.main(config_path="../conf", config_name="param_config", version_base=None)
def main(cfg: DictConfig):
    client: Client = hydra.utils.instantiate(cfg.client)
    logging.info(client.dashboard_link)
    try:
        params_cfg = _plain(cfg.parameters)
        name = params_cfg.pop("name", "spectral_dask")
        data = xr.open_dataset(
            cfg.get("input_fn", "autofocus.h5"),
            chunks={
                "time": cfg.get("time_chunk", -1),
                "x": params_cfg.get("block_size", 32),
                "y": params_cfg.get("block_size", 32),
            },
        )
        logging.info(data)
        params = estimate_named_parameters(name, data, params_cfg)
        params.to_zarr(cfg.get("output_fn", "params.zarr"), mode="w")
        logging.info("SUCCESS")
    finally:
        client.close()


if __name__ == "__main__":
    main()
