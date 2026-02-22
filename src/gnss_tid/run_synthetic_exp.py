import logging
import shutil
from datetime import datetime
from math import ceil
from time import perf_counter
from typing import TYPE_CHECKING

import hydra
import xarray
from zarr.codecs import BloscCodec, BloscCname, BloscShuffle

import gnss_tid.synthetic
import gnss_tid.parameter

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from dask.distributed import Client


def save_data(data_fn, wave_type, n_trials):
    logging.info("#" * 80)
    logging.info("DATA")
    logging.info("#" * 80)
    shutil.rmtree(data_fn, ignore_errors=True)
    model_options = dict(
        snr_lim=[-6, 6],
        lam_lim=[100, 500],
        tau_lim=[10, 60],
        n_trials=n_trials,
        xlim=(-1500, 1500),
        ylim=(-1500, 1500),
        hres=20,
        batch_size=10,
    )
    if wave_type == "spherical":
        data = gnss_tid.synthetic.spherical_model(center=(0, 0), **model_options)
    elif wave_type == "planar":
        data = gnss_tid.synthetic.planar_model(**model_options)
    else:
        raise Exception(wave_type)
    
    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {var: {"compressors": compressor} for var in data.data_vars}
    logging.info("saving data to zarr")
    data.to_zarr(data_fn, mode="w", encoding=encoding)


@hydra.main(config_path="conf", config_name="exp_config", version_base=None)
def main(cfg: DictConfig):
    try:
        client: Client = hydra.utils.instantiate(cfg.client)
        logging.info(client.dashboard_link)

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        data_fn = f"/disk1/tid/users/starr/results/{stamp}_data_{cfg.wave_type}.zarr"
        save_data(data_fn, cfg.wave_type, cfg.n_trials)
        data = xarray.open_zarr(data_fn)
        logging.info("DATA")
        logging.info(data)

        output_fn = f"/disk1/tid/users/starr/results/{stamp}_results_{cfg.wave_type}.zarr"
        logging.info("RUNNING")
        n = data.sizes["trial"]
        n_batches = ceil(n / cfg.batch_size)
        total_time = 0
        for b in range(n_batches):
            logging.info(f"running batch {b + 1} / {n_batches}")
            t0 = perf_counter()
            batch_trials = slice(b * cfg.batch_size, min((b + 1) * cfg.batch_size, n))
            params = gnss_tid.parameter.estimate_parameters_dask(
                data.isel(trial=batch_trials),
                Nfft=cfg.nfft,
                block_size=cfg.block_size,
                step_size=cfg.step_size,
                normalize=cfg.norm,
            )
            if b == 0:
                params.to_zarr(output_fn, mode="w")
            else:
                params.to_zarr(output_fn, append_dim="trial")
            
            batch_duration = perf_counter() - t0
            total_time += batch_duration
            logging.info(f"batch {b+1}: {batch_duration:.2f} s")
            logging.info(f"total_time {total_time/60:.2f} min")
            logging.info(f"average time per batch {total_time/(b+1):.2f} s")
            logging.info(f"average time per trial {total_time/batch_trials.stop:.2f} s")

        client.close()

        logging.info("FINISHED")
        logging.info("OUTPUT: %s", output_fn)

        if cfg.get("save", True):
            return
        
        logging.info("CLEANUP")
        shutil.rmtree(data_fn, ignore_errors=True)
        shutil.rmtree(output_fn, ignore_errors=True)
    except Exception as e:
        logging.info("CLEANUP")
        shutil.rmtree(data_fn, ignore_errors=True)
        shutil.rmtree(output_fn, ignore_errors=True)
        logging.exception("error occurred", exc_info=True, stack_info=True)
        raise e


if __name__ == "__main__":
    main()
