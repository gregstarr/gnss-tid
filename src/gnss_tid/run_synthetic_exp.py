import logging
import functools
import shutil
from datetime import datetime

import zarr
import hydra
import dask
from dask.distributed import Client
import xarray
from numcodecs import Blosc

import gnss_tid.synthetic
import gnss_tid.parameter


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
    )
    if wave_type == "spherical":
        data = gnss_tid.synthetic.spherical_model(center=(0, 0), **model_options)
    elif wave_type == "planar":
        data = gnss_tid.synthetic.planar_model(**model_options)
    else:
        raise Exception(wave_type)
    
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    encoding = {var: {"compressor": compressor} for var in data.data_vars}
    logging.info("saving data to zarr")
    data.to_zarr(data_fn, mode="w", encoding=encoding, consolidated=True)


def initialize(estimator, data: xarray.Dataset, wave_type: str):
    logging.info("INITIALIZING")
    params: xarray.Dataset = estimator(data.isel(trial=0))
    params = params.expand_dims(trial=data.trial)
    params = params.assign_coords(
        snr=("trial", data.snr.values),
        tau=("trial", data.tau.values),
        lam=("trial", data.lam.values),
    )
    if wave_type == "planar":
        params = params.assign_coords(dir=("trial", data.dir.values))
    params = params.chunk(px=-1, py=-1, time=-1, trial=1)
    return params


@hydra.main(config_path="conf", config_name="exp_config", version_base=None)
def main(cfg):
    try:
        print("ZARR:", zarr.__version__)
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        dask.config.set({
            "distributed.worker.memory.target": 0.50,
            "distributed.worker.memory.spill": 0.60,
            "distributed.worker.memory.pause": 0.85,
            "distributed.worker.memory.terminate": 0.95,
        })
        
        client: Client = hydra.utils.instantiate(cfg.client)
        print(client.dashboard_link)

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        data_fn = f"/disk1/tid/users/starr/results/{stamp}_data_{cfg.wave_type}.zarr"
        save_data(data_fn, cfg.wave_type, cfg.n_trials)
        data = xarray.open_zarr(data_fn).chunk(trial=1)
        hres = float(data.x.isel(x=1).values - data.x.isel(x=0).values)
        print(hres)
        
        estimator = functools.partial(
            gnss_tid.parameter.estimate_parameters_block,
            hres=hres,
            Nfft=cfg.nfft,
            block_size=cfg.block_size,
            step_size=cfg.step_size,
            normalize=cfg.norm,
        )

        def func(block):
            params = estimator(block)
            params = params.assign_coords(
                snr=block.coords["snr"],
                lam=block.coords["lam"],
                tau=block.coords["tau"],
            )
            if cfg.wave_type == "planar":
                params = params.assign_coords(dir=block.coords["dir"])
            return params
            
        output_fn = f"/disk1/tid/users/starr/results/{stamp}_results_{cfg.wave_type}.zarr"
        template = initialize(estimator, data, cfg.wave_type)

        logging.info("RUNNING TRIALS")
        store = zarr.DirectoryStore(output_fn)
        params = data.map_blocks(func, template=template)
        print(params)
        params.to_zarr(store, mode="w")

        client.close()

        logging.info("FINISHED")
    except:
        logging.info("CLEANUP")
        shutil.rmtree(data_fn, ignore_errors=True)
        shutil.rmtree(output_fn, ignore_errors=True)



if __name__ == "__main__":
    main()
