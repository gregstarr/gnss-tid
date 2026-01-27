import logging
import functools
import shutil
from datetime import datetime

import hydra
from dask.distributed import Client
import zarr
from tqdm.std import trange
from tqdm.dask import TqdmCallback
import xarray
from zarr.codecs import BloscCodec, BloscCname, BloscShuffle

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
    
    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {var: {"compressors": compressor} for var in data.data_vars}
    logging.info("saving data to zarr")
    data.to_zarr(data_fn, mode="w", encoding=encoding, consolidated=True)


def initialize(estimator, data: xarray.Dataset, output_fn: str, wave_type: str):
    logging.info("INITIALIZING")
    shutil.rmtree(output_fn, ignore_errors=True)
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
    logging.info("initialize zarr")
    params.to_zarr(output_fn, mode="w", consolidated=False)


@hydra.main(config_path="conf", config_name="exp_config", version_base=None)
def main(cfg):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    client: Client = hydra.utils.instantiate(cfg.client)
    print(client.dashboard_link)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    data_fn = f"/disk1/tid/users/starr/results/{stamp}_data_{cfg.wave_type}.zarr"
    save_data(data_fn, cfg.wave_type, cfg.n_trials)
    data = xarray.open_zarr(data_fn)
    
    estimator = functools.partial(
        gnss_tid.parameter.estimate_parameters_block_unopt,
        Nfft=cfg.nfft,
        block_size=cfg.block_size,
        step_size=cfg.step_size,
        normalize=cfg.norm,
    )
    output_fn = f"/disk1/tid/users/starr/results/{stamp}_results_{cfg.wave_type}.zarr"
    initialize(estimator, data, output_fn, cfg.wave_type)

    logging.info("RUNNING TRIALS")
    for ii in trange(data.sizes["trial"], desc="TRIALS"):
        params = estimator(data.isel(trial=ii))
        params = params.expand_dims(trial=[data.trial[ii].item()])
        params = params.assign_coords(
            snr=("trial", [data.snr[ii].values.item()]),
            lam=("trial", [data.lam[ii].values.item()]),
            tau=("trial", [data.tau[ii].values.item()]),
        )
        if cfg.wave_type == "planar":
            params = params.assign_coords(
                dir=("trial", [data.dir[ii].values.item()]),
            )
        params = params.chunk(px=-1, py=-1, time=-1, trial=1)
        with TqdmCallback(position=1, leave=False):
            params.to_zarr(
                output_fn,
                region={
                    "trial": slice(ii, ii+1),
                    "px": slice(None),
                    "py": slice(None),
                    "time": slice(None),
                },
                consolidated=False,
            )
    zarr.consolidate_metadata(output_fn)

    client.close()


if __name__ == "__main__":
    main()
