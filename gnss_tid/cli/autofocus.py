from __future__ import annotations

import logging
import traceback

import hydra
from omegaconf import DictConfig, OmegaConf

from gnss_tid.autofocus import autofocus_smoothed_spectral, autofocus_spectral_blocks
from gnss_tid.plotting import make_animation


def _plain(cfg) -> dict:
    return {} if cfg is None else OmegaConf.to_container(cfg, resolve=True)


def build_autofocus_kwargs(cfg: DictConfig) -> dict:
    heights = range(cfg.heights.min, cfg.heights.max, cfg.heights.step)
    image = _plain(cfg.get("image"))
    center = _plain(cfg.get("center"))
    kwargs = dict(
        heights=heights,
        window=cfg.sample.window,
        step=cfg.sample.step,
        tec_name=cfg.get("tec_name", "dtec1"),
        image_method=image.pop("name", "rbf"),
        image_kwargs=image,
        center_algorithm=center.pop("name", "stationary"),
        center_kwargs=center,
        hres=cfg.get("hres", 20.0),
        block_size=cfg.get("block_size", 32),
        block_step=cfg.get("block_step", 16),
        kaiser_beta=cfg.get("kaiser_beta", 1.0),
        logscale_objective=cfg.get("logscale_objective", False),
        n_jobs=cfg.get("n_jobs", 1),
        diagnostics_dir=cfg.get("diagnostics_dir"),
    )
    if "density_thresh" in cfg:
        kwargs["density_thresh"] = cfg.density_thresh
    if "time_window" in cfg:
        kwargs["time_window"] = cfg.time_window
    return kwargs


def run_named_autofocus(name: str, points, kwargs: dict):
    if name == "smoothed_spectral":
        return autofocus_smoothed_spectral(points, **kwargs)
    if name == "spectral_blocks":
        kwargs.pop("density_thresh", None)
        kwargs.pop("time_window", None)
        return autofocus_spectral_blocks(points, **kwargs)
    raise ValueError(f"unknown autofocus algorithm: {name}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        points = hydra.utils.instantiate(cfg.event.pointdata)
        kwargs = build_autofocus_kwargs(cfg.focus)
        focus_data = run_named_autofocus(cfg.focus.name, points, kwargs)
        focus_data.to_netcdf(cfg.output_fn)
        if cfg.get("animation_fn"):
            make_animation(focus_data, cfg.animation_fn)
    except Exception as e:
        logging.error("An error occurred:\n%s", traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
