import logging
import traceback

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.signal.windows import kaiser

from gnss_tid.plotting import make_animation
from gnss_tid.pointdata import load_observations
from gnss_tid.spectral import run_spectral_focusing

logger = logging.getLogger(__name__)


def _build_pipeline_kwargs(obs, rx, cfg: DictConfig, lat_limits, lon_limits):
    """Build keyword arguments for :func:`~gnss_tid.spectral.run_spectral_focusing`.

    Args:
        obs: Observation DataFrame returned by load_observations.
        rx: Receiver lookup DataFrame returned by load_observations.
        cfg: Hydra DictConfig containing ``sample`` and ``focus`` sub-configs.
        lat_limits: Tuple of (min_lat, max_lat) drawn from the event config.
        lon_limits: Tuple of (min_lon, max_lon) drawn from the event config.

    Returns:
        Dictionary of keyword arguments accepted by
        :func:`~gnss_tid.spectral.run_spectral_focusing`.
    """
    heights = np.arange(cfg.focus.height_min, cfg.focus.height_max, cfg.focus.height_step)
    block_size = cfg.focus.block_size
    k = kaiser(block_size, cfg.focus.kaiser_beta)
    window = np.outer(k, k).reshape(1, 1, block_size, block_size)

    image_maker = hydra.utils.instantiate(cfg.focus.image_maker)
    center_finder = hydra.utils.instantiate(cfg.focus.center_finder)

    return {
        "obs": obs,
        "rx": rx,
        "window_size": cfg.sample.window,
        "step": cfg.sample.step,
        "image_maker": image_maker,
        "center_finder": center_finder,
        "heights": heights,
        "block_shape": (block_size, block_size),
        "block_step": cfg.focus.block_step,
        "window": window,
        "logscale_objective": cfg.focus.logscale_objective,
        "n_jobs": cfg.focus.n_jobs,
        "tec_name": cfg.focus.tec_name,
        "lat_limits": lat_limits,
        "lon_limits": lon_limits,
        "time_window": cfg.focus.time_window,
        "density_thresh": cfg.focus.density_thresh,
    }


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run the autofocus spectral pipeline.

    Loads GNSS observations from the event config, builds all pipeline
    components from the focus config, and calls
    :func:`~gnss_tid.spectral.run_spectral_focusing`.

    The ``time_window`` config key controls height smoothing:
    ``time_window=1`` (default) applies no smoothing (equivalent to the
    former ``block_spectral`` pipeline); ``time_window>1`` applies a rolling
    mean over the objective surface (equivalent to the former
    ``smoothed_patch`` pipeline).

    Args:
        cfg: Hydra DictConfig composed from ``config.yaml`` and its defaults.
    """
    try:
        lat_limits = tuple(cfg.event.latitude_limits)
        lon_limits = tuple(cfg.event.longitude_limits)

        logger.info("loading observations")
        obs, rx = load_observations(
            files=list(cfg.event.files),
            lat_limits=lat_limits,
            lon_limits=lon_limits,
            time_limits=list(cfg.event.time_limits),
            el_min=cfg.event.el_min,
            q_thresh=cfg.event.q_thresh,
            n_jobs=cfg.event.n_jobs,
        )
        logger.info("loaded %d observations from %d receivers", len(obs), len(rx))

        kwargs = _build_pipeline_kwargs(obs, rx, cfg, lat_limits, lon_limits)

        logger.info(
            "starting run_spectral_focusing (time_window=%d)", kwargs["time_window"]
        )
        result = run_spectral_focusing(**kwargs)

        logger.info("saving results to %s", cfg.output_fn)
        result.to_netcdf(cfg.output_fn)
        make_animation(result, "event.gif")
    except Exception:
        logger.error("An error occurred:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
