import logging
import traceback

import hydra
import numpy as np
from omegaconf import DictConfig

from gnss_tid.coords import Local2D
from gnss_tid.fft import make_kaiser_2d
from gnss_tid.plotting import make_animation
from gnss_tid.pointdata import load_observations
from gnss_tid.spectral import SpectralConfig, run_spectral_focusing

logger = logging.getLogger(__name__)


def _build_pipeline_inputs(obs, rx, cfg: DictConfig, lat_limits, lon_limits):
    """Build inputs for :func:`~gnss_tid.spectral.run_spectral_focusing`.

    Args:
        obs: Observation DataFrame returned by load_observations.
        rx: Receiver lookup DataFrame returned by load_observations.
        cfg: Hydra DictConfig containing ``sample`` and ``focus`` sub-configs.
        lat_limits: Tuple of (min_lat, max_lat) drawn from the event config.
        lon_limits: Tuple of (min_lon, max_lon) drawn from the event config.

    Returns:
        A 2-tuple ``(spectral_cfg, orchestration_kwargs)``:

        - ``spectral_cfg``: :class:`~gnss_tid.spectral.SpectralConfig` bundling
          the per-worker static inputs.
        - ``orchestration_kwargs``: dict of the remaining keyword arguments
          accepted by :func:`~gnss_tid.spectral.run_spectral_focusing`.
    """
    heights = np.arange(cfg.focus.height_min, cfg.focus.height_max, cfg.focus.height_step)
    block_size = cfg.focus.block_size
    window = make_kaiser_2d(block_size, cfg.focus.kaiser_beta).reshape(
        1, 1, block_size, block_size
    )

    image_maker = hydra.utils.instantiate(cfg.focus.image_maker)
    center_finder = hydra.utils.instantiate(cfg.focus.center_finder)

    proj = Local2D.from_geodetic(
        float(np.mean(lat_limits)),
        float(np.mean(lon_limits)),
        float(np.median(heights)),
    )

    spectral_cfg = SpectralConfig(
        obs=obs,
        rx=rx,
        image_maker=image_maker,
        tec_name=cfg.focus.tec_name,
        block_size=block_size,
        block_step=cfg.focus.block_step,
        window=window,
        logscale_objective=cfg.focus.logscale_objective,
        lat_limits=lat_limits,
        lon_limits=lon_limits,
        proj=proj,
    )
    orchestration_kwargs = {
        "window_size": cfg.sample.window,
        "step": cfg.sample.step,
        "center_finder": center_finder,
        "heights": heights,
        "n_jobs": cfg.focus.n_jobs,
        "time_window": cfg.focus.time_window,
        "density_thresh": cfg.focus.density_thresh,
    }
    return spectral_cfg, orchestration_kwargs


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

        spectral_cfg, orchestration_kwargs = _build_pipeline_inputs(
            obs, rx, cfg, lat_limits, lon_limits
        )

        logger.info(
            "starting run_spectral_focusing (time_window=%d)",
            orchestration_kwargs["time_window"],
        )
        result = run_spectral_focusing(spectral_cfg, **orchestration_kwargs)

        logger.info("saving results to %s", cfg.output_fn)
        result.to_netcdf(cfg.output_fn)
        make_animation(result, "event.gif")
    except Exception:
        logger.error("An error occurred:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
