"""Generate a time-stack of TEC images at a chosen IPP height.

Produces the same image-data structure as the autofocus pipeline (variables
``image``, ``density``, ``height`` along ``time``) without doing FFT focusing
or centre finding.  Heights can be supplied as a single scalar (km) via the
``height`` key, or per-timestep via ``heights_file`` (a netCDF with a
``height(time)`` variable, e.g. an autofocus output).
"""

import logging
import traceback
from pathlib import Path

import hydra
import numpy as np
import xarray
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from gnss_tid.coords import Local2D
from gnss_tid.image import generate_image_stack
from gnss_tid.pointdata import load_observations, make_time_windows

logger = logging.getLogger(__name__)


def _resolve_heights(
    cfg: DictConfig, time_windows
) -> tuple[float | np.ndarray, float]:
    """Return ``(heights, init_height)`` where ``heights`` is scalar or per-window.

    ``init_height`` is the representative height passed to
    :meth:`ImageMakerBase.initialize_from_bounds` for sizing the interpolation
    grid (median of per-timestep heights, or the scalar itself).
    """
    if cfg.heights_file:
        ds = xarray.open_dataset(cfg.heights_file)
        heights = ds["height"].values
        if heights.shape[0] != len(time_windows):
            raise ValueError(
                f"heights_file '{cfg.heights_file}' has {heights.shape[0]} "
                f"entries but {len(time_windows)} time windows were generated; "
                f"check that sample.window and sample.step match the source run"
            )
        if np.any(~np.isfinite(heights)):
            raise ValueError(
                f"heights_file '{cfg.heights_file}' contains non-finite values"
            )
        return heights, float(np.median(heights))
    return float(cfg.height), float(cfg.height)


def _write_pngs(images: xarray.Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ii in range(images.sizes["time"]):
        frame = images.isel(time=ii)
        fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
        frame.image.plot(ax=ax)
        ax.set_title(
            f"t={frame.time.values}  h={float(frame.height):.0f} km"
        )
        fig.savefig(out_dir / f"frame_{ii:04d}.png")
        plt.close(fig)


@hydra.main(config_path="conf", config_name="make_images", version_base=None)
def main(cfg: DictConfig):
    """Run the image-stack pipeline.

    Loads observations, builds time windows from ``sample.window``/
    ``sample.step``, resolves the IPP height(s) from config, initializes the
    image maker, and produces a per-time-window image stack saved to netCDF.
    Optionally also writes per-frame PNGs when ``write_pngs`` is true.
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

        image_maker = hydra.utils.instantiate(cfg.image_maker)

        time_windows = make_time_windows(
            obs["time"], cfg.sample.window, cfg.sample.step
        )
        logger.info("built %d time windows", len(time_windows))

        heights, init_height = _resolve_heights(cfg, time_windows)
        logger.info(
            "height source: %s (init_height=%.1f km)",
            "per-timestep" if isinstance(heights, np.ndarray) else f"constant {heights:.1f} km",
            init_height,
        )

        proj = Local2D.from_geodetic(
            float(np.mean(lat_limits)),
            float(np.mean(lon_limits)),
            init_height,
        )

        image_maker.initialize_from_bounds(
            proj=proj,
            lat_limits=lat_limits,
            lon_limits=lon_limits,
        )

        images = generate_image_stack(
            obs=obs,
            rx=rx,
            image_maker=image_maker,
            time_windows=time_windows,
            heights=heights,
            tec_name=cfg.tec_name,
            lat_limits=lat_limits,
            lon_limits=lon_limits,
            proj=proj,
            n_jobs=cfg.n_jobs,
        )

        logger.info("saving images to %s", cfg.output_fn)
        images.to_netcdf(cfg.output_fn)

        if cfg.write_pngs:
            out_dir = Path(cfg.plots_dir)
            logger.info("writing per-frame PNGs to %s", out_dir)
            _write_pngs(images, out_dir)
    except Exception:
        logger.error("An error occurred:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
