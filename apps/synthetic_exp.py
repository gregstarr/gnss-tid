from __future__ import annotations


import logging
import shutil
import uuid
from datetime import datetime
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
import xarray
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle

import gnss_tid.parameter
import gnss_tid.simulate

if TYPE_CHECKING:
    from dask.distributed import Client
    from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


def _resolve_scenario(cfg: DictConfig, reference_time: np.datetime64):
    """Resolve the analytical scenario from config.

    Builds the scenario from explicit ``scenario.sources`` when supplied;
    otherwise falls back to the legacy flat ``wave_type`` grid preset
    (``planar`` or ``spherical``) with the historical parameter ranges.
    Per-trial [min, max] ranges are sampled from the parameter seed stream.

    Args:
        cfg: Hydra config with ``scenario`` (or legacy ``wave_type``),
            ``n_trials``, and ``seed``.
        reference_time: Scenario epoch (UTC, timezone-naive).

    Returns:
        Tuple ``(scenario, ranges)`` where ``scenario`` is the resolved
        :class:`gnss_tid.simulate.Scenario` and ``ranges`` maps
        ``"{source}.{field}"`` keys to [min, max] pairs.

    Raises:
        ValueError: If neither explicit sources nor a known legacy
            ``wave_type`` is configured.
    """
    n_trials = int(cfg.n_trials)
    sources_cfg = cfg.get("scenario", {}).get("sources") if cfg.get("scenario") else None
    if sources_cfg:
        sources = []
        for src in sources_cfg:
            wave_cfg = src.wave
            wave_type = str(wave_cfg.type)
            if wave_type == "spherical":
                wave_type = "circular"
            common = {
                "wavelength_km": _as_float(wave_cfg.wavelength_km)
                if not _is_range(wave_cfg.wavelength_km)
                else 1.0,
                "period_s": _as_float(wave_cfg.period_min) * 60
                if not _is_range(wave_cfg.period_min)
                else 1.0,
                "phase_rad": _as_float(wave_cfg.get("phase_rad", 0.0)),
                "center_km": (
                    _as_float(wave_cfg.get("center_x_km", 0.0)),
                    _as_float(wave_cfg.get("center_y_km", 0.0)),
                ),
            }
            if wave_type == "planar":
                direction = wave_cfg.get("direction_deg", 0.0)
                common["direction_rad"] = (
                    float(direction) * np.pi / 180 if not _is_range(direction) else 0.0
                )
                wave = gnss_tid.simulate.PlanarWave(**common)
            elif wave_type == "circular":
                wave = gnss_tid.simulate.CircularWave(**common)
            else:
                raise ValueError(f"unknown wave type: {wave_type}")

            envelope = None
            if src.get("envelope") and str(src.envelope.type) == "gaussian":
                sigma_v = src.envelope.get("sigma_v_km")
                spreading = src.envelope.get("spreading_time_s")
                envelope = gnss_tid.simulate.GaussianEnvelope(
                    sigma_u_km=float(src.envelope.sigma_u_km)
                    if not _is_range(src.envelope.sigma_u_km)
                    else 1.0,
                    sigma_v_km=float(sigma_v) if sigma_v is not None else None,
                    speed_ratio=float(src.envelope.get("speed_ratio", 0.9)),
                    center_u_km=float(src.envelope.get("center_u_km", 0.0)),
                    spreading_time_s=float(spreading) if spreading is not None else None,
                )
            amplitude = 1.0
            if wave_cfg.get("amplitude_tecu") is not None:
                amplitude = (
                    _as_float(wave_cfg.amplitude_tecu)
                    if not _is_range(wave_cfg.amplitude_tecu)
                    else 1.0
                )
            sources.append(gnss_tid.simulate.Source(wave, envelope, amplitude))

        ranges = _collect_ranges(cfg, len(sources))
        scenario = gnss_tid.simulate.Scenario(tuple(sources), reference_time)
        scenario = gnss_tid.simulate.sample_parameters(
            scenario, n_trials, int(cfg.seed), ranges
        )
        return scenario, ranges

    wave_type = str(cfg.wave_type)
    if wave_type == "spherical":
        wave = gnss_tid.simulate.CircularWave(300.0, 1800.0, center_km=(0.0, 0.0))
    elif wave_type == "planar":
        wave = gnss_tid.simulate.PlanarWave(300.0, 1800.0)
    else:
        raise ValueError(f"unknown legacy wave_type: {wave_type}")
    scenario = gnss_tid.simulate.Scenario(
        (gnss_tid.simulate.Source(wave, None, 1.0),), reference_time
    )
    return scenario, {}


def _is_range(value) -> bool:
    """Return True when a config value is a two-element [min, max] list.

    Args:
        value: Config value (scalar, list, tuple, or ListConfig).

    Returns:
        True when the value is a list-like container of exactly two
        elements (interpreted as a range only for scenario parameter
        fields, not coordinate limits or two-component vectors).
    """
    if isinstance(value, (list, tuple)):
        return len(value) == 2
    if hasattr(value, "__len__") and not isinstance(value, str):
        try:
            return len(value) == 2
        except TypeError:
            return False
    return False


def _as_float(value) -> float:
    """Convert a config scalar value to float.

    Args:
        value: Scalar config value (int, float, str, or ListConfig scalar).

    Returns:
        Float value.

    Raises:
        TypeError: If the value is not a scalar.
    """
    if hasattr(value, "__len__") and not isinstance(value, str):
        raise TypeError(f"expected a scalar value, got list-like: {value}")
    return float(value)


def _collect_ranges(cfg: DictConfig, n_sources: int) -> dict:
    """Collect per-trial [min, max] ranges from resolved scenario config.

    Args:
        cfg: Hydra config with ``scenario.sources``.
        n_sources: Number of configured sources.

    Returns:
        Mapping of ``"{source_index}.{field}"`` keys to [min, max] pairs.

    Raises:
        ValueError: If a range references an unknown source.
    """
    ranges = {}
    for i, src in enumerate(cfg.scenario.sources):
        wave_cfg = src.wave
        for field in ("wavelength_km", "period_min", "direction_deg", "amplitude_tecu"):
            value = wave_cfg.get(field)
            if value is not None and _is_range(value):
                lo, hi = float(value[0]), float(value[1])
                if field == "period_min":
                    ranges[f"{i}.period_s"] = [lo * 60, hi * 60]
                elif field == "direction_deg":
                    ranges[f"{i}.direction_rad"] = [
                        lo * np.pi / 180,
                        hi * np.pi / 180,
                    ]
                else:
                    ranges[f"{i}.{field}"] = [lo, hi]
        env_cfg = src.get("envelope")
        if env_cfg:
            for field in ("sigma_u_km", "sigma_v_km"):
                value = env_cfg.get(field)
                if value is not None and _is_range(value):
                    ranges[f"{i}.{field}"] = [float(value[0]), float(value[1])]
    for key in list(ranges):
        idx = int(key.split(".", 1)[0])
        if not 0 <= idx < n_sources:
            raise ValueError(f"range references unknown source {idx}")
    return ranges


def _resolve_sampler(cfg: DictConfig):
    """Resolve the sampler from config.

    Args:
        cfg: Hydra config with a ``sampler`` group (``grid``, ``real_gnss``,
            or ``orbit_gnss`` type).

    Returns:
        A :class:`gnss_tid.simulate.GridSampler`,
        :class:`gnss_tid.simulate.RealGNSSSampler`, or
        :class:`gnss_tid.simulate.OrbitGNSSSampler` configured from the
        sampler group.

    Raises:
        ValueError: If the sampler type is unknown.
    """
    sampler_type = str(cfg.sampler.type)
    if sampler_type == "grid":
        return gnss_tid.simulate.GridSampler(
            start_time=cfg.sampler.start_time,
            end_time=cfg.sampler.end_time,
            cadence_s=float(cfg.sampler.cadence_s),
            xlim=tuple(cfg.sampler.xlim),
            ylim=tuple(cfg.sampler.ylim),
            hres=float(cfg.sampler.hres),
        )
    if sampler_type == "real_gnss":
        from gnss_tid.coords import Local2D
        from gnss_tid.pointdata import load_observations

        obs, rx = load_observations(
            files=list(cfg.sampler.files),
            lat_limits=list(cfg.sampler.latitude_limits),
            lon_limits=list(cfg.sampler.longitude_limits),
            time_limits=list(cfg.sampler.time_limits),
            el_min=float(cfg.sampler.get("el_min", 0.0)),
            q_thresh=float(cfg.sampler.get("q_thresh", 0.99)),
            n_jobs=int(cfg.sampler.get("n_jobs", 1)),
            pbar=False,
        )
        proj = Local2D.from_geodetic(
            float(cfg.sampler.projection.lat),
            float(cfg.sampler.projection.lon),
            float(cfg.sampler.projection.height_km),
        )
        return gnss_tid.simulate.RealGNSSSampler(
            obs=obs,
            rx=rx,
            proj=proj,
            true_height_km=float(cfg.sampler.true_height_km),
            start_time=cfg.sampler.start_time,
            end_time=cfg.sampler.end_time,
            cadence_s=float(cfg.sampler.cadence_s),
            el_min=float(cfg.sampler.get("el_min", 0.0)),
            availability=str(cfg.sampler.get("availability", "geometry")),
            tec_column=cfg.sampler.get("tec_column"),
            timestamp_tolerance_s=float(cfg.sampler.get("timestamp_tolerance_s", 0.0)),
        )
    if sampler_type == "orbit_gnss":
        receivers = cfg.sampler.get("receivers")
        satellites = cfg.sampler.get("satellites")
        return gnss_tid.simulate.OrbitGNSSSampler(
            n_rx=int(cfg.sampler.get("n_rx", 20)),
            n_sat=int(cfg.sampler.get("n_sat", 8)),
            lat_range=tuple(cfg.sampler.get("lat_range", (35.0, 45.0))),
            lon_range=tuple(cfg.sampler.get("lon_range", (-110.0, -100.0))),
            el_min=float(cfg.sampler.get("el_min", 15.0)),
            seed=int(cfg.sampler.get("seed", 0)),
            receivers=np.asarray(receivers, dtype=float) if receivers else None,
            satellites=np.asarray(satellites, dtype=float) if satellites else None,
            start_time=cfg.sampler.get("start_time"),
            end_time=cfg.sampler.get("end_time"),
            cadence_s=float(cfg.sampler.get("cadence_s", 60.0)),
        )
    raise ValueError(f"unknown sampler type: {sampler_type}")


def _reserve_output_dir(root: Path, experiment_id: str) -> Path:
    """Reserve a unique experiment output directory exclusively.

    Args:
        root: Configurable output root directory.
        experiment_id: Collision-resistant experiment ID.

    Returns:
        The reserved experiment directory path (created).

    Raises:
        FileExistsError: If the destination already exists, including
            concurrent same-time launches.
    """
    exp_dir = root / experiment_id
    if exp_dir.exists():
        raise FileExistsError(
            f"experiment directory {exp_dir} already exists; reserve a new "
            "unique experiment_id"
        )
    exp_dir.mkdir(parents=True)
    return exp_dir


def _validate_detector_view(data: xarray.Dataset, cfg: DictConfig) -> None:
    """Preflight-validate the grid dataset for the estimator.

    Args:
        data: Dataset with ``image(trial, time, y, x)`` and ``density``.
        cfg: Hydra config with detector settings (``block_size``,
            ``step_size``, ``nfft``, ``smooth_win``, ``q_threshold``,
            ``density_threshold``).

    Raises:
        ValueError: If the time axis is not regularly spaced/increasing,
            spatial axes are not uniform, spatial axes are smaller than
            ``block_size``, ``smooth_win`` is even, or ``nfft``/``step_size``
            settings are incompatible.
    """
    time = data["time"].values
    if time.size < 2:
        raise ValueError("image time axis needs at least two epochs")
    dt = np.diff(time).astype("timedelta64[s]").astype(float)
    if np.any(dt <= 0):
        raise ValueError("image time axis must be strictly increasing")
    if not np.allclose(dt, dt[0]):
        raise ValueError("image time axis must be regularly spaced for the estimator")

    for axis in ("x", "y"):
        values = data[axis].values
        if values.size < int(cfg.block_size):
            raise ValueError(
                f"{axis} axis has {values.size} pixels; needs >= block_size "
                f"{cfg.block_size}"
            )
        spacing = np.diff(values)
        if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0]):
            raise ValueError(f"{axis} axis must be uniformly increasing in km")

    smooth_win = int(cfg.get("smooth_win", 9))
    if smooth_win % 2 == 0:
        raise ValueError(
            "smooth_win must be odd; the smoothing helper returns one fewer "
            "time value for even widths"
        )
    if (
        float(cfg.get("q_threshold", 0.95)) <= 0
        or float(cfg.get("q_threshold", 0.95)) >= 1
    ):
        raise ValueError("q_threshold must be in (0, 1)")
    if int(cfg.get("density_threshold", 5)) <= 0:
        raise ValueError("density_threshold must be positive")


def sampler_schedule(cfg: DictConfig, sampler_type: str, samples) -> np.ndarray | None:
    """Return the master requested raw-epoch schedule for GNSS samplers.

    Args:
        cfg: Hydra config.
        sampler_type: Resolved sampler type.
        samples: GNSSSamples from the sampler (unused for the schedule).

    Returns:
        Master epoch array from the requested start/end/cadence, or ``None``
        when the sampler does not define a requested schedule.

    Raises:
        ValueError: If the schedule settings are invalid.
    """
    if sampler_type not in ("real_gnss", "orbit_gnss"):
        return None
    start = cfg.sampler.get("start_time")
    end = cfg.sampler.get("end_time")
    cadence = cfg.sampler.get("cadence_s")
    if start is None or end is None or cadence is None:
        return None
    return gnss_tid.simulate.master_epochs(start, end, float(cadence))


def run_gnss_experiment(
    cfg: DictConfig,
    data_fn: Path,
    output_fn: Path,
    client: Client | None,
) -> None:
    """Run a GNSS simulation experiment end to end (real or orbit geometry).

    Generates geometry-only GNSS samples, evaluates the analytical scenario
    per original observation, adds batch-invariant seeded noise, reconstructs
    images with the existing image maker at one reconstruction height, and
    runs the estimator with app-side gap masking for missing frames.

    Args:
        cfg: Hydra config with ``sampler.type`` ``real_gnss`` or
            ``orbit_gnss``.
        data_fn: Output Zarr path for the image dataset.
        output_fn: Output Zarr path for the estimator results.
        client: Optional initialized Dask client (closed by the caller).

    Raises:
        ValueError: If no observations survive filtering.
        Exception: Any generation/estimation failure; owned partial
            artifacts are preserved.
    """
    from gnss_tid.coords import Local2D
    from gnss_tid.image import ImageMakerBase

    reference_time = pd.Timestamp(cfg.reference_time).to_datetime64()
    scenario, _ranges = _resolve_scenario(cfg, reference_time)
    sampler = _resolve_sampler(cfg)
    sampler_type = str(cfg.sampler.type)

    if sampler_type == "real_gnss":
        samples = sampler.sample(reference_time)
        proj = samples.proj
        float(samples.true_height_km)
    else:
        proj = Local2D.from_geodetic(
            float(np.mean(cfg.sampler.get("lat_range", (35.0, 45.0)))),
            float(np.mean(cfg.sampler.get("lon_range", (-110.0, -100.0)))),
            float(cfg.image.reconstruction_height_km),
        )
        samples = sampler.sample(reference_time, proj)
        float(cfg.image.reconstruction_height_km)
    LOGGER.info("GNSS samples: %d retained observations", len(samples.obs))

    trials = np.arange(int(cfg.n_trials))
    forward = gnss_tid.simulate.AnalyticalTECForwardModel(scenario, trials)
    field = forward.measure(samples)
    noise_std = cfg.get("noise", {}).get("std_tecu") if cfg.get("noise") else None
    if noise_std:
        field = gnss_tid.simulate.add_noise(field, float(noise_std), int(cfg.seed) + 2)

    image_maker: ImageMakerBase = hydra.utils.instantiate(
        cfg.get("image", {}).get("image_maker")
    )
    lat_limits = tuple(cfg.image.get("latitude_limits", (30.0, 50.0)))
    lon_limits = tuple(cfg.image.get("longitude_limits", (-115.0, -95.0)))
    if cfg.image.hp_freq is not None:
        image_maker.hp_freq = float(cfg.image.hp_freq)
    image_maker.initialize_from_bounds(
        proj=proj,
        lat_limits=(float(lat_limits[0]), float(lat_limits[1])),
        lon_limits=(float(lon_limits[0]), float(lon_limits[1])),
    )

    data = gnss_tid.simulate.build_gnss_dataset(
        samples,
        field,
        scenario,
        image_maker,
        window=int(cfg.image.get("window", 1)),
        step=int(cfg.image.get("step", 1)),
        schedule=sampler_schedule(cfg, sampler_type, samples),
        output_dtype=str(cfg.get("output_dtype", "float32")),
    )
    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {
        "image": {"compressors": compressor},
        "density": {"compressors": compressor},
    }
    data.to_zarr(data_fn, mode="w", encoding=encoding, zarr_format=3)
    LOGGER.info("saved image dataset to %s", data_fn)

    reopened = xarray.open_zarr(data_fn)
    _validate_detector_view(reopened, cfg)

    smooth_win = int(cfg.get("smooth_win", 9))
    norm = cfg.get("norm", None)
    n = reopened.sizes["trial"]
    n_batches = ceil(n / int(cfg.batch_size))
    total_time = 0.0
    output_started = False
    for b in range(n_batches):
        LOGGER.info("running estimation batch %d / %d", b + 1, n_batches)
        t0 = perf_counter()
        batch = slice(b * int(cfg.batch_size), min((b + 1) * int(cfg.batch_size), n))
        params = gnss_tid.parameter.estimate_parameters_dask(
            reopened.isel(trial=batch),
            Nfft=int(cfg.nfft),
            block_size=int(cfg.block_size),
            step_size=int(cfg.step_size),
            normalize=norm,
            smooth_win=smooth_win,
            q_threshold=float(cfg.get("q_threshold", 0.95)),
            density_threshold=int(cfg.get("density_threshold", 5)),
        )
        params = gnss_tid.simulate.apply_gap_mask(
            params, reopened.isel(trial=batch), smooth_win=smooth_win
        )
        if not output_started:
            params.to_zarr(output_fn, mode="w", safe_chunks=True)
            output_started = True
        else:
            params.to_zarr(
                output_fn, append_dim="trial", align_chunks=True, safe_chunks=True
            )
        batch_duration = perf_counter() - t0
        total_time += batch_duration
        LOGGER.info("batch %d: %.2f s", b + 1, batch_duration)

    reopened_results = xarray.open_zarr(output_fn)
    coverage = int(reopened_results.sizes["trial"])
    if coverage != n:
        raise RuntimeError(f"incomplete run: {coverage} of {n} trials written")
    LOGGER.info("completed %d / %d trials", coverage, n)


def run_grid_experiment(
    cfg: DictConfig,
    data_fn: Path,
    output_fn: Path,
    client: Client | None,
) -> None:
    """Run the uniform-grid simulation experiment end to end.

    Resolves the scenario/sampler, generates the requested trials with
    batch-invariant seeded noise, saves the image/density/truth product,
    runs the estimator by trial batch with safe aligned appends, and marks
    completion only after all results are written and reopenable.

    Args:
        cfg: Hydra config (grid mode).
        data_fn: Output Zarr path for the image dataset.
        output_fn: Output Zarr path for the estimator results.
        client: Optional initialized Dask client (closed by the caller).

    Raises:
        Exception: Any generation/estimation failure; owned partial
            artifacts are preserved.
    """
    reference_time = pd.Timestamp(cfg.reference_time).to_datetime64()
    scenario, _ranges = _resolve_scenario(cfg, reference_time)
    sampler = _resolve_sampler(cfg)
    samples = sampler.sample(reference_time)
    LOGGER.info(
        "scenario: %d source(s); sampler grid %s; %d trials",
        len(scenario.sources),
        samples.x.sizes,
        int(cfg.n_trials),
    )

    trials = np.arange(int(cfg.n_trials))
    forward = gnss_tid.simulate.AnalyticalTECForwardModel(scenario, trials)
    generation_batch = int(cfg.get("generation_batch_size", cfg.batch_size))
    noise_std = cfg.get("noise", {}).get("std_tecu") if cfg.get("noise") else None

    # fixed RNG layout against the full logical trial axis: batch selection
    # keeps sample noise identical when batch sizes or chunks change
    noise_full = None
    if noise_std:
        noise_shape = (
            int(cfg.n_trials),
            samples.t_seconds.sizes["time"],
            samples.y.sizes["y"],
            samples.x.sizes["x"],
        )
        noise_full = gnss_tid.simulate.noise_stream(
            noise_shape, float(noise_std), int(cfg.seed) + 2
        )

    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {
        "image": {"compressors": compressor},
        "density": {"compressors": compressor},
    }

    started = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    for start in range(0, int(cfg.n_trials), generation_batch):
        stop = min(start + generation_batch, int(cfg.n_trials))
        batch_trials = trials[start:stop]
        field = forward.measure(samples).isel(trial=batch_trials)
        if noise_full is not None:
            noise = noise_full.isel(trial=batch_trials)
            chunks = dict(zip(field.dims, field.chunks, strict=True))
            noisy = field + noise.chunk(chunks)
        else:
            noisy = field
        data = gnss_tid.simulate.build_grid_dataset(
            samples,
            noisy,
            scenario,
            smooth_pixels=int(cfg.get("image", {}).get("grid_smoothing_pixels", 0) or 0)
            if cfg.get("image")
            else 0,
            output_dtype=str(cfg.get("output_dtype", "float32")),
        )
        if start == 0:
            data.to_zarr(data_fn, mode="w", encoding=encoding, zarr_format=3)
            LOGGER.info("saved image dataset to %s", data_fn)
        else:
            data.to_zarr(data_fn, append_dim="trial", align_chunks=True, safe_chunks=True)

    data = xarray.open_zarr(data_fn)
    _validate_detector_view(data, cfg)

    smooth_win = int(cfg.get("smooth_win", 9))
    norm = cfg.get("norm", None)
    n = data.sizes["trial"]
    n_batches = ceil(n / int(cfg.batch_size))
    total_time = 0.0
    output_started = False
    for b in range(n_batches):
        LOGGER.info("running estimation batch %d / %d", b + 1, n_batches)
        t0 = perf_counter()
        batch = slice(b * int(cfg.batch_size), min((b + 1) * int(cfg.batch_size), n))
        params = gnss_tid.parameter.estimate_parameters_dask(
            data.isel(trial=batch),
            Nfft=int(cfg.nfft),
            block_size=int(cfg.block_size),
            step_size=int(cfg.step_size),
            normalize=norm,
            smooth_win=smooth_win,
            q_threshold=float(cfg.get("q_threshold", 0.95)),
            density_threshold=int(cfg.get("density_threshold", 5)),
        )
        if not output_started:
            params.to_zarr(output_fn, mode="w", safe_chunks=True)
            output_started = True
        else:
            params.to_zarr(
                output_fn, append_dim="trial", align_chunks=True, safe_chunks=True
            )
        batch_duration = perf_counter() - t0
        total_time += batch_duration
        LOGGER.info(
            "batch %d: %.2f s (total %.2f min, %.2f s/trial)",
            b + 1,
            batch_duration,
            total_time / 60,
            total_time / batch.stop,
        )

    # completion marker: reopen both products and record coverage
    reopened_images = xarray.open_zarr(data_fn)
    reopened_results = xarray.open_zarr(output_fn)
    coverage = int(reopened_results.sizes["trial"])
    LOGGER.info(
        "completed %d / %d trials; images=%s results=%s",
        coverage,
        n,
        data_fn,
        output_fn,
    )
    if coverage != n:
        raise RuntimeError(f"incomplete run: {coverage} of {n} trials written")
    status = {
        "experiment_id": str(cfg.get("output", {}).get("experiment_id") or ""),
        "started": started,
        "completed": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "trials": n,
        "status": "complete",
    }
    xarray.Dataset({k: ((), v) for k, v in status.items() if k != "status"}).assign_attrs(
        status="complete"
    ).to_zarr(str(data_fn) + "_status.zarr", mode="w")
    assert reopened_images.sizes["trial"] == n


def save_data(data_fn, wave_type, n_trials):
    """Legacy grid save path retained for backward compatibility.

    Args:
        data_fn: Output Zarr path.
        wave_type: Legacy ``planar`` or ``spherical`` preset.
        n_trials: Number of trials.
    """
    LOGGER.info("#" * 80)
    LOGGER.info("DATA (legacy path)")
    shutil.rmtree(data_fn, ignore_errors=True)
    model_options = {
        "snr_lim": [-6, 6],
        "lam_lim": [100, 500],
        "tau_lim": [10, 60],
        "n_trials": n_trials,
        "xlim": (-1500, 1500),
        "ylim": (-1500, 1500),
        "hres": 20,
        "batch_size": 10,
    }
    if wave_type == "spherical":
        data = gnss_tid.synthetic.spherical_model(center=(0, 0), **model_options)
    elif wave_type == "planar":
        data = gnss_tid.synthetic.planar_model(**model_options)
    else:
        raise Exception(wave_type)

    compressor = BloscCodec(cname=BloscCname.lz4, clevel=5, shuffle=BloscShuffle.shuffle)
    encoding = {var: {"compressors": compressor} for var in data.data_vars}
    LOGGER.info("saving data to zarr")
    data.to_zarr(data_fn, mode="w", encoding=encoding)


@hydra.main(config_path="conf", config_name="exp_config", version_base=None)
def main(cfg: DictConfig):
    """Run the synthetic simulation experiment.

    Grid mode (``sampler.type == grid``) uses the new pipeline with explicit
    trials, batch-invariant seeded noise, and safe aligned Zarr writes. The
    legacy flat ``wave_type`` fallback (no ``scenario.sources``) is retained.
    """
    use_new = bool(cfg.get("scenario", {}).get("sources")) or str(
        cfg.get("sampler", {}).get("type", "grid")
    ) in ("grid", "real_gnss", "orbit_gnss")
    try:
        client: Client = hydra.utils.instantiate(cfg.client)
        LOGGER.info(client.dashboard_link)

        output_cfg = cfg.get("output", {})
        root = Path(str(output_cfg.get("root", "./results")))
        experiment_id = output_cfg.get("experiment_id") or uuid.uuid4().hex[:12]
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        exp_dir = _reserve_output_dir(root, f"{stamp}_{experiment_id}")
        data_fn = exp_dir / "images.zarr"
        output_fn = exp_dir / "results.zarr"

        sampler_type = str(cfg.get("sampler", {}).get("type", "grid"))
        if use_new and sampler_type == "grid":
            run_grid_experiment(cfg, data_fn, output_fn, client)
        elif use_new and sampler_type in ("real_gnss", "orbit_gnss"):
            run_gnss_experiment(cfg, data_fn, output_fn, client)
        else:
            save_data(data_fn, cfg.wave_type, cfg.n_trials)
            data = xarray.open_zarr(data_fn)
            LOGGER.info("DATA\n%s", data)
            n = data.sizes["trial"]
            n_batches = ceil(n / int(cfg.batch_size))
            total_time = 0.0
            for b in range(n_batches):
                LOGGER.info("running batch %d / %d", b + 1, n_batches)
                t0 = perf_counter()
                batch_trials = slice(
                    b * int(cfg.batch_size), min((b + 1) * int(cfg.batch_size), n)
                )
                params = gnss_tid.parameter.estimate_parameters_dask(
                    data.isel(trial=batch_trials),
                    Nfft=int(cfg.nfft),
                    block_size=int(cfg.block_size),
                    step_size=int(cfg.step_size),
                    normalize=cfg.norm,
                )
                if b == 0:
                    params.to_zarr(output_fn, mode="w")
                else:
                    params.to_zarr(output_fn, append_dim="trial", align_chunks=True)
                batch_duration = perf_counter() - t0
                total_time += batch_duration
                LOGGER.info("batch %d: %.2f s", b + 1, batch_duration)

        LOGGER.info("FINISHED; output %s", exp_dir)

        if cfg.get("save", True):
            client.close()
            return

        LOGGER.info("CLEANUP")
        shutil.rmtree(data_fn, ignore_errors=True)
        shutil.rmtree(output_fn, ignore_errors=True)
        client.close()
    except Exception as e:
        LOGGER.exception("error occurred", exc_info=True, stack_info=True)
        raise e


if __name__ == "__main__":
    main()
