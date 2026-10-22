"""Tests for direct spectral dispatch with categorical, memmapped inputs."""

from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
import xarray
from joblib import Parallel, delayed

import gnss_tid.spectral as spectral
from gnss_tid.coords import Local2D
from gnss_tid.fft import make_kaiser_2d
from gnss_tid.image import ScipyRbfImageMaker
from gnss_tid.pointdata import get_obs_in_window, make_time_windows
from gnss_tid.simulation import ShellGrid, SimulationScenario
from gnss_tid.spectral import (
    SpectralConfig,
    run_spectral_focusing,
)

NAV_FILE = "tests/data/BRDC00IGS_R_20253500000_01D_MN.rnx"
# A timestamp within the RINEX file's coverage window (DOY 350 of 2025).
BASE_TIME = datetime(2025, 12, 16, 12, 0, 0)
LAT_RANGE = (25.0, 45.0)
LON_RANGE = (-110.0, -85.0)
TRUE_HEIGHT = 290.0
HEIGHTS = np.arange(250.0, 340.0, 20.0)


def _probe_transport(cfg: SpectralConfig, time_window):
    """Inspect transported arrays and copy a window without changing inputs.

    Args:
        cfg: Configuration with categorical identifiers and large arrays.
        time_window: Observation row indices and timestamps for one window.

    Returns:
        Mapping from numeric/datetime column names to memmap and writeability
        flags, plus categorical-identifier and copy-isolation checks.
    """
    result = {}
    for name in ("az", "time"):
        array = cfg.obs[name].to_numpy(copy=False)
        writeable = array.flags.writeable
        while array is not None and not isinstance(array, np.memmap):
            array = getattr(array, "base", None)
        result[name] = (isinstance(array, np.memmap), writeable)
    result["categorical"] = all(
        isinstance(cfg.obs[name].dtype, pd.CategoricalDtype) for name in ("rx", "sv")
    )
    original = cfg.obs.az.to_numpy(copy=True)
    subset = get_obs_in_window(cfg.obs, time_window)
    subset["az"] = -1.0
    result["isolated"] = np.array_equal(cfg.obs.az.to_numpy(), original)
    return result


def _make_transport_cfg(tec_name: str = "stec") -> SpectralConfig:
    """Build a minimal picklable ``SpectralConfig`` for transport-only tests.

    No image computation is performed; its arrays exceed joblib's default
    memmapping threshold so real process transport can be inspected.

    Args:
        tec_name: Configuration marker and TEC column name.

    Returns:
        A configuration with large numeric/datetime observation columns.
    """
    obs = pd.DataFrame(
        {
            "time": np.full(140_000, np.datetime64("2025-12-16T12:00:00", "ns")),
            "rx": np.tile(["rx0", "rx1"], 70_000),
            "sv": np.tile(["G01", "G02"], 70_000),
            "az": np.arange(140_000, dtype=float),
        }
    )
    rx = pd.DataFrame({"rx": ["rx0"], "lat": [30.0], "lon": [-100.0], "alt": [0.0]})
    image_maker = ScipyRbfImageMaker(hres=20.0, hp_freq=0.05, neighbor_radius=100.0)
    window = make_kaiser_2d(8, beta=1.0).reshape(1, 1, 8, 8)
    proj = Local2D.from_geodetic(30.0, -100.0, 300.0)
    return SpectralConfig(
        obs=obs,
        rx=rx,
        image_maker=image_maker,
        tec_name=tec_name,
        block_size=8,
        block_step=4,
        window=window,
        logscale_objective=False,
        lat_limits=LAT_RANGE,
        lon_limits=LON_RANGE,
        proj=proj,
    )


@pytest.mark.parametrize("time_window", [1, 2])
def test_pipeline_prepares_memmapped_inputs(time_window, tmp_path, monkeypatch):
    """Both branches receive categorical inputs without changing the caller."""
    cfg = _make_transport_cfg()
    original = cfg.obs.copy(deep=True)
    monkeypatch.chdir(tmp_path)

    def inspect_inputs(windows, times, heights, prepared, n_jobs, *args):
        """Probe prepared configuration in real workers, then stop orchestration."""
        assert prepared is not cfg
        assert np.shares_memory(prepared.obs.az.to_numpy(), cfg.obs.az.to_numpy())
        with Parallel(n_jobs=n_jobs, max_nbytes="1M", mmap_mode="r") as parallel:
            results = parallel(
                delayed(_probe_transport)(prepared, windows[0]) for _ in range(2)
            )
        assert (
            results
            == [
                {
                    "az": (True, False),
                    "time": (True, False),
                    "categorical": True,
                    "isolated": True,
                }
            ]
            * 2
        )
        raise RuntimeError("transport probe complete")

    branch = "_focus_per_slice" if time_window == 1 else "_focus_with_time_smoothing"
    monkeypatch.setattr(spectral, branch, inspect_inputs)
    with pytest.raises(RuntimeError, match="transport probe complete"):
        run_spectral_focusing(
            cfg,
            window_size=1,
            step=1,
            heights=HEIGHTS,
            n_jobs=2,
            time_window=time_window,
        )
    pd.testing.assert_frame_equal(cfg.obs, original)


@lru_cache(maxsize=1)
def _cached_multi_time_obs():
    """Run the RINEX-based multi-time scenario build once per test session."""
    np.random.seed(42)
    grid = ShellGrid(res_lat=0.2, res_lon=0.2, lat_range=LAT_RANGE, lon_range=LON_RANGE)
    grid.add_wave(alt=TRUE_HEIGHT, k_lat=3.0, k_lon=3.0, amplitude=10.0)
    scenario = SimulationScenario(grid, delta_h_eff=10.0, nav_file=NAV_FILE)
    scenario.generate_uniform_grid_receivers(LAT_RANGE, LON_RANGE, n_lat=10, n_lon=10)
    obs = scenario.generate_observations_time_series(
        start_time=BASE_TIME,
        end_time=BASE_TIME + timedelta(minutes=30),
        sample_rate=900.0,
    )
    obs = obs.loc[obs["el"] > 15.0].dropna(subset=["az", "el"]).reset_index(drop=True)
    rx = scenario.generate_receiver_df()
    return obs, rx


def _build_pipeline_cfg(obs, rx) -> SpectralConfig:
    """Build the ``SpectralConfig`` for the two-pass simulated pipeline run."""
    image_maker = ScipyRbfImageMaker(
        hres=20.0,
        hp_freq=0.05,
        neighbor_radius=100.0,
        neighbors=20,
        kernel="multiquadric",
        smoothing=0.3,
    )
    window = make_kaiser_2d(32, beta=1.0).reshape(1, 1, 32, 32)
    proj = Local2D.from_geodetic(
        float(np.mean(LAT_RANGE)), float(np.mean(LON_RANGE)), float(np.median(HEIGHTS))
    )
    return SpectralConfig(
        obs=obs,
        rx=rx,
        image_maker=image_maker,
        tec_name="stec",
        block_size=32,
        block_step=16,
        window=window,
        logscale_objective=False,
        lat_limits=LAT_RANGE,
        lon_limits=LON_RANGE,
        proj=proj,
    )


def _run_pipeline(n_jobs: int, obs, rx, time_window: int):
    """Run autofocus using the supplied observations, job count, and smoothing.

    Args:
        n_jobs: Joblib worker count.
        obs: Observation DataFrame with time, rx, sv, az, el, and stec columns.
        rx: Receiver DataFrame with rx, lat, lon, and alt columns.
        time_window: Smoothing length; one selects the single-pass branch.

    Returns:
        Focused Dataset with time, x, y, px, py, kx, and ky coordinates and
        image, density, patch, objective, and height variables.
    """
    cfg = _build_pipeline_cfg(obs, rx)
    return run_spectral_focusing(
        cfg,
        window_size=2,
        step=1,
        heights=HEIGHTS,
        n_jobs=n_jobs,
        time_window=time_window,
    )


@pytest.mark.parametrize("time_window", [1, 2])
def test_pipeline_equivalent_n_jobs_1_and_2(time_window, tmp_path, monkeypatch):
    """Both focusing modes agree for ``n_jobs=1`` and ``n_jobs=2``.

    A small multi-time simulated wave case runs through the full pipeline for
    both job counts; selected height tracks must match exactly, objectives and
    image/patch arrays must match to tight tolerance, and both modes retain
    their expected output variables.
    """
    obs, rx = _cached_multi_time_obs()
    monkeypatch.chdir(tmp_path)
    r1 = _run_pipeline(1, obs.copy(), rx.copy(), time_window)
    r2 = _run_pipeline(2, obs.copy(), rx.copy(), time_window)

    np.testing.assert_array_equal(r1.time.values, r2.time.values)

    required = {
        "image",
        "patch",
        "F",
        "Fx",
        "Fy",
        "objective",
        "height",
    }
    if time_window > 1:
        required |= {
            "objective_smoothed_linear",
            "objective_smoothed_log",
            "height_linear",
            "height_log",
            "image_log",
            "density_log",
        }
    for result in (r1, r2):
        missing = required - set(result.data_vars)
        assert not missing, f"missing focused variables: {missing}"

    for var in (
        ("height", "height_linear", "height_log") if time_window > 1 else ("height",)
    ):
        np.testing.assert_array_equal(r1[var].values, r2[var].values, err_msg=var)
    for var in required - {"height", "height_linear", "height_log"}:
        np.testing.assert_allclose(
            r1[var].values, r2[var].values, rtol=1e-12, atol=0, err_msg=var
        )


@pytest.mark.parametrize(
    "worker",
    [
        spectral.compute_height_objectives_for_slice,
        spectral._select_best_height_for_slice,
        spectral._rebuild_slice_at_height,
        spectral._build_image_at_height,
    ],
)
def test_categorical_workers_match_string_inputs(worker):
    """Every direct worker preserves numerical results with categorical IDs."""
    obs, rx = _cached_multi_time_obs()
    cfg = _build_pipeline_cfg(obs, rx)
    cfg.image_maker.initialize_from_bounds(
        proj=cfg.proj, lat_limits=cfg.lat_limits, lon_limits=cfg.lon_limits
    )
    prepared = replace(cfg, obs=obs.astype({"rx": "category", "sv": "category"}))
    ts = make_time_windows(obs.time, 2, 1)[0]
    heights = (
        HEIGHTS
        if worker
        in (
            spectral.compute_height_objectives_for_slice,
            spectral._select_best_height_for_slice,
        )
        else TRUE_HEIGHT
    )
    expected = worker(ts, ts.start_time, heights, cfg)
    with Parallel(n_jobs=2, max_nbytes=1, mmap_mode="r") as parallel:
        actual = parallel([delayed(worker)(ts, ts.start_time, heights, prepared)])[0]
    assert expected is not None
    assert actual is not None
    if not isinstance(expected, tuple):
        expected, actual = (expected,), (actual,)
    for left, right in zip(expected, actual, strict=True):
        xarray.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
