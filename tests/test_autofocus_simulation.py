"""Integration tests for the spectral autofocus pipeline.

These tests use :mod:`gnss_tid.simulation_3d` to plant a known disturbance at a
single shell altitude, generate simulated TEC observations, and verify that the
spectral autofocus's per-height objective is maximised at the true disturbance
height.
"""

from datetime import datetime

import numpy as np
import pytest

from gnss_tid.coords import Local2D
from gnss_tid.fft import make_kaiser_2d
from gnss_tid.image import ScipyRbfImageMaker
from gnss_tid.pointdata import make_time_windows
from gnss_tid.simulation import ShellGrid, SimulationScenario
from gnss_tid.spectral import (
    SpectralConfig,
    build_patch_stack,
    extract_patch_peaks,
)

NAV_FILE = "tests/data/BRDC00IGS_R_20253500000_01D_MN.rnx"
# A timestamp within the RINEX file's coverage window (DOY 350 of 2025).
BASE_TIME = datetime(2025, 12, 16, 12, 0, 0)


def _build_scenario_obs(
    true_height: float,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_rx_per_side: int,
    k_lat: float,
    k_lon: float,
    amplitude: float,
    seed: int,
):
    """Build a SimulationScenario with a single-shell wave and produce obs/rx.

    Uses the bundled GPS RINEX navigation file to place satellites at a
    realistic constellation geometry for ``BASE_TIME``.
    """
    np.random.seed(seed)

    grid = ShellGrid(
        res_lat=0.2, res_lon=0.2, lat_range=lat_range, lon_range=lon_range
    )
    grid.add_wave(
        alt=true_height, k_lat=k_lat, k_lon=k_lon, amplitude=amplitude
    )

    scenario = SimulationScenario(grid, delta_h_eff=10.0, nav_file=NAV_FILE)
    scenario.generate_uniform_grid_receivers(
        lat_range, lon_range, n_lat=n_rx_per_side, n_lon=n_rx_per_side
    )

    # Single time step: the time-series helper handles populating
    # ``scenario.satellites`` / ``sv_names`` from the nav file.
    obs = scenario.generate_observations_time_series(
        start_time=BASE_TIME, end_time=BASE_TIME, sample_rate=1.0,
    )
    # Drop below-horizon rays and any satellites whose ephemeris produced NaN
    # geometry — neither carries usable TEC information.
    obs = obs.loc[obs["el"] > 15.0].dropna(subset=["az", "el"])
    obs = obs.reset_index(drop=True)
    obs["time"] = np.datetime64(BASE_TIME)

    rx = scenario.generate_receiver_df()
    return obs, rx


def _build_spectral_cfg(
    obs,
    rx,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    center_height: float,
    block_size: int = 32,
    block_step: int = 16,
    hres: float = 20.0,
) -> SpectralConfig:
    image_maker = ScipyRbfImageMaker(
        hres=hres,
        hp_freq=0.05,
        neighbor_radius=100,
        neighbors=20,
        kernel="multiquadric",
        smoothing=0.3,
    )
    window = make_kaiser_2d(block_size, beta=1.0).reshape(
        1, 1, block_size, block_size
    )
    proj = Local2D.from_geodetic(
        float(np.mean(lat_range)),
        float(np.mean(lon_range)),
        float(center_height),
    )
    image_maker.initialize_from_bounds(
        proj=proj, lat_limits=lat_range, lon_limits=lon_range
    )
    return SpectralConfig(
        obs=obs,
        rx=rx,
        image_maker=image_maker,
        tec_name="stec",
        block_size=block_size,
        block_step=block_step,
        window=window,
        logscale_objective=False,
        lat_limits=lat_range,
        lon_limits=lon_range,
        proj=proj,
    )


@pytest.mark.parametrize("true_height", [220.0, 320.0])
def test_autofocus_recovers_wave_height(true_height):
    """A wave planted at a known altitude should be focused at that altitude.

    Plant a 2-D cosine wave in a single shell.  Run the autofocus's per-height
    FFT-patch objective sweep and assert that the maximum lands on the true
    height.
    """
    lat_range = (25.0, 45.0)
    lon_range = (-110.0, -85.0)
    candidate_heights = np.arange(150, 450, 10)

    obs, rx = _build_scenario_obs(
        true_height=true_height,
        lat_range=lat_range,
        lon_range=lon_range,
        n_rx_per_side=20,
        k_lat=3.0,
        k_lon=3.0,
        amplitude=10.0,
        seed=42,
    )

    cfg = _build_spectral_cfg(
        obs, rx, lat_range, lon_range,
        center_height=float(np.median(candidate_heights)),
    )

    time_windows = make_time_windows(obs["time"], window=1, step=1)
    assert len(time_windows) == 1, (
        f"expected one time window for the single simulated epoch, "
        f"got {len(time_windows)}"
    )
    patches = build_patch_stack(time_windows[0], candidate_heights, cfg)
    assert patches is not None, "patches computation failed — no IPP data?"

    summary = extract_patch_peaks(patches)
    objectives = summary.objective.values
    best_idx = int(np.argmax(objectives))
    recovered = float(candidate_heights[best_idx])

    assert recovered == pytest.approx(true_height), (
        f"autofocus picked {recovered} km, expected {true_height} km. "
        f"objectives: "
        f"{dict(zip(candidate_heights.tolist(), objectives.tolist(), strict=True))}"
    )


def test_autofocus_objective_peaks_above_noise():
    """The objective at the true height should comfortably exceed the floor.

    A success in :func:`test_autofocus_recovers_wave_height` could in principle
    be a tie broken in favour of the right bin.  This test checks that the peak
    is actually distinct from the noise floor across the candidate sweep.
    """
    true_height = 300.0
    lat_range = (25.0, 45.0)
    lon_range = (-110.0, -85.0)
    candidate_heights = np.arange(150, 450, 10)

    obs, rx = _build_scenario_obs(
        true_height=true_height,
        lat_range=lat_range,
        lon_range=lon_range,
        n_rx_per_side=20,
        k_lat=3.0,
        k_lon=3.0,
        amplitude=10.0,
        seed=7,
    )

    cfg = _build_spectral_cfg(
        obs, rx, lat_range, lon_range, center_height=true_height,
    )

    time_windows = make_time_windows(obs["time"], window=1, step=1)
    assert len(time_windows) == 1, (
        f"expected one time window for the single simulated epoch, "
        f"got {len(time_windows)}"
    )
    patches = build_patch_stack(time_windows[0], candidate_heights, cfg)
    assert patches is not None

    summary = extract_patch_peaks(patches)
    objectives = summary.objective.values
    best_idx = int(np.argmax(objectives))
    true_idx = int(np.argmin(np.abs(candidate_heights - true_height)))

    assert best_idx == true_idx, (
        f"argmax at {candidate_heights[best_idx]} km, expected "
        f"{true_height} km. objectives: "
        f"{dict(zip(candidate_heights.tolist(), objectives.tolist(), strict=True))}"
    )
    assert objectives[true_idx] > 1.5 * objectives.min(), (
        f"focus signal too weak: peak={objectives[true_idx]:.3g}, "
        f"min={objectives.min():.3g}"
    )
