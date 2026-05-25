from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from gnss_tid.coords import ecef2spherical, spherical2ecef
from gnss_tid.pointdata import OBS_COLUMNS
from gnss_tid.simulation import (
    ShellGrid,
    SimulationScenario,
    calculate_single_shell_tec,
    intersect_shell,
)

NAV_FILE = "tests/data/BRDC00IGS_R_20253500000_01D_MN.rnx"


@pytest.fixture
def scenario():
    grid = ShellGrid()
    return SimulationScenario(grid)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_intersect_shell():
    r = np.array([6378.137, 0, 0])
    s = np.array([8378.137, 0, 0])
    shell_radius = 6378.137 + 350

    t = intersect_shell(r, s, shell_radius)
    expected_t = 350 / 2000
    assert np.isclose(t, expected_t)


def test_single_shell_tec_constant_ne():
    r = np.array([6378.137, 0, 0])
    s = np.array([8378.137, 0, 0])
    shell_radius = 6378.137 + 350

    def const_ne(lat, lon, h):
        return 1.0

    tec = calculate_single_shell_tec(r, s, shell_radius, const_ne, delta_h_eff=10.0)
    assert np.isclose(tec, 10.0)


# ---------------------------------------------------------------------------
# ShellGrid
# ---------------------------------------------------------------------------

def test_shell_grid_indexing():
    grid = ShellGrid(res_lat=1.0, res_lon=1.0)
    grid.add_uniform(300, 10.0)
    assert grid.get_delta_ne(0, 0, 300) == 10.0

    grid.set_value(300, 90, 180, 1.0)
    assert grid.get_delta_ne(90, 180, 300) == 1.0


def test_shell_grid_patterns_compose():
    grid = ShellGrid(res_lat=1.0, res_lon=1.0)
    grid.add_gaussian(
        alt=300, lat0=0, lon0=0, sigma_lat=1.0, sigma_lon=1.0, amplitude=1.0
    )
    assert np.isclose(grid.get_delta_ne(0, 0, 300), 1.0, atol=1e-2)

    grid.add_uniform(alt=300, value=5.0)
    grid.add_uniform(alt=400, value=5.0)
    assert grid.get_delta_ne(0, 0, 300) == 6.0
    assert grid.get_delta_ne(45, 45, 400) == 5.0


def test_tec_scales_linearly_with_shells():
    """Two filled shells along a vertical ray should double single-shell TEC."""
    grid = ShellGrid(res_lat=1.0, res_lon=1.0)
    grid.add_uniform(alt=300, value=1.0)
    grid.add_uniform(alt=400, value=1.0)

    r = np.array([6378.137, 0, 0])
    s = np.array([6378.137 + 20000, 0, 0])
    scenario = SimulationScenario(
        grid, delta_h_eff=10.0, receivers=np.array([r]), satellites=np.array([s])
    )

    tec = scenario.generate_tec()
    assert np.isclose(tec[0, 0], 20.0)


# ---------------------------------------------------------------------------
# SimulationScenario — receiver generation
# ---------------------------------------------------------------------------

def test_generate_random_receivers(scenario):
    n_rx = 1000
    lat_range = (0.0, 45.0)
    lon_range = (-10.0, 10.0)

    receivers = scenario.generate_random_receivers(n_rx, lat_range, lon_range)

    assert receivers.shape == (n_rx, 3)
    assert scenario.receivers is not None
    np.testing.assert_array_equal(scenario.receivers, receivers)

    lats, lons, _ = ecef2spherical(receivers[:, 0], receivers[:, 1], receivers[:, 2])
    assert np.all(lats >= lat_range[0])
    assert np.all(lats <= lat_range[1])
    assert np.all(lons >= lon_range[0])
    assert np.all(lons <= lon_range[1])

    sin_lats = np.sin(np.radians(lats))
    expected_mean = (
        np.sin(np.radians(lat_range[0])) + np.sin(np.radians(lat_range[1]))
    ) / 2
    assert np.isclose(np.mean(sin_lats), expected_mean, atol=0.05)


def test_generate_uniform_grid_receivers(scenario):
    lat_range = (0.0, 10.0)
    lon_range = (0.0, 10.0)
    n_lat = 5
    n_lon = 5

    receivers = scenario.generate_uniform_grid_receivers(
        lat_range, lon_range, n_lat, n_lon
    )

    assert receivers.shape == (n_lat * n_lon, 3)
    assert scenario.receivers is not None

    lats, lons, _ = ecef2spherical(receivers[:, 0], receivers[:, 1], receivers[:, 2])

    expected_lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    expected_lons = np.linspace(lon_range[0], lon_range[1], n_lon)

    unique_lats = np.unique(np.round(lats, 5))
    unique_lons = np.unique(np.round(lons, 5))

    assert len(unique_lats) == n_lat
    assert len(unique_lons) == n_lon
    np.testing.assert_allclose(unique_lats, expected_lats, atol=1e-5)
    np.testing.assert_allclose(unique_lons, expected_lons, atol=1e-5)


def test_generate_receiver_df_success(scenario):
    lats = np.array([0.0, 10.0])
    lons = np.array([0.0, 10.0])
    alts = np.full(2, 6378.137)
    scenario.receivers = spherical2ecef(lats, lons, alts)

    df = scenario.generate_receiver_df()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["rx", "lat", "lon", "alt"]
    assert len(df) == 2
    assert df.iloc[0]["rx"] == "rx0"
    assert df.iloc[1]["rx"] == "rx1"
    np.testing.assert_allclose(df["lat"].values, lats, atol=1e-5)
    np.testing.assert_allclose(df["lon"].values, lons, atol=1e-5)
    np.testing.assert_allclose(df["alt"].values, 0.0, atol=1e-5)


def test_generate_receiver_df_no_receivers(scenario):
    scenario.receivers = None
    with pytest.raises(ValueError, match="No receivers defined in the scenario."):
        scenario.generate_receiver_df()


# ---------------------------------------------------------------------------
# TEC + observation table (manual satellites)
# ---------------------------------------------------------------------------

def test_generate_tec_and_observation_table():
    lat_range = (30.0, 40.0)
    lon_range = (-100.0, -90.0)

    grid = ShellGrid(
        res_lat=0.5, res_lon=0.5, lat_range=lat_range, lon_range=lon_range
    )
    grid.add_gaussian(
        alt=300.0, lat0=35.0, lon0=-95.0,
        sigma_lat=2.0, sigma_lon=2.0, amplitude=1e12,
    )
    scenario = SimulationScenario(grid, delta_h_eff=10.0)

    n_rx = 100
    n_sat = 12
    scenario.generate_random_receivers(n_rx, lat_range, lon_range)
    scenario.generate_gps_satellites(n_sat)

    tec = scenario.generate_tec()
    assert tec.shape == (n_rx, n_sat)
    assert np.any(tec != 0)
    assert np.all(tec >= 0)

    obs_df = scenario.generate_observation_table()
    assert isinstance(obs_df, pd.DataFrame)
    assert list(obs_df.columns) == list(OBS_COLUMNS)
    assert len(obs_df) == n_rx * n_sat
    assert not obs_df["az"].isna().all()
    assert not obs_df["el"].isna().all()
    assert np.allclose(obs_df["stec"].values, tec.ravel())

    test_time = np.datetime64("2026-05-23T12:00:00")
    obs_df_time = scenario.generate_observation_table(time=test_time)
    assert np.all(obs_df_time["time"] == test_time)


# ---------------------------------------------------------------------------
# RINEX nav-file integration
# ---------------------------------------------------------------------------

def test_observations_time_series_with_rinex():
    grid = ShellGrid()
    grid.add_uniform(300, 1.0)

    scenario = SimulationScenario(
        shell_grid=grid, nav_file=NAV_FILE, delta_h_eff=10.0
    )
    scenario.generate_uniform_grid_receivers(
        lat_range=(30, 40), lon_range=(-10, 0), n_lat=2, n_lon=2
    )

    start_time = datetime(2025, 12, 17, 12, 0, 0)
    end_time = start_time + timedelta(minutes=10)
    df = scenario.generate_observations_time_series(
        start_time=start_time, end_time=end_time, sample_rate=300
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ("time", "rx", "sv", "stec"):
        assert col in df.columns
    for col in ("time", "rx", "sv"):
        assert df[col].isna().sum() == 0
    assert len(df["time"].unique()) > 1


def test_satellite_positions_are_physically_plausible():
    grid = ShellGrid()
    scenario = SimulationScenario(grid, nav_file=NAV_FILE)
    sats, _ = scenario._get_satellites_at_time(datetime(2025, 12, 16, 0, 0, 0))

    valid = ~np.isnan(sats).any(axis=1)
    assert valid.sum() > 20, f"Expected >20 valid satellites, got {valid.sum()}"
    sats = sats[valid]

    radii = np.linalg.norm(sats, axis=1)
    assert np.all(radii > 26000)
    assert np.all(radii < 27500)

    lats, lons, _ = ecef2spherical(sats[:, 0], sats[:, 1], sats[:, 2])
    # GPS inclination is ~55° so no satellite should exceed ±60°
    assert np.all(np.abs(lats) < 60)
    # A full constellation spans more than 180° in longitude
    assert np.ptp(lons) > 180


def test_observations_have_above_horizon_elevations():
    grid = ShellGrid()
    grid.add_uniform(300, 1.0)
    scenario = SimulationScenario(grid, nav_file=NAV_FILE)
    scenario.generate_uniform_grid_receivers((30, 40), (-100, -90), 2, 2)

    obs = scenario.generate_observations_time_series(
        datetime(2025, 12, 16, 0, 0, 0),
        datetime(2025, 12, 16, 0, 0, 0),
        60,
    )

    assert (obs["el"] > 0).any()
    assert (obs["el"] > 30).any()
