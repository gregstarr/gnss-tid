import numpy as np
import pandas as pd
import pytest

from gnss_tid.simulation_3d import ShellGrid, SimulationScenario


@pytest.fixture
def scenario():
    grid = ShellGrid()
    return SimulationScenario(grid)

def test_generate_random_receivers(scenario):
    n_rx = 1000
    lat_range = (0.0, 45.0)
    lon_range = (-10.0, 10.0)

    receivers = scenario.generate_random_receivers(n_rx, lat_range, lon_range)

    assert receivers.shape == (n_rx, 3)
    assert scenario.receivers is not None
    np.testing.assert_array_equal(scenario.receivers, receivers)

    # Convert back to spherical to verify ranges
    from gnss_tid.coords import ecef2spherical
    lats, lons, _ = ecef2spherical(receivers[:, 0], receivers[:, 1], receivers[:, 2])

    assert np.all(lats >= lat_range[0])
    assert np.all(lats <= lat_range[1])
    assert np.all(lons >= lon_range[0])
    assert np.all(lons <= lon_range[1])

    # Verify uniform area distribution: sin(lat) should be roughly uniform
    sin_lats = np.sin(np.radians(lats))
    sin_lat_min = np.sin(np.radians(lat_range[0]))
    sin_lat_max = np.sin(np.radians(lat_range[1]))

    # The mean of a uniform distribution is (min + max) / 2
    expected_mean = (sin_lat_min + sin_lat_max) / 2
    actual_mean = np.mean(sin_lats)

    # With 1000 samples, we expect it to be close
    assert np.isclose(actual_mean, expected_mean, atol=0.05)

def test_generate_uniform_grid_receivers(scenario):
    lat_range = (0.0, 10.0)
    lon_range = (0.0, 10.0)
    n_lat = 5
    n_lon = 5

    receivers = scenario.generate_uniform_grid_receivers(lat_range, lon_range, n_lat, n_lon)

    assert receivers.shape == (n_lat * n_lon, 3)
    assert scenario.receivers is not None

    from gnss_tid.coords import ecef2spherical
    lats, lons, _ = ecef2spherical(receivers[:, 0], receivers[:, 1], receivers[:, 2])

    # Since it's a grid, we expect specific values
    expected_lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    expected_lons = np.linspace(lon_range[0], lon_range[1], n_lon)

    # Meshgrid with indexing='ij' means lat changes slower than lon?
    # Wait, np.meshgrid(lats, lons, indexing='ij') -> lat_grid is (n_lat, n_lon), lon_grid is (n_lat, n_lon)
    # ravel() will give [lat0, lon0], [lat0, lon1]...

    # Check unique values and their counts
    unique_lats = np.unique(np.round(lats, 5))
    unique_lons = np.unique(np.round(lons, 5))

    assert len(unique_lats) == n_lat
    assert len(unique_lons) == n_lon
    np.testing.assert_allclose(unique_lats, expected_lats, atol=1e-5)
    np.testing.assert_allclose(unique_lons, expected_lons, atol=1e-5)

def test_generate_receiver_df_success(scenario):
    # Manually set receivers
    from gnss_tid.coords import spherical2ecef
    lats = np.array([0.0, 10.0])
    lons = np.array([0.0, 10.0])
    alts = np.full(2, 6378.137)
    # The implementation of spherical2ecef now expects degrees
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
