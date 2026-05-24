import numpy as np
import pandas as pd

from gnss_tid.pointdata import OBS_COLUMNS
from gnss_tid.simulation_3d import ShellGrid, SimulationScenario


def test_simulation_e2e():
    # Define a regional area
    lat_range = (30.0, 40.0)
    lon_range = (-100.0, -90.0)
    altitudes = np.arange(100, 500, 100)

    grid = ShellGrid(
        res_lat=0.5,
        res_lon=0.5,
        lat_range=lat_range,
        lon_range=lon_range
    )

    # Add a Gaussian blob in the middle of the area
    grid.add_gaussian(
        alt=300.0, lat0=35.0, lon0=-95.0,
        sigma_lat=2.0, sigma_lon=2.0, amplitude=1e12
    )

    scenario = SimulationScenario(grid, delta_h_eff=10.0)

    # Generate 100 receivers in the area and 12 GPS satellites
    n_rx = 100
    n_sat = 12
    scenario.generate_random_receivers(n_rx, lat_range, lon_range)
    scenario.generate_gps_satellites(n_sat)

    # Generate TEC
    tec = scenario.generate_tec()

    assert tec.shape == (n_rx, n_sat)
    # Ensure we have some non-zero values (at least some rays should hit the blob)
    assert np.any(tec != 0), "Simulated TEC should not be all zeros"
    assert np.all(tec >= 0), "TEC should be non-negative"

    # Test generate_observation_table
    obs_df = scenario.generate_observation_table()

    assert isinstance(obs_df, pd.DataFrame)
    assert list(obs_df.columns) == list(OBS_COLUMNS)
    assert len(obs_df) == n_rx * n_sat
    assert not obs_df["az"].isna().all()
    assert not obs_df["el"].isna().all()
    assert np.allclose(obs_df["stec"].values, tec.ravel())

    # Test with specific time
    test_time = np.datetime64("2026-05-23T12:00:00")
    obs_df_time = scenario.generate_observation_table(time=test_time)
    assert np.all(obs_df_time["time"] == test_time)

if __name__ == "__main__":
    test_simulation_e2e()

