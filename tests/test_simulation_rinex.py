import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from gnss_tid.simulation_3d import SimulationScenario, ShellGrid
from gnss_tid.coords import ecef2spherical

def test_simulation_with_rinex():
    # Setup a simple ShellGrid
    grid = ShellGrid()
    
    # Create a simple blob of delta Ne to ensure we get non-zero TEC
    grid.add_uniform(300, 1.0)


    nav_file = "tests/data/BRDC00IGS_R_20253500000_01D_MN.rnx"
    
    # Initialize scenario with RINEX file
    scenario = SimulationScenario(
        shell_grid=grid,
        nav_file=nav_file,
        delta_h_eff=10.0
    )
    
    # Generate 2 receivers
    scenario.generate_uniform_grid_receivers(
        lat_range=(30, 40), 
        lon_range=(-10, 0), 
        n_lat=1, 
        n_lon=1
    )
    # Actually generate a few more to be sure
    scenario.generate_uniform_grid_receivers(
        lat_range=(30, 40), 
        lon_range=(-10, 0), 
        n_lat=2, 
        n_lon=2
    )

    # Define time span
    # Based on the RINEX file header: 20251217
    start_time = datetime(2025, 12, 17, 12, 0, 0)
    end_time = start_time + timedelta(minutes=10)
    sample_rate = 300  # 5 minutes
    
    df = scenario.generate_observations_time_series(
        start_time=start_time, 
        end_time=end_time, 
        sample_rate=sample_rate
    )
    
    # Verifications
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "time" in df.columns
    assert "rx" in df.columns
    assert "sv" in df.columns
    assert "stec" in df.columns
    
    # Check for NaNs in critical columns
    assert df["time"].isna().sum() == 0
    assert df["rx"].isna().sum() == 0
    assert df["sv"].isna().sum() == 0
    
    # Verify that we have multiple time steps
    unique_times = df["time"].unique()
    assert len(unique_times) > 1
    
    # Verify that satellite positions change (implicitly by checking that the
    # result for different times is different, although TEC depends on the grid)
    # Since we set the grid to a constant, stec will only change if the satellite
    # enters/leaves the shell or if the intersection point moves.
    # For a constant grid, we just check that the process completes and produces data.


def test_satellite_positions_are_physically_plausible():
    grid = ShellGrid()
    scenario = SimulationScenario(grid, nav_file="tests/data/BRDC00IGS_R_20253500000_01D_MN.rnx")
    sats, _ = scenario._get_satellites_at_time(datetime(2025, 12, 16, 0, 0, 0))

    valid = ~np.isnan(sats).any(axis=1)
    assert valid.sum() > 20, f"Expected >20 valid satellites, got {valid.sum()}"
    sats = sats[valid]

    radii = np.linalg.norm(sats, axis=1)
    assert np.all(radii > 26000), f"GPS satellites should be above 26000 km, min was {radii.min():.0f} km"
    assert np.all(radii < 27500), f"GPS satellites should be below 27500 km, max was {radii.max():.0f} km"

    lats, lons, _ = ecef2spherical(sats[:, 0], sats[:, 1], sats[:, 2])
    # GPS inclination is ~55° so no satellite should exceed ±60°
    assert np.all(np.abs(lats) < 60), f"GPS latitudes should be within ±60°, got {np.abs(lats).max():.1f}°"
    # With a full constellation the satellites span more than 180° in longitude
    assert np.ptp(lons) > 180, f"Satellites should span >180° in longitude, got {np.ptp(lons):.1f}°"


def test_observations_have_above_horizon_elevations():
    grid = ShellGrid()
    grid.add_uniform(300, 1.0)
    scenario = SimulationScenario(grid, nav_file="tests/data/BRDC00IGS_R_20253500000_01D_MN.rnx")
    scenario.generate_uniform_grid_receivers((30, 40), (-100, -90), 2, 2)

    obs = scenario.generate_observations_time_series(
        datetime(2025, 12, 16, 0, 0, 0),
        datetime(2025, 12, 16, 0, 0, 0),
        60,
    )

    assert (obs["el"] > 0).any(), "Some satellites should be above the horizon"
    assert (obs["el"] > 30).any(), "Some satellites should have elevation > 30°"
