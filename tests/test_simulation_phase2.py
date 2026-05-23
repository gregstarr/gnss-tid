import numpy as np
from gnss_tid.simulation_3d import (
    ShellGrid,
    SimulationScenario,
    generate_gaussian_blob,
    generate_slab,
)


def test_shell_grid_indexing():
    grid = ShellGrid(altitudes=[300, 400], res_lat=1.0, res_lon=1.0)
    # Test point (0, 0) should be at index (90, 180) since lat -90..90 and lon -180..180
    # lat 0: (0 + 90) / 1 = 90
    # lon 0: (0 + 180) / 1 = 180
    grid.set_value(300, 0, 0, 10.0)
    assert grid.get_delta_ne(0, 0, 300) == 10.0

    # Test boundary
    grid.set_value(300, 90, 180, 1.0)
    assert grid.get_delta_ne(90, 180, 300) == 1.0
    print("test_shell_grid_indexing passed")


def test_pattern_verification():
    grid = ShellGrid(altitudes=[300, 400], res_lat=1.0, res_lon=1.0)

    # Gaussian Blob: peak at (0, 0, 300)
    generate_gaussian_blob(
        grid, lat0=0, lon0=0, h0=300, A=1.0, sigma_lat=1.0, sigma_lon=1.0, sigma_h=10.0
    )
    # Peak value should be approximately 1.0 at (0,0,300)
    assert np.isclose(grid.get_delta_ne(0, 0, 300), 1.0, atol=1e-2)

    # Slab: shells 300 and 400 should be filled
    generate_slab(grid, alt_start=200, alt_end=500, value=5.0)
    # The slab generator adds to existing values if we were using additive logic,
    # but in current implementation it sets values.
    # Wait, generate_slab uses `grid.shells[alt][:] = value` which overwrites.
    # Let's check a few points.
    assert grid.get_delta_ne(0, 0, 300) == 5.0
    assert grid.get_delta_ne(45, 45, 400) == 5.0
    print("test_pattern_verification passed")


def test_linearity():
    # Verify that adding two shells with same delta Ne doubles the TEC
    # We'll use the simulation scenario to check this.
    grid = ShellGrid(altitudes=[300, 400], res_lat=1.0, res_lon=1.0)
    generate_slab(grid, 200, 500, 1.0)

    scenario = SimulationScenario(grid, delta_h_eff=10.0)

    # Receiver at (0,0), Sat at (0,0, 20000) - essentially vertical
    r = np.array([6378.137, 0, 0])
    s = np.array([6378.137 + 20000, 0, 0])

    # Single shell result
    # For this setup, both shells (300 and 400) will be intersected
    # delta Ne = 1.0, delta H = 10.0, 2 shells = 20.0 TEC
    tec = scenario.generate_tec(np.array([r]), np.array([s]))

    assert np.isclose(tec[0, 0], 20.0), f"Expected 20.0, got {tec[0, 0]}"
    print("test_linearity passed")


if __name__ == "__main__":
    test_shell_grid_indexing()
    test_pattern_verification()
    test_linearity()
    print("All Phase 2 tests passed!")
