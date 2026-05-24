import numpy as np
from gnss_tid.simulation_3d import (
    ShellGrid,
    SimulationScenario,
)


def test_shell_grid_indexing():
    grid = ShellGrid(res_lat=1.0, res_lon=1.0)
    # Test point (0, 0) should be at index (90, 180) since lat -90..90 and lon -180..180
    # lat 0: (0 + 90) / 1 = 90
    # lon 0: (0 + 180) / 1 = 180
    grid.add_uniform(300, 10.0)
    assert grid.get_delta_ne(0, 0, 300) == 10.0


    # Test boundary
    grid.set_value(300, 90, 180, 1.0)
    assert grid.get_delta_ne(90, 180, 300) == 1.0
    print("test_shell_grid_indexing passed")


def test_pattern_verification():
    grid = ShellGrid(res_lat=1.0, res_lon=1.0)

    # Gaussian Blob: peak at (0, 0, 300)
    grid.add_gaussian(
        alt=300, lat0=0, lon0=0, sigma_lat=1.0, sigma_lon=1.0, amplitude=1.0
    )
    # Peak value should be approximately 1.0 at (0,0,300)
    assert np.isclose(grid.get_delta_ne(0, 0, 300), 1.0, atol=1e-2)

    # Uniform: shells 300 and 400 should be filled
    grid.add_uniform(alt=300, value=5.0)
    grid.add_uniform(alt=400, value=5.0)
    # The uniform setter sets values.
    assert grid.get_delta_ne(0, 0, 300) == 6.0 # 1.0 from gaussian + 5.0 from uniform
    assert grid.get_delta_ne(45, 45, 400) == 5.0
    print("test_pattern_verification passed")


def test_linearity():
    # Verify that adding two shells with same delta Ne doubles the TEC
    # We'll use the simulation scenario to check this.
    grid = ShellGrid(res_lat=1.0, res_lon=1.0)
    grid.add_uniform(alt=300, value=1.0)
    grid.add_uniform(alt=400, value=1.0)

    scenario = SimulationScenario(grid, delta_h_eff=10.0)

    # Receiver at (0,0), Sat at (0,0, 20000) - essentially vertical
    r = np.array([6378.137, 0, 0])
    s = np.array([6378.137 + 20000, 0, 0])

    scenario = SimulationScenario(grid, delta_h_eff=10.0, receivers=np.array([r]), satellites=np.array([s]))

    # Single shell result
    # For this setup, both shells (300 and 400) will be intersected
    # delta Ne = 1.0, delta H = 10.0, 2 shells = 20.0 TEC
    tec = scenario.generate_tec()

    assert np.isclose(tec[0, 0], 20.0), f"Expected 20.0, got {tec[0, 0]}"
    print("test_linearity passed")


if __name__ == "__main__":
    test_shell_grid_indexing()
    test_pattern_verification()
    test_linearity()
    print("All Phase 2 tests passed!")
