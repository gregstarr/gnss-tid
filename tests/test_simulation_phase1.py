import numpy as np

from gnss_tid.simulation_3d import calculate_single_shell_tec, intersect_shell


def test_intersect_shell():
    # Receiver at ground (approx), Satellite at altitude
    r = np.array([6378.137, 0, 0])
    # Satellite directly above receiver at 2000km
    s = np.array([8378.137, 0, 0])
    # Shell at 350km
    shell_radius = 6378.137 + 350

    t = intersect_shell(r, s, shell_radius)
    expected_t = 350 / 2000
    assert np.isclose(t, expected_t), f"Expected {expected_t}, got {t}"
    print("test_intersect_shell passed")


def test_single_shell_e2e():
    # Setup
    r = np.array([6378.137, 0, 0])
    s = np.array([8378.137, 0, 0])
    shell_radius = 6378.137 + 350
    delta_h_eff = 10.0

    # Constant delta Ne function
    def const_ne(lat, lon, h):
        return 1.0

    tec = calculate_single_shell_tec(r, s, shell_radius, const_ne, delta_h_eff)
    expected_tec = 1.0 * 10.0
    assert np.isclose(tec, expected_tec), f"Expected {expected_tec}, got {tec}"
    print("test_single_shell_e2e passed")


if __name__ == "__main__":
    test_intersect_shell()
    test_single_shell_e2e()
    print("All Phase 1 tests passed!")
