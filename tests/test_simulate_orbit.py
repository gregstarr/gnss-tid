"""Stage 3 tests: OrbitGNSSSampler geometry and validated conversions."""

import numpy as np
import pymap3d
import pytest

from gnss_tid.coords import Local2D, spherical2ecef
from gnss_tid.simulate import (
    AnalyticalTECForwardModel,
    OrbitGNSSSampler,
    PlanarWave,
    Scenario,
    Source,
    build_rx_metadata_from_ecef,
    ecef_to_geodetic,
    ray_aer,
)

REF = np.datetime64("2025-01-01T00:00:00")


@pytest.fixture
def proj():
    """Build a shared Local2D projection.

    Returns:
        Local2D frame at a midlatitude receiver region.
    """
    return Local2D.from_geodetic(40.0, -105.0, 350.0)


class TestECEFGroundTrips:
    """Validated ECEF-to-geodetic round trips."""

    @pytest.mark.parametrize(
        "lat,lon,h", [(40.0, -105.0, 100.0), (0.0, 0.0, 0.0), (-45.0, 120.0, 500.0)]
    )
    def test_round_trip_midlatitudes(self, lat, lon, h):
        """Verify round trips at midlatitudes and edge cases."""
        x, y, z = pymap3d.geodetic2ecef(lat, lon, h * 1000)
        la, lo, al = ecef_to_geodetic(
            np.array([x / 1000]), np.array([y / 1000]), np.array([z / 1000])
        )
        assert abs(la[0] - lat) < 1e-8
        assert abs(lo[0] - lon) < 1e-8
        assert abs(al[0] - h) < 1e-8

    def test_geodetic_not_spherical(self):
        """Verify geodetic latitude differs from the legacy spherical value."""
        lat, lon, h = 40.0, -105.0, 100.0
        x, y, z = pymap3d.geodetic2ecef(lat, lon, h * 1000)
        la, _lo, _al = ecef_to_geodetic(
            np.array([x / 1000]), np.array([y / 1000]), np.array([z / 1000])
        )
        # the legacy spherical latitude is measurably different at 40 degrees
        assert abs(la[0] - lat) < 1e-8
        assert abs(la[0] - 39.8135) > 0.1

    def test_rx_metadata_from_ecef(self):
        """Verify canonical receiver metadata from exact ECEF coordinates."""
        lats = np.array([39.0, 40.0, 41.0])
        lons = np.array([-106.0, -105.0, -104.0])
        receivers = spherical2ecef(lats, lons, np.full(3, 6378.137))
        rx = build_rx_metadata_from_ecef(receivers)
        assert rx["rx"].is_unique
        assert len(rx) == 3
        np.testing.assert_allclose(rx["x_ecef"].values, receivers[:, 0])
        assert (rx["alt"].abs() < 20.0).all()
        # geodetic latitudes are derived from the exact ECEF coordinates, so
        # they differ from the input spherical latitudes at midlatitudes but
        # stay within the range
        assert rx["lat"].between(38.5, 41.5).all()


class TestRayConsistency:
    """Satellite-ray azimuth/elevation consistency."""

    def test_ray_agrees_with_pymap3d(self):
        """Verify ray AER agrees with a direct pymap3d call."""
        receivers = spherical2ecef(
            np.array([40.0]), np.array([-105.0]), np.array([6378.137])
        )
        satellites = np.array([[26560.0, 0.0, 0.0]])
        rx_meta = build_rx_metadata_from_ecef(receivers, ["rx0"])
        az, el = ray_aer(receivers, satellites, rx_meta)
        la, lo, al = rx_meta.iloc[0][["lat", "lon", "alt"]]
        az2, el2, _ = pymap3d.ecef2aer(
            satellites[0, 0] * 1000,
            satellites[0, 1] * 1000,
            satellites[0, 2] * 1000,
            la,
            lo,
            al * 1000,
        )
        np.testing.assert_allclose(az, [az2])
        np.testing.assert_allclose(el, [el2])

    def test_ray_check_rejects_inconsistent_metadata(self):
        """Verify a conflicting observer raises through the ray check."""
        receivers = spherical2ecef(
            np.array([40.0]), np.array([-105.0]), np.array([6378.137])
        )
        satellites = np.array([[26560.0, 0.0, 0.0]])
        # metadata from different ECEF coordinates: observer disagrees with rays
        rx_meta = build_rx_metadata_from_ecef(
            spherical2ecef(np.array([10.0]), np.array([10.0]), np.array([6378.137])),
            ["rx0"],
        )
        with pytest.raises(ValueError, match="ray agreement check failed"):
            ray_aer(receivers, satellites, rx_meta)

    def test_nonfinite_rays_rejected(self):
        """Verify nonfinite satellite coordinates raise."""
        receivers = spherical2ecef(
            np.array([40.0]), np.array([-105.0]), np.array([6378.137])
        )
        satellites = np.array([[np.nan, 0.0, 0.0]])
        rx_meta = build_rx_metadata_from_ecef(receivers, ["rx0"])
        with pytest.raises(ValueError, match="finite"):
            ray_aer(receivers, satellites, rx_meta)


class TestOrbitGNSSSampler:
    """Geometry-only orbit sampler behavior."""

    def test_rows_same_shape_as_real_sampler(self, proj):
        """Verify output shape matches the RealGNSSSampler contract."""
        sampler = OrbitGNSSSampler(
            n_rx=5,
            n_sat=8,
            seed=42,
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T02:00:00",
            cadence_s=60.0,
        )
        samples = sampler.sample(REF, proj)
        for col in ("obs_id", "time", "rx", "sv", "az", "el", "x", "y", "t_seconds"):
            assert col in samples.obs.columns
        assert samples.obs["obs_id"].is_unique
        assert samples.true_height_km == 350.0

    def test_seed_reproducible(self, proj):
        """Verify random geometry is reproducible from a seed."""
        kwargs = {
            "n_rx": 5,
            "n_sat": 8,
            "seed": 42,
            "start_time": "2025-01-01T00:00:00",
            "end_time": "2025-01-01T02:00:00",
            "cadence_s": 60.0,
        }
        a = OrbitGNSSSampler(**kwargs).sample(REF, proj)
        b = OrbitGNSSSampler(**kwargs).sample(REF, proj)
        np.testing.assert_array_equal(a.obs["az"].values, b.obs["az"].values)

    def test_deterministic_receivers(self, proj):
        """Verify explicit receiver ECEF gives deterministic geometry."""
        fixed = np.array([[6378.137, 0.0, 0.0], [0.0, 6378.137, 0.0]])
        samples = OrbitGNSSSampler(seed=1, receivers=fixed, n_sat=4).sample(REF, proj)
        assert samples.rx["rx"].is_unique
        assert len(samples.rx) == 2
        np.testing.assert_allclose(samples.rx["x_ecef"].values, fixed[:, 0], atol=1e-9)

    def test_min_elevation_filter(self, proj):
        """Verify satellites below the minimum elevation are filtered."""
        sampler = OrbitGNSSSampler(
            n_rx=5,
            n_sat=8,
            el_min=15.0,
            seed=42,
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T02:00:00",
            cadence_s=60.0,
        )
        samples = sampler.sample(REF, proj)
        assert (samples.obs["el"].to_numpy() >= 15.0).all()

    def test_ecef_metadata_round_trip(self, proj):
        """Verify retained ECEF coordinates match the geodetic metadata."""
        sampler = OrbitGNSSSampler(
            n_rx=5,
            n_sat=8,
            seed=42,
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T02:00:00",
            cadence_s=60.0,
        )
        samples = sampler.sample(REF, proj)
        for _, row in samples.rx.iterrows():
            la, _lo, al = ecef_to_geodetic(
                np.array([row["x_ecef"]]),
                np.array([row["y_ecef"]]),
                np.array([row["z_ecef"]]),
            )
            assert abs(la[0] - row["lat"]) < 1e-8
            assert abs(al[0] - row["alt"]) < 1e-8

    def test_forward_model_aligned(self, proj):
        """Verify the analytical forward model aligns to orbit obs rows."""
        sampler = OrbitGNSSSampler(
            n_rx=5,
            n_sat=8,
            seed=42,
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T02:00:00",
            cadence_s=60.0,
        )
        samples = sampler.sample(REF, proj)
        scenario = Scenario((Source(PlanarWave(300.0, 1800.0), None, 1.0),), REF)
        out = AnalyticalTECForwardModel(scenario, np.arange(2)).measure(samples)
        assert out.dims == ("trial", "obs")
        assert out.shape == (2, len(samples.obs))

    def test_ipp_agrees_with_real_sampler_helpers(self, proj):
        """Verify IPPs use the same projection helpers as the real sampler."""
        from gnss_tid.pointdata import convert_to_local_coords, project_to_ipp

        sampler = OrbitGNSSSampler(
            n_rx=3,
            n_sat=8,
            seed=7,
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T02:00:00",
            cadence_s=60.0,
        )
        samples = sampler.sample(REF, proj)
        # recompute the projection from the receiver metadata (not the IPP
        # lat/lon stored on obs) with the same helpers
        geometry = samples.obs[["rx", "sv", "az", "el", "time", "obs_id"]]
        merged = geometry.merge(samples.rx, on="rx", how="left")
        reprojected = project_to_ipp(
            merged,
            merged[["lat", "lon", "alt"]].to_numpy(),
            350.0,
        )
        reprojected = convert_to_local_coords(reprojected, proj)
        np.testing.assert_allclose(
            reprojected["x"].to_numpy(), samples.obs["x"].to_numpy()
        )
        np.testing.assert_allclose(
            reprojected["y"].to_numpy(), samples.obs["y"].to_numpy()
        )

    def test_no_usable_rays_raises(self, proj):
        """Verify a geometry with no usable rays raises clearly."""
        sampler = OrbitGNSSSampler(
            n_rx=1,
            n_sat=1,
            el_min=89.0,
            seed=0,
            receivers=np.array([[6378.137, 0.0, 0.0]]),
        )
        with pytest.raises(ValueError, match="no rays above"):
            sampler.sample(REF, proj)
