"""Stage 1 tests: analytical core, grid pathway, noise, and dataset builder."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gnss_tid import simulate
from gnss_tid.simulate import (
    AnalyticalTECForwardModel,
    CircularWave,
    GaussianEnvelope,
    GridSampler,
    PlanarWave,
    Scenario,
    Source,
)

REF = np.datetime64("2025-01-01T00:00:00")


def make_scenario(wave=None, envelope=None, amplitude=1.0):
    """Build a single-source scenario with a planar wave.

    Args:
        wave: Optional pre-built wave.
        envelope: Optional Gaussian envelope.
        amplitude: Source amplitude in TECU.

    Returns:
        Scenario with one source referenced to ``REF``.
    """
    if wave is None:
        wave = PlanarWave(300.0, 1800.0)
    return Scenario((Source(wave, envelope, amplitude),), REF)


# ---------------------------------------------------------------------------
# Legacy cosine equivalence
# ---------------------------------------------------------------------------


class TestLegacyCosine:
    """Unbounded carriers must equal the original cosine formulas."""

    def test_planar_matches_formula(self):
        """Verify the planar carrier equals the analytical cosine formula."""
        wave = PlanarWave(300.0, 1800.0, direction_rad=0.7, phase_rad=0.5)
        scenario = make_scenario(wave)
        x, y, t = np.meshgrid(
            np.linspace(-100, 100, 5),
            np.linspace(-100, 100, 4),
            np.array([0.0, 900.0, 1800.0]),
            indexing="ij",
        )
        u = x * np.cos(0.7) + y * np.sin(0.7)
        expected = np.cos(2 * np.pi / 300 * u - 2 * np.pi / 1800 * t + 0.5)
        np.testing.assert_allclose(scenario.evaluate(x, y, t), expected)

    def test_planar_matches_legacy_orientation(self):
        """Verify orientation matches planar_model's k.conj()*r convention."""
        wave = PlanarWave(250.0, 1000.0, direction_rad=1.1, phase_rad=0.0)
        scenario = make_scenario(wave)
        X, Y, T = np.meshgrid(
            np.linspace(-200, 200, 7),
            np.linspace(-200, 200, 5),
            np.array([0.0, 300.0]),
            indexing="ij",
        )
        k = np.exp(1j * 1.1) * 2 * np.pi / 250
        r = X + 1j * Y
        w = 2 * np.pi / 1000
        expected = np.cos((k.conj() * r).real - w * T)
        np.testing.assert_allclose(scenario.evaluate(X, Y, T), expected)

    def test_circular_matches_formula(self):
        """Verify the circular carrier equals the radial cosine formula."""
        wave = CircularWave(300.0, 1800.0, center_km=(50.0, -20.0), phase_rad=0.25)
        scenario = make_scenario(wave)
        x, y, t = np.meshgrid(
            np.linspace(-100, 200, 5),
            np.linspace(-100, 100, 4),
            np.array([0.0, 1800.0]),
            indexing="ij",
        )
        r = np.hypot(x - 50, y + 20)
        expected = np.cos(2 * np.pi / 300 * r - 2 * np.pi / 1800 * t + 0.25)
        np.testing.assert_allclose(scenario.evaluate(x, y, t), expected)


# ---------------------------------------------------------------------------
# Wave propagation
# ---------------------------------------------------------------------------


class TestPropagation:
    """Crests move at lambda/T; envelope centers at alpha*lambda/T."""

    def test_planar_crest_moves_at_phase_speed(self):
        """Verify a planar crest advances lambda along direction per period."""
        theta = 0.7
        wave = PlanarWave(300.0, 1800.0, direction_rad=theta)
        scenario = make_scenario(wave)
        t0 = scenario.evaluate(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        t1 = scenario.evaluate(
            np.array([300 * np.cos(theta)]),
            np.array([300 * np.sin(theta)]),
            np.array([1800.0]),
        )
        assert np.isclose(t0[0], 1.0)
        assert np.isclose(t1[0], 1.0)

    def test_circular_crest_moves_outward(self):
        """Verify circular crests expand outward at lambda/T."""
        wave = CircularWave(300.0, 1800.0)
        scenario = make_scenario(wave)
        t0 = scenario.evaluate(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        t1 = scenario.evaluate(np.array([300.0]), np.array([0.0]), np.array([1800.0]))
        assert np.isclose(t0[0], 1.0)
        assert np.isclose(t1[0], 1.0)

    def test_gaussian_center_moves_at_alpha_phase_speed(self):
        """Verify the envelope center moves at alpha*lambda/T along direction."""
        theta = 0.3
        alpha = 0.9
        wave = PlanarWave(300.0, 1800.0, direction_rad=theta)
        env = GaussianEnvelope(40.0, 600.0, speed_ratio=alpha, spreading_time_s=None)
        scenario = make_scenario(wave, env, amplitude=1.0)
        phase_speed = 300.0 / 1800.0
        shift = alpha * phase_speed * 900.0
        a0 = scenario.evaluate(np.array([0.0]), np.array([0.0]), np.array([0.0, 900.0]))
        a1 = scenario.evaluate(
            np.array([shift * np.cos(theta)]),
            np.array([shift * np.sin(theta)]),
            np.array([900.0]),
        )
        # the amplitude at the moved center should exceed the stationary value
        assert abs(a1[0]) > abs(a0[1])

    def test_circular_envelope_moves_outward(self):
        """Verify the circular envelope center moves outward radially."""
        wave = CircularWave(300.0, 1800.0)
        env = GaussianEnvelope(40.0, None, speed_ratio=0.9, center_u_km=0.0)
        scenario = make_scenario(wave, env)
        phase_speed = 300.0 / 1800.0
        shift = 0.9 * phase_speed * 900.0
        at_center = scenario.evaluate(
            np.array([shift]), np.array([0.0]), np.array([900.0])
        )
        # near the moved ring the field is strong; at the origin it is weak
        at_origin = scenario.evaluate(np.array([0.0]), np.array([0.0]), np.array([900.0]))
        assert abs(at_center[0]) > abs(at_origin[0])

    def test_alpha_zero_stationary_envelope(self):
        """Verify alpha=0 keeps the envelope center stationary."""
        wave = PlanarWave(300.0, 1800.0)
        env = GaussianEnvelope(40.0, 600.0, speed_ratio=0.0, spreading_time_s=None)
        scenario = make_scenario(wave, env)
        a0 = scenario.evaluate(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        a1 = scenario.evaluate(np.array([0.0]), np.array([0.0]), np.array([900.0]))
        # carrier phase differs but the envelope multiplier is identical
        assert np.isclose(a0[0], 1.0)
        assert abs(a1[0]) > 0.99


# ---------------------------------------------------------------------------
# Spreading
# ---------------------------------------------------------------------------


class TestSpreading:
    """Gaussian spreading follows sigma_u0*sqrt(1+(t/t_s)^2)."""

    def test_sigma_doubles_sqrt2_at_spreading_time(self):
        """Verify sigma_u = sqrt(2)*sigma_u0 and prefactor 2**(-1/4) at t_s."""
        env = GaussianEnvelope(350.0, 600.0, speed_ratio=0.0, spreading_time_s=3600.0)
        u = np.array([0.0])
        v = np.array([0.0])
        amp = env.amplitude(u, v, np.array([3600.0]), 300.0 / 1800.0)
        np.testing.assert_allclose(amp[0], 2 ** (-0.25))

    def test_unit_prefactor_without_spreading(self):
        """Verify no spreading gives constant width and unit prefactor."""
        env = GaussianEnvelope(350.0, 600.0, speed_ratio=0.0, spreading_time_s=None)
        amp = env.amplitude(
            np.array([0.0]), np.array([0.0]), np.array([0.0, 3600.0]), 1.0
        )
        np.testing.assert_allclose(amp, 1.0)

    def test_envelope_none_gives_unit_multiplier(self):
        """Verify envelope None reproduces the unbounded cosine exactly."""
        wave = PlanarWave(300.0, 1800.0)
        without = make_scenario(wave, None)
        with_flat = make_scenario(
            wave, GaussianEnvelope(1e6, None, speed_ratio=0.0, center_u_km=0.0)
        )
        # a flat envelope centered at u=0 with huge width is E~1 everywhere;
        # meshgrid NumPy inputs
        x, y, t = np.meshgrid(
            np.array([-100.0, 0.0, 100.0]),
            np.array([0.0]),
            np.array([0.0, 900.0]),
            indexing="ij",
        )
        np.testing.assert_allclose(without.evaluate(x, y, t), with_flat.evaluate(x, y, t))

    def test_one_dimensional_spreading_conserves_energy(self):
        """Verify the sqrt prefactor conserves packet energy for 1-D spreading.

        With sigma growing by sqrt(2), the prefactor sqrt(sigma0/sigma)
        gives a peak amplitude of 2**(-1/4) and an exactly conserved
        integral of E^2 for the longitudinal Gaussian.
        """
        env = GaussianEnvelope(100.0, None, speed_ratio=0.0, spreading_time_s=1000.0)
        u = np.linspace(-800, 800, 4001)
        v = np.zeros_like(u)
        phase_speed = 1.0
        e0 = env.amplitude(u, v, np.array([0.0]), phase_speed)
        e1 = env.amplitude(u, v, np.array([1000.0]), phase_speed)
        i0 = np.trapezoid(e0**2, u)
        i1 = np.trapezoid(e1**2, u)
        np.testing.assert_allclose(i1 / i0, 1.0, rtol=1e-6)
        np.testing.assert_allclose(e1.max() / e0.max(), 2 ** (-0.25), rtol=1e-6)


# ---------------------------------------------------------------------------
# Superposition and validation
# ---------------------------------------------------------------------------


class TestSuperposition:
    """Multi-source fields sum elementwise; zero sources give zero field."""

    def test_two_source_field_is_elementwise_sum(self):
        """Verify the two-source field equals the sum of individual fields."""
        s1 = make_scenario(PlanarWave(300.0, 1800.0, direction_rad=0.5))
        s2 = make_scenario(CircularWave(400.0, 2400.0), amplitude=2.0)
        both = Scenario(
            (
                Source(PlanarWave(300.0, 1800.0, direction_rad=0.5), None, 1.0),
                Source(CircularWave(400.0, 2400.0), None, 2.0),
            ),
            REF,
        )
        x = xr.DataArray(np.linspace(-200, 200, 9), dims="x")
        y = xr.DataArray(np.linspace(-200, 200, 5), dims="y")
        t = xr.DataArray(np.array([0.0, 600.0]), dims="time")
        summed = s1.evaluate(x, y, t) + s2.evaluate(x, y, t)
        xr.testing.assert_allclose(both.evaluate(x, y, t), summed)

    def test_zero_sources_gives_zero_field(self):
        """Verify the empty-source control produces zeros."""
        scenario = Scenario((), REF)
        out = scenario.evaluate(
            np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0, 5.0])
        )
        np.testing.assert_array_equal(out, 0.0)

    def test_validation_rejects_bad_parameters(self):
        """Verify nonpositive/nonfinite parameters raise ValueError."""
        with pytest.raises(ValueError):
            PlanarWave(0.0, 1800.0)
        with pytest.raises(ValueError):
            PlanarWave(300.0, -1.0)
        with pytest.raises(ValueError):
            CircularWave(np.nan, 1800.0)
        with pytest.raises(ValueError):
            GaussianEnvelope(0.0)
        with pytest.raises(ValueError):
            GaussianEnvelope(350.0, 0.0)
        with pytest.raises(ValueError):
            GaussianEnvelope(350.0, spreading_time_s=0.0)
        with pytest.raises(ValueError):
            GaussianEnvelope(350.0, speed_ratio=-0.1)
        with pytest.raises(ValueError):
            Source(PlanarWave(300.0, 1800.0), None, np.nan)
        with pytest.raises(ValueError):
            GridSampler("2025-01-01", "2025-01-01T01:00", cadence_s=0)
        with pytest.raises(ValueError):
            GridSampler("2025-01-01", "2025-01-01T00:30:30", cadence_s=60)


# ---------------------------------------------------------------------------
# Grid sampler and forward model
# ---------------------------------------------------------------------------


class TestGridPathway:
    """GridSampler + AnalyticalTECForwardModel produce canonical output."""

    @pytest.fixture
    def samples(self):
        """Build a small grid sample geometry.

        Returns:
            GridSamples with a 5x5 spatial grid and 61 time steps.
        """
        sampler = GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T01:00:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        return sampler.sample(REF)

    def test_grid_sampler_coordinates(self):
        """Verify labeled 1-D coordinates with datetime and t_seconds."""
        sampler = GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:10:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(REF)
        assert samples.x.dims == ("x",)
        assert samples.y.dims == ("y",)
        assert samples.t_seconds.dims == ("time",)
        assert samples.time.dims == ("time",)
        np.testing.assert_array_equal(samples.t_seconds.values, np.arange(11) * 60.0)
        assert str(samples.time.values[0]).startswith("2025-01-01T00:00:00")
        np.testing.assert_allclose(samples.x.values, [-100, -50, 0, 50, 100])

    def test_grid_forward_model_dims(self, samples):
        """Verify (trial, time, y, x) output dims aligned to trials."""
        forward = AnalyticalTECForwardModel(make_scenario(), np.arange(3))
        out = forward.measure(samples)
        assert out.dims == ("trial", "time", "y", "x")
        assert out.shape == (3, 61, 5, 5)
        np.testing.assert_array_equal(out.trial.values, [0, 1, 2])

    def test_grid_matches_direct_evaluation(self, samples):
        """Verify grid output equals direct scenario evaluation at same coords."""
        scenario = make_scenario(
            PlanarWave(300.0, 1800.0, direction_rad=0.7, phase_rad=0.3)
        )
        forward = AnalyticalTECForwardModel(scenario, np.arange(2))
        out = forward.measure(samples)
        xe = float(samples.x.values[1])
        ye = float(samples.y.values[2])
        te = float(samples.t_seconds.values[3])
        direct = scenario.evaluate(np.array([xe]), np.array([ye]), np.array([te]))
        from_grid = out.isel(trial=0, time=3).sel(x=xe, y=ye).values
        np.testing.assert_allclose(from_grid, direct[0])

    def test_single_trial_keeps_trial_dim(self, samples):
        """Verify every experiment has a trial dimension."""
        forward = AnalyticalTECForwardModel(make_scenario(), np.arange(1))
        out = forward.measure(samples)
        assert out.dims == ("trial", "time", "y", "x")
        assert out.sizes["trial"] == 1

    def test_zero_source_keeps_requested_dims(self, samples):
        """Verify the noise-only control preserves trial/sample dimensions."""
        forward = AnalyticalTECForwardModel(Scenario((), REF), np.arange(4))
        out = forward.measure(samples)
        assert out.dims == ("trial", "time", "y", "x")
        assert out.shape == (4, 61, 5, 5)
        np.testing.assert_array_equal(out.values, 0.0)

    def test_grid_path_is_lazy(self, samples):
        """Verify the grid output is Dask-backed (no eager evaluation)."""
        scenario = make_scenario()
        forward = AnalyticalTECForwardModel(scenario, np.arange(2))
        out = forward.measure(samples)
        assert out.chunks is not None

    def test_gnss_forward_model_aligned_to_obs(self):
        """Verify GNSS output is (trial, obs) aligned to obs rows."""
        obs = pd.DataFrame(
            {
                "x": [0.0, 10.0, 20.0],
                "y": [0.0, 5.0, 10.0],
                "t_seconds": [0.0, 60.0, 120.0],
            }
        )
        samples = simulate.GNSSSamples(
            obs=obs, rx=pd.DataFrame(), proj=None, true_height_km=350.0
        )
        scenario = make_scenario(PlanarWave(300.0, 1800.0, direction_rad=0.2))
        forward = AnalyticalTECForwardModel(scenario, np.arange(2))
        out = forward.measure(samples)
        assert out.dims == ("trial", "obs")
        assert out.shape == (2, 3)
        direct = scenario.evaluate(
            obs["x"].to_numpy(), obs["y"].to_numpy(), obs["t_seconds"].to_numpy()
        )
        np.testing.assert_allclose(out.isel(trial=0).values, direct)

    def test_forward_model_rejects_unknown_samples(self):
        """Verify an unsupported sample type raises TypeError."""
        with pytest.raises(TypeError):
            AnalyticalTECForwardModel(make_scenario(), np.arange(1)).measure("bad")


# ---------------------------------------------------------------------------
# Sampling equivalence and noise
# ---------------------------------------------------------------------------


class TestSamplingEquivalence:
    """Grid and point coordinates give equal noise-free dTEC."""

    def test_grid_vs_point_equivalence(self):
        """Verify evaluating the same (x, y, t) through both paths matches."""
        scenario = make_scenario(
            PlanarWave(300.0, 1800.0, direction_rad=1.0, phase_rad=0.2)
        )
        sampler = GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T01:00:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(REF)
        grid_out = AnalyticalTECForwardModel(scenario, np.arange(1)).measure(samples)
        point_out = scenario.evaluate(samples.x, samples.y, samples.t_seconds)
        np.testing.assert_allclose(
            grid_out.isel(trial=0).transpose("time", "y", "x").values,
            point_out.transpose("time", "y", "x").values,
        )


class TestNoise:
    """Noise is added once, reproducibly, and batch-invariantly."""

    @pytest.fixture
    def field(self):
        """Build a small noise-free field.

        Returns:
            (trial, time, y, x) DataArray from the forward model.
        """
        sampler = GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:10:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(REF)
        return AnalyticalTECForwardModel(make_scenario(), np.arange(4)).measure(samples)

    def test_noise_reproducible_for_same_seed(self, field):
        """Verify the same seed gives identical noise."""
        a = simulate.add_noise(field, 0.02, 999)
        b = simulate.add_noise(field, 0.02, 999)
        xr.testing.assert_allclose(a, b)

    def test_noise_differs_across_seeds(self, field):
        """Verify different seeds give different noise realizations."""
        a = simulate.add_noise(field, 0.02, 1)
        b = simulate.add_noise(field, 0.02, 2)
        assert not np.allclose(a.values, b.values)

    def test_none_noise_returns_field_unchanged(self, field):
        """Verify None/0 noise std returns the input field."""
        xr.testing.assert_allclose(simulate.add_noise(field, None, 1), field)
        xr.testing.assert_allclose(simulate.add_noise(field, 0, 1), field)

    def test_noise_batch_invariant(self, field):
        """Verify identical noise across output chunk layouts."""
        full = simulate.add_noise(field, 0.02, 999)
        split = simulate.add_noise(field.chunk(trial=2), 0.02, 999)
        xr.testing.assert_allclose(full, split)

    def test_noise_added_after_summation_once(self, field):
        """Verify noisy equals field plus exactly one noise draw."""
        noisy = simulate.add_noise(field, 0.05, 7)
        residual = (noisy - field).values
        std = residual.std()
        np.testing.assert_allclose(std, 0.05, rtol=0.3)


# ---------------------------------------------------------------------------
# Parameter sampling and dataset builder
# ---------------------------------------------------------------------------


class TestParameterSampling:
    """Per-trial ranges are sampled from a dedicated seed stream."""

    def test_ranges_sampled_per_trial(self):
        """Verify sampled parameters vary within their ranges."""
        base = make_scenario(PlanarWave(300.0, 1800.0, direction_rad=0.0))
        ranges = {
            "0.wavelength_km": [150, 400],
            "0.period_s": [900, 2700],
            "0.direction_rad": [0, 2 * np.pi],
            "0.amplitude_tecu": [0.1, 0.3],
        }
        scenario = simulate.sample_parameters(base, 20, 1234, ranges)
        lam = scenario.sources[0].wave.wavelength_km
        assert isinstance(lam, xr.DataArray)
        assert lam.dims == ("trial",)
        assert np.all(lam.values >= 150) and np.all(lam.values <= 400)
        assert np.all(scenario.sources[0].amplitude_tecu.values >= 0.1)

    def test_same_seed_reproducible(self):
        """Verify identical parameters for identical seeds."""
        base = make_scenario(PlanarWave(300.0, 1800.0))
        ranges = {"0.wavelength_km": [150, 400]}
        a = simulate.sample_parameters(base, 5, 42, ranges)
        b = simulate.sample_parameters(base, 5, 42, ranges)
        np.testing.assert_array_equal(
            a.sources[0].wave.wavelength_km.values,
            b.sources[0].wave.wavelength_km.values,
        )

    def test_unknown_parameter_rejected(self):
        """Verify unknown range fields raise ValueError."""
        base = make_scenario(PlanarWave(300.0, 1800.0))
        with pytest.raises(ValueError):
            simulate.sample_parameters(base, 2, 1, {"0.not_a_field": [0, 1]})

    def test_unknown_source_rejected(self):
        """Verify ranges naming missing sources raise ValueError."""
        base = make_scenario(PlanarWave(300.0, 1800.0))
        with pytest.raises(ValueError):
            simulate.sample_parameters(base, 2, 1, {"1.wavelength_km": [0, 1]})

    def test_no_ranges_keeps_constants(self):
        """Verify an empty range dict returns the template scenario."""
        base = make_scenario(PlanarWave(300.0, 1800.0))
        scenario = simulate.sample_parameters(base, 5, 1, {})
        assert np.isscalar(scenario.sources[0].wave.wavelength_km) or isinstance(
            scenario.sources[0].wave.wavelength_km, float
        )


class TestBuildGridDataset:
    """build_grid_dataset produces the canonical detector product."""

    @pytest.fixture
    def dataset(self):
        """Build a small noisy dataset with per-trial parameters.

        Returns:
            Dataset from build_grid_dataset with 3 trials.
        """
        base = make_scenario(PlanarWave(300.0, 1800.0, direction_rad=0.5))
        scenario = simulate.sample_parameters(
            base, 3, 5, {"0.wavelength_km": [200, 400], "0.period_s": [900, 2700]}
        )
        sampler = GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:10:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(REF)
        field = AnalyticalTECForwardModel(scenario, np.arange(3)).measure(samples)
        noisy = simulate.add_noise(field, 0.02, 11)
        return simulate.build_grid_dataset(samples, noisy, scenario)

    def test_dims_and_dtype(self, dataset):
        """Verify image/density dims, canonical order, and dtype."""
        assert dataset.image.dims == ("trial", "time", "y", "x")
        assert dataset.density.dims == ("trial", "time", "y", "x")
        assert dataset.image.dtype == np.float32
        assert dataset.time.dtype.kind == "M"

    def test_truth_attached_by_trial_source(self, dataset):
        """Verify generating parameters are attached on (trial, source)."""
        lam = dataset["wavelength_km"]
        assert lam.dims == ("trial", "source")
        assert lam.attrs["units"] == "km"
        np.testing.assert_allclose(dataset.lam.values, lam.isel(source=0).values)
        np.testing.assert_allclose(
            dataset.tau.values, dataset.period_s.isel(source=0).values / 60
        )

    def test_direct_pixels_equal_expected(self, dataset):
        """Verify pixels equal transient evaluation at declared dtype."""
        scenario = make_scenario(PlanarWave(300.0, 1800.0, direction_rad=0.5))
        sampler = GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:10:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(REF)
        field = AnalyticalTECForwardModel(scenario, np.arange(1)).measure(samples)
        data = simulate.build_grid_dataset(samples, field, scenario)
        expected = field.isel(trial=0).transpose("time", "y", "x").values
        np.testing.assert_allclose(data.image.isel(trial=0).values, expected, atol=1e-6)

    def test_density_is_constant_placeholder(self, dataset):
        """Verify density is the documented constant 20."""
        np.testing.assert_array_equal(dataset.density.values, 20)

    def test_smoothing_optional_default_off(self, dataset):
        """Verify smoothing is disabled by default and applied on request."""
        assert "none" in dataset.image.attrs["smoothing"]
        sampler = GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:10:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(REF)
        field = AnalyticalTECForwardModel(make_scenario(), np.arange(1)).measure(samples)
        smoothed = simulate.build_grid_dataset(
            samples, field, make_scenario(), smooth_pixels=3
        )
        assert "3x3" in smoothed.image.attrs["smoothing"]
        assert not np.allclose(smoothed.image.values, field.isel(trial=0).values)

    def test_zero_source_dataset(self):
        """Verify the noise-only control dataset keeps full dims and truth."""
        sampler = GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:10:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(REF)
        field = AnalyticalTECForwardModel(Scenario((), REF), np.arange(2)).measure(
            samples
        )
        data = simulate.build_grid_dataset(samples, field, Scenario((), REF))
        assert data.image.dims == ("trial", "time", "y", "x")
        assert "wavelength_km" not in data.coords

    def test_zarr_round_trip(self, dataset, tmp_path):
        """Verify Zarr v3 round trips retain truth coords and dims."""
        fn = tmp_path / "grid.zarr"
        dataset.to_zarr(fn, mode="w", zarr_format=3)
        back = xr.open_zarr(fn)
        assert back.image.dims == ("trial", "time", "y", "x")
        np.testing.assert_allclose(back.lam.values, dataset.lam.values)
        np.testing.assert_allclose(
            back.spreading_time_s.values, dataset.spreading_time_s.values
        )
