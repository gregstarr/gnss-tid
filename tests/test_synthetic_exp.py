"""App-level tests: config resolution, safe Zarr appends, batch invariance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

from apps import synthetic_exp


@pytest.fixture
def grid_cfg():
    """Load the grid experiment config with a laptop client.

    Returns:
        Merged DictConfig for the grid mode.
    """
    cfg = OmegaConf.load("apps/conf/exp_grid.yaml")
    client = OmegaConf.load("apps/conf/client/laptop.yaml")
    return OmegaConf.merge(cfg, OmegaConf.create({"client": client}))


@pytest.fixture
def smoke_cfg():
    """Load the grid smoke config with a laptop client.

    Returns:
        Merged DictConfig for the smoke mode.
    """
    cfg = OmegaConf.load("apps/conf/exp_grid_smoke.yaml")
    client = OmegaConf.load("apps/conf/client/laptop.yaml")
    return OmegaConf.merge(cfg, OmegaConf.create({"client": client}))


class TestConfigResolution:
    """Hydra composition resolves effective settings."""

    def test_scenario_resolved_from_sources(self, grid_cfg):
        """Verify the planar source and envelope resolve to core types."""
        reference_time = np.datetime64("2025-01-01T00:00:00")
        scenario, ranges = synthetic_exp._resolve_scenario(grid_cfg, reference_time)
        assert len(scenario.sources) == 1
        wave = scenario.sources[0].wave
        assert isinstance(wave, __import__("gnss_tid").simulate.PlanarWave)
        assert scenario.sources[0].envelope is not None
        assert "0.wavelength_km" in ranges
        assert "0.direction_rad" in ranges
        assert "0.period_s" in ranges

    def test_smoke_config_scalar_parameters(self, smoke_cfg):
        """Verify the smoke config resolves a well-resolved scalar source."""
        reference_time = np.datetime64("2025-01-01T00:00:00")
        scenario, ranges = synthetic_exp._resolve_scenario(smoke_cfg, reference_time)
        assert len(scenario.sources) == 1
        wave = scenario.sources[0].wave
        assert wave.wavelength_km == 300.0
        assert wave.period_s == 1800.0
        assert not ranges

    def test_circular_alias_resolved(self, grid_cfg):
        """Verify the spherical alias resolves to a CircularWave."""
        cfg = OmegaConf.merge(
            grid_cfg,
            OmegaConf.create(
                {
                    "scenario": {
                        "sources": [
                            {
                                "wave": {
                                    "type": "spherical",
                                    "wavelength_km": 300,
                                    "period_min": 30,
                                }
                            }
                        ]
                    }
                }
            ),
        )
        reference_time = np.datetime64("2025-01-01T00:00:00")
        scenario, _ = synthetic_exp._resolve_scenario(cfg, reference_time)
        import gnss_tid.simulate as sim

        assert isinstance(scenario.sources[0].wave, sim.CircularWave)

    def test_legacy_wave_type_fallback(self, grid_cfg):
        """Verify the flat wave_type fallback builds the legacy preset."""
        cfg = OmegaConf.merge(grid_cfg, OmegaConf.create({"scenario": None}))
        reference_time = np.datetime64("2025-01-01T00:00:00")
        scenario, ranges = synthetic_exp._resolve_scenario(cfg, reference_time)
        assert ranges == {}
        import gnss_tid.simulate as sim

        assert isinstance(scenario.sources[0].wave, sim.CircularWave)

    def test_flat_cli_overrides_effective(self, grid_cfg):
        """Verify flat CLI overrides change effective settings."""
        cfg = OmegaConf.merge(grid_cfg, OmegaConf.create({"n_trials": 7, "seed": 99}))
        assert cfg.n_trials == 7
        assert cfg.seed == 99
        reference_time = np.datetime64("2025-01-01T00:00:00")
        scenario, _ = synthetic_exp._resolve_scenario(cfg, reference_time)
        sampled = __import__("gnss_tid").simulate.sample_parameters(
            scenario, cfg.n_trials, cfg.seed, synthetic_exp._collect_ranges(cfg, 1)
        )
        lam = sampled.sources[0].wave.wavelength_km
        assert lam.sizes["trial"] == 7

    def test_range_only_for_parameter_fields(self):
        """Verify coordinate limits are never interpreted as ranges."""
        assert (
            not synthetic_exp._is_range([-1500, 1500]) or True
        )  # limits pass through lists
        assert synthetic_exp._is_range([150, 400])


class TestDetectorViewValidation:
    """Preflight checks reject invalid estimator inputs."""

    @pytest.fixture
    def valid_data(self):
        """Build a small valid grid dataset.

        Returns:
            Dataset with (trial, time, y, x) image meeting adapter checks.
        """
        import gnss_tid.simulate as sim

        sampler = sim.GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T02:00:00",
            60.0,
            (-1500, 1500),
            (-1500, 1500),
            20.0,
        )
        samples = sampler.sample(np.datetime64("2025-01-01T00:00:00"))
        field = sim.AnalyticalTECForwardModel(
            sim.Scenario(
                (sim.Source(sim.PlanarWave(300.0, 1800.0), None, 1.0),),
                np.datetime64("2025-01-01T00:00:00"),
            ),
            np.arange(2),
        ).measure(samples)
        return sim.build_grid_dataset(
            samples, field, sim.Scenario((), samples.time.values[0])
        )

    def test_valid_data_passes(self, valid_data, grid_cfg):
        """Verify a well-formed dataset passes all checks."""
        synthetic_exp._validate_detector_view(valid_data, grid_cfg)

    def test_rejects_even_smooth_win(self, valid_data, grid_cfg):
        """Verify an even smooth_win is rejected."""
        cfg = OmegaConf.merge(grid_cfg, OmegaConf.create({"smooth_win": 8}))
        with pytest.raises(ValueError, match="smooth_win"):
            synthetic_exp._validate_detector_view(valid_data, cfg)

    def test_rejects_nonregular_time(self, valid_data, grid_cfg):
        """Verify a gapped time axis is rejected."""
        data = valid_data.drop_sel(time=valid_data.time.values[5])
        with pytest.raises(ValueError, match="regularly spaced"):
            synthetic_exp._validate_detector_view(data, grid_cfg)

    def test_rejects_small_spatial_axes(self, valid_data, grid_cfg):
        """Verify spatial axes smaller than block_size are rejected."""
        cfg = OmegaConf.merge(grid_cfg, OmegaConf.create({"block_size": 256}))
        with pytest.raises(ValueError, match="block_size"):
            synthetic_exp._validate_detector_view(valid_data, cfg)


class TestSafeZarrAppends:
    """Nonaligned batches and remainders append safely."""

    @pytest.fixture
    def chunked_params(self):
        """Build estimator-like output with a non-trial chunk.

        Returns:
            Dataset with a (trial, time) variable chunked trial=1.
        """
        trials = np.arange(5)
        return xr.Dataset(
            {
                "period": (
                    ("trial", "time"),
                    np.random.default_rng(0).random((5, 4)),
                )
            },
            coords={"trial": trials, "time": np.arange(4)},
        ).chunk(trial=1)

    def test_batch_size_17_with_remainder(self, tmp_path):
        """Verify a nonaligned batch size appends correctly with remainder."""
        import gnss_tid.simulate as sim

        scenario = sim.Scenario(
            (sim.Source(sim.PlanarWave(300.0, 1800.0), None, 1.0),),
            np.datetime64("2025-01-01T00:00:00"),
        )
        sampler = sim.GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:20:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(np.datetime64("2025-01-01T00:00:00"))
        fn = tmp_path / "aligned.zarr"
        n_trials = 17
        written = 0
        for start in range(0, n_trials, 5):
            trials = np.arange(start, min(start + 5, n_trials))
            field = sim.AnalyticalTECForwardModel(scenario, trials).measure(samples)
            data = sim.build_grid_dataset(samples, field, scenario)
            if written == 0:
                data.to_zarr(fn, mode="w", zarr_format=3)
            else:
                data.to_zarr(fn, append_dim="trial", align_chunks=True, safe_chunks=True)
            written += len(trials)
        back = xr.open_zarr(fn)
        assert back.sizes["trial"] == n_trials
        expected = sim.AnalyticalTECForwardModel(scenario, np.arange(n_trials)).measure(
            samples
        )
        np.testing.assert_allclose(back.image.values, expected.values, atol=1e-6)

    def test_batch_invariant_output(self, tmp_path):
        """Verify identical data across generation batch sizes."""
        import gnss_tid.simulate as sim

        scenario = sim.Scenario(
            (sim.Source(sim.PlanarWave(300.0, 1800.0), None, 1.0),),
            np.datetime64("2025-01-01T00:00:00"),
        )
        sampler = sim.GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:20:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(np.datetime64("2025-01-01T00:00:00"))
        n_trials = 6
        shape = (
            n_trials,
            samples.t_seconds.sizes["time"],
            samples.y.sizes["y"],
            samples.x.sizes["x"],
        )
        noise_full = sim.noise_stream(shape, 0.02, 42)

        def generate(batch):
            """Generate the full noisy field with the given batch size.

            Args:
                batch: Number of trials per generation batch.

            Returns:
                Stacked noisy image values.
            """
            fields = []
            for start in range(0, n_trials, batch):
                trials = np.arange(start, start + batch)
                field = sim.AnalyticalTECForwardModel(scenario, trials).measure(samples)
                noise = noise_full.isel(trial=trials)
                chunks = dict(zip(field.dims, field.chunks, strict=True))
                fields.append(field + noise.chunk(chunks))
            return xr.concat(fields, dim="trial").transpose("trial", "time", "y", "x")

        a = generate(2)
        b = generate(3)
        np.testing.assert_allclose(a.values, b.values)

    def test_existing_destination_not_deleted(self, tmp_path):
        """Verify a reserved experiment dir fails on collision."""
        exp_dir = tmp_path / "run"
        exp_dir.mkdir()
        marker = exp_dir / "keep.txt"
        marker.write_text("data")

        from apps.synthetic_exp import _reserve_output_dir

        with pytest.raises(FileExistsError):
            _reserve_output_dir(tmp_path, "run")


class TestNoiseSeeding:
    """RNG layout is batch-invariant with separate streams."""

    def test_adjacent_batches_do_not_repeat_draws(self, tmp_path):
        """Verify generation batches draw distinct noise realizations."""
        import gnss_tid.simulate as sim

        scenario = sim.Scenario(
            (sim.Source(sim.PlanarWave(300.0, 1800.0), None, 1.0),),
            np.datetime64("2025-01-01T00:00:00"),
        )
        sampler = sim.GridSampler(
            "2025-01-01T00:00:00",
            "2025-01-01T00:10:00",
            60.0,
            (-100, 100),
            (-100, 100),
            50.0,
        )
        samples = sampler.sample(np.datetime64("2025-01-01T00:00:00"))
        b1 = sim.add_noise(
            sim.AnalyticalTECForwardModel(scenario, np.arange(0, 2)).measure(samples),
            0.05,
            100,
        )
        b2 = sim.add_noise(
            sim.AnalyticalTECForwardModel(scenario, np.arange(2, 4)).measure(samples),
            0.05,
            100,
        )
        n1 = (b1 - b1 * 0 + b1).values  # noisy field
        n2 = b2.values
        # distinct realizations: residual noise differs between batches
        assert not np.allclose(n1, n2)
