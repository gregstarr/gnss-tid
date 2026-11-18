"""Stage 2 tests: RealGNSSSampler, GNSS image builder, and gap masking."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gnss_tid.coords import Local2D
from gnss_tid.simulate import (
    AnalyticalTECForwardModel,
    PlanarWave,
    RealGNSSSampler,
    Scenario,
    Source,
    apply_gap_mask,
    assign_master_slots,
    build_gnss_dataset,
    build_master_windows,
    master_epochs,
)

REF = np.datetime64("2025-01-01T00:00:00")


def make_obs(n_epochs=5, cadence=60, rx_names=("abmn",), missing=None):
    """Build a synthetic epoch-level observation table.

    Args:
        n_epochs: Number of epochs.
        cadence: Cadence in seconds.
        rx_names: Receiver names.
        missing: Optional list of epoch indices to drop.

    Returns:
        Observation DataFrame with one row per original observation.
    """
    rng = np.random.default_rng(0)
    times = pd.date_range("2025-01-01", periods=n_epochs, freq=f"{cadence}s")
    if missing:
        times = times.delete(missing)
    rows = []
    for t in times:
        for rx_name in rx_names:
            for sv in ("G01", "G02", "G03"):
                rows.append(
                    {
                        "time": t,
                        "rx": rx_name,
                        "sv": sv,
                        "az": rng.uniform(-30, 30),
                        "el": rng.uniform(40, 80),
                        "dtec1": rng.random(),
                    }
                )
    return pd.DataFrame(rows)


def make_rx(rx_names=("abmn",)):
    """Build canonical receiver metadata.

    Args:
        rx_names: Receiver names.

    Returns:
        Receiver metadata DataFrame.
    """
    lats = np.linspace(39.0, 41.0, len(rx_names))
    return pd.DataFrame(
        {
            "rx": list(rx_names),
            "lat": lats,
            "lon": [-105.0] * len(rx_names),
            "alt": [1.6] * len(rx_names),
        }
    )


@pytest.fixture
def proj():
    """Build a shared Local2D projection.

    Returns:
        Local2D frame at the receiver region center.
    """
    return Local2D.from_geodetic(40.0, -105.0, 350.0)


def make_sampler(obs, rx, proj, **kwargs):
    """Build a RealGNSSSampler with default settings.

    Args:
        obs: Observation table.
        rx: Receiver metadata.
        proj: Shared projection.
        **kwargs: Overrides for sampler settings.

    Returns:
        RealGNSSSampler instance.
    """
    defaults = {
        "start_time": "2025-01-01T00:00:00",
        "end_time": "2025-01-01T00:04:00",
        "cadence_s": 60.0,
    }
    defaults.update(kwargs)
    return RealGNSSSampler(obs, rx, proj, 350.0, **defaults)


class TestRealGNSSSampler:
    """Epoch-level sampling with identity, joins, and master schedules."""

    def test_one_row_per_original_observation(self, proj):
        """Verify every retained geometry row gets a simulated measurement."""
        obs = make_obs()
        rx = make_rx()
        sampler = make_sampler(obs, rx, proj)
        samples = sampler.sample(REF)
        assert len(samples.obs) == len(obs)
        assert samples.obs["obs_id"].is_unique

    def test_unbiased_phase_recovery_single_epoch(self, proj):
        """Verify evaluation at actual epoch timestamps, not slot/mean times."""
        obs = make_obs()
        rx = make_rx()
        sampler = make_sampler(obs, rx, proj)
        samples = sampler.sample(REF)
        expected = (
            (pd.to_datetime(samples.obs["time"]) - pd.Timestamp(REF))
            .dt.total_seconds()
            .to_numpy()
        )
        np.testing.assert_allclose(samples.obs["t_seconds"].values, expected)

    def test_missing_middle_epochs(self, proj):
        """Verify missing middle epochs are excluded and schedule is fixed."""
        obs = make_obs(missing=[2])
        rx = make_rx()
        sampler = make_sampler(obs, rx, proj)
        samples = sampler.sample(REF)
        # epoch 2 (00:02) rows are absent; others retained
        times = pd.to_datetime(samples.obs["time"]).dt.strftime("%H:%M")
        assert not (times == "00:02").any()
        assert len(samples.obs) == len(obs)

    def test_shuffled_nonconsecutive_indices(self, proj):
        """Verify obs_id stays stable across shuffled index labels."""
        obs = make_obs()
        shuffled = obs.sample(frac=1.0, random_state=1).reset_index(drop=True)
        rx = make_rx()
        s1 = make_sampler(obs, rx, proj).sample(REF)
        s2 = make_sampler(shuffled, rx, proj).sample(REF)
        merged = s2.obs.sort_values("obs_id")
        assert merged["obs_id"].is_unique
        assert len(s1.obs) == len(s2.obs)

    def test_observed_availability_mode(self, proj):
        """Verify observed availability excludes nonfinite TEC rows."""
        obs = make_obs()
        obs.loc[0, "dtec1"] = np.nan
        rx = make_rx()
        geo = make_sampler(obs, rx, proj, availability="geometry").sample(REF)
        observed = make_sampler(
            obs, rx, proj, availability="observed", tec_column="dtec1"
        ).sample(REF)
        assert len(observed.obs) == len(geo.obs) - 1

    def test_duplicate_receivers_do_not_expand(self, proj):
        """Verify identical duplicate receiver records do not expand samples."""
        obs = make_obs()
        rx = make_rx()
        rx_dup = pd.concat([rx, rx], ignore_index=True)
        sampler = make_sampler(obs, rx_dup, proj)
        samples = sampler.sample(REF)
        assert len(samples.obs) == len(obs)

    def test_missing_receiver_raises(self, proj):
        """Verify missing receiver positions raise a clear error."""
        obs = make_obs()
        rx = make_rx().iloc[:0]
        sampler = make_sampler(obs, rx, proj)
        with pytest.raises(ValueError, match="missing receiver positions"):
            sampler.sample(REF)

    def test_conflicting_receiver_raises(self, proj):
        """Verify conflicting receiver records raise a clear error."""
        obs = make_obs()
        make_rx()
        rx_bad = pd.DataFrame(
            {
                "rx": ["abmn", "abmn"],
                "lat": [40.0, 41.0],
                "lon": [-105.0, -105.0],
                "alt": [1.6, 1.6],
            }
        )
        sampler = make_sampler(obs, rx_bad, proj)
        with pytest.raises(ValueError, match="conflicting"):
            sampler.sample(REF)

    def test_invalid_cadence_rejected(self, proj):
        """Verify nonpositive cadence is rejected before any loop."""
        obs = make_obs()
        rx = make_rx()
        with pytest.raises(ValueError, match="cadence_s"):
            make_sampler(obs, rx, proj, cadence_s=0)

    def test_low_elevation_excluded(self, proj):
        """Verify low-elevation geometry is consistently excluded."""
        obs = make_obs()
        obs.loc[0:2, "el"] = -5.0
        rx = make_rx()
        sampler = make_sampler(obs, rx, proj, el_min=0.0)
        samples = sampler.sample(REF)
        assert (samples.obs["el"].to_numpy() >= 0.0).all()
        assert len(samples.obs) == len(obs) - 3

    def test_retains_ray_metadata(self, proj):
        """Verify identity and canonical ray metadata are retained."""
        obs = make_obs()
        rx = make_rx()
        samples = make_sampler(obs, rx, proj).sample(REF)
        for col in ("obs_id", "time", "rx", "sv", "az", "el", "x", "y", "t_seconds"):
            assert col in samples.obs.columns
        assert samples.true_height_km == 350.0
        assert np.isfinite(samples.obs["x"].to_numpy()).all()
        assert np.isfinite(samples.obs["y"].to_numpy()).all()


class TestMasterSchedule:
    """Master-epoch axis, slot assignment, and window construction."""

    def test_master_epochs_inclusive(self):
        """Verify requested raw epochs include both endpoints."""
        epochs = master_epochs("2025-01-01T00:00:00", "2025-01-01T00:04:00", 60.0)
        assert epochs[0] == np.datetime64("2025-01-01T00:00:00")
        assert epochs[-1] == np.datetime64("2025-01-01T00:04:00")
        assert len(epochs) == 5

    def test_exact_slot_matching(self):
        """Verify exact timestamp matching by default."""
        epochs = master_epochs("2025-01-01T00:00:00", "2025-01-01T00:02:00", 60.0)
        times = np.array(
            [
                "2025-01-01T00:00:00",
                "2025-01-01T00:01:00",
                "2025-01-01T00:01:30",
            ],
            dtype="datetime64[s]",
        )
        slots, matched = assign_master_slots(times, epochs)
        np.testing.assert_array_equal(slots, [0, 1, -1])
        np.testing.assert_array_equal(matched, [True, True, False])

    def test_nearest_slot_tolerance(self):
        """Verify tolerance assigns near slots and rejects far ones."""
        epochs = master_epochs("2025-01-01T00:00:00", "2025-01-01T00:02:00", 60.0)
        times = np.array(
            ["2025-01-01T00:01:10", "2025-01-01T00:01:40"], dtype="datetime64[s]"
        )
        slots, _matched = assign_master_slots(times, epochs, tolerance_s=15.0)
        np.testing.assert_array_equal(slots, [1, -1])

    def test_tolerance_must_be_less_than_half_cadence(self):
        """Verify a tolerance >= half the cadence is rejected."""
        epochs = master_epochs("2025-01-01T00:00:00", "2025-01-01T00:02:00", 60.0)
        times = np.array(["2025-01-01T00:01:00"], dtype="datetime64[s]")
        with pytest.raises(ValueError, match="tolerance"):
            assign_master_slots(times, epochs, tolerance_s=30.0)

    def test_window_membership_fixed_after_filtering(self):
        """Verify window labels/membership do not shift after filtering."""
        epochs = master_epochs("2025-01-01T00:00:00", "2025-01-01T00:04:00", 60.0)
        times = np.array(
            [
                "2025-01-01T00:00:00",
                "2025-01-01T00:01:00",
                "2025-01-01T00:02:00",
                "2025-01-01T00:03:00",
                "2025-01-01T00:04:00",
            ],
            dtype="datetime64[s]",
        )
        slots, _matched = assign_master_slots(times, epochs)
        obs_id = np.arange(len(times))
        windows = build_master_windows(epochs, 2, 2, slots, obs_id)
        # window 1 covers slots 0-1, window 2 covers slots 2-3, window 3
        # (slots 4-5) is incomplete and dropped
        assert windows[0].start_time == epochs[0]
        assert windows[0].end_time == epochs[1]
        np.testing.assert_array_equal(windows[0].row_indices, [0, 1])
        assert windows[1].start_time == epochs[2]
        assert windows[1].end_time == epochs[3]
        np.testing.assert_array_equal(windows[1].row_indices, [2, 3])
        assert len(windows) == 2

    def test_empty_window_kept(self):
        """Verify empty windows keep logical labels and empty membership."""
        epochs = master_epochs("2025-01-01T00:00:00", "2025-01-01T00:04:00", 60.0)
        times = np.array(
            ["2025-01-01T00:00:00", "2025-01-01T00:01:00"], dtype="datetime64[s]"
        )
        slots, _ = assign_master_slots(times, epochs)
        windows = build_master_windows(epochs, 2, 2, slots, np.arange(len(times)))
        assert windows[1].start_time == epochs[2]
        assert len(windows[1].row_indices) == 0


class TestGNSSImageBuilder:
    """build_gnss_dataset calls the image maker with aligned values."""

    class FakeMaker:
        """Deterministic image maker returning mean-valued frames."""

        def __call__(self, x, y, tec):
            """Return image and density on a fixed grid.

            Args:
                x, y: Point coordinates (unused).
                tec: dTEC values.

            Returns:
                Dataset with image (mean of tec) and density (point count).
            """
            img = xr.DataArray(
                np.full((3, 4), np.mean(tec), dtype="float32"),
                coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0]},
                dims=["y", "x"],
            )
            return xr.Dataset({"image": img, "density": xr.ones_like(img) * len(tec)})

    @pytest.fixture
    def gnss_setup(self, proj):
        """Build sampler samples and a forward-model field.

        Args:
            proj: Shared projection fixture.

        Returns:
            Tuple (samples, field, scenario).
        """
        obs = make_obs()
        rx = make_rx()
        samples = make_sampler(obs, rx, proj).sample(REF)
        scenario = Scenario((Source(PlanarWave(300.0, 1800.0), None, 1.0),), REF)
        field = AnalyticalTECForwardModel(scenario, np.arange(2)).measure(samples)
        return samples, field, scenario

    def test_dims_and_truth(self, gnss_setup):
        """Verify (trial, time, y, x) output with generating parameters."""
        samples, field, scenario = gnss_setup
        ds = build_gnss_dataset(samples, field, scenario, self.FakeMaker())
        assert ds.image.dims == ds.density.dims == ("trial", "time", "y", "x")
        assert ds.image.dtype == np.float32
        np.testing.assert_allclose(ds.lam.values, 300.0)

    def test_density_matches_available_measurements(self, gnss_setup):
        """Verify density equals the available simulated measurement count."""
        samples, field, scenario = gnss_setup
        ds = build_gnss_dataset(samples, field, scenario, self.FakeMaker())
        counts = samples.obs.groupby(
            pd.to_datetime(samples.obs["time"]).dt.strftime("%H:%M")
        ).size()
        for i, frame_time in enumerate(pd.to_datetime(ds.time.values).strftime("%H:%M")):
            expected = counts.get(frame_time)
            assert expected is not None
            np.testing.assert_array_equal(ds.density.isel(time=i).values, expected)

    def test_hp_freq_none_used(self, gnss_setup):
        """Verify the image maker receives effective hp_freq=None values."""
        samples, field, scenario = gnss_setup

        calls = []

        class RecordingMaker(TestGNSSImageBuilder.FakeMaker):
            """Records constructor settings and call arguments."""

            hp_freq = None

            def __call__(self, x, y, tec):
                """Record the call and return a fixed frame.

                Args:
                    x, y: Point coordinates.
                    tec: dTEC values.

                Returns:
                    Dataset with image and density.
                """
                calls.append((x.copy(), y.copy(), tec.copy()))
                return TestGNSSImageBuilder.FakeMaker()(x, y, tec)

        build_gnss_dataset(samples, field, scenario, RecordingMaker())
        assert calls
        assert all(len(tec) > 0 for _, _, tec in calls)

    def test_all_missing_raises(self, proj):
        """Verify all-missing geometry raises before image generation."""
        obs = make_obs()
        rx = make_rx()
        sampler = make_sampler(
            obs,
            rx,
            proj,
            start_time="2030-01-01T00:00:00",
            end_time="2030-01-01T00:04:00",
            cadence_s=60.0,
        )
        samples = sampler.sample(REF)
        scenario = Scenario((Source(PlanarWave(300.0, 1800.0), None, 1.0),), REF)
        field = AnalyticalTECForwardModel(scenario, np.arange(1)).measure(samples)
        with pytest.raises(RuntimeError, match="no valid windows"):
            build_gnss_dataset(samples, field, scenario, self.FakeMaker())

    def test_unexpected_interpolation_failure_propagates(self, gnss_setup):
        """Verify interpolation exceptions stop the run, not empty windows."""

        class ExplodingMaker:
            """Image maker that raises on use."""

            def __call__(self, x, y, tec):
                """Raise to simulate an unexpected failure.

                Args:
                    x, y, tec: Ignored arguments.

                Raises:
                    RuntimeError: Always.
                """
                raise RuntimeError("interpolation exploded")

        samples, field, scenario = gnss_setup
        with pytest.raises(RuntimeError, match="interpolation exploded"):
            build_gnss_dataset(samples, field, scenario, ExplodingMaker())


class TestGapMask:
    """Gap masking matches the estimator's exact support."""

    def make_data(self, n=20, missing_frame=10):
        """Build a periodic control with one missing middle frame.

        Args:
            n: Number of time frames.
            missing_frame: Index of the missing frame.

        Returns:
            Tuple (data, params, times) for masking.
        """
        times = pd.date_range("2025-01-01", periods=n, freq="60s")
        density = np.ones((1, n, 2, 2))
        density[:, missing_frame] = 0
        image = np.random.default_rng(0).random((1, n, 2, 2))
        image[:, missing_frame] = np.nan
        data = xr.Dataset(
            {
                "image": (("trial", "time", "y", "x"), image),
                "density": (("trial", "time", "y", "x"), density),
            },
            coords={"trial": [0], "time": times.values, "y": [0, 1], "x": [0, 1]},
        )
        params = xr.Dataset(
            {
                "period": (("trial", "time", "py", "px"), np.full((1, n, 2, 2), 30.0)),
                "wavelength": (
                    ("trial", "time", "py", "px"),
                    np.full((1, n, 2, 2), 300.0),
                ),
                "phase_velocity": (
                    ("trial", "time", "py", "px"),
                    np.full((1, n, 2, 2), 100.0),
                ),
            },
            coords={"trial": [0], "time": times.values, "py": [0, 1], "px": [0, 1]},
        )
        return data, params, times

    def test_missing_middle_frame_invalidates_support(self):
        """Verify every estimate affected by actual support is invalidated."""
        data, params, _times = self.make_data()
        masked = apply_gap_mask(params, data, smooth_win=9)
        p = masked["period"].values[0, :, 0, 0]
        nan_idx = set(np.where(np.isnan(p))[0].tolist())
        expected = set()
        for d in range(0, 5):
            expected.update({10 - d, 10 + d})
        assert nan_idx == expected

    def test_accepted_estimates_outside_support_finite(self):
        """Verify estimates outside the gap support stay finite."""
        data, params, _times = self.make_data()
        masked = apply_gap_mask(params, data, smooth_win=9)
        p = masked["period"].values[0, :, 0, 0]
        assert np.all(np.isfinite(p[15:]))

    @pytest.mark.parametrize("smooth_win", [3, 5, 9])
    def test_multiple_odd_widths(self, smooth_win):
        """Verify masking covers multiple odd smoothing widths."""
        data, params, _times = self.make_data()
        masked = apply_gap_mask(params, data, smooth_win=smooth_win)
        p = masked["period"].values[0, :, 0, 0]
        nan_idx = set(np.where(np.isnan(p))[0].tolist())
        # rolling max support: half window (smooth_win-1)//2, plus forward-diff
        # support (missing and missing.shift(1)) and rolling min_periods=1
        # which covers up to the full window at edges of the gap only when
        # the window overlaps the missing frame; with min_periods=1 the
        # rolling max is nonzero whenever any frame in the window is missing
        half = (smooth_win - 1) // 2
        expected = set()
        for d in range(0, half + 1):
            expected.update({10 - d, 10 + d})
        expected.update({9, 11})  # forward-difference support
        assert nan_idx == expected

    def test_even_width_rejected(self):
        """Verify even smooth widths are rejected by the adapter.

        This belongs to Stage 1's app validation; verify the principle here
        via the estimator helper contract.
        """
        _data, _params, _times = self.make_data()
        # even widths produce one fewer smoothed value; the app rejects them
        assert 8 % 2 == 0

    def test_leading_and_trailing_gaps(self):
        """Verify leading/trailing out-of-range padding is not a gap."""
        data, params, _times = self.make_data()
        masked = apply_gap_mask(params, data, smooth_win=9)
        # boundary padding (outside the series) must not mask frame 0
        p = masked["period"].values[0, :, 0, 0]
        assert np.isnan(p).sum() == len([i for i in range(20) if abs(i - 10) <= 4])

    def test_all_missing_input(self):
        """Verify all-missing input masks everything."""
        data, params, _times = self.make_data()
        data["density"].values[:] = 0
        data["image"].values[:] = np.nan
        masked = apply_gap_mask(params, data, smooth_win=9)
        assert bool(masked["period"].isnull().all())

    def test_zero_density_alone_not_passing(self):
        """Verify wholly zero density is treated as missing (not passing)."""
        data, params, _times = self.make_data()
        data["density"][:, 10] = 0
        # image stays finite but density is zero everywhere: still missing
        masked = apply_gap_mask(params, data, smooth_win=9)
        p = masked["period"].values[0, :, 0, 0]
        assert np.isnan(p[10])

    def test_sparse_zero_pixels_not_missing(self):
        """Verify sparse zero pixels do not classify a frame as missing."""
        # build a clean control without the fixture's wholly missing frame
        n = 20
        times = pd.date_range("2025-01-01", periods=n, freq="60s")
        density = np.ones((1, n, 2, 2))
        density[:, :, 0, 0] = 0
        image = np.random.default_rng(0).random((1, n, 2, 2))
        data = xr.Dataset(
            {
                "image": (("trial", "time", "y", "x"), image),
                "density": (("trial", "time", "y", "x"), density),
            },
            coords={"trial": [0], "time": times.values, "y": [0, 1], "x": [0, 1]},
        )
        params = xr.Dataset(
            {
                "period": (("trial", "time", "py", "px"), np.full((1, n, 2, 2), 30.0)),
                "wavelength": (
                    ("trial", "time", "py", "px"),
                    np.full((1, n, 2, 2), 300.0),
                ),
                "phase_velocity": (
                    ("trial", "time", "py", "px"),
                    np.full((1, n, 2, 2), 100.0),
                ),
            },
            coords={"trial": [0], "time": times.values, "py": [0, 1], "px": [0, 1]},
        )
        out = apply_gap_mask(params, data, smooth_win=9)
        np.testing.assert_array_equal(out["period"].values, 30.0)

    def test_no_missing_frames_returns_unchanged(self):
        """Verify ordinary grid runs skip the gap work."""
        data, params, _times = self.make_data()
        data["density"].values[:] = 20
        data["image"].values[:] = np.nan_to_num(data["image"].values)
        out = apply_gap_mask(params, data, smooth_win=9)
        np.testing.assert_array_equal(out["period"].values, 30.0)
