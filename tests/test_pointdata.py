from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_tid.pointdata import (
    aggregate_by_receiver_satellite,
    convert_to_local_coords,
    get_data,
    get_obs_in_window,
    load_observations,
    lookup_rx_location,
    make_time_windows,
    project_to_ipp,
)

DATA_DIR = Path("tests/data")
V2_FILES = list(DATA_DIR.glob("*.nc"))


def test_load_observations_v2_fixture_returns_canonical_table():
    obs, _ = load_observations(
        V2_FILES,
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )

    assert len(obs) == 1560
    assert "stec" in obs.columns
    assert "dtec1" in obs.columns
    assert "tec_noise" in obs.columns
    assert "tec_snr" in obs.columns
    assert "roti" in obs.columns
    assert obs["stec"].notna().any()
    assert obs["dtec1"].notna().any()
    assert obs["tec_noise"].notna().any()
    assert obs["tec_snr"].notna().any()
    assert obs["roti"].notna().any()


def test_load_observations_legacy_fixture_maps_res_to_dtec1():
    obs, _ = load_observations(
        DATA_DIR / "legacy.h5",
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )

    assert len(obs) == 17923
    assert obs["dtec1"].notna().any()
    assert obs["stec"].isna().all()
    assert obs["dtec0"].isna().all()


def test_make_time_windows_can_keep_or_drop_incomplete_windows():
    times = np.array(
        [
            "2020-01-01T00:00:00",
            "2020-01-01T00:00:00",
            "2020-01-01T00:01:00",
            "2020-01-01T00:01:00",
            "2020-01-01T00:01:00",
            "2020-01-01T00:02:00",
            "2020-01-01T00:03:00",
            "2020-01-01T00:03:00",
            "2020-01-01T00:03:00",
            "2020-01-01T00:03:00",
            "2020-01-01T00:04:00",
        ],
        dtype="datetime64[m]",
    )
    unique_times = np.unique(times)

    windows_drop = make_time_windows(times, 3, 2, drop_incomplete=True)
    windows_keep = make_time_windows(times, 3, 2, drop_incomplete=False)

    assert len(windows_drop) == 2
    assert windows_drop[0].start_time == unique_times[0]
    assert windows_drop[0].end_time == unique_times[2]
    np.testing.assert_array_equal(windows_drop[0].row_indices, [0, 1, 2, 3, 4, 5])
    assert windows_drop[1].start_time == unique_times[2]
    assert windows_drop[1].end_time == unique_times[4]
    np.testing.assert_array_equal(windows_drop[1].row_indices, [5, 6, 7, 8, 9, 10])

    assert len(windows_keep) == 3
    assert windows_keep[0].start_time == unique_times[0]
    assert windows_keep[0].end_time == unique_times[2]
    assert windows_keep[1].start_time == unique_times[2]
    assert windows_keep[1].end_time == unique_times[4]
    assert windows_keep[2].start_time == unique_times[4]
    assert windows_keep[2].end_time == unique_times[4]


def test_get_data_aggregates_and_projects_v2_fixture():
    obs, rx = load_observations(
        V2_FILES,
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )
    windows = make_time_windows(obs, 4, 2)

    out = get_data(obs, rx, windows[0], 350, [0.0, 90.0], [-180.0, 180.0])

    assert out is not None
    assert len(out) > 0
    assert "lat" in out.columns
    assert "lon" in out.columns
    assert "rx" in out.columns
    assert "sv" in out.columns


def test_lookup_rx_location_returns_merged_positions():
    obs, rx = load_observations(
        V2_FILES,
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )
    result = lookup_rx_location(obs, rx)

    assert "rx" in result.columns
    assert "lat" in result.columns
    assert "lon" in result.columns
    assert "alt" in result.columns
    assert len(result) == len(obs)


def test_aggregate_by_receiver_satellite_groups_correctly():
    obs, _ = load_observations(
        V2_FILES,
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )
    aggregated = aggregate_by_receiver_satellite(obs)

    assert aggregated is not None
    assert "rx" in aggregated.columns
    assert "sv" in aggregated.columns
    assert len(aggregated) < len(obs)
    assert "rx" in aggregated.columns
    assert "sv" in aggregated.columns
    assert "az" in aggregated.columns
    assert "stec" in aggregated.columns


def test_aggregate_by_receiver_satellite_returns_none_when_empty():
    empty_df = pd.DataFrame(columns=["rx", "sv", "stec", "dtec1"])
    result = aggregate_by_receiver_satellite(empty_df)
    assert result is None


@pytest.fixture
def sample_obs_df():
    return pd.DataFrame(
        {
            "time": [
                "2020-01-01T00:00:00",
                "2020-01-01T00:01:00",
                "2020-01-01T00:02:00",
                "2020-01-01T00:03:00",
            ],
            "rx": ["rx1", "rx1", "rx2", "rx1"],
            "sv": ["G01", "G02", "G01", "G02"],
            "az": [45.0, 50.0, 55.0, 60.0],
            "el": [30.0, 35.0, 40.0, 45.0],
            "stec": [10.0, 11.0, 12.0, 13.0],
            "dtec1": [1.0, 1.1, 1.2, 1.3],
        }
    ).assign(time=lambda df: pd.to_datetime(df["time"]))


def test_get_obs_in_window_with_timewindow(sample_obs_df):
    windows = make_time_windows(sample_obs_df, 2, 1)
    result = get_obs_in_window(sample_obs_df, windows[0])

    assert result is not None
    assert len(result) > 0


def test_get_obs_in_window_with_tuple(sample_obs_df):
    start = np.datetime64("2020-01-01T00:00:00")
    end = np.datetime64("2020-01-01T00:01:00")
    result = get_obs_in_window(sample_obs_df, (start, end))

    assert result is not None
    assert len(result) == 2


def test_get_obs_in_window_returns_none_when_empty():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2020-01-01T00:00:00"]),
            "rx": ["rx1"],
            "sv": ["G01"],
        }
    )
    result = get_obs_in_window(
        df, (np.datetime64("2030-01-01"), np.datetime64("2030-12-31"))
    )
    assert result is None


def test_get_obs_in_window_rejects_invalid_type(sample_obs_df):
    with pytest.raises(TypeError):
        get_obs_in_window(sample_obs_df, "invalid")


def test_project_to_ipp_adds_lat_lon():
    obs_df = pd.DataFrame(
        {
            "rx": ["rx1", "rx2"],
            "sv": ["G01", "G02"],
            "az": [45.0, 90.0],
            "el": [45.0, 60.0],
        }
    )
    rx_positions = np.array([[30.0, -90.0, 0.0], [40.0, -100.0, 0.0]])
    result = project_to_ipp(obs_df, rx_positions, 350.0)

    assert "lat" in result.columns
    assert "lon" in result.columns
    assert len(result) == 2
    assert result["lat"].notna().all()
    assert result["lon"].notna().all()


def test_convert_to_local_coords_adds_x_y():
    obs_df = pd.DataFrame(
        {
            "rx": ["rx1"],
            "sv": ["G01"],
            "lat": [35.0],
            "lon": [-95.0],
        }
    )
    result = convert_to_local_coords(obs_df, [0.0, 90.0], [-180.0, 180.0], 350.0)

    assert "x" in result.columns
    assert "y" in result.columns
    assert len(result) == 1
    assert np.isfinite(result["x"].iloc[0])
    assert np.isfinite(result["y"].iloc[0])


def test_get_data_with_use_local_cs_false():
    obs, rx = load_observations(
        V2_FILES,
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )
    windows = make_time_windows(obs, 4, 2)
    out = get_data(
        obs, rx, windows[0], 350, [0.0, 90.0], [-180.0, 180.0], use_local_cs=False
    )

    assert out is not None
    assert "lat" in out.columns
    assert "lon" in out.columns
    assert "x" not in out.columns
    assert "y" not in out.columns


def test_get_data_returns_none_when_no_obs_in_window():
    obs, rx = load_observations(
        V2_FILES,
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )
    future_time = (
        np.datetime64("2099-01-01"),
        np.datetime64("2099-12-31"),
    )
    result = get_data(obs, rx, future_time, 350.0, [0.0, 90.0], [-180.0, 180.0])
    assert result is None
