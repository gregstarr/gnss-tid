from __future__ import annotations

from pathlib import Path

import numpy as np

from gnss_tid.pointdata import get_data, get_time_slices, load_observations

DATA_DIR = Path("tests/data")


def test_load_observations_v2_fixture_returns_canonical_table():
    df = load_observations(
        DATA_DIR / "zeeb.nc",
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )

    assert len(df) == 520
    assert "rx_lookup" in df.attrs
    assert "stec" in df.columns
    assert "dtec1" in df.columns
    assert "tec_noise" in df.columns
    assert "tec_snr" in df.columns
    assert "roti" in df.columns
    assert df["stec"].notna().any()
    assert df["dtec1"].notna().any()
    assert df["tec_noise"].notna().any()
    assert df["tec_snr"].notna().any()
    assert df["roti"].notna().any()


def test_load_observations_legacy_fixture_maps_res_to_dtec1():
    df = load_observations(
        DATA_DIR / "legacy.h5",
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )

    assert len(df) == 17923
    assert "rx_lookup" in df.attrs
    assert df["dtec1"].notna().any()
    assert df["stec"].isna().all()
    assert df["dtec0"].isna().all()


def test_get_time_slices_can_keep_or_drop_incomplete_windows():
    times = np.array(
        [
            "2020-01-01T00:00:00",
            "2020-01-01T00:01:00",
            "2020-01-01T00:02:00",
            "2020-01-01T00:03:00",
            "2020-01-01T00:04:00",
        ],
        dtype="datetime64[m]",
    )

    slices_drop, times_drop = get_time_slices(times, 3, 2, drop_incomplete=True)
    slices_keep, times_keep = get_time_slices(times, 3, 2, drop_incomplete=False)

    assert slices_drop == [slice(0, 3), slice(2, 5)]
    assert times_drop == [times[0], times[2]]
    assert slices_keep == [slice(0, 3), slice(2, 5), slice(4, 5)]
    assert times_keep == [times[0], times[2], times[4]]


def test_get_data_returns_none_for_empty_window():
    df = load_observations(
        DATA_DIR / "zeeb.nc",
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )

    assert get_data(df, slice(10_000, 10_001), 350, [0.0, 90.0], [-180.0, 180.0]) is None


def test_get_data_aggregates_and_projects_v2_fixture():
    df = load_observations(
        DATA_DIR / "zeeb.nc",
        [0.0, 90.0],
        [-180.0, 180.0],
        ["20150324_000000", "20150327_000000"],
        n_jobs=1,
        pbar=False,
    )
    slices, _ = get_time_slices(df, 4, 2)

    out = get_data(df, slices[0], 350, [0.0, 90.0], [-180.0, 180.0])

    assert out is not None
    assert len(out) > 0
    assert "lat" in out.columns
    assert "lon" in out.columns
    assert "rx" in out.columns
    assert "sv" in out.columns
