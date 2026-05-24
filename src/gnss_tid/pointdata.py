from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from .coords import Local2D, aer2ipp
from .utils import normalize_paths

LOGGER = logging.getLogger(__name__)
OBS_DIM = "obs"
RX_LOOKUP_DIM = "rx"
GEO_DIM = "geo"
TEC_FILTER_FIELDS = ("stec", "dtec0", "dtec1", "dtec2", "dtec3", "dtecp")
OBS_TIME_FIELDS = ("time", "rx", "sv", "az", "el")
LOOKUP_GEO = ("lat", "lon", "alt")
OBS_COLUMNS = (
    "time",
    "rx",
    "sv",
    "az",
    "el",
    "stec",
    "dtec0",
    "dtec1",
    "dtec2",
    "dtec3",
    "dtecp",
    "roti",
    "tec_noise",
    "tec_snr",
)


@dataclass
class TimeWindow:
    """Time window with pre-computed row indices into the observation DataFrame."""

    start_time: np.datetime64
    end_time: np.datetime64
    row_indices: np.ndarray


def _normalize_time_value(value) -> datetime | np.datetime64:
    """Normalize a time value to datetime or np.datetime64.

    Args:
        value: Time value as string, pd.Timestamp, np.datetime64, or datetime

    Returns:
        Normalized datetime or np.datetime64 object
    """
    if isinstance(value, str):
        return datetime.strptime(value, "%Y%m%d_%H%M%S")
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.datetime64):
        return value
    return value


def _normalize_time_limits(
    time_limits,
) -> tuple[datetime | np.datetime64, datetime | np.datetime64]:
    """Normalize time limits to a tuple of start and end datetime values.

    Args:
        time_limits: Sequence of two time values

    Returns:
        Tuple of (start_time, end_time) as datetime or np.datetime64

    Raises:
        ValueError: If time_limits does not contain exactly two values
    """
    if len(time_limits) != 2:
        raise ValueError("time_limits must contain exactly two values")
    return _normalize_time_value(time_limits[0]), _normalize_time_value(time_limits[1])


def _as_array(values):
    """Convert values to a flattened numpy array.

    Args:
        values: Input values (list, array, or array-like)

    Returns:
        Flattened numpy array
    """
    arr = np.asarray(values)
    return arr.reshape(-1)


def _decode_strings(values) -> np.ndarray:
    """Decode string values from bytes or convert to string objects.

    Args:
        values: Input values that may be bytes, np.bytes_, or string types

    Returns:
        Array of decoded string objects
    """
    arr = np.asarray(values).reshape(-1)
    decoded = []
    for item in arr:
        if isinstance(item, (bytes, np.bytes_)):
            decoded.append(item.decode())
        else:
            decoded.append(str(item))
    return np.asarray(decoded, dtype=object)


def _broadcast_flat(values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Broadcast values to a shape and flatten.

    Args:
        values: Input array to broadcast
        shape: Target shape (time, sv, rx)

    Returns:
        Flattened broadcast array
    """
    return np.broadcast_to(values, shape).ravel()


def _ensure_columns(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ensure all expected columns exist, filling missing with NaN arrays.

    Args:
        obs: Dictionary of observation arrays

    Returns:
        Dictionary with all OBS_COLUMNS present
    """
    if not obs:
        return obs
    n = len(next(iter(obs.values())))
    for col in OBS_COLUMNS:
        if col not in obs:
            obs[col] = np.full(n, np.nan, dtype=float)
    return obs


def _detect_file_kind(file: Path) -> str:
    """Detect the format version of an observation file.

    Args:
        file: Path to observation file

    Returns:
        Format version string ("v1" or "v2")

    Raises:
        ValueError: If file format is not supported
    """
    with h5py.File(file, "r") as handle:
        keys = set(handle.keys())
        if {"flat", "time", "sv"}.issubset(keys) and "stec" in keys:
            return "v2"
        if {"az", "el", "res", "rx_positions", "rx_name", "obstimes"}.issubset(keys):
            return "v1"
    raise ValueError(f"unsupported observation file format: {file}")


def _load_v2_file(
    file: Path,
    latitude_limits,
    longitude_limits,
    time_limits,
    el_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Load observation data from a v2 format file.

    Args:
        file: Path to v2 format observation file
        latitude_limits: (min, max) latitude bounds
        longitude_limits: (min, max) longitude bounds
        time_limits: (start, end) time bounds
        el_min: Minimum elevation angle

    Returns:
        Tuple of (observation dict, lookup dict), or None if no valid data
    """
    f = xr.open_dataset(file)
    if "position_geodetic" not in f.attrs and "position_geodetic" not in f:
        raise ValueError(f"missing receiver position metadata in {file}")

    rx_position = f.attrs.get("position_geodetic")
    if rx_position is None:
        rx_position = f["position_geodetic"].values
    rx_position = np.asarray(rx_position, dtype=float).reshape(-1)[:3]
    rx_name = file.stem.split("_")[0]

    time = _as_array(f["time"].values)
    az = _as_array(f["az"].values).astype(float)
    el = _as_array(f["el"].values).astype(float)
    sv = _decode_strings(f["sv"].values)

    lat_min, lat_max = latitude_limits
    lon_min, lon_max = longitude_limits
    valid_rx = (
        (rx_position[0] <= lat_max + 20)
        and (rx_position[0] >= lat_min - 20)
        and (rx_position[1] <= lon_max + 20)
        and (rx_position[1] >= lon_min - 20)
    )

    time_start = np.datetime64(time_limits[0])
    time_end = np.datetime64(time_limits[1])
    valid_time = (time >= time_start) & (time <= time_end)
    valid = np.isfinite(time) & np.isfinite(az) & np.isfinite(el) & valid_time
    valid &= el >= el_min

    if not valid_rx or not valid.any():
        return None

    n = valid.sum()
    obs: dict[str, np.ndarray] = {
        "time": time[valid],
        "rx": np.full(n, rx_name, dtype=object),
        "sv": sv[valid],
        "az": az[valid],
        "el": el[valid],
    }

    for field in TEC_FILTER_FIELDS:
        obs[field] = _as_array(f[field].values).astype(float)[valid]

    obs["tec_noise"] = _as_array(f["tec_sigma"].values).astype(float)[valid]
    obs["tec_snr"] = _as_array(f["snr"].values).astype(float)[valid]
    obs["roti"] = _as_array(f["roti"].values).astype(float)[valid]
    obs = _ensure_columns(obs)

    rx: dict[str, np.ndarray] = {
        "rx": np.array([rx_name], dtype=object),
        LOOKUP_GEO[0]: rx_position[0],
        LOOKUP_GEO[1]: rx_position[1],
        LOOKUP_GEO[2]: rx_position[2],
    }
    return obs, rx


def _load_v1_file(
    file: Path,
    latitude_limits,
    longitude_limits,
    time_limits,
    el_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Load observation data from a v1 format file.

    Args:
        file: Path to v1 format observation file
        latitude_limits: (min, max) latitude bounds
        longitude_limits: (min, max) longitude bounds
        time_limits: (start, end) time bounds
        el_min: Minimum elevation angle

    Returns:
        Tuple of (observation dict, lookup dict), or None if no valid data
    """
    with h5py.File(file, "r") as handle:
        az = np.asarray(handle["az"][:], dtype=float)
        el = np.asarray(handle["el"][:], dtype=float)
        res = np.asarray(handle["res"][:], dtype=float)
        time = pd.to_datetime(handle["obstimes"][:], unit="s").to_numpy()
        rx_positions = np.asarray(handle["rx_positions"][:], dtype=float)
        rx_names = _decode_strings(handle["rx_name"][:])

    lat_min, lat_max = latitude_limits
    lon_min, lon_max = longitude_limits
    valid_rx = (
        (rx_positions[:, 0] <= lat_max + 20)
        & (rx_positions[:, 0] >= lat_min - 20)
        & (rx_positions[:, 1] <= lon_max + 20)
        & (rx_positions[:, 1] >= lon_min - 20)
    )

    time_start = np.datetime64(time_limits[0])
    time_end = np.datetime64(time_limits[1])
    valid_time = (time >= time_start) & (time <= time_end)

    if not valid_rx.any() or not valid_time.any():
        return None

    az = az[valid_time][:, :, valid_rx]
    el = el[valid_time][:, :, valid_rx]
    res = res[valid_time][:, :, valid_rx]
    time = time[valid_time]
    rx_positions = rx_positions[valid_rx]
    rx_names = rx_names[valid_rx]

    valid = np.isfinite(az) & np.isfinite(el) & np.isfinite(res)
    valid &= el >= el_min

    if not valid.any():
        return None

    shape = az.shape
    time_flat = _broadcast_flat(time[:, None, None], shape)
    sv_flat = _broadcast_flat(np.arange(shape[1], dtype=int)[None, :, None], shape)
    rx_flat = _broadcast_flat(rx_names[None, None, :], shape)
    obs: dict[str, np.ndarray] = {
        "time": time_flat[valid.ravel()],
        "rx": rx_flat[valid.ravel()],
        "sv": sv_flat[valid.ravel()],
        "az": az.ravel()[valid.ravel()],
        "el": el.ravel()[valid.ravel()],
        "dtec1": res.ravel()[valid.ravel()],
    }
    obs = _ensure_columns(obs)

    rx: dict[str, np.ndarray] = {
        "rx": rx_names,
        LOOKUP_GEO[0]: rx_positions[:, 0],
        LOOKUP_GEO[1]: rx_positions[:, 1],
        LOOKUP_GEO[2]: rx_positions[:, 2],
    }
    return obs, rx


def _load_file(
    file: Path,
    lat_limits,
    lon_limits,
    time_limits,
    el_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Load observation data from a file, auto-detecting format version.

    Args:
        file: Path to observation file
        latitude_limits: (min, max) latitude bounds
        longitude_limits: (min, max) longitude bounds
        time_limits: (start, end) time bounds
        el_min: Minimum elevation angle

    Returns:
        Tuple of (observation dict, lookup dict), or None if no valid data

    Raises:
        ValueError: If file format is not supported
    """
    kind = _detect_file_kind(file)
    try:
        if kind == "v2":
            return _load_v2_file(file, lat_limits, lon_limits, time_limits, el_min)
        if kind == "v1":
            return _load_v1_file(file, lat_limits, lon_limits, time_limits, el_min)
        raise ValueError(f"unsupported observation file format: {file}")
    except Exception:
        LOGGER.exception("failed to load file %s", file)


def _concat_obs_arrays(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate observation arrays from all files using numpy.

    Args:
        results: List of observation dictionaries from each file

    Returns:
        Combined dictionary with all observations concatenated

    Raises:
        ValueError: If results list is empty
    """
    if not results:
        raise ValueError("no results to concatenate")

    combined = {}
    for key in OBS_COLUMNS:
        arrays = [r[key] for r in results]
        combined[key] = np.concatenate(arrays)
    return combined


def _concat_lookups(lookups: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate receiver lookup dictionaries from all files.

    Args:
        lookups: List of lookup dictionaries from each file

    Returns:
        Combined lookup dictionary
    """
    if not lookups:
        return {
            "rx": np.array([], dtype=object),
            LOOKUP_GEO[0]: np.array([]),
            LOOKUP_GEO[1]: np.array([]),
            LOOKUP_GEO[2]: np.array([]),
        }

    result = {"rx": np.concatenate([lk["rx"] for lk in lookups])}
    if np.isscalar(lookups[0][LOOKUP_GEO[0]]):
        for g in LOOKUP_GEO:
            result[g] = np.array([lk[g] for lk in lookups])
    else:
        for g in LOOKUP_GEO:
            result[g] = np.concatenate([lk[g] for lk in lookups])
    return result


def _apply_tec_filtering_df(df: pd.DataFrame, q_thresh: float) -> pd.DataFrame:
    """Apply TEC filtering by masking outliers in DataFrame.

    Args:
        df: DataFrame with TEC fields
        q_thresh: Quantile threshold for filtering (0-1)

    Returns:
        Filtered DataFrame with TEC values masked above threshold
    """
    filtered = df.copy()
    for field in TEC_FILTER_FIELDS:
        if field not in filtered.columns:
            continue
        values = filtered[field]
        q = np.nanquantile(np.abs(values), q_thresh)
        if np.isnan(q):
            continue
        filtered.loc[values.abs() >= q, field] = np.nan
    return filtered


def load_observations(
    files: Path | list[Path],
    lat_limits,
    lon_limits,
    time_limits,
    el_min: float = 0,
    q_thresh: float = 0.99,
    n_jobs: int = 16,
    pbar: bool = True,
) -> pd.DataFrame:
    """Load GNSS observation data from files.

    Args:
        files: Path or list of paths to observation files
        latitude_limits: [min, max] latitude bounds
        longitude_limits: [min, max] longitude bounds
        time_limits: [start, end] time bounds
        el_min: Minimum elevation angle filter
        q_thresh: Quantile threshold for TEC filtering (default 0.99)
        noise_max: Maximum noise threshold (unused, retained for API compatibility)
        n_jobs: Number of parallel jobs for loading (default 16)
        pbar: Whether to show progress bar (default True)

    Returns:
        Tuple of (combined pandas DataFrame with observation data,
                 lookup dict with rx_name and rx_position arrays)

    Raises:
        ValueError: If no observation files provided or no valid data loaded
    """

    paths = normalize_paths(files)
    if len(paths) == 0:
        raise ValueError("no observation files were provided")

    time_limits = _normalize_time_limits(time_limits)
    load_args = (lat_limits, lon_limits, time_limits, el_min)

    if n_jobs == 1:
        results = [_load_file(file, *load_args) for file in paths]
    else:
        if pbar:
            LOGGER.info("loading files")
            with (
                tqdm_joblib(desc="loading files", total=len(paths)),
                Parallel(n_jobs=n_jobs) as pool,
            ):
                results = pool(delayed(_load_file)(f, *load_args) for f in paths)
        else:
            with Parallel(n_jobs=n_jobs) as pool:
                results = pool(delayed(_load_file)(f, *load_args) for f in paths)

    results = [result for result in results if result is not None]
    if not results:
        raise ValueError("no valid observations could be loaded")

    obs_dicts = [result[0] for result in results]
    lookups = [result[1] for result in results]

    combined_obs = _concat_obs_arrays(obs_dicts)
    obs = pd.DataFrame(combined_obs)
    obs = obs.sort_values("time").reset_index(drop=True)

    rx = _concat_lookups(lookups)
    rx = pd.DataFrame(rx)
    obs = _apply_tec_filtering_df(obs, q_thresh)

    return obs, rx


def make_time_windows(
    times: pd.DataFrame | np.ndarray,
    window: int,
    step: int,
    drop_incomplete: bool = True,
) -> list[TimeWindow]:
    if isinstance(times, pd.DataFrame):
        if "time" not in times.columns:
            raise ValueError("dataframe does not contain a time column")
        time_values = np.asarray(times["time"])
    else:
        time_values = np.asarray(times)
    unique_times, inverse = np.unique(time_values, return_inverse=True)
    n_unique = len(unique_times)
    if n_unique == 0:
        return []
    sorted_idx = np.argsort(inverse)
    _, group_sizes = np.unique(inverse, return_counts=True)
    split_pts = np.cumsum(group_sizes)[:-1]
    time_to_indices = dict(
        zip(unique_times, np.split(sorted_idx, split_pts), strict=True)
    )
    if window <= 0 or step <= 0:
        return []
    windows: list[TimeWindow] = []
    start = 0
    while start < n_unique:
        stop = start + window
        if stop <= n_unique:
            wtimes = unique_times[start:stop]
            row_idx = np.concatenate([time_to_indices[t] for t in wtimes])
            windows.append(TimeWindow(wtimes[0], wtimes[-1], row_idx))
        elif not drop_incomplete:
            wtimes = unique_times[start:]
            row_idx = np.concatenate([time_to_indices[t] for t in wtimes])
            windows.append(TimeWindow(wtimes[0], wtimes[-1], row_idx))
            break
        else:
            break
        start += step
    return windows


def lookup_rx_location(obs: pd.DataFrame, rx: pd.DataFrame) -> np.ndarray:
    """Get receiver positions for given receiver names.

    Args:
        obs: DataFrame with rx column
        rx: Array of receiver names

    Returns:
        Array of receiver positions (lat, lon, alt)
    """
    result = pd.merge(obs, rx, on="rx", how="left")
    return result.loc[:, ["rx", *LOOKUP_GEO]]


def aggregate_by_receiver_satellite(obs: pd.DataFrame) -> pd.DataFrame | None:
    """Aggregate observations by receiver-satellite pair using mean reduction.

    Args:
        df: DataFrame with observation data including time, rx, sv columns

    Returns:
        Aggregated DataFrame with rx and sv as grouping keys, or None if empty
    """
    grouped = (
        obs.drop(columns=["time"], errors="ignore")
        .groupby(["rx", "sv"], as_index=False)
        .mean(numeric_only=True)
    )
    if grouped.empty:
        return None
    return grouped.sort_values(["rx", "sv"]).reset_index(drop=True)


def project_to_ipp(
    obs: pd.DataFrame,
    rx_positions: np.ndarray,
    h: float,
) -> pd.DataFrame:
    """Project observations to Ionospheric Piercing Points.

    Args:
        grouped: Aggregated DataFrame with az, el columns
        rx_positions: Receiver positions array
        h: Height for IPP projection

    Returns:
        DataFrame with added lat, lon columns
    """
    lat, lon = aer2ipp(
        obs["az"].to_numpy(),
        obs["el"].to_numpy(),
        rx_positions,
        h,
    )
    return obs.assign(lat=lat, lon=lon)


def convert_to_local_coords(
    grouped: pd.DataFrame,
    proj: Local2D,
) -> pd.DataFrame:
    """Project geodetic coordinates into a shared local x/y frame.

    Args:
        grouped: DataFrame with ``lat`` and ``lon`` columns.
        proj: Local cartesian projection (typically constructed once at the
            orchestration layer so every slice and the image grid share a
            single frame).

    Returns:
        DataFrame with added ``x`` and ``y`` columns (km).
    """
    x, y = proj.convert_from_spherical(
        grouped["lat"].to_numpy(),
        grouped["lon"].to_numpy(),
    )
    return grouped.assign(x=x, y=y)


def get_obs_in_window(
    obs: pd.DataFrame, time_spec: tuple[np.datetime64, np.datetime64] | TimeWindow
):
    if isinstance(time_spec, TimeWindow):
        df = obs.iloc[time_spec.row_indices].copy()
    elif isinstance(time_spec, tuple):
        start_time, end_time = time_spec
        mask = (obs["time"] >= start_time) & (obs["time"] <= end_time)
        df = obs.loc[mask].copy()
    else:
        raise TypeError(
            f"get_data() time_spec must be a tuple (start_time, end_time) or TimeWindow, "
            f"not {type(time_spec).__name__}"
        )
    LOGGER.info("points in time range: %s", len(df))
    if df.empty:
        return
    return df


def get_data(
    obs: pd.DataFrame,
    rx: pd.DataFrame,
    time_spec: tuple[np.datetime64, np.datetime64] | TimeWindow,
    h: float,
    lat_limits: list | None = None,
    lon_limits: list | None = None,
    proj: Local2D | None = None,
) -> pd.DataFrame | None:
    """Return a per-slice observation DataFrame, optionally projected to local CS.

    When ``proj`` is provided, the slice's geodetic (lat, lon) are projected
    into ``proj``'s shared local cartesian frame (adding ``x``, ``y`` columns).
    When ``proj`` is ``None``, the local projection step is skipped and only
    geodetic coordinates are returned.
    """
    df = get_obs_in_window(obs, time_spec)
    if df is None:
        return
    df = aggregate_by_receiver_satellite(df)
    if df is None:
        return

    rx_positions = lookup_rx_location(df, rx)
    df = project_to_ipp(df, rx_positions.loc[:, LOOKUP_GEO].values, h)

    if not (lat_limits is None and lon_limits is None):
        valid_lat = (
            (df["lat"] > lat_limits[0]) & (df["lat"] < lat_limits[1])
            if lat_limits
            else np.ones(df.shape[0], bool)
        )
        valid_lon = (
            (df["lon"] > lon_limits[0]) & (df["lon"] < lon_limits[1])
            if lon_limits
            else np.ones(df.shape[0], bool)
        )
        df = df.loc[valid_lat & valid_lon]
    if df.empty:
        LOGGER.warning("empty lat data: %s", time_spec)
        return None

    if proj is not None:
        df = convert_to_local_coords(df, proj)

    return df
