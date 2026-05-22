from __future__ import annotations

import logging
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

    lookup: dict[str, np.ndarray] = {
        "rx_name": np.array([rx_name], dtype=object),
        "rx_position": np.array([rx_position], dtype=float),
    }
    return obs, lookup


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

    lookup: dict[str, np.ndarray] = {
        "rx_name": rx_names,
        "rx_position": rx_positions,
    }
    return obs, lookup


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
        return {"rx_name": np.array([], dtype=object), "rx_position": np.array([])}

    rx_names = np.concatenate([lk["rx_name"] for lk in lookups])
    rx_positions = np.concatenate([lk["rx_position"] for lk in lookups])
    return {"rx_name": rx_names, "rx_position": rx_positions}


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
    latitude_limits,
    longitude_limits,
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
    load_args = (latitude_limits, longitude_limits, time_limits, el_min)

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
    df = pd.DataFrame(combined_obs)
    df = df.sort_values("time").reset_index(drop=True)

    lookup = _concat_lookups(lookups)
    df = _apply_tec_filtering_df(df, q_thresh)

    return df, lookup


def _extract_times(source) -> np.ndarray:
    """Extract unique time values from various input types.

    Args:
        source: pd.DataFrame, xr.Dataset, xr.DataArray, or array-like with time values

    Returns:
        Sorted array of unique time values

    Raises:
        ValueError: If source is xr.Dataset without time variable
    """
    if isinstance(source, pd.DataFrame):
        if "time" not in source.columns:
            raise ValueError("dataframe does not contain a time column")
        return np.unique(np.asarray(source["time"].values))
    if isinstance(source, xr.Dataset):
        if "time" not in source:
            raise ValueError("dataset does not contain a time variable")
        return np.unique(np.asarray(source["time"].values))
    if isinstance(source, xr.DataArray):
        return np.unique(np.asarray(source.values))
    return np.unique(np.asarray(source))


def get_time_slices(data_or_times, window: int, step: int, drop_incomplete: bool = True):
    """Generate time window slices for processing.

    Args:
        data_or_times: Dataset, DataArray, or array with time values
        window: Number of time points per window
        step: Step size between windows
        drop_incomplete: Whether to drop incomplete final windows

    Returns:
        Tuple of (list of slice objects, list of start times)

    Raises:
        ValueError: If window or step is not positive
    """
    times = _extract_times(data_or_times)
    n_times = times.shape[0]
    if window <= 0:
        raise ValueError("window must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if n_times == 0:
        return [], []

    slices = []
    slice_times = []
    start = 0
    while start < n_times:
        stop = start + window
        if stop <= n_times:
            slices.append(slice(start, stop))
            slice_times.append(times[start])
        elif not drop_incomplete:
            slices.append(slice(start, n_times))
            slice_times.append(times[start])
            break
        else:
            break
        start += step

    return slices, slice_times


def _get_lookup_value(
    data: pd.DataFrame | xr.Dataset, rx_values: np.ndarray
) -> np.ndarray:
    """Get receiver positions for given receiver names.

    Args:
        data: DataFrame with rx_lookup in attrs, or xr.Dataset with rx_position lookup
        rx_values: Array of receiver names

    Returns:
        Array of receiver positions (lat, lon, alt)
    """
    if isinstance(data, pd.DataFrame):
        lookup = data.attrs.get("rx_lookup")
        if lookup is None:
            raise ValueError("dataframe does not contain rx_lookup attribute")
        rx_names = lookup["rx_name"]
        rx_positions = lookup["rx_position"]
        pos_dict = {
            str(name): pos for name, pos in zip(rx_names, rx_positions, strict=True)
        }
        result = np.array(
            [
                pos_dict.get(str(rx), np.array([np.nan, np.nan, np.nan]))
                for rx in rx_values
            ]
        )
        return result
    else:
        lookup = data["rx_position"].sel(
            {
                RX_LOOKUP_DIM: xr.DataArray(
                    np.asarray(rx_values, dtype=object), dims=OBS_DIM
                )
            }
        )
        return np.asarray(lookup.values, dtype=float)


def collect_time_window(
    data: pd.DataFrame, time_slice: slice
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract observation DataFrame for a time slice.

    Args:
        data: DataFrame with observation data
        time_slice: Slice selecting time points

    Returns:
        Tuple of (DataFrame with observations, selected time values)

    Raises:
        ValueError: If dataframe missing time column
    """
    if "time" not in data.columns:
        raise ValueError("dataframe does not contain a time column")

    unique_times = np.unique(np.asarray(data["time"].values))
    selected_times = unique_times[time_slice]
    if selected_times.size == 0:
        return pd.DataFrame(), selected_times

    mask = np.isin(np.asarray(data["time"].values), selected_times)
    if not np.any(mask):
        return pd.DataFrame(), selected_times

    df = data.loc[mask].copy()
    return df, selected_times


def aggregate_by_receiver_satellite(df: pd.DataFrame) -> pd.DataFrame | None:
    """Aggregate observations by receiver-satellite pair using mean reduction.

    Args:
        df: DataFrame with observation data including time, rx, sv columns

    Returns:
        Aggregated DataFrame with rx and sv as grouping keys, or None if empty
    """
    grouped = (
        df.drop(columns=["time"], errors="ignore")
        .groupby(["rx", "sv"], as_index=False)
        .mean(numeric_only=True)
    )
    if grouped.empty:
        return None
    return grouped.sort_values(["rx", "sv"]).reset_index(drop=True)


def project_to_ipp(
    grouped: pd.DataFrame,
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
        grouped["az"].to_numpy(),
        grouped["el"].to_numpy(),
        rx_positions,
        h,
    )
    return grouped.assign(lat=lat, lon=lon)


def convert_to_local_coords(
    grouped: pd.DataFrame,
    latitude_limits: list,
    longitude_limits: list,
    h: float,
) -> pd.DataFrame:
    """Convert geodetic coordinates to local x/y system.

    Args:
        grouped: DataFrame with lat, lon columns
        latitude_limits: [min, max] latitude bounds
        longitude_limits: [min, max] longitude bounds
        h: Height for coordinate conversion

    Returns:
        DataFrame with added x, y columns
    """
    center_lat = float(np.mean(latitude_limits))
    center_lon = float(np.mean(longitude_limits))
    local_coords = Local2D.from_geodetic(center_lat, center_lon, h)
    x, y = local_coords.convert_from_spherical(
        grouped["lat"].to_numpy(),
        grouped["lon"].to_numpy(),
    )
    return grouped.assign(x=x, y=y)


def get_data(
    data: pd.DataFrame,
    time_slice: slice,
    h: float,
    lat_limits: list | None = None,
    lon_limits: list | None = None,
    use_local_cs: bool = True,
) -> pd.DataFrame | None:
    df, selected_times = collect_time_window(data, time_slice)
    if df.empty or selected_times.size == 0:
        return None

    LOGGER.info("points in time range: %s", len(df))
    LOGGER.info("time range: %s, %s", selected_times[0], selected_times[-1])

    grouped = aggregate_by_receiver_satellite(df)
    if grouped is None:
        return None

    rx_positions = _get_lookup_value(data, grouped["rx"].to_numpy())
    grouped = project_to_ipp(grouped, rx_positions, h)

    if not (lat_limits is None and lon_limits is None):
        valid_lat = (
            (grouped["lat"] > lat_limits[0]) & (grouped["lat"] < lat_limits[1])
            if lat_limits
            else np.ones(grouped.shape[0], bool)
        )
        valid_lon = (
            (grouped["lon"] > lon_limits[0]) & (grouped["lon"] < lon_limits[1])
            if lon_limits
            else np.ones(grouped.shape[0], bool)
        )
        grouped = grouped.loc[valid_lat & valid_lon]
    if grouped.empty:
        LOGGER.warning("empty lat data: %s", time_slice)
        return None

    if use_local_cs:
        grouped = convert_to_local_coords(grouped, lat_limits, lon_limits, h)

    return grouped
