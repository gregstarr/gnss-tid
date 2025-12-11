from pathlib import Path
from datetime import datetime
import logging

import h5py
import pandas
import xarray
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from .coords import Local2D, aer2ipp
from .utils import normalize_paths

logger = logging.getLogger(__name__)

class PointData:
    """loads observations from file, computes IPPs
    """

    def __init__(
        self,
        files: Path | list[Path],
        latitude_limits: list,
        longitude_limits: list,
        time_limits: list,
        file_type: str = "v2",
        tec_var: str = "dtec0",
        n_jobs: int = 16,
    ):
        self.latitude_limits = latitude_limits
        self.longitude_limits = longitude_limits
        if isinstance(time_limits[0], str):
            time_limits = [
                datetime.strptime(x, "%Y%m%d_%H%M%S") for x in time_limits
            ]

        files = normalize_paths(files)

        if n_jobs == 1:
            results = []
            for file in files:
                results.append(load_file(file, self.latitude_limits, self.longitude_limits, time_limits, tec_var))
        else:
            @delayed
            def fn(file):
                return load_file(file, self.latitude_limits, self.longitude_limits, time_limits, tec_var)

            logger.info("loading files")
            with tqdm_joblib(desc="loading files", total=len(files)):
                with Parallel(n_jobs=n_jobs) as pool:
                    results = pool(fn(f) for f in files)

        logger.info("combining files")
        rx_names = []
        rx_positions = []
        data = []
        for ii, (d, pos, name) in enumerate(filter(lambda x: x is not None, results)):
            rx_names.append(name)
            rx_positions.append(pos)
            d["rx"] = ("n", np.full(d.sizes["n"], ii))
            data.append(d)
        self.rx_names = np.stack(rx_names)
        self.rx_positions = np.stack(rx_positions, 0)
        self._data = xarray.concat(data, dim="n")
        self.times = np.unique(self._data.time)
        logger.info("data ready")

    def get_time_slices(self, window: int, step: int):
        n_times = self.times.shape[0]
        slices = [slice(i, i + window) for i in range(0, n_times - window, step)]
        times = [self.times[i] for i in range(0, n_times - window, step)]
        return slices, times
    
    def get_coord_center(self):
        return np.mean(self.latitude_limits), np.mean(self.longitude_limits)
    
    def get_data(self, time_slice: slice, h: float) -> xarray.Dataset | None:
        time_mask = np.in1d(self._data.time, self.times[time_slice])
        data = self._data.isel(n=time_mask)
        los = pandas.MultiIndex.from_arrays([data.rx.values, data.sv.values], names=["rx", "sv"])
        data = (
            data.assign_coords(los=("n", los))
            .groupby("los").mean()
            .assign_attrs(time=self.times[time_slice.start], height=h)
        )
        if data.az.size == 0:
             logger.warning("empty az data: %s", time_slice)
             return None
        # aer2ipp requires rx_positions and az/el to have corresponding dimensions
        ipp_lat, ipp_lon = aer2ipp(
            data.az.values,
            data.el.values,
            self.rx_positions[data.rx.values, :],
            h,
        )
        
        # now reshape to (time, rx-prn pair)
        data = (
            data.assign(lat=("los", ipp_lat), lon=("los", ipp_lon))
            .drop_vars(["az", "el"])
            .query(los=f"lat > {self.latitude_limits[0]} & lat < {self.latitude_limits[1]}")
            .query(los=f"lon > {self.longitude_limits[0]} & lon < {self.longitude_limits[1]}")
        )
        if data.lat.size == 0:
            logger.warning("empty lat data: %s", time_slice)
            return None
        local_coords = Local2D.from_geodetic(*self.get_coord_center(), h)
        x, y = local_coords.convert_from_spherical(data.lat.values, data.lon.values)
        data = (
            data.assign(x=(data.tec.dims, x), y=(data.tec.dims, y))
            .where(abs(data.tec) < abs(data.tec).quantile(.998), drop=True)
        )
        
        return data


def load_file(file, latitude_limits, longitude_limits, time_limits, tec_var) -> xarray.Dataset:
    """loads v2 (single rx) file

    dtec0; 5-min high-pass filter
    dtec1; 30-min high-pass filter
    dtec2; 60-min high-pass filter
    dtec3; 90-min high-pass filter
    dtecp; polynomial filter

    Args:
        file (str | Path): file path

    Returns:
        xarray.Dataset: filtered file data
    """
    f = xarray.open_dataset(file)
    time = f.time.values
    rx_position = f.position_geodetic
    rx_name = file.stem.split("_")[0]
    valid_rx = (
        (rx_position[0] <= latitude_limits[1] + 20) &
        (rx_position[0] >= latitude_limits[0] - 20) &
        (rx_position[1] <= longitude_limits[1] + 20) &
        (rx_position[1] >= longitude_limits[0] - 20)
    )
    valid_time = (
        (time >= np.datetime64(time_limits[0])) & 
        (time <= np.datetime64(time_limits[1]))
    )
    if not valid_rx or not valid_time.any():
        return

    tec = f[tec_var].values.astype(float)
    valid = valid_time & np.isfinite(tec)
    tec = tec[valid]
    az = f.az.values.astype(float)[valid]
    el = f.el.values.astype(float)[valid]
    tec_noise  = f.tec_sigma.values.astype(float)[valid]
    tec_snr  = f.snr.values.astype(float)[valid]
    sv = f.sv.values[valid]
    time = time[valid]
    
    data = xarray.Dataset(
        data_vars={
            "az": ("n", az),
            "el": ("n", el),
            "tec": ("n", tec),
            "tec_noise": ("n", tec_noise),
            "tec_snr": ("n", tec_snr),
            "sv": ("n", sv),
            "time": ("n", time),
        },
    )
    return data, rx_position, rx_name

