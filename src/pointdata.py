from pathlib import Path
from datetime import datetime

from numpy.typing import ArrayLike
import h5py
import pandas
import xarray
import numpy as np

from coords import Local2D, aer2ipp



class PointData:
    """loads observations from file, computes IPPs
    """

    def __init__(
        self,
        file: Path,
        latitude_limits: ArrayLike,
        longitude_limits: ArrayLike,
        time_limits: ArrayLike,
        missing_data_threshold: float = .9,
    ):
        self.latitude_limits = latitude_limits
        self.longitude_limits = longitude_limits
        self.time_limits = time_limits
        if isinstance(self.time_limits[0], str):
            self.time_limits = [
                datetime.strptime(x, "%Y%m%d_%H%M%S") for x in self.time_limits
            ]

        self.missing_data_threshold = missing_data_threshold

        self.az, self.el, self.tid, self.time, self.rx_positions = self.load_file(file)
        self.data = pandas.DataFrame()
    
    def load_file(self, file):
        with h5py.File(file) as f:
            az = f['az'][:]  # time x prn x rx
            el = f['el'][:]
            tid = f['res'][:]
            time = pandas.to_datetime(f['obstimes'][:], unit='s')
            rx_positions = f['rx_positions'][:]
        
        valid_rx = (
            (rx_positions[:, 0] <= self.latitude_limits[1] + 10) &
            (rx_positions[:, 0] >= self.latitude_limits[0] - 10) &
            (rx_positions[:, 1] <= self.longitude_limits[1] + 10) &
            (rx_positions[:, 1] >= self.longitude_limits[0] - 10)
        )
        valid_time = (time >= self.time_limits[0]) & (time <= self.time_limits[1])
        return (
            az[valid_time][:, :, valid_rx].astype(float),
            el[valid_time][:, :, valid_rx].astype(float),
            tid[valid_time][:, :, valid_rx].astype(float),
            time[valid_time],
            rx_positions[valid_rx]
        )
    
    def get_data_at_height(self, h) -> xarray.Dataset:
        # aer2ipp requires rx_positions and az/el to have corresponding dimensions
        ipp_lat, ipp_lon = aer2ipp(self.az, self.el, self.rx_positions, h)
        # now reshape to (time, rx-prn pair)
        n_times = ipp_lat.shape[0]
        ipp_lat = ipp_lat.reshape((n_times, -1))
        ipp_lon = ipp_lon.reshape((n_times, -1))
        tid = self.tid.reshape((n_times, -1))
        # filter all-nan pairs, out-of-bounds pairs
        mask = (
            np.isnan(ipp_lat) | 
            np.isnan(ipp_lon) | 
            np.isnan(tid) |
            (ipp_lat > self.latitude_limits[1]) |
            (ipp_lat < self.latitude_limits[0]) |
            (ipp_lon > self.longitude_limits[1]) |
            (ipp_lon < self.longitude_limits[0])
        )
        mask = ~np.all(mask, axis=0)
        ipp_lat = ipp_lat[:, mask]
        ipp_lon = ipp_lon[:, mask]
        tid = tid[:, mask]
        # filter mostly-nan times
        mask = np.isnan(ipp_lat) | np.isnan(ipp_lon) | np.isnan(tid)
        time_mask = np.mean(mask, axis=1) < self.missing_data_threshold
        ipp_lat = ipp_lat[time_mask]
        ipp_lon = ipp_lon[time_mask]
        tid = tid[time_mask]
        
        local_coords = Local2D.from_geodetic(
            np.mean(self.latitude_limits),
            np.mean(self.longitude_limits),
            h
        )
        x, y = local_coords.convert_from_spherical(ipp_lat, ipp_lon)

        data = xarray.Dataset(
            {
                "x": (["time", "los"], x),
                "y": (["time", "los"], y),
                "lat": (["time", "los"], ipp_lat),
                "lon": (["time", "los"], ipp_lon),
                "tid": (["time", "los"], tid),
            },
            coords={"time": self.time[time_mask]},
        )

        return data
