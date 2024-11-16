from pathlib import Path
from datetime import datetime
from typing import Iterable

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
        files: Path | list[Path],
        latitude_limits: list,
        longitude_limits: list,
        time_limits: list,
        missing_data_threshold: float = .9,
        flatten_method: str = "mean",
    ):
        self.latitude_limits = latitude_limits
        self.longitude_limits = longitude_limits
        self.time_limits = time_limits
        if isinstance(self.time_limits[0], str):
            self.time_limits = [
                datetime.strptime(x, "%Y%m%d_%H%M%S") for x in self.time_limits
            ]

        self.missing_data_threshold = missing_data_threshold
        self.flatten_method = flatten_method

        if not isinstance(files, Iterable):
            files = [files]
        
        data = []
        for file in files:
            data.append(self.load_file(file))
        self._data = xarray.combine_by_coords(
            data,
            compat="override",
            data_vars="minimal",
            coords="minimal",
        )

    def get_time_slices(self, window: int, step: int):
        n_times = self._data.time.shape[0]
        for i in range(0, n_times - window, step):
            yield slice(i, i + window)

    def load_file(self, file):
        with h5py.File(file) as f:
            az = f['az'][:]  # time x prn x rx
            el = f['el'][:]
            tec = f['res'][:]
            time = pandas.to_datetime(f['obstimes'][:], unit='s')
            rx_positions = f['rx_positions'][:]
            rx_name = f["rx_name"][:]
        
        valid_rx = (
            (rx_positions[:, 0] <= self.latitude_limits[1] + 20) &
            (rx_positions[:, 0] >= self.latitude_limits[0] - 20) &
            (rx_positions[:, 1] <= self.longitude_limits[1] + 20) &
            (rx_positions[:, 1] >= self.longitude_limits[0] - 20)
        )
        valid_time = (time >= self.time_limits[0]) & (time <= self.time_limits[1])

        data = xarray.Dataset(
            data_vars={
                "az": (["time", "prn", "rx"], az[valid_time][:, :, valid_rx].astype(float)),
                "el": (["time", "prn", "rx"], el[valid_time][:, :, valid_rx].astype(float)),
                "tec": (["time", "prn", "rx"], tec[valid_time][:, :, valid_rx].astype(float)),
                "rx_position": (["rx", "geo"], rx_positions[valid_rx].astype(float)),
            },
            coords={
                "time": time[valid_time],
                "prn": np.arange(32),
                "rx": rx_name[valid_rx, 0].astype(str),
                "geo": ["lat", "lon", "alt"],
            },
        ).dropna("prn", how="all", subset=["tec"]).dropna("rx", how="all", subset=["tec"])
        return data
    
    def get_data(self, time_slice: slice, h: float) -> xarray.Dataset:
        data = (
            self._data.isel(time=time_slice)
            .dropna("prn", how="all", subset=["tec"])
            .dropna("rx", how="all", subset=["tec"])
            .dropna("prn", how="all", subset=["az"])
            .dropna("rx", how="all", subset=["az"])
            .mean(dim="time")
            .assign_attrs(time=self._data.time.values[0], height=h)
        )
        # aer2ipp requires rx_positions and az/el to have corresponding dimensions
        ipp_lat, ipp_lon = aer2ipp(data.az, data.el, data.rx_position, h)
        data["lat"] = (data.az.dims, ipp_lat)
        data["lon"] = (data.az.dims, ipp_lon)
        
        # now reshape to (time, rx-prn pair)
        data = (
            data.drop_dims(["geo"])
            .stack(los=("rx", "prn"))
            .dropna("los")
            .reset_index("los")
            .query(los=f"lat > {self.latitude_limits[0]}")
            .query(los=f"lat < {self.latitude_limits[1]}")
            .query(los=f"lon > {self.longitude_limits[0]}")
            .query(los=f"lon < {self.longitude_limits[1]}")
        )
        local_coords = Local2D.from_geodetic(
            np.mean(self.latitude_limits),
            np.mean(self.longitude_limits),
            h
        )
        x, y = local_coords.convert_from_spherical(data.lat.values, data.lon.values)
        data["x"] = (data.az.dims, x)
        data["y"] = (data.az.dims, y)

        return data
