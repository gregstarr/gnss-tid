from pathlib import Path
from datetime import datetime
import logging

import pandas
import xarray
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from .coords import Local2D, aer2ipp
from .utils import normalize_paths


class PointData:
    """loads observations from file, computes IPPs
    """

    def __init__(
        self,
        files: Path | list[Path],
        latitude_limits: list,
        longitude_limits: list,
        time_limits: list,
        el_min: float = 0,
        q_thresh: float = .99,
        noise_max: float = 100,
        n_jobs: int = 16,
        pbar: bool = True,
    ):
        self.latitude_limits = latitude_limits
        self.longitude_limits = longitude_limits
        self.q_thresh = q_thresh
        if isinstance(time_limits[0], str):
            time_limits = [
                datetime.strptime(x, "%Y%m%d_%H%M%S") for x in time_limits
            ]

        files = normalize_paths(files)

        def load_file_safe(file):
            try:
                return load_file(
                    file,
                    self.latitude_limits,
                    self.longitude_limits,
                    time_limits,
                    el_min,
                    noise_max,
                )
            except Exception:
                logging.exception("failed to load file %s", file)
                return None

        if n_jobs == 1:
            results = []
            for file in files:
                results.append(load_file_safe(file))
        else:
            @delayed
            def fn(file):
                return load_file_safe(file)
            if pbar:
                logging.info("loading files")
                with tqdm_joblib(desc="loading files", total=len(files)):
                    with Parallel(n_jobs=n_jobs) as pool:
                        results = pool(fn(f) for f in files)
            else:
                with Parallel(n_jobs=n_jobs) as pool:
                    results = pool(fn(f) for f in files)

        n_rejected = sum([1 for r in results if r is None])
        logging.info(f"files {n_rejected = }")
        logging.info(f"file {100*n_rejected/len(files):.2f}% reject rate")
        logging.info("combining files")
        rx_names = []
        rx_positions = []
        data = []
        debug_table = {}

        for ii, (d, pos, name, dbg) in enumerate(filter(lambda x: x is not None, results)):
            rx_names.append(name)
            rx_positions.append(pos)
            d["rx"] = ("n", np.full(d.sizes["n"], ii))
            data.append(d)
            debug_table[name] = dbg

        self.debug_table = pandas.DataFrame.from_dict(debug_table, orient="index")

        self.rx_names = np.stack(rx_names)
        self.rx_positions = np.stack(rx_positions, 0)
        self._data = xarray.concat(data, dim="n")
        los = pandas.MultiIndex.from_arrays(
            [self._data.rx.values, self._data.sv.values],
            names=["rx", "sv"],
        )
        los_id, self.unique_los = pandas.factorize(los)
        self._data = self._data.assign_coords(los_id=("n", los_id))
        self.times = np.unique(self._data.time)
        
        logging.info("data ready")

    def get_time_slices(self, window: int, step: int):
        n_times = self.times.shape[0]
        slices = [slice(i, i + window) for i in range(0, n_times - window, step)]
        times = [self.times[i] for i in range(0, n_times - window, step)]
        return slices, times
    
    def get_coord_center(self):
        return np.mean(self.latitude_limits), np.mean(self.longitude_limits)
    
    def get_data(self, time_slice: slice, h: float, use_local_cs: bool = True) -> xarray.Dataset | None:
        """gets TEC data of time window at IPP height=h

        Args:
            time_slice (slice): slice object over time slices, see 
                `PointData.get_time_slices`
            h (float): IPP height km
            use_local_cs (bool, optional): convert to local 2D coordinates. Defaults to True.

        Returns:
            xarray.Dataset
            Dimensions: (los_id, )
            Data variables:
                az:         float64
                el:         float64
                dtec0:      float64
                dtec1:      float64
                dtec2:      float64
                dtec3:      float64
                dtecp:      float64
                tec_noise:  float64
                tec_snr:    float64
                rx:         int64
                sv:         object
                lat:        float32
                lon:        float32
                [[if use_local_cs]]
                x:          float32
                y:          float32
        """
        time_mask = np.isin(self._data.time.values, self.times[time_slice])
        logging.info(f"points in time range: {time_mask.sum()}")
        t = self.times[time_slice]
        logging.info(f"time range: {t[0]}, {t[-1]}")

        data = (
            self._data.isel(n=time_mask)
            .drop_vars(["sv", "time", "rx"])
            .groupby("los_id").mean()
            .assign_attrs(time=self.times[time_slice.start], height=h)
        )
        for v in ["dtec0", "dtec1", "dtec3", "dtecp"]:
            z = abs(data[v])
            data[v] = data[v].where(z < z.quantile(self.q_thresh).drop_vars("quantile"))

        if data.az.size == 0:
             logging.warning("empty az data: %s", time_slice)
             return None
        # aer2ipp requires rx_positions and az/el to have corresponding dimensions
        data["rx"] = ("los_id", self.unique_los.get_level_values(0).values[data.los_id])
        data["sv"] = ("los_id", self.unique_los.get_level_values(1).values[data.los_id])
        ipp_lat, ipp_lon = aer2ipp(
            data.az.values,
            data.el.values,
            self.rx_positions[data.rx.values, :],
            h,
        )
        
        data = (
            data.assign(lat=("los_id", ipp_lat), lon=("los_id", ipp_lon))
            .query(los_id=f"lat > {self.latitude_limits[0]} & lat < {self.latitude_limits[1]}")
            .query(los_id=f"lon > {self.longitude_limits[0]} & lon < {self.longitude_limits[1]}")
        )
        if data.lat.size == 0:
            logging.warning("empty lat data: %s", time_slice)
            return None
        
        if use_local_cs:
            local_coords = Local2D.from_geodetic(*self.get_coord_center(), h)
            x, y = local_coords.convert_from_spherical(data.lat.values, data.lon.values)
            data = data.assign_coords(x=("los_id", x), y=("los_id", y))
        return data


def load_file(
        file,
        latitude_limits,
        longitude_limits,
        time_limits,
        el_min,
        noise_max,
    ) -> xarray.Dataset:
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

    el = f.el.values.astype(float)
    az = f.az.values.astype(float)
    sv = f.sv.values
    tec_noise  = f.tec_sigma.values.astype(float)
    
    fin_mask = np.isfinite(time) & np.isfinite(az) & np.isfinite(el)
    valid_el = el >= el_min
    valid_noise = tec_noise <= noise_max
    valid = fin_mask & valid_time & valid_el# & valid_noise
    
    az = az[valid]
    el = el[valid]
    time = time[valid]
    tec_noise  = tec_noise[valid]
    dtec0 = f.dtec0.values.astype(float)[valid]
    dtec1 = f.dtec1.values.astype(float)[valid]
    dtec2 = f.dtec2.values.astype(float)[valid]
    dtec3 = f.dtec3.values.astype(float)[valid]
    dtecp = f.dtecp.values.astype(float)[valid]
    roti = f.roti.values.astype(float)[valid]
    tec_snr  = f.snr.values.astype(float)[valid]
    sv = sv[valid]
    
    data = xarray.Dataset(
        data_vars={
            "az": ("n", az),
            "el": ("n", el),
            "dtec0": ("n", dtec0),
            "dtec1": ("n", dtec1),
            "dtec2": ("n", dtec2),
            "dtec3": ("n", dtec3),
            "dtecp": ("n", dtecp),
            "roti": ("n", roti),
            "tec_noise": ("n", tec_noise),
            "tec_snr": ("n", tec_snr),
            "sv": ("n", sv),
            "time": ("n", time),
        },
    )
    
    debug = dict(
        valid_mean=valid.mean(),
        valid_sum=valid.sum(),
        invalid_sum=(~valid).mean(),
        fin_mask_mean=fin_mask.mean(),
        fin_mask_sum=fin_mask.sum(),
        valid_time_mean=valid_time.mean(),
        valid_time_sum=valid_time.sum(),
        invalid_time_sum=(~valid_time).sum(),
        valid_el_mean=valid_el.mean(),
        valid_el_sum=valid_el.sum(),
        invalid_el_sum=(~valid_el).sum(),
        valid_noise_mean=valid_noise.mean(),
        valid_noise_sum=valid_noise.sum(),
        invalid_noise_sum=(~valid_noise).sum(),
        unique_valid_svs=np.unique(sv),
        unique_svs=np.unique(f.sv.values),
    )
    for v in ["dtec0", "dtec1", "dtec2", "dtec3", "dtecp", "roti"]:
        debug[f"{v}_null_sum"] = data[v].isnull().sum().item()
        debug[f"{v}_null_mean"] = data[v].isnull().mean().item()

    return data, rx_position, rx_name, debug