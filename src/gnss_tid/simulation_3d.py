from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import georinex
import warnings
import numpy as np
import pandas as pd
import pymap3d

from gnss_tid.coords import ecef2spherical, spherical2ecef
from gnss_tid.pointdata import OBS_COLUMNS

# GPS Constants
MU = 3.986005e14  # Earth's gravitational constant, m^3/s^2
OMEGA_E = 7.2921151467e-5  # Earth's rotation rate, rad/s


def calculate_gps_position(t: datetime, ephem: dict[str, Any]) -> np.ndarray:
    """
    Calculate GPS satellite ECEF position at time t using orbital parameters.
    Implements the GPS ICD Table 20-IV algorithm.

    Parameters
    ----------
    t : datetime
        The time for which to calculate the position.
    ephem : Dict[str, Any]
        Dictionary containing ephemeris parameters.

    Returns
    -------
    pos_ecef : np.ndarray
        ECEF position [x, y, z] in km.
    """
    tk = (t - ephem["toe"]).total_seconds()

    a = ephem["sqrtA"] ** 2
    n = np.sqrt(MU / a**3) + ephem["deltaN"]

    # All angles from RINEX are already in radians
    Mk = ephem["M0"] + n * tk
    Ek = Mk
    for _ in range(10):
        Ek = Mk + ephem["e"] * np.sin(Ek)

    nu_k = 2 * np.arctan2(
        np.sqrt(1 + ephem["e"]) * np.sin(Ek / 2),
        np.sqrt(1 - ephem["e"]) * np.cos(Ek / 2),
    )

    phi_k = nu_k + ephem["omega"]

    delta_uk = ephem["Cus"] * np.sin(2 * phi_k) + ephem["Cuc"] * np.cos(2 * phi_k)
    delta_rk = ephem["Crs"] * np.sin(2 * phi_k) + ephem["Crc"] * np.cos(2 * phi_k)
    delta_ik = ephem["Cis"] * np.sin(2 * phi_k) + ephem["Cic"] * np.cos(2 * phi_k)

    uk = phi_k + delta_uk
    rk = a * (1 - ephem["e"] * np.cos(Ek)) + delta_rk
    ik = ephem["i0"] + ephem["idot"] * tk + delta_ik

    xk_prime = rk * np.cos(uk)
    yk_prime = rk * np.sin(uk)

    # Corrected longitude of ascending node (GPS ICD eq. includes Earth-rotation at toe)
    omega_k = (
        ephem["Omega0"]
        + (ephem["OmegaDot"] - OMEGA_E) * tk
        - OMEGA_E * ephem["toe_gps"]
    )

    x_ecef = xk_prime * np.cos(omega_k) - yk_prime * np.cos(ik) * np.sin(omega_k)
    y_ecef = xk_prime * np.sin(omega_k) + yk_prime * np.cos(ik) * np.cos(omega_k)
    z_ecef = yk_prime * np.sin(ik)

    return np.array([x_ecef, y_ecef, z_ecef]) / 1000.0


def intersect_shell_vectorized(
    receivers: np.ndarray, satellites: np.ndarray, shell_radius: float
) -> np.ndarray:
    """
    Find the intersection of multiple LoS rays with a spherical shell.

    Parameters
    ----------
    receivers : np.ndarray
        Shape (N, 3) ECEF coordinates of receivers in km.
    satellites : np.ndarray
        Shape (N, 3) ECEF coordinates of satellites in km.
    shell_radius : float
        Radius of the shell from the Earth's center in km.

    Returns
    -------
    t_values : np.ndarray
        Shape (N,) intersection parameters t in [0, 1].
        Returns np.nan for rays that do not intersect the segment.
    """
    r = receivers
    s = satellites
    d = s - r

    a = np.sum(d * d, axis=1)
    b = 2 * np.sum(r * d, axis=1)
    c = np.sum(r * r, axis=1) - shell_radius**2

    discriminant = b**2 - 4 * a * c

    # Mask for rays that actually intersect the sphere
    mask = discriminant >= 0
    t_values = np.full(len(receivers), np.nan)

    if not np.any(mask):
        return t_values

    sqrt_disc = np.sqrt(np.maximum(0, discriminant[mask]))
    t1 = (-b[mask] - sqrt_disc) / (2 * a[mask])
    t2 = (-b[mask] + sqrt_disc) / (2 * a[mask])

    # Intersection candidates for the masked subset
    candidates = np.stack([t1, t2], axis=1)

    # Find candidates in [0, 1]
    valid_mask = (candidates >= 0) & (candidates <= 1)

    # Replace invalid candidates with infinity to find the minimum valid t
    masked_candidates = np.where(valid_mask, candidates, np.inf)
    min_ts = np.min(masked_candidates, axis=1)

    # Set nan for rays where no candidate was valid
    t_subset = np.where(min_ts == np.inf, np.nan, min_ts)
    t_values[mask] = t_subset

    return t_values


def intersect_shell(
    receiver_ecef: np.ndarray, satellite_ecef: np.ndarray, shell_radius: float
) -> float | None:
    """
    Find the intersection of a line-of-sight (LoS) ray with a spherical shell.

    Parameters
    ----------
    receiver_ecef : np.ndarray
        ECEF coordinates of the receiver [x, y, z] in km.
    satellite_ecef : np.ndarray
        ECEF coordinates of the satellite [x, y, z] in km.
    shell_radius : float
        Radius of the shell from the Earth's center in km.

    Returns
    -------
    t : float or None
        The intersection parameter t in [0, 1]. Returns None if no intersection exists in the segment.
    """
    r = np.asarray(receiver_ecef)
    s = np.asarray(satellite_ecef)
    d = s - r

    a = np.dot(d, d)
    b = 2 * np.dot(r, d)
    c = np.dot(r, r) - shell_radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    for t in [t1, t2]:
        if 0 <= t <= 1:
            return t

    return None


def calculate_single_shell_tec(
    receiver_ecef: np.ndarray,
    satellite_ecef: np.ndarray,
    shell_radius: float,
    delta_ne_func: Callable[[float, float, float], float],
    delta_h_eff: float,
) -> float:
    """
    Calculate delta TEC for a single shell.

    Parameters
    ----------
    receiver_ecef : np.ndarray
        ECEF coordinates of the receiver in km.
    satellite_ecef : np.ndarray
        ECEF coordinates of the satellite in km.
    shell_radius : float
        Radius of the shell in km.
    delta_ne_func : Callable[[float, float, float], float]
        Function that takes (lat, lon, height) and returns delta Ne.
    delta_h_eff : float
        Effective thickness of the shell in km.

    Returns
    -------
    delta_tec : float
        The simulated delta TEC.
    """
    t = intersect_shell(receiver_ecef, satellite_ecef, shell_radius)
    if t is None:
        return 0.0

    # Intersection point in ECEF
    p = receiver_ecef + t * (satellite_ecef - receiver_ecef)

    # Convert to spherical coordinates (lat, lon, radius)
    lat, lon, _ = ecef2spherical(p[0], p[1], p[2])

    # The height is implicit in shell_radius, but delta_ne_func might want it
    height = shell_radius - 6378.137  # Approximate earth radius

    delta_ne = delta_ne_func(lat, lon, height)
    return delta_ne * delta_h_eff


class ShellGrid:
    """
    Manages a collection of 2D arrays for different altitude shells.
    """

    def __init__(
        self,
        res_lat: float = 1.0,
        res_lon: float = 1.0,
        lat_range: tuple[float, float] = (-90.0, 90.0),
        lon_range: tuple[float, float] = (-180.0, 180.0),
    ) -> None:
        """
        Initialize the ShellGrid.

        Parameters
        ----------
        res_lat : float
            Latitude resolution in degrees.
        res_lon : float
            Longitude resolution in degrees.
        lat_range : Tuple[float, float]
            Latitude range [min, max] in degrees.
        lon_range : Tuple[float, float]
            Longitude range [min, max] in degrees.
        """
        self.res_lat = res_lat
        self.res_lon = res_lon
        self.lat_range = lat_range
        self.lon_range = lon_range

        # Create grid dimensions
        self.lat_bins = np.arange(lat_range[0], lat_range[1] + res_lat, res_lat)
        self.lon_bins = np.arange(lon_range[0], lon_range[1] + res_lon, res_lon)

        # Initialize shells: {altitude: 2D_array}
        self.shells: dict[float, np.ndarray] = {}

    @property
    def altitudes(self) -> np.ndarray:
        """Returns the altitudes of the shells currently in the grid."""
        return np.array(list(self.shells.keys()))

    def _ensure_shell(self, alt: float) -> np.ndarray:
        """Ensure a shell exists at the given altitude and return it."""
        if alt not in self.shells:
            self.shells[alt] = np.zeros((len(self.lat_bins), len(self.lon_bins)))
        return self.shells[alt]

    def _get_indices(
        self, lat: np.ndarray | float, lon: np.ndarray | float
    ) -> tuple[np.ndarray | int, np.ndarray | int]:
        """
        Map (lat, lon) to the index of the nearest bin.

        Parameters
        ----------
        lat : np.ndarray | float
            Latitude(s) in degrees.
        lon : np.ndarray | float
            Longitude(s) in degrees.

        Returns
        -------
        indices : Tuple[np.ndarray | int, np.ndarray | int]
            Indices for latitude and longitude bins.
        """
        # Map (lat, lon) to the index of the nearest bin
        idx_lat = np.round((lat - self.lat_range[0]) / self.res_lat).astype(int)
        idx_lon = np.round((lon - self.lon_range[0]) / self.res_lon).astype(int)

        # Clamp to grid boundaries
        idx_lat = np.clip(idx_lat, 0, len(self.lat_bins) - 1)
        idx_lon = np.clip(idx_lon, 0, len(self.lon_bins) - 1)

        return idx_lat, idx_lon

    def set_value(self, alt: float, lat: float, lon: float, value: float) -> None:
        """
        Set delta Ne value at a specific shell and location.

        Parameters
        ----------
        alt : float
            Altitude of the shell.
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        value : float
            Value to set.
        """
        if alt not in self.shells:
            raise ValueError(f"Altitude {alt} not in ShellGrid")
        idx_lat, idx_lon = self._get_indices(lat, lon)
        self.shells[alt][idx_lat, idx_lon] = value

    def get_delta_ne(self, lat: float, lon: float, height: float) -> float:
        """
        Get delta Ne for a specific location and height.

        Parameters
        ----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        height : float
            Height in km.

        Returns
        -------
        value : float
            The sampled delta Ne.
        """
        # Find the closest shell altitude
        closest_alt = self.altitudes[np.argmin(np.abs(self.altitudes - height))]
        idx_lat, idx_lon = self._get_indices(lat, lon)
        return self.shells[closest_alt][idx_lat, idx_lon]

    def get_delta_ne_vectorized(
        self, lats: np.ndarray, lons: np.ndarray, height: float
    ) -> np.ndarray:
        """
        Get delta Ne for multiple locations at a given height.

        Parameters
        ----------
        lats : np.ndarray
            Array of latitudes in degrees.
        lons : np.ndarray
            Array of longitudes in degrees.
        height : float
            Height in km.

        Returns
        -------
        values : np.ndarray
            Array of sampled delta Ne.
        """
        # Find the closest shell altitude
        closest_alt = self.altitudes[np.argmin(np.abs(self.altitudes - height))]
        idx_lat, idx_lon = self._get_indices(lats, lons)
        return self.shells[closest_alt][idx_lat, idx_lon]

    def add_uniform(self, alt: float, value: float) -> None:
        """Add constant delta Ne across the entire shell.

        Parameters
        ----------
        alt : float
            Altitude of the shell.
        value : float
            Value to add.
        """
        shell = self._ensure_shell(alt)
        shell += value

    def add_rectangle(
        self,
        alt: float,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        value: float,
    ) -> None:
        """Add delta Ne within a lat/lon rectangular region on the shell.

        Parameters
        ----------
        alt : float
            Altitude of the shell.
        lat_range : Tuple[float, float]
            Latitude range [min, max] in degrees.
        lon_range : Tuple[float, float]
            Longitude range [min, max] in degrees.
        value : float
            Value to add within the rectangle.
        """
        shell = self._ensure_shell(alt)
        lat_grid, lon_grid = np.meshgrid(self.lat_bins, self.lon_bins, indexing="ij")
        mask = (
            (lat_grid >= lat_range[0])
            & (lat_grid <= lat_range[1])
            & (lon_grid >= lon_range[0])
            & (lon_grid <= lon_range[1])
        )
        shell[mask] += value

    def add_gaussian(
        self,
        alt: float,
        lat0: float,
        lon0: float,
        sigma_lat: float,
        sigma_lon: float,
        amplitude: float,
    ) -> None:
        """Add a 2D Gaussian perturbation on the shell.

        Parameters
        ----------
        alt : float
            Altitude of the shell.
        lat0 : float
            Center latitude in degrees.
        lon0 : float
            Center longitude in degrees.
        sigma_lat : float
            Latitude standard deviation in degrees.
        sigma_lon : float
            Longitude standard deviation in degrees.
        amplitude : float
            Peak amplitude of delta Ne.
        """
        shell = self._ensure_shell(alt)
        lat_grid, lon_grid = np.meshgrid(self.lat_bins, self.lon_bins, indexing="ij")
        dist_sq = ((lat_grid - lat0) ** 2 / (2 * sigma_lat**2)) + (
            (lon_grid - lon0) ** 2 / (2 * sigma_lon**2)
        )
        shell += amplitude * np.exp(-dist_sq)

    def add_wave(
        self, alt: float, k_lat: float, k_lon: float, amplitude: float, phase: float = 0
    ) -> None:
        """Add a 2D cosine wave perturbation on the shell.

        Parameters
        ----------
        alt : float
            Altitude of the shell.
        k_lat : float
            Wave number in latitude.
        k_lon : float
            Wave number in longitude.
        amplitude : float
            Amplitude of the wave.
        phase : float
            Phase shift in radians.
        """
        shell = self._ensure_shell(alt)
        lat_grid, lon_grid = np.meshgrid(self.lat_bins, self.lon_bins, indexing="ij")
        val = amplitude * np.cos(k_lat * lat_grid + k_lon * lon_grid + phase)
        shell += val


class SimulationScenario:
    """
    Infrastructure to generate large datasets for autofocus algorithm testing.

    Example:

    >>> altitudes = np.array([300.0])
    >>> grid = ShellGrid()
    >>> grid.add_wave(300.0, k_lat=0.1, k_lon=0.1, amplitude=0.1)
    >>> scenario = SimulationScenario(grid, receivers=rx_ecef, satellites=sat_ecef)
    >>> tec = scenario.generate_tec()
    """

    def __init__(
        self,
        shell_grid: ShellGrid,
        delta_h_eff: float = 10.0,
        receivers: np.ndarray | None = None,
        satellites: np.ndarray | None = None,
        nav_file: str | None = None,
    ):
        self.grid = shell_grid
        self.delta_h_eff = delta_h_eff
        self.receivers = receivers  # Shape (N_rx, 3)
        self.satellites = satellites  # Shape (N_sat, 3)
        self.nav_file = nav_file
        self._nav_data = None

        if nav_file:
            self._load_nav_data()

    def _load_nav_data(self):
        """Load RINEX navigation data using georinex."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._nav_data = georinex.load(self.nav_file, use={"G"})

    def _get_satellites_at_time(self, t: datetime):
        """
        Calculate ECEF positions of all satellites in the nav file at time t.

        Returns
        -------
        satellites : np.ndarray
            Shape (N_sat, 3) ECEF coordinates in km.
        sv_names : list
            Names of the satellites.
        """
        if self._nav_data is None:
            raise ValueError("No navigation data loaded. Provide nav_file in __init__.")

        # In georinex, the nav dataset usually has coordinates 'sv' and 'time'
        # and variables for each orbital parameter.

        svs = self._nav_data.sv.values
        sat_pos = []
        sv_names = []

        # Convert t to numpy.datetime64 for comparison with xarray times
        t_np = np.datetime64(t)

        for sv in svs:
            # Get data for this satellite
            sv_data = self._nav_data.sel(sv=sv)

            # Find the epoch closest to t
            times = sv_data.time.values
            # Compute absolute difference in seconds
            diffs = np.abs((times - t_np).astype("timedelta64[s]").astype(float))
            idx = np.argmin(diffs)
            ephem_epoch = pd.to_datetime(times[idx])

            # Extract parameters for this epoch
            ephem = {
                "toe": ephem_epoch,
                "toe_gps": float(sv_data.Toe.values[idx]),
                "sqrtA": float(sv_data.sqrtA.values[idx]),
                "e": float(sv_data.Eccentricity.values[idx]),
                "i0": float(sv_data.Io.values[idx]),
                "Omega0": float(sv_data.Omega0.values[idx]),
                "omega": float(sv_data.omega.values[idx]),
                "M0": float(sv_data.M0.values[idx]),
                "deltaN": float(sv_data.DeltaN.values[idx]),
                "idot": float(sv_data.IDOT.values[idx]) if "IDOT" in sv_data else 0.0,
                "OmegaDot": float(sv_data.OmegaDot.values[idx]),
                "Crs": float(sv_data.Crs.values[idx]),
                "Crc": float(sv_data.Crc.values[idx]),
                "Cuc": float(sv_data.Cuc.values[idx]),
                "Cus": float(sv_data.Cus.values[idx]),
                "Cic": float(sv_data.Cic.values[idx]),
                "Cis": float(sv_data.Cis.values[idx]),
            }

            pos = calculate_gps_position(t, ephem)
            sat_pos.append(pos)
            sv_names.append(sv)

        return np.stack(sat_pos), sv_names

    def generate_tec(self):
        """
        Generate delta TEC for all receiver-satellite pairs.

        Returns
        -------
        tec_array : np.ndarray
            Shape (N_rx, N_sat) simulated delta TEC.
        """
        r = self.receivers
        s = self.satellites

        if r is None or s is None:
            raise ValueError(
                "Both receivers and satellites must be provided or set in the scenario"
            )
        n_rx = r.shape[0]
        n_sat = s.shape[0]
        tec_array = np.zeros((n_rx, n_sat))

        # Create flattened arrays of all rays for full vectorization
        r_flat = np.repeat(r, n_sat, axis=0)
        s_flat = np.tile(s, (n_rx, 1))

        for alt in self.grid.altitudes:
            shell_radius = 6378.137 + alt
            ts = intersect_shell_vectorized(r_flat, s_flat, shell_radius)

            mask = ~np.isnan(ts)
            if np.any(mask):
                # Intersection points in ECEF
                ps = r_flat[mask] + ts[mask][:, np.newaxis] * (
                    s_flat[mask] - r_flat[mask]
                )

                # Convert to spherical coordinates
                lats, lons, _ = ecef2spherical(ps[:, 0], ps[:, 1], ps[:, 2])

                # Sample delta Ne from the shell grid
                delta_nes = self.grid.get_delta_ne_vectorized(lats, lons, alt)

                # Map flat indices back to (receiver, satellite) indices
                flat_indices = np.where(mask)[0]
                i = flat_indices // n_sat
                j = flat_indices % n_sat

                tec_array[i, j] += delta_nes * self.delta_h_eff

        return tec_array

    def generate_observations_time_series(
        self, start_time: datetime, end_time: datetime, sample_rate: float
    ):
        """
        Generate a time series of observations.

        Parameters
        ----------
        start_time : datetime
            Start of the simulation.
        end_time : datetime
            End of the simulation.
        sample_rate : float
            Sampling rate in seconds.

        Returns
        -------
        df : pd.DataFrame
            Concatenated observation tables for all timestamps.
        """
        current_time = start_time
        dfs = []

        while current_time <= end_time:
            # Update satellites if using nav_file
            if self.nav_file:
                self.satellites, self.sv_names = self._get_satellites_at_time(
                    current_time
                )

            # Generate table for this timestamp
            df = self.generate_observation_table(time=current_time)
            dfs.append(df)

            current_time += timedelta(seconds=sample_rate)

        return pd.concat(dfs, ignore_index=True)

    def generate_random_receivers(self, n_rx, lat_range, lon_range):
        """
        Generate random receiver locations uniform on the sphere over the given range.
        """
        lat_min, lat_max = np.radians(lat_range)
        lon_min, lon_max = lon_range

        # Uniform sampling on sphere
        # Sample longitude uniformly
        lons = np.random.uniform(lon_min, lon_max, n_rx)

        # Sample sin(lat) uniformly to get uniform area distribution
        sin_lat_min = np.sin(lat_min)
        sin_lat_max = np.sin(lat_max)
        sin_lats = np.random.uniform(sin_lat_min, sin_lat_max, n_rx)
        lats = np.degrees(np.arcsin(sin_lats))

        # Convert to ECEF (radius = Earth radius)
        # Vectorized call to spherical2ecef
        receivers = spherical2ecef(lats, lons, np.full(n_rx, 6378.137))

        self.receivers = receivers
        return receivers

    def generate_uniform_grid_receivers(self, lat_range, lon_range, n_lat, n_lon):
        """
        Generate a uniform grid of receiver locations over the given range.

        Parameters
        ----------
        lat_range : tuple[float, float]
            Latitude range in degrees.
        lon_range : tuple[float, float]
            Longitude range in degrees.
        n_lat : int
            Number of receivers along latitude.
        n_lon : int
            Number of receivers along longitude.

        Returns
        -------
        receivers : np.ndarray
            Shape (n_lat * n_lon, 3) ECEF coordinates in km.
        """
        lats = np.linspace(lat_range[0], lat_range[1], n_lat)
        lons = np.linspace(lon_range[0], lon_range[1], n_lon)

        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

        # Convert to ECEF (radius = Earth radius)
        receivers = spherical2ecef(
            lat_grid.ravel(),
            lon_grid.ravel(),
            np.full(n_lat * n_lon, 6378.137),
        )

        self.receivers = receivers
        return receivers

    def generate_gps_satellites(self, n_sat):
        """
        Generate random GPS satellite locations at a typical GPS orbital altitude.
        """
        # Typical GPS orbital radius: ~26,560 km
        radius = 26560.0

        # Uniform distribution on the sphere
        # phi in [0, 2pi], cos(theta) in [-1, 1]
        phi = np.random.uniform(0, 2 * np.pi, n_sat)
        cos_theta = np.random.uniform(-1, 1, n_sat)
        sin_theta = np.sqrt(1 - cos_theta**2)

        # Convert to ECEF coordinates
        x = radius * sin_theta * np.cos(phi)
        y = radius * sin_theta * np.sin(phi)
        z = radius * cos_theta

        satellites = np.stack([x, y, z], axis=1)
        self.satellites = satellites
        return satellites

    def generate_receiver_df(self) -> pd.DataFrame:
        """
        Generate a DataFrame containing receiver names and their geodetic positions.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns ['rx', 'lat', 'lon', 'alt'].
        """
        if self.receivers is None:
            raise ValueError("No receivers defined in the scenario.")

        lats, lons, radii = ecef2spherical(
            self.receivers[:, 0], self.receivers[:, 1], self.receivers[:, 2]
        )

        df = pd.DataFrame({
            "rx": [f"rx{i}" for i in range(len(self.receivers))],
            "lat": lats,
            "lon": lons,
            "alt": radii - 6378.137,
        })
        return df

    def generate_observation_table(self, time=None):
        """
        Generate a canonical observation table for the current simulation scenario.
        """
        tec_array = self.generate_tec()
        n_rx = self.receivers.shape[0]
        n_sat = self.satellites.shape[0]
        n_total = n_rx * n_sat

        obs_time = time if time is not None else np.datetime64("now")

        # Pre-allocate data arrays for efficiency
        data = {col: np.full(n_total, np.nan) for col in OBS_COLUMNS}
        data["time"] = obs_time
        data["rx"] = np.repeat([f"rx{i}" for i in range(n_rx)], n_sat)

        if hasattr(self, "sv_names") and self.sv_names is not None:
            data["sv"] = np.tile(self.sv_names, n_rx)
        else:
            data["sv"] = np.tile([f"SV{j}" for j in range(n_sat)], n_rx)

        data["stec"] = tec_array.ravel()

        az_flat = np.empty(n_total)
        el_flat = np.empty(n_total)

        for i in range(n_rx):
            rx_ecef = self.receivers[i]
            rx_lat, rx_lon, rx_radius = ecef2spherical(rx_ecef[0], rx_ecef[1], rx_ecef[2])
            rx_alt_m = (rx_radius - 6378.137) * 1000.0
            # Compute Azimuth and Elevation for all satellites for this receiver.
            # ecef2aer expects target in meters and observer in geodetic (deg, deg, m).
            azs, els, _ = pymap3d.ecef2aer(
                self.satellites[:, 0] * 1000.0,
                self.satellites[:, 1] * 1000.0,
                self.satellites[:, 2] * 1000.0,
                rx_lat,
                rx_lon,
                rx_alt_m,
            )
            az_flat[i * n_sat : (i + 1) * n_sat] = azs
            el_flat[i * n_sat : (i + 1) * n_sat] = els

        data["az"] = az_flat
        data["el"] = el_flat

        df = pd.DataFrame(data)
        return df[list(OBS_COLUMNS)]
