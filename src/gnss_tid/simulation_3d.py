import numpy as np
import pandas as pd
import pymap3d

from gnss_tid.coords import ecef2spherical, spherical2ecef
from gnss_tid.pointdata import OBS_COLUMNS


def intersect_shell_vectorized(
    receivers: np.ndarray, satellites: np.ndarray, shell_radius: float
):
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
):
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
    delta_ne_func,
    delta_h_eff: float,
):
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
    delta_ne_func : callable
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
        altitudes: np.ndarray,
        res_lat: float = 1.0,
        res_lon: float = 1.0,
        lat_range: tuple[float, float] = (-90.0, 90.0),
        lon_range: tuple[float, float] = (-180.0, 180.0),
    ):
        self.altitudes = np.asarray(altitudes)
        self.res_lat = res_lat
        self.res_lon = res_lon
        self.lat_range = lat_range
        self.lon_range = lon_range

        # Create grid dimensions
        self.lat_bins = np.arange(lat_range[0], lat_range[1] + res_lat, res_lat)
        self.lon_bins = np.arange(lon_range[0], lon_range[1] + res_lon, res_lon)

        # Initialize shells: {altitude: 2D_array}
        self.shells = {
            alt: np.zeros((len(self.lat_bins), len(self.lon_bins)))
            for alt in self.altitudes
        }

    def _get_indices(self, lat, lon):
        # Map (lat, lon) to the index of the nearest bin
        idx_lat = np.round((lat - self.lat_range[0]) / self.res_lat).astype(int)
        idx_lon = np.round((lon - self.lon_range[0]) / self.res_lon).astype(int)

        # Clamp to grid boundaries
        idx_lat = np.clip(idx_lat, 0, len(self.lat_bins) - 1)
        idx_lon = np.clip(idx_lon, 0, len(self.lon_bins) - 1)

        return idx_lat, idx_lon

    def set_value(self, alt, lat, lon, value):
        if alt not in self.shells:
            raise ValueError(f"Altitude {alt} not in ShellGrid")
        idx_lat, idx_lon = self._get_indices(lat, lon)
        self.shells[alt][idx_lat, idx_lon] = value

    def get_delta_ne(self, lat, lon, height):
        # Find the closest shell altitude
        closest_alt = self.altitudes[np.argmin(np.abs(self.altitudes - height))]
        idx_lat, idx_lon = self._get_indices(lat, lon)
        return self.shells[closest_alt][idx_lat, idx_lon]

    def get_delta_ne_vectorized(self, lats, lons, height):
        # Find the closest shell altitude
        closest_alt = self.altitudes[np.argmin(np.abs(self.altitudes - height))]
        idx_lat, idx_lon = self._get_indices(lats, lons)
        return self.shells[closest_alt][idx_lat, idx_lon]


def generate_slab(grid: ShellGrid, alt_start: float, alt_end: float, value: float):
    """Assign constant delta Ne to shells within [alt_start, alt_end]."""
    for alt in grid.altitudes:
        if alt_start <= alt <= alt_end:
            grid.shells[alt][:] = value


def generate_gaussian_blob(
    grid: ShellGrid, lat0, lon0, h0, A, sigma_lat, sigma_lon, sigma_h
):
    """Generate a 3D Gaussian blob of delta Ne."""
    for alt in grid.altitudes:
        # Weight for this altitude shell
        h_weight = np.exp(-((alt - h0) ** 2) / (2 * sigma_h**2))
        if h_weight < 1e-4:
            continue

        # Create 2D Gaussian for this shell
        # Efficiently calculate for all grid points
        lat_grid, lon_grid = np.meshgrid(grid.lat_bins, grid.lon_bins, indexing="ij")

        dist_sq = ((lat_grid - lat0) ** 2 / (2 * sigma_lat**2)) + (
            (lon_grid - lon0) ** 2 / (2 * sigma_lon**2)
        )

        grid.shells[alt] += A * np.exp(-dist_sq) * h_weight


def generate_tid_wave(grid: ShellGrid, k_lat, k_lon, k_h, A, phase=0):
    """Generate a 3D cosine wave sampled onto discrete shells."""
    for alt in grid.altitudes:
        lat_grid, lon_grid = np.meshgrid(grid.lat_bins, grid.lon_bins, indexing="ij")

        val = A * np.cos(k_lat * lat_grid + k_lon * lon_grid + k_h * alt + phase)
        grid.shells[alt] += val


class SimulationScenario:
    """
    Infrastructure to generate large datasets for autofocus algorithm testing.
    """

    def __init__(self, shell_grid: ShellGrid, delta_h_eff: float = 10.0):
        self.grid = shell_grid
        self.delta_h_eff = delta_h_eff
        self.receivers = None  # Shape (N_rx, 3)
        self.satellites = None  # Shape (N_sat, 3)

    def generate_tec(self, receivers: np.ndarray = None, satellites: np.ndarray = None):
        """
        Generate delta TEC for all receiver-satellite pairs.

        Parameters
        ----------
        receivers : np.ndarray, optional
            Shape (N_rx, 3) ECEF coordinates in km. If None, uses self.receivers.
        satellites : np.ndarray, optional
            Shape (N_sat, 3) ECEF coordinates in km. If None, uses self.satellites.

        Returns
        -------
        tec_array : np.ndarray
            Shape (N_rx, N_sat) simulated delta TEC.
        """
        r = receivers if receivers is not None else self.receivers
        s = satellites if satellites is not None else self.satellites

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

    def generate_random_receivers(self, n_rx, lat_range, lon_range):
        """
        Generate random receiver locations uniform on the sphere over the given range.
        """
        lat_min, lat_max = np.radians(lat_range)
        lon_min, lon_max = np.radians(lon_range)

        # Uniform sampling on sphere
        # Sample longitude uniformly
        lons = np.random.uniform(lon_min, lon_max, n_rx)

        # Sample sin(lat) uniformly to get uniform area distribution
        sin_lat_min = np.sin(lat_min)
        sin_lat_max = np.sin(lat_max)
        sin_lats = np.random.uniform(sin_lat_min, sin_lat_max, n_rx)
        lats = np.arcsin(sin_lats)

        # Convert to ECEF (radius = Earth radius)
        # Vectorized call to spherical2ecef
        receivers = spherical2ecef(lats, lons, np.full(n_rx, 6378.137))

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

    def generate_observation_table(self, time=None):
        """
        Generate a canonical observation table for the current simulation scenario.

        Parameters
        ----------
        time : np.datetime64, optional
            Timestamp for the observations. If None, uses current time.

        Returns
        -------
        df : pd.DataFrame
            Observation table containing columns defined in OBS_COLUMNS.
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
        data["sv"] = np.tile([f"SV{j}" for j in range(n_sat)], n_rx)
        data["stec"] = tec_array.ravel()

        az_flat = np.empty(n_total)
        el_flat = np.empty(n_total)

        for i in range(n_rx):
            rx_ecef = self.receivers[i]
            # Compute Azimuth and Elevation for all satellites for this receiver
            # pymap3d.ecef2aer supports arrays for target coordinates
            azs, els, _ = pymap3d.ecef2aer(
                rx_ecef[0],
                rx_ecef[1],
                rx_ecef[2],
                self.satellites[:, 0],
                self.satellites[:, 1],
                self.satellites[:, 2],
            )
            az_flat[i * n_sat : (i + 1) * n_sat] = azs
            el_flat[i * n_sat : (i + 1) * n_sat] = els

        data["az"] = az_flat
        data["el"] = el_flat

        df = pd.DataFrame(data)
        return df[list(OBS_COLUMNS)]
