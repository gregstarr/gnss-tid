import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
import pymap3d


class Local2D:

    def __init__(self, x: float, y: float, z: float):
        """
        Parameters
        ----------
        x, y, z: ECEF coordinates of local origin
        """
        self.origin = np.array([x, y, z])
        self.radius = np.linalg.norm(self.origin)
        self.phi = np.rad2deg(np.atan2(y, x))
        self.theta = np.rad2deg(np.asin(z / self.radius))
        self.r = Rotation.from_euler(
            "ZX", [self.phi + 90, 90 - self.theta], degrees=True
        )
        test = self.r.apply(self.origin, inverse=True)
        assert np.all(np.isclose(test[:2], 0))
        self.i_geo = self.r.apply([1, 0, 0])

    @classmethod
    def from_geodetic(
        cls, lat: float, lon: float, height: float, km: bool = True
        ) -> "Local2D":
        """creates Local2D object from geodetic coordinates, converts meters to km

        Parameters
        ----------
        lat, lon: geodetic coordinates of local origin
        height: height in meters above ellipsoid

        Returns
        -------
        Local2D coordinate system object
        """
        if km:
            height = height * 1000
        x, y, z = pymap3d.geodetic2ecef(lat, lon, height)
        return cls(x / 1000, y / 1000, z / 1000)

    @classmethod
    def from_spherical(cls, lat: float, lon: float, radius: float):
        """creates Local2D object from spherical coordinates

        Parameters
        ----------
        lat, lon: spherical coordinates of local origin
        radius: radius from earth center

        Returns
        -------
        Local2D coordinate system object
        """
        vec = spherical2ecef(lat, lon, radius)
        return cls(vec[0], vec[1], vec[2])

    def convert_from_spherical(self, lat: ArrayLike, lon: ArrayLike):
        """convert spherical coordinates to local cartesian

        Parameters
        ----------
        lat, lon: spherical coordinates to convert

        Returns
        -------
        x, y: local cartesian coordinates
        """
        base_shape = lat.shape
        vec = spherical2ecef(lat.flatten(), lon.flatten(), self.radius)
        vec_geo = self.r.apply(vec, inverse=True)
        phi = np.atan2(vec_geo[:, 0], vec_geo[:, 1])
        lengths = np.acos(vec_geo[:, 2] / self.radius) * self.radius
        x = lengths * np.sin(phi)
        y = lengths * np.cos(phi)
        return x.reshape(base_shape), y.reshape(base_shape)

    def convert_to_spherical(self, xi: ArrayLike, yi: ArrayLike):
        """convert spherical coordinates to local cartesian

        Parameters
        ----------
        x, y: local cartesian coordinates to convert

        Returns
        -------
        lat, lon: spherical coordinates
        """
        base_shape = xi.shape
        x = xi.flatten()
        y = yi.flatten()
        theta = np.hypot(x, y) / self.radius
        phi = np.atan2(x, y)
        org = np.array([0, 0, self.radius])
        rotations = np.column_stack([theta, phi])
        vec_geo = Rotation.from_euler("XZ", rotations).apply(org, inverse=True)
        vec = self.r.apply(vec_geo)
        lon = np.rad2deg(np.atan2(vec[:, 1], vec[:, 0]))
        lat = np.rad2deg(np.asin(vec[:, 2] / self.radius))
        return lat.reshape(base_shape), lon.reshape(base_shape)


def spherical2ecef(lat: ArrayLike, lon: ArrayLike, radius: ArrayLike):
    """convert spherical coordinates to ECEF

    Parameters
    ----------
    lat, lon: spherical coordinates
    radius: radius from earth center

    Returns
    -------
    Nx3 vector of ecef coordinates
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    rotations = np.column_stack([lon, -1 * lat])
    base = np.column_stack([radius, np.zeros_like(radius), np.zeros_like(radius)])
    return Rotation.from_euler("ZY", rotations, degrees=True).apply(base)


def ecef2spherical(x: ArrayLike, y: ArrayLike, z: ArrayLike):
    """convert ECEF coordinates to spherical

    Parameters
    ----------
    x, y, z: ecef coordinates

    Returns
    -------
    lat, lon: spherical coordinates
    radius: radius from earth center
    """
    radius = np.linalg.norm(np.column_stack((x, y, z)))
    lon = np.rad2deg(np.atan2(y, x))
    lat = np.rad2deg(np.asin(z / radius))
    return lat, lon, radius


def aer2ipp(az, el, rxp, H=350):
    """
    Compute poistion of Ionospheric piercing points (IPPs) at height H [km]
    for a given mulitdimensional vecotr of azimuth/elevation/rang (aer) and 
    the receiver position (rxp = [lat, lon, h0]).
    
    * Prol, F., et al (2017), COMPARATIVE STUDY OF METHODS FOR CALCULATING 
    IONOSPHERIC POINTS AND DESCRIBING THE GNSS SIGNAL PATH. 
    Doi:10.1590/s1982-21702017000400044
    Web: http://www.scielo.br/scielo.php?script=sci_arttext&pid=S1982-21702017000400669&lng=en&tlng=en
    """
    az = np.asarray(az, dtype=np.float32)
    el = np.asarray(el, dtype=np.float32)
    rxp = np.asarray(rxp, dtype=np.float32)
    
    Req = 6378.137
    f = 1/298.257223563
    
    if len(rxp.shape) == 1:
        lat0 = rxp[0]
        lon0 = rxp[1]
    else:
        lat0 = rxp[:,0]
        lon0 = rxp[:,1]
    
    R = np.sqrt(Req**2 / (1 + (1/(1-f)**2 -1) * np.sin(np.radians(lat0))**2))
    
    psi = (np.pi/2 - np.radians(el)) - np.arcsin(R / (R+H) * np.cos(np.radians(el)))
    
    lat = np.arcsin(np.sin(np.radians(lat0)) * np.cos(psi) + \
                    np.cos(np.radians(lat0)) * np.sin(psi) * np.cos(np.radians(az)))
    
    lon = np.radians(lon0) + np.arcsin(np.sin(psi) * np.sin(np.radians(az)) / np.cos(lat))
    
    return np.degrees(lat), np.degrees(lon)
