import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from coords import Local2D

def main():
    ORIGIN_LAT = 80
    ORIGIN_LON = -100
    coords = Local2D.from_geodetic(ORIGIN_LAT, ORIGIN_LON, 100)
    X, Y = np.meshgrid(np.arange(-1000, 1200, 200), np.arange(-1000, 1200, 200))
    xy = np.column_stack((X.ravel(), Y.ravel()))
    lat, lon = coords.convert_to_spherical(xy[:, 0], xy[:, 1])
    lat = lat.reshape(X.shape)
    lon = lon.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection=ccrs.NearsidePerspective(ORIGIN_LON, ORIGIN_LAT, satellite_height=10_000_000))
    ax.coastlines(resolution="50m")
    ax.plot(0, 90, "kx", transform=ccrs.PlateCarree())
    for i in range(X.shape[0]):
        ax.plot(lon[i], lat[i], "b-", transform=ccrs.Geodetic())
    for i in range(X.shape[0]):
        ax.plot(lon[:, i], lat[:, i], "b-", transform=ccrs.Geodetic())
    ax.plot(ORIGIN_LON, ORIGIN_LAT, "rx", transform=ccrs.PlateCarree())

    ax.set_global()

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.Mercator())
    ax.coastlines(resolution="50m")
    ax.plot(0, 90, "kx", transform=ccrs.PlateCarree())
    for i in range(X.shape[0]):
        ax.plot(lon[i], lat[i], "b-", transform=ccrs.Geodetic())
    for i in range(X.shape[0]):
        ax.plot(lon[:, i], lat[:, i], "b-", transform=ccrs.Geodetic())
    ax.plot(ORIGIN_LON, ORIGIN_LAT, "rx", transform=ccrs.PlateCarree())
    ax.set_global()
    plt.show()

if __name__ == "__main__":
    main()
