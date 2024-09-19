import sys
import pathlib
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import binned_statistic_2d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage import filters

from pointdata import PointData


class FftMethod:

    def __init__(self, name):
        self.name = name
        self.focus = {}

    def get_image(self, x, y, tid):
        ...

    def get_power(self, img, scale="linear"):
        FFT = np.fft.fftshift(np.fft.fft2(img))
        if scale == "linear":
            power = abs(FFT) ** 2
            locs = np.column_stack(np.unravel_index(np.argsort(power, axis=None)[::-1], power.shape))
            locs = locs[locs[:, 1] > power.shape[1] // 2]
            locs = locs[:20]
            dist = squareform(pdist(locs))
            top = np.argmax(np.sum(dist <= 3.1, axis=1) >= 4)
            pmax = locs[top]
            peak = power[pmax[0], pmax[1]]
            half_power_level = peak / 4
            noise_cutoff = peak / 10
        elif scale == "log":
            power = 10 * np.log10(abs(FFT) ** 2)
            peak = np.max(power)
            half_power_level = np.nanmax(power) - 3
            noise_cutoff = peak  - 10
        else:
            raise ValueError("specify scale as log or linear")
        return power, peak, pmax, half_power_level, noise_cutoff


    def analyze_power_spectra(self, scale="linear"):
        self.power, self.peak, self.pmax, half_power_level, noise_cutoff = self.get_power(self.img, scale)
        hpcs = measure.find_contours(self.power, half_power_level)
        for hpc in hpcs:
            if hpc.shape[0] < 3:
                continue
            if measure.points_in_poly(self.pmax.reshape((1, 2)), hpc):
                hull = ConvexHull(hpc)
                v = np.concatenate([np.arange(hull.vertices.shape[0]), [0]])
                self.hp_contour = hpc[hull.vertices[v]]
                break
        
        self.cutoff_contour = measure.find_contours(convex_hull_image(self.power > noise_cutoff, offset_coordinates=False), .75)[0]

    def run(self, data):
        self.X, self.Y, self.img = self.get_image(data.x, data.y, data.tid)
        if self.X is None:
            return None
        
        self.img[np.isnan(self.img)] = 0
        self.img = filters.butterworth(self.img, .05, high_pass=True)

        self.analyze_power_spectra()

        hp_mask = measure.grid_points_in_poly(self.power.shape, self.hp_contour, binarize=True)
        noise_mask = measure.grid_points_in_poly(self.power.shape, self.cutoff_contour, binarize=True) == 0

        cost = {}
        # cost["peak"] = self.peak
        cost["hpbw"] = self.power[hp_mask].sum()
        cost["sum"] = self.power.sum()
        cost["hpbw/area_3db"] = self.power[hp_mask].mean()
        # cost["hpbw/area_10db"] = cost["hpbw"] / (~noise_mask).sum()
        cost["hpbw/sum"] = cost["hpbw"] / cost["sum"]
        cost["hpbw/noise"] = cost["hpbw"] / self.power[noise_mask].sum()
        print("###################################")
        print(self.name)
        print(cost)
        print(f"area: {hp_mask.sum()}")
        print(f"noise: {self.power[noise_mask].sum()}")
        print(f"noise area: {noise_mask.sum()}")
        print()

        for k, v in cost.items():
            if k not in self.focus:
                self.focus[k] = []
            self.focus[k].append(v)
        
        self.xf = np.fft.fftshift(np.fft.fftfreq(self.X.shape[1], self.resolution))
        self.yf = np.fft.fftshift(np.fft.fftfreq(self.X.shape[0], self.resolution))

        cx = np.interp(self.hp_contour[:, 1], np.arange(len(self.xf)), self.xf)
        cy = np.interp(self.hp_contour[:, 0], np.arange(len(self.yf)), self.yf)
        self.hp_contour = np.column_stack((cx, cy))
        self.pmax = np.array([self.xf[self.pmax[1]], self.yf[self.pmax[0]]])

        cx = np.interp(self.cutoff_contour[:, 1], np.arange(len(self.xf)), self.xf)
        cy = np.interp(self.cutoff_contour[:, 0], np.arange(len(self.yf)), self.yf)
        self.cutoff_contour = np.column_stack((cx, cy))


class GriddataMethod(FftMethod):
    def __init__(self, name, resolution, **kwargs):
        super().__init__(name)
        self.resolution = resolution
        self.kwargs = kwargs

    def get_image(self, x, y, tid):
        x = np.nanmedian(x, axis=0)
        y = np.nanmedian(y, axis=0)
        tid = np.nanmedian(tid, axis=0)
        # x = x.flatten()
        # y = y.flatten()
        # tid = tid.flatten()
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(tid))
        x = x[mask]
        y = y[mask]
        tid = tid[mask]

        pts = np.column_stack((x, y))
        
        print(pts.shape)
        if pts.shape[0] > 1000:
            X, Y = np.meshgrid(
                np.arange(x.min(), x.max(), self.resolution),
                np.arange(y.min(), y.max(), self.resolution)
            )
            grid_pts = np.column_stack((X.flatten(), Y.flatten()))
            img = interpolate.griddata(pts, tid, grid_pts, **self.kwargs).reshape(X.shape)
            return X, Y, img
        return None, None, None
    

class RbfMethod(FftMethod):
    def __init__(self, name, resolution, **kwargs):
        super().__init__(name)
        self.resolution = resolution
        self.kwargs = kwargs

    def get_image(self, x, y, tid):
        x = np.nanmean(x, axis=0)
        y = np.nanmean(y, axis=0)
        tid = np.nanmean(tid, axis=0)
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(tid))
        x = x[mask]
        y = y[mask]
        tid = tid[mask]

        pts = np.column_stack((x, y))
        
        if pts.shape[0] > 1000:
            X, Y = np.meshgrid(
                np.arange(x.min(), x.max(), self.resolution),
                np.arange(y.min(), y.max(), self.resolution)
            )
            grid_pts = np.column_stack((X.flatten(), Y.flatten()))
            rbf = interpolate.RBFInterpolator(pts, tid, **self.kwargs)
            img = rbf(grid_pts).reshape(X.shape)
            return X, Y, img
        return None, None, None


class BinnedStatMethod(FftMethod):
    def __init__(self, name, resolution):
        super().__init__(name)
        self.resolution = resolution

    def get_image(self, x, y, tid):
        x = x.flatten()
        y = y.flatten()
        tid = tid.flatten()
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(tid))
        x = x[mask]
        y = y[mask]
        tid = tid[mask]
        if x.shape[0] > 1000:
            img, ye, xe, binnumber = binned_statistic_2d(y, x, tid, "median", [np.floor((y.max() - y.min()) / self.resolution), np.floor((x.max() - x.min()) / self.resolution)])
            X = xe[:-1] + np.diff(xe) / 2
            Y = ye[:-1] + np.diff(ye) / 2
            X, Y = np.meshgrid(X, Y)
            return X, Y, img
        return None, None, None


class MetPyMethod(FftMethod):
    ...


def plot_points(ax, data, h, clim):
    x = np.nanmean(data.x, axis=0)
    y = np.nanmean(data.y, axis=0)
    tid = np.nanmean(data.tid, axis=0)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(tid))
    x = x[mask]
    y = y[mask]
    tid = tid[mask]
    ax.scatter(x, y, c=tid, s=6, vmin=clim[0], vmax=clim[1], cmap='bwr')
    ax.set_ylabel(f"height={h}km")

def plot_img(method, ax, clim):
    if method.X is None:
        return
    ax.pcolormesh(method.X, method.Y, method.img, vmin=clim[0], vmax=clim[1], cmap="grey")

def plot_psd(method, ax):
    if method.X is None:
        return
    vlims = np.quantile(method.power, [.01, .99])
    ax.pcolormesh(method.xf, method.yf, method.power, vmin=vlims[0], vmax=vlims[1])
    ax.plot(method.hp_contour[:, 0], method.hp_contour[:, 1], 'r-')
    ax.plot(method.pmax[0], method.pmax[1], 'k.')
    ax.plot(method.cutoff_contour[:, 0], method.cutoff_contour[:, 1], 'w-')

    
def main():
    file = sys.argv[1]
    tlim = [datetime(2015, 3, 25, 23, 45), datetime(2015, 3, 26, 0, 0)]
    latlim = [27, 46]
    lonlim = [-110, -85]
    dalt = 20
    ipp_heights = np.arange(100, 375.1, dalt)

    points = PointData(file, latlim, lonlim, tlim)

    resolution = 10
    methods = [
        # BinnedStatMethod("binned_stat", resolution*2),
        GriddataMethod("griddata_10km", resolution, method="linear"),
        GriddataMethod("griddata_20km", 2*resolution, method="linear"),
        GriddataMethod("griddata_30km", 3*resolution, method="linear"),
        # RbfMethod("rbf", resolution, neighbors=30, smoothing=100),
    ]

    PLOT_ENABLE = True
    plots = {}

    clim = [-.3, .3]
    for i, h in enumerate(ipp_heights):
        data = points.get_data_at_height(h)
        data = data.isel(time=slice(0, 4))
        for method in methods:
            method.run(data)
            
            if PLOT_ENABLE:
                if method.name not in plots:
                    fig, ax = plt.subplots(len(ipp_heights), 3, tight_layout=True, figsize=(9, len(ipp_heights)*2.2), sharex='col', sharey='col')
                    ax[0, 1].set_title(method.__class__.__name__)
                    plots[method.name] = fig, ax
                plot_points(plots[method.name][1][i, 0], data, h, clim)
                plot_img(method, plots[method.name][1][i, 1], clim)
                plot_psd(method, plots[method.name][1][i, 2])
    
    
    fig, ax = plt.subplots(len(methods), 1, tight_layout=True, figsize=(8, 10))
    for i, method in enumerate(methods):
        ax[i].set_ylabel("cost")
        ax[i].set_title(method.name)
        ax[i].grid(True)
        
        for cost_name, vals in method.focus.items():
            vals = np.array(vals)
            ax[i].plot(ipp_heights, vals / vals.max(), label=cost_name)
        ax[i].legend()

        if PLOT_ENABLE:
            plots[method.name][1][0, 0].set_title(f"t=[{data.time.values[0]} - {data.time.values[-1]}]")
            plots[method.name][1][-1, 0].set_xlabel("km")
            plots[method.name][1][-1, 1].set_xlabel("km")
            plots[method.name][1][-1, 2].set_xlabel("wavenumber (1/km)")
            plots[method.name][0].savefig(f"results/{method.name}.png")
            plt.close(plots[method.name][0])
    ax[-1].set_xlabel("height km")
    fig.savefig("results/focus.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
