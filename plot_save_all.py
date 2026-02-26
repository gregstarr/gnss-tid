import gnss_tid.pointdata
import xarray as xr
import subprocess, os
import numpy as np
from scipy import ndimage
from astropy.convolution import convolve, Gaussian2DKernel
from datetime import datetime, timedelta
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import warnings, re

warnings.filterwarnings('ignore')

def getNeighbours(image,i,j,N=3):
    """
    Return an array of <=9 neighbour pixel of an image with a center at (i,j)
    """
    nbg = []
    m = int(np.floor(N/2))
    M = int(np.ceil(N/2))
    for k in np.arange(i-m, i+M):
        for l in np.arange(j-m, j+M):
            try:
                nbg.append(image[k,l])
            except:
                pass
    return np.array(nbg)

def fillPixels(im, N=1):
    """
    Fill in the dead pixels. If a dead pixel has a least 4 finite neighbour
    pixel, than replace the center pixel with a mean valuse of the neighbours
    """
    X = im.shape[0]-1
    Y = im.shape[1]-1
    imcopy = np.copy(im)
    for n in range(N):
        skip = int(np.floor((3+n)/2))
        starti = 0
        startj = 0
        forwardi = int(np.floor(0.7*X))
        backwardi = int(np.floor(0.3*X))
        if n%2 == 0:
            for i in np.arange(starti, forwardi, skip):
                for j in np.arange(startj, Y, skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.median(nbg[ix])
                            im[i,j] = avg
            for i in np.arange(X, backwardi, -skip):
                for j in np.arange(Y, 0, -skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.median(nbg[ix])
                            im[i,j] = avg
        else:
            for j in np.arange(startj, Y, skip):
                for i in np.arange(starti, forwardi, skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.median(nbg[ix])
                            im[i,j] = avg

            for j in np.arange(Y, 0, -skip):
                for i in np.arange(X, backwardi, -skip):
                    # Check if th epixel is dead, i.e. empty
                    if np.isnan(im[i,j]):
                        # Get its neighbours as a np array
                        nbg = getNeighbours(imcopy,i,j,N=(3+n))
                        # If there are at leas 4 neighbours, replace the value with a mean
                        if sum(np.isfinite(nbg)) >= 4:
                            ix = np.where(np.isfinite(nbg))[0]
                            avg = np.mean(nbg[ix])
                            im[i,j] = avg
    return im

def variance_weighted_mean(x, var):
    """
    Compute inverse-variance weighted mean and its variance.
    Parameters
    ----------
    x : array_like
        Observations.
    var : array_like
        Variances associated with each observation.

    Returns
    -------
    mean : float
        Variance-weighted mean.
    mean_var : float
        Variance of the weighted mean.
    """
    x = np.asarray(x, dtype=float)
    var = np.asarray(var, dtype=float)

    w = 1.0 / np.square(var)                   # inverse-variance weights
    mean = np.sum(w * x) / np.sum(w)
    mean_var = 1.0 / np.sum(w)      # variance of the weighted mean

    return mean, mean_var

def ImageNew(glon, glat, tid, variance,
            latlim=None, lonlim=None, 
            res=None, filter_type='median', filter_size=3,
            sigma=2):
    """
    """
    xgrid, ygrid = np.meshgrid(np.arange(lonlim[0], lonlim[1]+.01, res),
                                np.arange(latlim[0], latlim[1]+.01, res))
    im = np.empty(xgrid.shape, dtype=object)
    var = np.empty(xgrid.shape, dtype=object)
    cnt = np.zeros(xgrid.shape, dtype=int)
    # Fill out the image pixels
    for i in range(glon.size):
        idx = abs(xgrid[0, :] - glon[i]).argmin() if abs(xgrid[0, :] - glon[i]).min() < 3*res else np.nan
        idy = abs(ygrid[:, 0] - glat[i]).argmin() if abs(ygrid[:, 0] - glat[i]).min() < 3*res else np.nan
        # If image indexes are valid
        if np.isfinite(idx) and np.isfinite(idy):
            # Assign the value to the pixel
            if im[idy,idx] is None:
                im[idy,idx] = [tid[i]]
                var[idy,idx] = [variance[i]]
            # If this is not the first value to assign, assign a
            # mean of both values
            else:
    #                im[idy, idx] = np.nanmedian( [im[idy, idx], + tid[i]])
                im[idy,idx].append(tid[i])
                var[idy,idx].append(variance[i])
            cnt[idy,idx] += 1

    imout = np.nan * np.empty(xgrid.shape)
    varout = np.nan * np.empty(xgrid.shape)
    for i in range(xgrid.shape[0]):
        for j in range(xgrid.shape[1]):
            if im[i,j] is not None:
                # imout[i,j] = np.nanmedian(im[i,j])
                imout[i,j], varout[i,j] = variance_weighted_mean(im[i,j], var[i,j])           
    #    
    if filter_type == 'median':
        imout = fillPixels(imout)
        imout = ndimage.median_filter(imout, filter_size)
    elif filter_type == 'gaussian':
        kernel = Gaussian2DKernel(x_stddev=sigma, y_stddev=sigma, x_size=filter_size, y_size=filter_size)
        imout = convolve(imout, kernel)
        varout = convolve(varout, kernel)
        
        imout[:filter_size, :] = np.nan
        imout[:, :filter_size] = np.nan
        imout[-filter_size:, :] = np.nan
        imout[:, -filter_size:] = np.nan
        
        varout[:filter_size, :] = np.nan
        varout[:, :filter_size] = np.nan
        varout[-filter_size:, :] = np.nan
        varout[:, -filter_size:] = np.nan
        
    del im
    return xgrid, ygrid, imout, varout, cnt

def main(P):
  folder = P.folder+os.sep #"/disk1/tid/sharon/poly/2025/0925/"
  latlim = P.latlim #[-90, 90]
  lonlim = P.lonlim #[-180, 180]
  el_mask = P.elmask
  res = P.res # 0.15
  avg = P.avg
  skip = P.skip
  ipp_alt = P.altkm
  filter = P.filter
  sigma1, sigma2, sigma3 = P.filtersigma 
  filter_size1, filter_size2, filter_size3 = P.filtersize
  datefmt = re.search(r'(\d{4})/(\d{4})', folder).group(0)
  if P.tlim is None:
    tlim = [datetime.strptime(datefmt, "%Y/%m%d").strftime("%Y%m%d_0000"), (datetime.strptime(datefmt, "%Y/%m%d")+timedelta(days=1)).strftime("%Y%m%d_0000")]
  else:
    tlim = P.tlim
  ncfn = folder + f"out{os.sep}save{os.sep}"
  imfn = folder + f"out/{os.sep}img{os.sep}"
  #print (tlim)
  #return
  if not os.path.exists(ncfn):
    subprocess.call(f"mkdir -p '{ncfn}'", shell=True)
  if not os.path.exists(imfn):
    subprocess.call(f"mkdir -p '{imfn}'", shell=True)

  print (tlim)
  return

  points = gnss_tid.pointdata.PointData(
  folder+"*.nc",
  latitude_limits=latlim,
  longitude_limits=lonlim,
  time_limits=tlim,
  el_min=el_mask,
  q_thresh=.95,
  noise_max=10,
  n_jobs=P.j,
  pbar=False
  )
  time_slices, times = points.get_time_slices(avg, skip)

  for i in range(len(time_slices)):
    print (f"{i+1}/{len(times)}, Processing {times[i]}")
    ofn = ncfn + "grid_" + times[i].astype("datetime64[s]").astype(datetime).strftime("%Y%m%dT%H%M") + ".nc4"
    if os.path.exists(ofn):
      continue
    point_data = points.get_data(time_slices[i], h=ipp_alt, use_local_cs=0)
    grid, ygrid, imout1, varout, cnt = ImageNew(point_data.lon.values, 
                                                point_data.lat.values, 
                                                point_data.dtec1.values, 
                                                point_data.tec_noise.values, 
                                                latlim=latlim, lonlim=lonlim, res=res, filter_type='gaussian', 
                                                filter_size=filter_size1, sigma=sigma1)
    xgrid, ygrid, imout2, varout, cnt = ImageNew(point_data.lon.values, 
                                                point_data.lat.values, 
                                                point_data.dtec2.values, 
                                                point_data.tec_noise.values, 
                                                latlim=latlim, lonlim=lonlim, res=res, filter_type='gaussian', 
                                                filter_size=filter_size2, sigma=sigma2)
    xgrid, ygrid, imout3, varout, cnt = ImageNew(point_data.lon.values, 
                                                point_data.lat.values, 
                                                point_data.dtec3.values, 
                                                point_data.tec_noise.values, 
                                                latlim=latlim, lonlim=lonlim, res=res, filter_type='gaussian', 
                                                filter_size=filter_size3, sigma=sigma3)
    
    O = xr.Dataset(coords={"lon": (["lat","lon"], xgrid), "lat": (["lat","lon"], ygrid),
                       "ipp_lon": point_data.lon.values, "ipp_lat": point_data.lat.values,
                       },
                  data_vars={"dtec30": (["ipp"], point_data.dtec1.values),
                             "dtec60": (["ipp"], point_data.dtec2.values),
                             "dtec90": (["ipp"], point_data.dtec3.values),
                             "dtec30grid": (["lat", "lon"], imout1),
                             "dtec60grid": (["lat", "lon"], imout2),
                             "dtec90grid": (["lat", "lon"], imout3),})
    attrs = {
    "filter": filter,
    "filter_size30": filter_size1,
    "filter_size60": filter_size2,
    "filter_size90": filter_size3,
    "filter_sigma30": sigma1,
    "filter_sigma60": sigma2,
    "filter_sigma90": sigma3,
    "resolution": res,
    "ipp_altkm": ipp_alt,
    "el_mask": el_mask,
    "author": "smrak",
    "processed:": datetime.now(),

    }
    O.assign_attrs(attrs)

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in O.data_vars}
    O.to_netcdf(ofn, mode='w', encoding=encoding)
    
    fig = plt.figure(figsize=[9,12])
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(413, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(414, sharex=ax1, sharey=ax1)
    ax1.set_xlim(lonlim)
    ax1.set_ylim(latlim)

    plt.suptitle(f"{point_data.time}")
    ax1.set_title(f"TID Scatter Plot, {np.nansum(point_data.dtec1.values.size)} data points")
    ax2.set_title(f"30min running-mean, {np.nansum(point_data.dtec1.values.size)} data points")
    ax3.set_title(f"60min running-mean, {np.nansum(point_data.dtec2.values.size)} data points")
    ax4.set_title(f"90min running-mean, {np.nansum(point_data.dtec3.values.size)} data points")
    # cnt = cnt.astype(float)
    # cnt[cnt==0] = np.nan

    order = np.argsort(abs(point_data.dtec1.values))
    im1 = ax1.scatter(point_data.lon.values[order], point_data.lat.values[order], c=point_data.dtec1.values[order], cmap='bwr', vmin=-.5, vmax=.5, s=2)
    im2 = ax2.pcolormesh(xgrid, ygrid, imout1, cmap='grey', vmin=-.2, vmax=.2)
    im3 = ax3.pcolormesh(xgrid, ygrid, imout2, cmap='grey', vmin=-.5, vmax=.5)
    im4 = ax4.pcolormesh(xgrid, ygrid, imout3, cmap='grey', vmin=-1, vmax=1)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()

    plt.tight_layout()

    posn = ax1.get_position()
    cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
    fig.colorbar(im1, cax=cax, label='dTEC [TECu]')
    posn = ax2.get_position()
    cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
    fig.colorbar(im2, cax=cax, label='dTEC [TECu]')
    posn = ax3.get_position()
    cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
    fig.colorbar(im3, cax=cax, label='dTEC [TECu]')
    posn = ax4.get_position()
    cax = fig.add_axes([posn.x0+posn.width+0.01, posn.y0, 0.02, posn.height])
    fig.colorbar(im4, cax=cax, label='dTEC [TECu]')

    fig.savefig(imfn + "img_" + times[i].astype("datetime64[s]").astype(datetime).strftime("%Y%m%dT%H%M") + ".png", dpi=90)
    plt.close(fig)

if __name__ == "__main__":
  p = ArgumentParser()
  p.add_argument('folder', help = 'Root Obs Dirctory')
  p.add_argument('-latlim', help = 'latlim, 2 aruments. Default = -90, 90', nargs=2, type=int, default=[-90, 90])
  p.add_argument('-lonlim', help = 'lonlim, 2 aruments. Default = -180, 180', nargs=2, type=int, default=[-180, 180])
  p.add_argument('-elmask', help = 'Elevation mask. Default=30', type=int, default=30)
  p.add_argument('-altkm', help = 'IPP Height. Default=250 km', type=int, default=250)
  p.add_argument('-res', help = 'Grid resolution. Float. Default=0.2', default=0.2, type=float)
  p.add_argument('-filter', help = 'Gaussian or median.', default='gaussian')
  p.add_argument('-skip', help="Time Resolution? Default=30. 30*30sec = 15min", default=30, type=int)
  p.add_argument('-avg', help="Average in Time? Default=10. 10*30sec = 5min", default=10, type=int)
  p.add_argument('-tlim', help="start and end times, 2 args. YYYYmmdd_HHMM YYYYmmdd_HHMM", default=None, nargs=2, type=str)
  p.add_argument('-filtersize', help = 'filter size. Needs three arguments. Default = 9,11,13', default=[5, 9, 9], nargs=3, type=int)
  p.add_argument('-filtersigma', help = 'filter sigma. sigma_y=sigma_x. Needs three arguments. Default = 0.75, 1.5, 3', default=[0.75, 1, 1.5], nargs=3, type=float)
  p.add_argument('-j', help = 'Number of processes for reading in the data" Default=4', default=4, type=int)

  main(p.parse_args())
