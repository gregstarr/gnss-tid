# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:03:45 2021

@author: smrak@bu.edu
"""

import h5py, os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import interpolate
from metpy.interpolate import remove_nan_observations, interpolate_to_grid, remove_repeat_coordinates
from scipy import ndimage
import pyGnss
import cv2
from dateutil import parser
from skimage import measure
from pymap3d import aer2geodetic
from astropy.convolution import Gaussian2DKernel, convolve, RickerWavelet2DKernel,Tophat2DKernel, Ring2DKernel, interpolate_replace_nans

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
    Req = 6378.137
    f = 1/298.257223563
    
    if not isinstance(rxp, np.ndarray):
        rxp = np.array(rxp)
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
    
    # ipp = np.vstack((np.degrees(lat), np.degrees(lon)))
    # print (az.shape, R.shape, psi.shape, lat.shape, lon.shape)
    return np.degrees(lat), np.degrees(lon)

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
                            avg = np.mean(nbg[ix])
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
                            avg = np.mean(nbg[ix])
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
                            avg = np.mean(nbg[ix])
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

def ImageNew(glon, glat, tid, 
             latlim=None, lonlim=None, 
             res=None, filter_type='median', filter_size=3,
             sigma=2):
    """
    """
    xgrid, ygrid = np.meshgrid(np.arange(lonlim[0], lonlim[1]+.01, res),
                               np.arange(latlim[0], latlim[1]+.01, res))
    im = np.empty(xgrid.shape, dtype=object)
    # Fill out the image pixels
    for i in range(glon.size):
        idx = abs(xgrid[0, :] - glon[i]).argmin() if abs(xgrid[0, :] - glon[i]).min() < 3*res else np.nan
        idy = abs(ygrid[:, 0] - glat[i]).argmin() if abs(ygrid[:, 0] - glat[i]).min() < 3*res else np.nan
        # If image indexes are valid
        if np.isfinite(idx) and np.isfinite(idy):
            # Assign the value to the pixel
            if im[idy,idx] is None:
                im[idy,idx] = [tid[i]]
            # If this is not the first value to assign, assign a
            # mean of both values
            else:
                im[idy,idx].append(tid[i])
    
    imout = np.nan * np.empty(xgrid.shape)
    for i in range(xgrid.shape[0]):
        for j in range(xgrid.shape[1]):
            if im[i,j] is not None:
                imout[i,j] = np.nanmedian(im[i,j])
#    
    if filter_type == 'median':
        imout = fillPixels(imout)
        imout = ndimage.median_filter(imout, filter_size)
    elif filter_type == 'gaussian':
        kernel = Gaussian2DKernel(x_stddev=sigma, y_stddev=sigma, x_size=filter_size, y_size=filter_size)
        imout = convolve(imout, kernel)
        
        imout[:filter_size, :] = np.nan
        imout[:, :filter_size] = np.nan
        imout[-filter_size:, :] = np.nan
        imout[:, -filter_size:] = np.nan
        
    del im
    return xgrid, ygrid, imout

def polyfit2d(x, y, z, kx=13, ky=13, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''
    # finite values
    idf = np.isfinite(z)
    x = x[idf]
    y = y[idf]
#     grid coords
#    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z[idf]), rcond=None)

def interpolate_image(xg, yg, im, method='cubic'):
    x0 = xg
    y0 = yg
    mask = np.ma.masked_invalid(im)
    x0 = x0[~mask.mask]
    y0 = y0[~mask.mask]
    X = im[~mask.mask]
    
    return interpolate.griddata((x0,y0), X.ravel(), (xg, yg), 
                        method=method, fill_value=0)
    
def get_contour_area(x, y, xc, yc):
    cimg = np.zeros((x.size,y.size), dtype=bool)
    eps = np.diff(x)[0] / 2
    
    for i, x0 in enumerate(x):
        
        idx = (abs(xc-x0) <= eps)
        if np.sum(idx) > 0:
            min_yc = np.round(yc[idx].min(),1)
            max_yc = np.round(yc[idx].max(),1)
            
            idy = (y >= min_yc) & (y <= max_yc)
            cimg[i,idy] = True
    return cimg

def get_contour_coords(spectra, threshold):
    
    hpbws = measure.find_contours(spectra, threshold)
    
    return hpbws
    if len(hpbws.shape) == 3 and hpbws.shape[0] == 1:
            x, y = hpbws[0,:,0], hpbws[0,:,1]
    elif len(hpbws.shape) == 3 and hpbws.shape[0] > 1:
        distance = np.nan*np.ones(hpbws.shape[0])
        for ii in range(hpbws.shape[0]):
            d = 0
            for jj in range(hpbws.shape[1]-1):
                d += np.sqrt((hpbws[ii,jj,0]**2 - hpbws[ii,jj,1]**2) + \
                             (hpbws[ii,jj+1,0]**2 - hpbws[ii,jj+1,1]**2))
            distance[ii] = d
        ihpbw = np.argmax(distance)
        x, y = hpbws[ihpbw][:,0], hpbws[ihpbw][:,1]
    else:
        ihpbw = np.argmax(np.array([h.size/2 for h in hpbws]))
        x, y = hpbws[ihpbw][:,0], hpbws[ihpbw][:,1]
        
    return y, x

date = '20150325'
datedt = parser.parse(date)
root = os.path.split(os.getcwd())[0] + os.sep + 'data' + os.sep
file = f'{root}{os.sep}{date}{os.sep}2015_{datedt.strftime("%m%d")}T0000-{(datedt+timedelta(days=1)).strftime("%m%d")}T0000_all{datedt.strftime("%m%d")}.yaml_30el_30s_ra.h5'


odir = f'{root}focusing{os.sep}'
save = 0

tlim = [datetime(2015, 3, 25, 23, 45), datetime(2015, 3, 26, 0, 0)]

average = 3
skip = 5
dalt = 25
altkm = np.arange(150, 375.1, dalt)
# altkm = [200]
fl = altkm.size

latlim = [27, 46]
lonlim = [-110, -85]
cmap = 'gray'
clim = [-0.3, 0.3]
filter_sigma = 1.5
resolution = 0.15
filter_size = 5

D = h5py.File(file, 'r')
time = D['obstimes'][:]
dt = np.array([datetime.utcfromtimestamp(t) for t in time])
idt = (dt >= tlim[0]) & (dt <= tlim[1])
rxp = D['rx_positions'][:]
D.close()

iterate = np.arange(np.where(idt==1)[0][0], np.where(idt==1)[0][-1]+1, skip)

for i in iterate:
    # Cost/focusing functions?
    L_total_power = []
    L_hpbw_power = []
    L_hpbw_power_normalized = []
    L_normalized_power = []
    L_normalized_hpf_power = []
    # Single figure 
    fig = plt.figure(figsize = [12, 3*fl])
    for ia, alt in enumerate(altkm):
        # Read the Data from hdf5
        D = h5py.File(file, 'r')
        az = np.nanmedian(D['az'][i-average : i+1, :, :], axis=0)
        el = np.nanmedian(D['el'][i-average : i+1, :, :], axis=0)
        tid = np.nanmedian(D['res'][i-average : i+1, :, :], axis=0).flatten()
        D.close()
        # coordainte conversion 
        ipp_lla = aer2ipp(az, el, rxp, alt)
        glon = ipp_lla[1].flatten()
        glat = ipp_lla[0].flatten()
        # Remove nan values
        idf = np.isfinite(glon) & np.isfinite(glat)
        glon = glon[idf]
        glat = glat[idf]
        tid = tid[idf]
        # Interpolate only in the regional domain
        idx = (glon >= lonlim[0]-2) & (glon <= lonlim[1]+2)
        idy = (glat >= latlim[0]-2) & (glat <= latlim[1]+2)
        idin = np.logical_and(idx, idy)
        # Interpolate
        # MetPy Natural Neighbour
        xmet, ymet, zmet = remove_nan_observations(glon[idin], glat[idin], tid[idin])
        xmet, ymet, zmet = remove_repeat_coordinates(xmet, ymet, zmet)
        gxmet, gymet, zimmet = interpolate_to_grid(xmet, ymet, zmet, interp_type='natural_neighbor', hres = 0.15)
        idyy = (gymet[:,0] >= latlim[0]) & (gymet[:,0] <= latlim[1])
        idxx = (gxmet[0,:] >= lonlim[0]) & (gxmet[0,:] <= lonlim[1])
        idy1, idy2 = np.where(idyy == 1)[0][0], np.where(idyy == 1)[0][-1]
        idx1, idx2 = np.where(idxx == 1)[0][0], np.where(idxx == 1)[0][-1]
        xg_nn = gxmet[idy1:idy2, idx1:idx2]
        yg_nn = gymet[idy1:idy2, idx1:idx2]
        # im_nn = np.nan_to_num(zimmet[idy1:idy2, idx1:idx2])
        im_nn = interpolate_image(xg_nn, yg_nn, zimmet[idy1:idy2, idx1:idx2])
        # Moving Gaussian window
        xg, yg, im0 = ImageNew(glon[idin], glat[idin], tid[idin], lonlim=lonlim, latlim=latlim, 
                           res=resolution, filter_type='gaussian', filter_size=5, sigma=0.75)
        # Intepolate moving window filter image so that there is no NaNs before taking the FFT
        im2 = interpolate_image(xg, yg, im0)
        
        # Use MetPy Intepolation this time
        im_shift = im_nn - np.nanmin(im_nn)
        FFT = np.fft.fftshift(np.fft.fft2(im_shift))
        # FFT Filter
        # fftfilter = np.ones(FFT.shape)
        x_0 = int(FFT.shape[0]/2)
        y_0 = int(FFT.shape[1]/2)
        xaxis = np.arange(FFT.shape[0]) - x_0
        yaxis = np.arange(FFT.shape[1]) - y_0
        # Radius of DC values?
        r0 = 8
        #
        power_spectra_iter = abs(FFT)**2
        # spectra_iter = abs(FFT)
        
        # Remove DC
        DC_mask = (((np.ones(FFT.shape) + yaxis*yaxis).T + xaxis*xaxis) > r0**2).T
        power_spectra = power_spectra_iter * DC_mask
        
        spectra_hp = np.nanmax(power_spectra)/2
        spectra_cutoff = np.nanmax(power_spectra)/10
        
        max_ixx, max_ixy = np.where(power_spectra==np.nanmax(power_spectra))
        hpo_ixx, hpo_ixy = np.where(power_spectra>=spectra_hp)
        cto_ixx, cto_ixy = np.where(power_spectra>=spectra_cutoff)
        
        
        # fig= plt.figure(figsize = [12,4])
        ax0 = fig.add_subplot(fl+1, 4, (ia*4)+1)
        ax1 = fig.add_subplot(fl+1, 4, (ia*4)+2)
        ax2 = fig.add_subplot(fl+1, 4, (ia*4)+3)
        ax3 = fig.add_subplot(fl+1, 4, (ia*4)+4)
#        ax4 = fig.add_subplot(325)
#        ax5 = fig.add_subplot(326)
        order = np.argsort(abs(tid))
        ax0.scatter(glon[order], glat[order], c=tid[order], s=7, vmin=clim[0], vmax=clim[1], cmap='bwr')
        ax0.set_ylim(latlim)
        ax0.set_xlim(lonlim)
        
        ax0.set_xlim(lonlim)
        ax0.set_ylim(latlim)
        # METPY Interpolation
        ax1.pcolormesh(xg_nn, yg_nn, im_nn, cmap=cmap, vmin=clim[0], vmax=clim[1])
        
        ax1.set_xlim(lonlim)
        ax1.set_ylim(latlim)
        # ax1.pcolormesh(xg_nn, yg_nn, im_nn, cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax2.pcolormesh(xg, yg, im2, cmap=cmap, vmin=clim[0], vmax=clim[1])
        
        ax2.set_xlim(lonlim)
        ax2.set_ylim(latlim)

        ax3.pcolormesh(xaxis, yaxis, power_spectra.T)
        
        # plt.plot(max_ix[0][1]-x_0, max_ix[1][1]-y_0, '.r', ms=3)
        ax3.plot(cto_ixx-x_0, cto_ixy-y_0, '.w', ms=1)
        ax3.plot(hpo_ixx-x_0, hpo_ixy-y_0, '.r', ms=1)
        ax3.plot(max_ixx-x_0, max_ixy-y_0, '.r', ms=5)
        
        if ia == 0:
            # fig.suptitle(f'{dt[i]}')
            ax1.set_title('Metpy - NN')
            ax2.set_title('Gaussian')
            ax3.set_title("2D FFT")
            ax0.set_title(f'Alt:{alt} + Nx{dalt} km')
        # break
    break
# fig.tight_layout()
# fig.subplots_adjust(top=0.88)

        # im_shift = im_nn - np.nanmin(im_nn)
        # FFT = np.fft.fftshift(np.fft.fft2(im_shift))
        
        # FFT Filter
        # fftfilter = np.ones(FFT.shape)
        # x_0 = int(FFT.shape[0]/2)
        # y_0 = int(FFT.shape[1]/2)
        # # r_outer = x_0-10
        # xaxis = np.arange(FFT.shape[0]) - x_0
        # yaxis = np.arange(FFT.shape[1]) - y_0
        # mask_outer = ((np.ones(FFT.shape) + yaxis*yaxis).T + xaxis*xaxis) < r_outer*r_outer
        # r_inner = 7
        # xaxis = np.arange(FFT.shape[0]) - x_0
        # yaxis = np.arange(FFT.shape[1]) - y_0
        # mask_inner = ((np.ones(FFT.shape) + yaxis*yaxis).T + xaxis*xaxis) < r_inner*r_inner
        # mask = np.multiply(mask_outer, ~mask_inner)
        # filter_fft = FFT * mask.T
        # inv_image = FFT * mask_outer.T
        
        # power_spectra_iter = abs(filter_fft)**2
        # spectra_iter = abs(filter_fft)
        # power_spectra = abs(FFT)**2
        # spectra_hp = np.nanmax(spectra_iter)/2
        # spectra_cutoff = np.nanmax(spectra_iter)/10
        # x_hp, y_hp = get_contour_coords(spectra_iter, spectra_hp)
        # t = np.arange(len(x_hp))
        # ti = np.linspace(0, len(x_hp)-1, int(10 * len(x_hp)))
        # x_hpi = interpolate.interp1d(t, x_hp, kind='quadratic')(ti)
        # y_hpi = interpolate.interp1d(t, y_hp, kind='quadratic')(ti)
        # r_hp = np.sqrt((x_hp-FFT.shape[0]/2)**2 + (y_hp-FFT.shape[1]/2)**2)
        # r_width = r_hp.max() - r_hp.min()
        
        # # Radius of DC values?
        # r0 = 8
        # #
        # power_spectra_iter = abs(FFT)**2
        # # spectra_iter = abs(FFT)
        
        # # Remove DC
        # DC_mask = (((np.ones(FFT.shape) + yaxis*yaxis).T + xaxis*xaxis) > r0**2).T
        # power_spectra = power_spectra_iter * DC_mask
        
        # spectra_hp = np.nanmax(power_spectra)/2
        # spectra_cutoff = np.nanmax(power_spectra)/10
        
        # max_ixx, max_ixy = np.where(power_spectra==np.nanmax(power_spectra))
        # hpo_ixx, hpo_ixy = np.where(power_spectra>=spectra_hp)
        # cto_ixx, cto_ixy = np.where(power_spectra>=spectra_cutoff)
        
        # plt.figure()
        # plt.pcolormesh(xaxis, yaxis, power_spectra.T)
        
        # # plt.plot(max_ix[0][1]-x_0, max_ix[1][1]-y_0, '.r', ms=3)
        # plt.plot(cto_ixx-x_0, cto_ixy-y_0, '.w', ms=1)
        # plt.plot(hpo_ixx-x_0, hpo_ixy-y_0, '.r', ms=1)
        # plt.plot(max_ixx-x_0, max_ixy-y_0, '.r', ms=5)
        # x_hp, y_hp = get_contour_coords(power_spectra, spectra_hp)
#         xxxxxx = get_contour_coords(power_spectra, spectra_hp)
#         t = np.arange(len(x_hp))
#         ti = np.linspace(0, len(x_hp)-1, int(10 * len(x_hp)))
#         x_hpi = interpolate.interp1d(t, x_hp, kind='quadratic')(ti)
#         y_hpi = interpolate.interp1d(t, y_hp, kind='quadratic')(ti)
#         r_hp = np.sqrt((x_hp-FFT.shape[0]/2)**2 + (y_hp-FFT.shape[1]/2)**2)
#         r_width = r_hp.max() - r_hp.min()
        
        
#         x_10, y_10 = get_contour_coords(power_spectra, spectra_cutoff)
#         t = np.arange(len(x_10))
#         ti = np.linspace(0, len(x_10)-1, int(0.05 * len(x_10)))
#         x_10i = interpolate.interp1d(t, x_10, kind='quadratic')(ti)
#         y_10i = interpolate.interp1d(t, y_10, kind='quadratic')(ti)
#         r_10i = np.median(np.sqrt((x_10i-FFT.shape[0]/2)**2 + (y_10i-FFT.shape[1]/2)**2))
#         max_rxy_mask = ((np.ones(FFT.shape) + yaxis*yaxis).T + xaxis*xaxis) > r_10i*r_10i
#         hpf_power = np.nansum(power_spectra.T * max_rxy_mask)
#         hpbw_mask = get_contour_area(np.arange(FFT.shape[0]),np.arange(FFT.shape[1]), x_hp, y_hp)
# #        r_hp = np.sqrt((x_hp-FFT.shape[0]/2)**2 + (y_hp-FFT.shape[1]/2)**2)
# #        hpbw_mask = get_contour_area(np.arange(FFT.shape[0]),np.arange(FFT.shape[1]), x_hp, y_hp)
# #        r_xy = np.sqrt((x_hp-FFT.shape[0]/2)**2 + (y_hp-FFT.shape[1]/2)**2)/2
        
# #        r_width = r_xy.max()-r_xy.min()
        
        
        
#         z_hp = np.copy(power_spectra)
#         z_hp[~hpbw_mask] = np.nan
#         total_power = np.nansum(power_spectra[:,:int(power_spectra.shape[1]/2)])
# #        z_hp = np.array([power_spectra_iter[round(x_hp[i]), round(y_hp[i])] for i in range(x_hp.size)])
# #        C = spectra_max * np.nansum(z_hp) / x_hp.size
# #        L.append(np.nansum(np.abs(filter_fft)))
# #        L_lp_hpbw_power.append(C)
#         L_total_power.append(total_power)
#         L_hpbw_power.append(np.nansum(z_hp))
#         L_hpbw_power_normalized.append(np.nansum(z_hp)/ np.nansum(hpbw_mask))
#         L_normalized_power.append(np.nansum(z_hp) / total_power)
#         L_normalized_hpf_power.append(np.nansum(z_hp) / hpf_power / r_width)
        
#         ax2.set_title("{}, {}, {}, {}".format(np.round(np.nansum(power_spectra),2), 
#                       np.round(np.nansum(z_hp),2),
#                       np.round(np.nansum(z_hp) / x_hp.size,2),
#                       np.round(np.nansum(z_hp) / np.nansum(power_spectra),8)))
#         ax2.pcolormesh(np.arange(FFT.shape[1]),np.arange(FFT.shape[0]), np.log(power_spectra), vmin=0, vmax=12, cmap='nipy_spectral')
#         ax2.plot(x_hpi, y_hpi, 'r')
#         ax2.plot(x_10i, y_10i, 'w')
# #        ax2.pcolormesh(np.arange(FFT.shape[0]),np.arange(FFT.shape[1]), np.log(abs(FFT)**2).T * mask, vmin=0, vmax=12, cmap='nipy_spectral')
        
# #         image_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(inv_image)))
# #         ax3.set_title('LPF image')
# #         ax3.pcolormesh(xg_nn, yg_nn, np.abs(image_filtered)+np.nanmin(im_nn), cmap=cmap, vmin=clim[0], vmax=clim[1])
# # #        ax3.contour(xg_nn, yg_nn, np.abs(image_filtered), levels=10, cmap=cmap)
# # #        plt.contour(xg_nn, yg_nn, np.abs(image_filtered)+np.nanmedian(np.abs(image_filtered)), levels=5, cmap=cmap)
# #         ax3.set_ylim(latlim)
# #         ax3.set_xlim(lonlim)
#         if save:
#             fig.savefig(odir + '{}_{}km_panels.png'.format(dt[i].strftime("%Y%m%d%H%M"), alt), dpi=250)
        
#    break
    # plt.figure()
    # plt.plot(altkm, L_total_power/max(L_total_power), label = '$P_{sum}$')
    # plt.plot(altkm, L_hpbw_power/max(L_hpbw_power), label = '$P_{HPBW}$')
    # plt.plot(altkm, L_hpbw_power_normalized/max(L_hpbw_power_normalized), label = '$P_{HPBW} / area$')
    # plt.plot(altkm, L_normalized_power/max(L_normalized_power), label = '$P_{HPBW} / P_{sum}$')
    # plt.plot(altkm, L_normalized_hpf_power /max(L_normalized_hpf_power), label = '$P_{HPBW} / P_{hpf}$')
    # plt.legend()
    # plt.xlabel('Altitude [km]')
    # plt.ylabel('Normalized optimization value')
    # break

