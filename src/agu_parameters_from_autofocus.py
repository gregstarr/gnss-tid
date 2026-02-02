#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:29:41 2024

@author: mraks1
"""

import numpy as np
import pandas as pd
import h5py, os
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from scipy.interpolate import CubicSpline
#Vadas center = 36.75N, 94.85W, time = 23:15-23:22, Phi = 180, 190, 208
#Vadas: X = 100 W, 34 N: lambda_H = 259,66\pm8.2, tau_r=22.25\pm5.6min
#Vadas: Red Box, -106 -- -92, 32 - 40N. 
def tdft(y, dx, nfft=128, window='hann', scaling='spectrum'):
    
    def _cubicSplineFit(x):
        idf = np.isfinite(x)
        x0 = np.where(idf)[0]
        x1 = np.arange(x.size)
        CSp = CubicSpline(x0, x[idf])
        y = CSp(x1)
        return y
    
    fs = 1 / dx
    k_max = fs / 2
    dk = 2 * k_max / nfft
    k = np.round(np.arange(-k_max, k_max, dk), 5)
    k = k[int(nfft/2):]
    
    if np.sum(np.isnan(y)) > 0:
        ix1 = np.where(np.isfinite(y))[0][0]
        ix2 = np.where(np.isfinite(y))[0][-1]
        y[ix1:ix2] = _cubicSplineFit(y[ix1:ix2])
    idf = np.isfinite(y)
    window_size = np.sum(idf)
    Stmp = np.fft.fft(y[idf]*signal.get_window(window, window_size), n=nfft)
    if scaling == 'spectrum':
        Sx = abs(Stmp[0:int(nfft/2)])**2 / window_size**2
    elif scaling == 'psd':
        Sx = abs(Stmp[0:int(nfft/2)])**2 / window_size * dx
    else:
        Sx = abs(Stmp[0:int(nfft/2)])**2

    return k, Sx

def tdftt(t,y,T=30,nfft=1024,Nw=240,Nskip=1,window='hann', scaling='spectrum'):
    
    def _cubicSplineFit(x):
        idf = np.isfinite(x)
        x0 = np.where(idf)[0]
        x1 = np.arange(x.size)
        CSp = CubicSpline(x0, x[idf])
        y = CSp(x1)
        return y
    
    Wn = signal.get_window(window, Nw)
    fs = 1 /T
    f_max = fs / 2
    df = 2 * f_max / nfft
    f = np.round(np.arange(-f_max, f_max, df),5)
#    f = np.fft.fftfreq(nfft) # to mHz
    f = f[int(nfft/2):]
    
    if np.sum(np.isnan(y)) > 0:
        y = _cubicSplineFit(y)
    
#    Treducted = t[:-Nw]
#    Tspecto = Treducted[::Nskip] + timedelta(seconds=Nw/2*T)
    
    Sx = np.nan * np.ones((t.size, int(nfft/2)))
    start = int(Nw/2)
    for i in np.arange(0, y.shape[0]-Nw, Nskip):
        Stmp = np.fft.fft(y[i:i+Nw]*Wn, n=nfft)
        if scaling == 'spectrum':
            Sx[start+i, :] = abs(Stmp[0:int(nfft/2)])**2 / Nw**2
        elif scaling == 'psd':
            Sx[start+i, :] = abs(Stmp[0:int(nfft/2)])**2 / Nw * T
        else:
            Sx[start+i, :] = abs(Stmp[0:int(nfft/2)])**2
            
#        Sx1 = abs(Stmp)**2
#        if i == 0:
#            Sx = Sx1
#        else:
#            Sx = np.vstack((Sx,Sx1))
            
    return f, Sx

def get_line_coordinates(start_x, start_y, azimuth, distance):
    """Calculates the end coordinates of a line given the start coordinates, azimuth, and distance."""

    # Convert azimuth to radians
    azimuth_rad = np.radians(azimuth)

    # Calculate end coordinates
    end_x = start_x + distance * np.sin(azimuth_rad)
    end_y = start_y + distance * np.cos(azimuth_rad)

    return end_x, end_y

# x0, y0 = -250, -300
azimuth = [230, 240, 250, 260]#, 280]
distance = np.arange(400,1200,25)


root = os.path.split(os.getcwd())[0] + '/data/'
date = '20150325'

file = "outputs/2024-12-06/22-20-41/autofocus.h5"

F = xr.open_dataset(file)
time = F.time.values.astype('datetime64[s]').astype(datetime)
x = F.x.values
y = F.y.values
px = F.px.values
py = F.py.values

cx0, cy0 = F.center.values[0], F.center.values[1]
for ia, az in enumerate(azimuth):
    if ia == 0:
        line = np.array([get_line_coordinates(cx0, cy0, azimuth, d) for d in distance])
    else:
        line = np.dstack((line, np.array([get_line_coordinates(cx0, cy0, azimuth, d) for d in distance])))

i = 75
im = F.image.values[i]
density = F.density.values[i]
isnan = density == 0
im[isnan]=np.nan

fig = plt.figure()
plt.title(f'{time[i]}, Alt={F.height.values[i]}km')
plt.pcolormesh(x,y, im, cmap='gray', vmin=-.2, vmax=.2)
plt.plot(F.center.values[0], F.center.values[1], '.r')
for z in range(len(azimuth)):
    plt.plot(line[:,0,z], line[:,1,z], 'r')
# plt.plot(x[idx-Ndx:idx+Ndx], y0*np.ones(Ndx*2))
# plt.plot(x0, y0, 'xr')
plt.xlabel('Longitude [km]')
plt.ylabel('Latitude [km]')
# fig = plt.figure()
# plt.title(time[i])
# plt.pcolormesh(px,py, F.Fy.values[i])
tlim = [datetime(2015,3,25,22,50), datetime(2015,3,25,23,45)]
figtau = plt.figure()
figspeed = plt.figure()
figlambda = plt.figure()
axtau = figtau.add_subplot(111)
axch = figspeed.add_subplot(111)
axlambda = figlambda.add_subplot(111)

dist = np.arange(400, 1201, 50)
tau = np.nan * np.ones((time.size, len(azimuth), len(dist)))
lambh = np.nan * np.ones((time.size, len(azimuth), len(dist)))
ch = np.nan * np.ones((time.size, len(azimuth), len(dist)))
for ll in range(len(azimuth)):
    # ll = 1
    for ixx, d in enumerate(dist):
        idd = abs(distance-d).argmin()
        x0, y0 = np.round(line[idd, :, ll], 1)
        idx, idy = abs(x-x0).argmin(), abs(y-y0).argmin()
        # Period
        dx = np.round(np.nanmedian(np.diff(x)))
        L = 800
        Ndx = round(L/2/dx)
        tid_center = np.nanmedian(F.image.values[:, idy-2:idy+2, idx-2:idx+2], axis=(1,2))
        tid_zonal = np.nanmedian(F.image.values[:, idy-2:idy+2, idx-Ndx:idx+Ndx], axis=1)
        tid_meridional = np.nanmedian(F.image.values[:, idy-Ndx:idy+Ndx, idx-2:idx+2], axis=1)
        F_period, Stt = tdftt(time, tid_center, T=60, nfft=128, Nw=30, Nskip=1)
        period_th = 0.0001
        period_max = np.nanargmax(np.nan_to_num(Stt), axis=1)
        period_max[np.nanmax(np.nan_to_num(Stt), axis=1) <= period_th] = 0
        f_r = F_period[period_max]
        f_r[f_r==0] = np.nan
        tau_r = 1 / f_r
        tau[:,ll,ixx] = tau_r/60
        # fig = plt.figure()
        # ax1 = fig.add_subplot(211)
        # ax2 = fig.add_subplot(212, sharex=ax1)
        # ax1.set_title(f"Periodogram and the dominant period at x={x0}, y={y0}")
        # ax1.pcolormesh(time, F_period, np.log10(Stt).T, cmap='nipy_spectral')
        # ax1.plot(time, F_period[period_max], 'b')
        # ax1.set_ylabel("Intrinsic Frequency [Hz]")
        # ax2.scatter(time, tau_r/60)
        # ax2.set_ylabel("Intrinsic Period [min]")
        # ax2.set_xlabel("Time in 25 March 2015")
        # ax2.set_ylim(10, 40)
        # ax2.set_xlim(tlim)
        
        # axtau.scatter(time, tau_r/60, label=f'{azimuth[ll]}deg')
        
        # Wavenumber
        nfft = 1024
        kx_spectra = np.nan * np.zeros((time.size, int(nfft/2)))
        ky_spectra = np.nan * np.zeros((time.size, int(nfft/2)))
        
        for i in range(1, time.size-1):
            try:
                kx, kx_spectra[i,:] = tdft(np.nanmedian(tid_zonal[i-1:i+1,:], axis=0), dx, nfft=nfft)
            except:
                pass
            try:
                ky, ky_spectra[i,:] = tdft(np.nanmedian(tid_meridional[i-1:i+1,:], axis=0), dx, nfft=nfft)
            except:
                pass
        kx_th = 0.0001
        ky_th = 0.0001
        kx_max = np.nanargmax(np.nan_to_num(kx_spectra), axis=1)
        kx_max[np.nanmax(np.nan_to_num(kx_spectra), axis=1) <= kx_th] = 0
        #kx_max[kx_max <= 2] = 0
        
        ky_max = np.nanargmax(np.nan_to_num(ky_spectra), axis=1)
        ky_max[np.nanmax(np.nan_to_num(ky_spectra), axis=1) <= ky_th] = 0
        #ky_max[kx_max <= 2] = 0
        
        lambda_h = np.nan * np.ones(time.size).astype(np.float32)
        lambda_x = np.nan * np.ones(time.size).astype(np.float32)
        lambda_y = np.nan * np.ones(time.size).astype(np.float32)
        c_x = np.nan * np.ones(time.size).astype(np.float32)
        c_y = np.nan * np.ones(time.size).astype(np.float32)
        c_h = np.nan * np.ones(time.size).astype(np.float32)
        theta = np.nan * np.ones(time.size).astype(np.float32)
        
        for i in range(time.size):
            if kx_max[i] >= 0 and ky_max[i] >= 0:
                lambda_x[i] = 1 / kx[kx_max[i]]
                lambda_y[i] = 1 / ky[ky_max[i]]
                if lambda_y[i] < L or lambda_x[i] < L:
                    lambda_h[i] = 1 / np.sqrt(1/lambda_x[i]**2 +  1/lambda_y[i]**2)
        
        l_h = np.asanyarray(pd.Series(lambda_h).rolling(10, min_periods=4, center=True).median())
        l_x = np.asanyarray(pd.Series(lambda_x).rolling(10, min_periods=4, center=True).median())
        l_y = np.asanyarray(pd.Series(lambda_y).rolling(10, min_periods=4, center=True).median())
        l_h_std = np.asanyarray(pd.Series(lambda_h).rolling(10, min_periods=4, center=True).std())
        
        for i in range(time.size):
            if np.isfinite(l_h[i]) and period_max[i] > 0:
                c_x[i] = l_x[i]*1e3 / (tau_r[i])
                c_y[i] = l_y[i]*1e3 / (tau_r[i])
                if np.logical_and(np.isfinite(c_x[i]), np.isfinite(c_y[i])):
                    c_h[i] = 1 / np.sqrt((1/c_x[i])**2 +  (1/c_y[i])**2)
                elif np.logical_or(np.isfinite(c_x[i]), np.isfinite(c_y[i])):
                    c_h[i] = 1 / np.sqrt(1/c_x[i])**2 if np.isfinite(c_x[i]) else 1 / np.sqrt(1/c_y[i])**2 
                else:
                    pass
                theta[i] = np.degrees(np.arctan2(c_y[i], c_x[i]))
                
        # fig = plt.figure()
        # plt.pcolormesh(time, kx, kx_spectra.T, cmap='nipy_spectral', vmin = 0.00001, vmax = 0.005)
        # plt.plot(time, kx[kx_max], 'b')
        
        # fig = plt.figure()
        # plt.pcolormesh(time, ky, ky_spectra.T, cmap='nipy_spectral', vmin = 0.00001, vmax = 0.001)
        # plt.plot(time, ky[ky_max], 'b')
        
        # fig = plt.figure()
        # plt.plot(time, lambda_h, '.m')
        # plt.errorbar(time, l_h, yerr=l_h_std, c='k')
        # plt.xlim(tlim)
        lambh[:,ll,ixx] = l_h
        ch[:,ll,ixx] = c_h
# axtau.scatter(time, tau_r/60, label=f'{azimuth[ll]}deg')
# axlambda.errorbar(time, l_h, yerr=l_h_std, label=f'{azimuth[ll]}deg')
# axch.scatter(time, c_h, label=f'{azimuth[ll]}deg')

#     # break

# axtau.legend()
# axlambda.legend()
# axch.legend()

t1 = datetime(2015,3,25,23,30)
t2 = datetime(2015,3,25,23,45)

idt = (time>=t1) & (time<=t2)

#average in this time for angels around 210

plt.figure()
for zz, azm in enumerate(azimuth):
    plt.plot(dist, np.nanmedian(tau[idt,zz,:], axis=0), '.-', label=f'{azm}deg')
plt.legend()
plt.xlabel('Radius [km]')
plt.ylabel('Intrinsic Period [min]')
plt.ylim(0, 40)

plt.figure()
for zz, azm in enumerate(azimuth):
    plt.plot(dist, np.nanmedian(ch[idt,zz,:], axis=0), '.-', label=f'{azm}deg')
    # plt.plot(dist, np.nanmedian(ch[idt,zz,:], axis=0))
plt.legend()
plt.xlabel('Radius [km]')
plt.ylabel('Phase Speed [m/s]')

plt.figure()
for zz, azm in enumerate(azimuth):
    plt.plot(dist, np.nanmedian(lambh[idt,zz,:], axis=0), '.-', label=f'{azm}deg')
    # plt.plot(dist, np.nanmedian(ch[idt,zz,:], axis=0))
plt.legend()
plt.xlabel('Radius [km]')
plt.ylabel('Horizontal Wavelength [km]')


