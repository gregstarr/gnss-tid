#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 13:35:18 2025

@author: mraks1
"""
import xarray as xr
import numpy as np

def worker_init(file_list):
    """
    Runs ONCE per worker.
    Each worker loads all files and stores open datasets in GLOBAL_DATASETS.
    """
    return [xr.open_dataset(f, engine="netcdf4", chunks={}) 
                       for f in file_list]

def get_tid_data(file_list, time_0, time_1, param=None)->xr.Dataset:
    """    

    Parameters
    ----------
    file_list : list / np.array
        List/np.array of input files
    time_0 : numpy.datetime64
        interval start time
    time_1 : numpy.datetime64
        interval end time
    param : str, optional        TBD
        What dtec paramter to return? 
        Optinos are 
        dtec0; 5-min high-pass filter
        dtec1; 30-min high-pass filter
        dtec2; 60-min high-pass filter
        dtec3; 90-min high-pass filter
        dtecp; polynomial filter
    Returns
    -------
    Xr.Dataset()

    """
    
    first = True
    dataset = worker_init(file_list)
    X = xr.Dataset()
    for D in dataset:
        if len(D.dims) == 0:
            continue
        
        idt = (D.time.values > time_0) & (D.time.values < time_1)
        
        if not np.any(idt):
            continue
        
        if first:
            time = D.time[idt].values
            az = D.az[idt].values
            el = D.el[idt].values
            dtec0 = D.dtec0[idt].values
            dtecp = D.dtecp[idt].values
            dtec1 = D.dtec1[idt].values
            dtec2 = D.dtec2[idt].values
            dtec3 = D.dtec3[idt].values
            tec_sigma = D['tec_sigma'][idt].values
            rxp_lat, rxp_lon, rxp_alt = np.tile(D.position_geodetic, (D.dtec0[idt].values.size, 1)).T
            first = False
        else:
            time = np.hstack((time, D.time[idt].values))            
            az = np.hstack((az, D.az[idt].values))            
            el = np.hstack((el, D.el[idt].values))            
            dtec0 = np.hstack((dtec0, D.dtec0[idt].values))
            dtecp = np.hstack((dtecp, D.dtecp[idt].values))
            dtec1 = np.hstack((dtec1, D.dtec1[idt].values))
            dtec2 = np.hstack((dtec2, D.dtec2[idt].values))
            dtec3 = np.hstack((dtec3, D.dtec3[idt].values))
            tec_sigma = np.hstack((tec_sigma, D['tec_sigma'][idt].values))
            x, y, z = np.tile(D.position_geodetic, (D.dtec0[idt].values.size, 1)).T
            rxp_lat = np.hstack((rxp_lat, x))
            rxp_lon = np.hstack((rxp_lon, y))
            rxp_alt = np.hstack((rxp_alt, z))
    
    X["az"] = ("time", az)
    X["el"] = ("time", el)
    X["dtec0"] = ("time", dtec0)
    X["dtecp"] = ("time", dtecp)
    X["dtec1"] = ("time", dtec1)
    X["dtec2"] = ("time", dtec2)
    X["dtec3"] = ("time", dtec3)
    X["tec_sigma"] = ("time", tec_sigma)
    X["rxp_lat"] = (["time"], rxp_lat)
    X["rxp_lon"] = (["time"], rxp_lon)
    X["rxp_alt"] = (["time"], rxp_alt)
    
    return X