#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on December 1 2018
@author: Philippe Delandmeter
Function creating the unbeach velocity for the CMEMS data (A-grid)
"""

import xarray as xr
import numpy as np


readdir_ocean = '/data/oceanparcels/input_data/CESM/0.1_deg/rcp8.5/ocean/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'
fieldfile_ocean = 'daily_CESM_0.1degree_rcp8.5run_years_2000-2010_arctic.nc'
meshfile = 'POP_grid_coordinates.nc'

inputData = xr.open_dataset(readdir_ocean + fieldfile_ocean)
meshData = xr.open_dataset(readdir_mesh + meshfile)

dataArrayLonF = meshData.ULON
dataArrayLatF = meshData.ULAT

U = np.array(inputData.UVEL_5m[0,0,:,:])
V = np.array(inputData.VVEL_5m[0,0,:,:])

U[np.isnan(U)] = 0
V[np.isnan(V)] = 0

unBeachU = np.zeros(U.shape)
unBeachV = np.zeros(V.shape)

def island(U, V, j, i):
    if U[j, i] == 0 and U[j, i+1] == 0 and U[j+1, i+1] == 0 and U[j+1, i+1] == 0 and\
       V[j, i] == 0 and V[j, i+1] == 0 and V[j+1, i+1] == 0 and V[j+1, i+1] == 0:
        return True
    else:
        return False


for j in range(1, U.shape[0]-2):
    for i in range(1, U.shape[1]-2):
        if island(U, V, j, i):
            if not island(U, V, j, i-1):
                unBeachU[j, i] = -1
                unBeachU[j+1, i] = -1
            if not island(U, V, j, i+1):
                unBeachU[j, i+1] = 1
                unBeachU[j+1, i+1] = 1
            if not island(U, V, j-1, i):
                unBeachV[j, i] = -1
                unBeachV[j, i+1] = -1
            if not island(U, V, j+1, i):
                unBeachV[j+1, i] = 1
                unBeachV[j+1, i+1] = 1
#             if not island(U, V, j, i-1) and not island(U, V, j+1, i) and island(U, V, j+1, i-1):
#                 print('Watch out: one cell width land [%d %d]: %g %g' %
#                       (j, i, dataArrayLonF[i], dataArrayLatF[j]))
#             if not island(U, V, j, i+1) and not island(U, V, j+1, i) and island(U, V, j+1, i+1):
#                 print('Watch out: one cell width land [%d %d]: %g %g' %
#                       (j, i, dataArrayLonF[i], dataArrayLatF[j]))
#             if not island(U, V, j, i-1) and not island(U, V, j-1, i) and island(U, V, j-1, i-1):
#                 print('Watch out: one cell width land [%d %d]: %g %g' %
#                       (j, i, dataArrayLonF[i], dataArrayLatF[j]))
#             if not island(U, V, j, i+1) and not island(U, V, j-1, i) and island(U, V, j-1, i+1):
#                 print('Watch out: one cell width land [%d %d]: %g %g' %
#                       (j, i, dataArrayLonF[i], dataArrayLatF[j]))

coords = {'lon': dataArrayLonF,
          'lat': dataArrayLatF}
dataArrayUnBeachU = xr.DataArray(unBeachU, name='unBeachU', dims=('lat', 'lon'))
dataArrayUnBeachV = xr.DataArray(unBeachV, name='unBeachV', dims=('lat', 'lon'))

dataset = xr.Dataset()
dataset[dataArrayLonF.name] = dataArrayLonF
dataset[dataArrayLatF.name] = dataArrayLatF
dataset[dataArrayUnBeachU.name] = dataArrayUnBeachU
dataset[dataArrayUnBeachV.name] = dataArrayUnBeachV

dataset.to_netcdf(path='CESM_POP_0.1degree_rcp8.5run_j1800_antibeach_vel.nc', engine='scipy')