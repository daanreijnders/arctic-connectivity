"""
Created on December 1 2018
@author: Philippe Delandmeter
Function creating the unbeach velocity for the CMEMS data (A-grid)
"""

import xarray as xr
import numpy as np


data_dir = '/Volumes/oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/'
datasetM = xr.open_dataset(data_dir + 'mercatorglorys12v1_gl12_mean_20180101_R20180103.nc')

dataArrayLonF = datasetM.longitude
dataArrayLatF = datasetM.latitude

U = np.array(datasetM.uo)
V = np.array(datasetM.vo)
U[np.isnan(U)] = 0
V[np.isnan(V)] = 0

unBeachU = np.zeros(U.shape[2:])
unBeachV = np.zeros(V.shape[2:])
Mask = np.ones(V.shape[2:])


def island(U, V, j, i):
    if U[0, 0, j, i] == 0 and U[0, 0, j, i+1] == 0 and U[0, 0, j+1, i+1] == 0 and U[0, 0, j+1, i+1] == 0 and\
       V[0, 0, j, i] == 0 and V[0, 0, j, i+1] == 0 and V[0, 0, j+1, i+1] == 0 and V[0, 0, j+1, i+1] == 0:
        return True
    else:
        return False


for j in range(1, U.shape[2]-2):
    for i in range(1, U.shape[3]-2):
        if U[0, 0, j, i] == 0 and V[0, 0, j, i] == 0:
            Mask[j, i] = 0
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
            if not island(U, V, j, i-1) and not island(U, V, j+1, i) and island(U, V, j+1, i-1):
                print('Watch out: one cell width land [%d %d]: %g %g' %
                      (j, i, dataArrayLonF[i], dataArrayLatF[j]))
            if not island(U, V, j, i+1) and not island(U, V, j+1, i) and island(U, V, j+1, i+1):
                print('Watch out: one cell width land [%d %d]: %g %g' %
                      (j, i, dataArrayLonF[i], dataArrayLatF[j]))
            if not island(U, V, j, i-1) and not island(U, V, j-1, i) and island(U, V, j-1, i-1):
                print('Watch out: one cell width land [%d %d]: %g %g' %
                      (j, i, dataArrayLonF[i], dataArrayLatF[j]))
            if not island(U, V, j, i+1) and not island(U, V, j-1, i) and island(U, V, j-1, i+1):
                print('Watch out: one cell width land [%d %d]: %g %g' %
                      (j, i, dataArrayLonF[i], dataArrayLatF[j]))

dataArrayUnBeachU = xr.DataArray(unBeachU, name='unBeachU', dims=('lat', 'lon'))
dataArrayUnBeachV = xr.DataArray(unBeachV, name='unBeachV', dims=('lat', 'lon'))
dataArrayMask = xr.DataArray(Mask, name='Mask', dims=('lat', 'lon'))

dataset = xr.Dataset()
dataset[dataArrayLonF.name] = dataArrayLonF
dataset[dataArrayLatF.name] = dataArrayLatF
dataset[dataArrayUnBeachU.name] = dataArrayUnBeachU
dataset[dataArrayUnBeachV.name] = dataArrayUnBeachV
dataset[dataArrayMask.name] = dataArrayMask

dataset.to_netcdf(path=data_dir + 'mercatorglorys12v1_gl12_unbeaching_vel.nc', engine='scipy')
