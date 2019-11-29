import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import SphericalVoronoi, cKDTree
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean

from datetime import timedelta as delta
from datetime import datetime

import networkx as nx

from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)
import sys
import pickle
from glob import glob
from importlib import reload

sys.path.append('/science/users/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repository/tools')
import plot
import community
import fieldsetter_cmems
import advectParticles

readdir_ocean = "/data/oceanparcels/input_data/CMEMS/CMEMS-GLORYS12V1-Arctic/"
fieldfiles = sorted(glob(readdir_ocean+'GLOBAL_REANALYSIS_PHY_001_030-TDS_*.nc'))

writedir = '/scratch/DaanR/psets/'

psetdir = "/data/oceanparcels/output_data/data_Daan/psets/"
matdir = "/data/oceanparcels/output_data/data_Daan/matrices/"
netdir = "/data/oceanparcels/output_data/data_Daan/networks/"
comdir = "/data/oceanparcels/output_data/data_Daan/communities/"


with open('/scratch/DaanR/meshes/ico_mesh_hex_r7.pickle', 'rb') as meshPick:
    meshDict = pickle.load(meshPick)
myBins = community.hexCountBins(meshDict['points'], 
                                np.degrees(meshDict['lons']), 
                                np.degrees(meshDict['lats']), 
                                meshDict['permutation'], 
                                meshDict['simplices'])
myMask = community.hexMask(myBins, -180, 180, 60, 90)
myMask.growToLevel(4)
myBins.calculate_voronoi(myMask, innerMaskLevel=2, outerMaskLevel=3)
#myBins.add_regular_rim()
myBins.calculate_neighbors()
del meshDict

with open("oceanMask_no_rim.pickle", 'rb') as pickFile:
    myBins.oceanMask = pickle.load(pickFile)

myParts = community.particles.from_pickle('/scratch/DaanR/meshes/ico_mesh_parts_deg_arctic_r11_delland.pickle')    
    
for year in range(2004, 2013):
    for month in [3, 9]:
        runName = f"Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24"
        transMat = community.transMat.from_pset(psetdir + f"pset_{runName}.nc", 
                                                       myBins, 
                                                       timedelta64 = np.timedelta64(30, 'D'),
                                                       mask = myBins.oceanMask)
        transMat.save_counter(matdir + f"masked_counter_{runName}_Cico7_subD30")
        transMat.save_network(netdir + f"masked_network_{runName}_Cico7_subD30.net")
        
        runName = f"Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24"
        transMat = community.transMat.from_pset(psetdir + f"pset_{runName}.nc", 
                                                       myBins, 
                                                       timedelta64 = np.timedelta64(90, 'D'),
                                                       mask = myBins.oceanMask)
        transMat.save_counter(matdir + f"masked_counter_{runName}_Cico7")
        transMat.save_network(netdir + f"masked_network_{runName}_Cico7.net")

for year in [1993, 2005]:
    for month in range(1,13):
        runName = f"Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24"
        transMat = community.transMat.from_pset(psetdir + f"pset_{runName}.nc", 
                                                       myBins, 
                                                       timedelta64 = np.timedelta64(30, 'D'),
                                                       mask = myBins.oceanMask)
        transMat.save_counter(matdir + f"masked_counter_{runName}_Cico7_subD30")
        transMat.save_network(netdir + f"masked_network_{runName}_Cico7_subD30.net")
        
        runName = f"Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24"
        transMat = community.transMat.from_pset(psetdir + f"pset_{runName}.nc", 
                                                       myBins, 
                                                       timedelta64 = np.timedelta64(90, 'D'),
                                                       mask = myBins.oceanMask)
        transMat.save_counter(matdir + f"masked_counter_{runName}_Cico7")
        transMat.save_network(netdir + f"masked_network_{runName}_Cico7.net")
