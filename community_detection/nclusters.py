import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import SphericalVoronoi, cKDTree
from scipy import sparse
import xarray as xr
import pandas as pd

from itertools import combinations

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
import copy
from glob import glob
from importlib import reload

sys.path.append('/science/users/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repository/tools')
import plot
import community
import fieldsetter_cmems
import advectParticles

readDir = "/data/oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/"
meanDir = "/data/oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030_monthly/"
fieldFiles = sorted(glob(readDir + "mercatorglorys12v1_gl12_mean_*.nc"))

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
print("Number of particles:", myParts.n)

ensembleCommunityID = {}
codelengths = []
for run in range(1, 101):
    myBins.load_communities(comdir + f"infomap_ensemble/masked_network_Rcmems_Pico11_S2018-3-1_D90_DT20_ODT24_Cico7_mt2_multirunN{run}.clu")
    ensembleCommunityID[run-1] = myBins.communityID
    codelengths.append(myBins.codelength)
    
with open("/data/oceanparcels/output_data/data_Daan/misc/jaccard_masked_network_Rcmems_Pico11_S2018-3-1_D90_DT20_ODT24_Cico7_mt2_ensemble100.pickle", "rb") as pickFile:
    jaccardDistances = pickle.load(pickFile)
    
nSolutionClusters = np.ma.masked_array(np.zeros(myBins.bindex.shape), ~myBins.oceanMask)
combos = list(combinations(list(range(0,100)), 2))
clSorter = np.argsort(codelengths)

tolerance = 0.25

for idx in myBins.bindex[myBins.oceanMask]:
    localJaccardDistances = jaccardDistances[idx].copy()
    sortedIdx = clSorter.tolist()
    clustered = []
    nClusters = 0
    while len(sortedIdx) > 0 and len(clustered) <= 95:
        clusterCenter = sortedIdx.pop(0)
        clustered.append(clusterCenter)
        nClusters += 1
        for unclustered in sortedIdx:
            if unclustered < clusterCenter:
                pair = (unclustered, clusterCenter)
            else:
                pair = (clusterCenter, unclustered)
            pairIdx = combos.index(pair)
            if jaccardDistances[idx][pairIdx] > 1 - tolerance:
                clustered.append(sortedIdx.pop(sortedIdx.index(unclustered)))
    nSolutionClusters[idx] = nClusters
        
with open("/data/oceanparcels/output_data/data_Daan/misc/nclusters_masked_network_Rcmems_Pico11_S2018-3-1_D90_DT20_ODT24_Cico7_mt2_ensemble100_threshold_0.25_95percent.pickle", "wb") as pickFile:
    pickle.dump(nSolutionClusters, pickFile)