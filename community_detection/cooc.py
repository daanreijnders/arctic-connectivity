"""Create a matrix that shows how often bin pairs fall in the same degenerate solution."""

import numpy as np
from scipy import sparse
import xarray as 

from itertools import combinations

import sys
import pickle
from glob import glob
from importlib import reload

sys.path.append('/science/users/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repository/tools')
import community

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
    
N = np.sum(myBins.oceanMask)
cooc = np.zeros((N, N))
for run in range(0,100):
    for comm in np.unique(ensembleCommunityID[run].data[~ensembleCommunityID[run].mask]):
        coocCurrCom = np.expand_dims((ensembleCommunityID[run].data[~ensembleCommunityID[run].mask] == comm).astype('int'), axis=1)
        cooc = cooc + np.matmul(coocCurrCom, coocCurrCom.T)
        
with open("cooccurance.pickle", "wb") as pickFile:
    pickle.dump(cooc, pickFile)
