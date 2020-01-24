"""Automatically analyse degenerate solutions."""

import numpy as np

import sys
import pickle
from glob import glob
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
myBins.calculate_neighbors()
del meshDict
with open("oceanMask_no_rim.pickle", 'rb') as pickFile:
    myBins.oceanMask = pickle.load(pickFile)
    
for year in range(2017, 2019):
    if year == 2017:
        months = np.arange(1,13)
    else:
        months = [3, 9]
    for month in months:
        for sub in ['', '_subD30']:
            transMat = community.transMat.from_counter_npz(matdir + f"masked_counter_Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24_Cico7{sub}.npz")
            print(f"Now running for ensemble_masked_network_Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24_Cico7{sub}_mt2")
            codelengths = []
            globalMixing = []
            globalCoherence = []

            avg_mixing = np.zeros_like(myBins.bindex)
            avg_coherence = np.zeros_like(myBins.bindex)
            avg_boundary = np.zeros_like(myBins.bindex)
            avg_global_mixing = 0
            avg_global_coherence = 0
            for run in range(1, 101):
                myBins.load_communities(comdir + f"ensemble_masked_network_Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24_Cico7{sub}_mt2/masked_network_Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24_Cico7{sub}_mt2_multirunN{run}.clu")
                
                codelengths.append(myBins.codelength)
                myBins.find_adjacency();
                avg_boundary = avg_boundary + myBins.flag_on_boundary();
                myBins.color_communities();
                avg_coherence = avg_coherence + myBins.calculate_coherence_ratio(transMat);
                avg_mixing = avg_mixing + myBins.calculate_mixing(transMat);
                avg_global_coherence = avg_global_coherence + myBins.calculate_global_coherence()
                avg_global_mixing = avg_global_mixing + myBins.calculate_global_mixing()
                globalMixing.append(myBins.globalMixing)
                globalCoherence.append(myBins.globalCoherenceRatio)

            avg_mixing = avg_mixing/100
            avg_coherence = avg_coherence/100
            avg_boundary = avg_boundary/100
            avg_global_mixing = avg_global_mixing/100
            avg_global_coherence = avg_global_coherence/100
            
            exportDict = {}
            exportDict['run'] = f"masked_network_Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24_Cico7{sub}_mt2"
            exportDict['codelengths'] = codelengths
            exportDict['globalMixing'] = globalMixing
            exportDict['globalCoherence'] = globalCoherence
            exportDict['avg_mixing'] = avg_mixing
            exportDict['avg_coherence'] = avg_coherence
            exportDict['avg_boundary'] = avg_boundary
            exportDict['avg_global_mixing'] = avg_global_mixing
            exportDict['avg_global_coherence'] = avg_global_coherence
            
            with open(comdir + f"ensemble_masked_network_Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24_Cico7{sub}_mt2/ensembleResults_network_Rcmems_Pico11_S{year}-{month}-1_D90_DT20_ODT24_Cico7{sub}_mt2.pickle", "wb") as pickFile:
                pickle.dump(exportDict, pickFile)
            
            
