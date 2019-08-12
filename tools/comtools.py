# Imports
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
import pandas as pd
import networkx as nx
import warnings
import matplotlib.pyplot as plt
import cartopy as cart
import os
from datetime import datetime

class myGrid:
    """
    Grid defined by a latitude-longitude range and a number indicating the (discreet) amount of latitudes and longitudes used for dividing the grid.
    
    Parameters
    ----------
    nlat : int
        number of latitudes.
    nlon : int
        number of longitudes.
    minLat : float
        minimum latitude of grid (southern boundary)
    maxLat : float
        maximum latitude of grid (northern boundary)
    minLon : float
        minimum longitude of grid (western boundary)
    maxLon : float
        maximum longitude of grid (eastern boundary)
    
    Attributes
    ----------
    nlat : int
        number of latitudes
    nlon : int
        number of longitudes
    dlat : float
        distance at which latitudes are spaced
    dlon : float
        distance at which longitudes are spaced
    """
    def __init__(self, nlon, nlat, minLat=60., maxLat=90., minLon=-180, maxLon=180):
        self.nlat = nlat
        self.nlon = nlon
        self.dlat = (maxLat - minLat)/nlat
        self.dlon = (maxLon - minLon)/nlon
        
class countGrid(myGrid):
    """
    Grid used as spatial particle counting bins
    """
    def __init__(self, nlon, nlat, minLat=60., maxLat=90., minLon=-180, maxLon=180):
        myGrid.__init__(self, nlon, nlat, minLat, maxLat, minLon, maxLon)
        self.lonOffset = self.dlon/2
        self.latOffset = self.dlat/2
        self.lonBounds = np.linspace(minLon, maxLon, nlon+1)
        self.latBounds = np.linspace(minLat, maxLat, nlat+1)
        self.lonarr = np.linspace(minLon+self.lonOffset, maxLon-self.lonOffset, nlon) # center
        self.latarr = np.linspace(minLat+self.latOffset, maxLat-self.latOffset, nlat) # center
        self.lons, self.lats = np.meshgrid(self.lonarr, self.latarr)
        self.lonIdx, self.latIdx = np.meshgrid(np.arange(self.nlon), np.arange(self.nlat))
        self.flatIdx = np.arange(len(self.lonIdx.flatten())).reshape(self.lonIdx.shape)
        self.nIdx = len(self.lonIdx.flatten())
        
    def particleCount(self, particleGrid, tindex=0):
        count = np.histogram2d(particleGrid.lonlat[tindex,:,0], particleGrid.lonlat[tindex,:,1], bins=[self.lonBounds, self.latBounds])[0]
        if tindex == 0:
            self.initCount = count
        return count
    
    def load_communities(self, cluFile):
        df = pd.read_csv(cluFile, delimiter=" ").set_index('node')
        self.community_map = np.zeros(self.flatIdx.shape) #maybe NaNs everywhere unless the node is in the csv file.
        for i in range(self.flatIdx.shape[0]):
            for j in range(self.flatIdx.shape[1]):
                self.community_map[i,j] = int(df['module'].loc[self.flatIdx[i,j]+1])
    
class particleGrid(myGrid):
    def __init__(self, nlon, nlat, release_time, minLat=60.5, maxLat=89.5, minLon=-179.5, maxLon=179.5):
        myGrid.__init__(self, nlon, nlat, minLat, maxLat, minLon, maxLon)
        self.advected = False
        if release_time:
                self.release_time = release_time
        # Create mesh
        self.lonarr = np.linspace(minLon, maxLon, nlon) # particle position
        self.latarr = np.linspace(minLat, maxLat, nlat) # particle position
        self.lons, self.lats = np.meshgrid(self.lonarr, self.latarr)
        
        # Create pairs and flatten using a reshape
        self.lonlat_3d = np.dstack((self.lons, self.lats))
        self.initialParticleCount = self.countParticles()
        self.lonlat = np.array([np.reshape(self.lonlat_3d, (self.particleCount, 2))])
        
        if self.release_time:
            self.release_times = [self.release_time for part in range(self.particleCount)]
        
        # Create labels
        self.lonlat_labels = np.arange(self.particleCount)
        
    def countParticles(self):
        if not hasattr(self, 'lonlat') and self.advected == False:
            self.particleCount = self.lonlat_3d.shape[0]*self.lonlat_3d.shape[1]
        else:
            self.particleCount = self.lonlat.shape[1]
        return self.particleCount
    
    def remove_on_land(self, fieldset):
        # Load landmask and initialize mask for particles on land 
        lm = fieldset.landMask
        # Use scipy.interpolate.griddata to have particles adopt value of landmask from nearest neighbor
        lonlatMask = griddata(np.dstack((fieldset.U.grid.lon.flatten(), fieldset.U.grid.lat.flatten()))[0,:,:], 
                              lm.flatten(), 
                              self.lonlat[0,:,:], method='nearest')
        self.lonlat = self.lonlat[:, ~lonlatMask, :]
        self.removedParticleCount = self.countParticles()
        # recreate labels
        self.lonlat_labels = np.arange(self.particleCount)
    
    def show(self, tindex=0, export=False):
        fig = plt.figure()
        ax = plt.axes(projection=cart.crs.PlateCarree())
        ax.scatter(self.lonlat[tindex, :, 0], self.lonlat[tindex, :, 1], transform=cart.crs.PlateCarree(), s=0.2)
        ax.add_feature(cart.feature.COASTLINE)
        if export:
            if not os.path.exists('figures'):
                os.makedirs('figures')
            if export[-4] == '.':
                plt.savefig(f'figures/{export}', dpi=300)
            else:
                plt.savefig(f'figures/{export}.png', dpi=300)
        return ax
    
    def add_advected(self, pset):
        lonlat_init, lonlat_final = loadLonlat(pset)
        self.lonlat = np.concatenate((self.lonlat, lonlat_final), axis=0)

class transMat:
    def __init__(self, counter):
        self.counter = counter
        self.sums = np.tile(self.counter.sum(axis=1),(self.counter.shape[1],1)).T
        self.data = np.divide(self.counter, self.sums, out=np.zeros_like(self.sums), where=self.sums!=0)
        
def loadLonlat(pset, timedelta64=None):
    ds = xr.open_dataset(pset)
    timedelta64 = False
    lons = ds['lon'].data
    lats = ds['lat'].data
    ids = ds['traj'].data
    times = ds['time'].data

    if np.any(np.diff(times[:,0]).astype(bool)):
        warning.warn("Not all starting times are equal. Behaviour may not be as expected.", Warning)
    if timedelta64:
        final_tidx = np.searchsorted(times[0,:], times[0,0]+timedelta64)
        if final_tidx == times.shape[1]:
            warning.warn("`final_tidx` lies outside of time window. Choosing last index instead", Warning)
            final_tidx = times.shape[1]-1
    else:
        final_tidx = times.shape[1]-1
    lonlat_init = np.dstack((lons[:,0], lats[:,0]))
    lonlat_final = np.dstack((lons[:,final_tidx], lats[:, final_tidx]))
    ds.close()
    return lonlat_init, lonlat_final


def createTransition(pset, countGrid, timedelta64=None):
    lonlat_init, lonlat_final = loadLonlat(pset, timedelta64)
    bindex_init = np.dstack((np.searchsorted(countGrid.lonBounds, lonlat_init[0,:,0]), np.searchsorted(countGrid.latBounds, lonlat_init[0,:,1])))[0]-1
    bindex_final = np.dstack((np.searchsorted(countGrid.lonBounds, lonlat_final[0,:,0]), np.searchsorted(countGrid.latBounds, lonlat_final[0,:,1])))[0]-1
    """2 issues:
    lonlats with NaN values will be put at index 60, 30 respectively. For these we don't want to look up the bindex
    """

    counter = np.zeros((countGrid.nIdx, countGrid.nIdx))
    N = bindex_init.shape[0]
    for i in range(N):
        print (f"\r Determining particle bins. {np.ceil(i/(N-1)*100)}%", end="")
        if (    bindex_final[i,0] < countGrid.flatIdx.shape[1] - 1
            and bindex_final[i,0] >= 0
            and bindex_final[i,1] < countGrid.flatIdx.shape[0]
            and bindex_final[i,1] >= 0): # moved outside domain
            sourceIdx = countGrid.flatIdx[bindex_init[i,1], bindex_init[i,0]]
            destIdx = countGrid.flatIdx[bindex_final[i,1], bindex_final[i,0]]
            counter[sourceIdx, destIdx] += 1
    return transMat(counter)

