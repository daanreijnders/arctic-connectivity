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
        # Rename communityMap to communityIDs. the map should have color labels.
        df = pd.read_csv(cluFile, delimiter=" ").set_index('node')
        self.communityMap = np.zeros(self.flatIdx.shape) #maybe NaNs everywhere unless the node is in the csv file.
        for i in range(self.flatIdx.shape[0]):
            for j in range(self.flatIdx.shape[1]):
                self.communityMap[i,j] = int(df['module'].loc[self.flatIdx[i,j]+1])
    
    def find_adjacency(self, mode='Neumann'):
        """
        Create an adjacency list: for each node (grid cell), determine which nodes are bordering this node.
        
        Parameters
        ----------
        mode : string
            Either 'Neumann' or 'Moore'. Indicates the pixel neighborhood used for determining
            neighbors. The Von Neumann neighborhood only considers pixels touching the edges to
            be neighbors, while the Moore neighborhood also considers pixels touching the 
            corners.

        Returns
        -------
        dict
            Containing keys corresponding to community IDs and values being `set` objects
            containing IDs of bordering communities.
        """
        # Construct empty adjacency dictionary
        # Using dictionary so that labels coincide labels created by InfoMap, rather than being 
        # indices, which might not coincide with labels.
        self.adjacencyDict = {}
        # Iterate over all cells
        for i in range(self.flatIdx.shape[0]):
            for j in range(self.flatIdx.shape[1]):
                # Save current community in variable
                currentCommunity = int(self.communityMap[i,j])
                # If the current community doesn't have a key and value yet, add an empty
                # set to the dictionary, with the key being the community ID.
                if currentCommunity not in self.adjacencyDict:
                    self.adjacencyDict[currentCommunity] = set()
                self.adjacencyDict[currentCommunity].add(int(self.communityMap[i, j+1//self.flatIdx.shape[1]]))
                self.adjacencyDict[currentCommunity].add(int(self.communityMap[i, j-1]))
                # Careful at northern and southern boundaries. 
                if i<self.flatIdx.shape[0]-1:
                    self.adjacencyDict[currentCommunity].add(int(self.communityMap[i+1, j]))
                    if mode == 'Moore':
                        self.adjacencyDict[currentCommunity].add(int(self.communityMap[i+1, j+1]))
                        self.adjacencyDict[currentCommunity].add(int(self.communityMap[i+1, j-1]))
                if i>0:
                    self.adjacencyDict[currentCommunity].add(int(self.communityMap[i-1, j]))
                    if mode == 'Moore':
                        self.adjacencyDict[currentCommunity].add(int(self.communityMap[i-1, j+1]))
                        self.adjacencyDict[currentCommunity].add(int(self.communityMap[i-1, j-1]))
        return self.adjacencyDict
                
    def color_communities(self, num_colors=4):
        """Associate new colors to existing communities by using graph coloring.
        
        Parameters
        ----------
        num_colors : int
            Number of colors that will be used for coloring the map. Currently, if `num_colors` is less than or 
            equal to the maximum degree, `num_colors` is increased to maxDegree+1.

        Returns
        -------
        np.array
            2D array containing new community IDs, corresponding to different colors.
        """
        try:
            self.communityNetwork = nx.Graph()
            for community in self.adjacencyDict:
                for neighbor in self.adjacencyDict[community]:
                    self.communityNetwork.add_edge(community, neighbor)
            # Remove self-loops
            self.communityNetwork.remove_edges_from(self.communityNetwork.selfloop_edges())
        except NameError:
            raise RuntimeError('The counting grid does not yet have an adjacency dictionary for determining the coloring of communities. Try calling the `find_adjacency()` method first.')
        maxDegree = len(nx.degree_histogram(self.communityNetwork))-1
        if not nx.algorithms.planarity.check_planarity(self.communityNetwork)[0]:
            print('Graph is not planar!')
            if maxDegree >= num_colors:
                num_colors = maxDegree+1
                print(f'Using {maxDegree+1} colors instead.')
        self.colorMapping = nx.coloring.equitable_color(self.communityNetwork, num_colors=num_colors)
        self.recoloredCommunityMap = np.array([self.colorMapping[index] for index in self.communityMap.flatten()]).reshape(self.communityMap.shape)
        return self.recoloredCommunityMap
    
class particleGrid(myGrid):
    def __init__(self, nlon, nlat, release_time=False, minLat=60.5, maxLat=89.5, minLon=-179.5, maxLon=179.5):
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
        
        if release_time:
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
        self.release_times = [self.release_time for part in range(self.removedParticleCount)]
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
    
    def add_advected(self, pset, timedelta64=None):
        """Does not work well, since particles can be deleted throughout the run, causing concatenation to break."""
        lonlat_init, lonlat_final = loadLonlat(pset, timedelta64)
        self.lonlat = np.concatenate((self.lonlat, lonlat_final), axis=0)

class transMat:
    def __init__(self, counter):
        self.counter = counter
        self.sums = np.tile(self.counter.sum(axis=1),(self.counter.shape[1],1)).T
        self.data = np.divide(self.counter, self.sums, out=np.zeros_like(self.sums), where=self.sums!=0)
        
def loadLonlat(pset, timedelta64=None):
    """
    Extract latitude and longitude
    """
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


def create_transMat(pset, countGrid, timedelta64=None):
    """
    Create transition matrix from particle trajectories (from `pset`) given a `countGrid`
    
    Parameters
    ----------
    pset : parcels.ParticleSet
        Particle set containing particle trajectories.
    countGrid : comtools.countGrid
        Grid containing cells on which the transition matrix is to be created.
    timedelta64 : np.timedelta64
        Timedelta relating to the elapsed time of the particle run for which the transition 
        matrix is to be determined. Example: np.timedelta64(30,'D') for 30 days.
        
    Returns
    -------
    comtools.transmat
        Transition matrix object, including attributes `counter` containing particle
        tranistions, and  `sums` used for normalization.
        
    Issues
    ------
    lonlats with NaN values will be put at index 60, 30 respectively. 
    For these we don't want to look up the bindex
    """
    
    lonlat_init, lonlat_final = loadLonlat(pset, timedelta64)
    # Find initial and final counting bin index for each particle
    bindex_init = np.dstack((np.searchsorted(countGrid.lonBounds, lonlat_init[0,:,0]),
                             np.searchsorted(countGrid.latBounds, lonlat_init[0,:,1])))[0]-1
    bindex_final = np.dstack((np.searchsorted(countGrid.lonBounds, lonlat_final[0,:,0]), 
                              np.searchsorted(countGrid.latBounds, lonlat_final[0,:,1])))[0]-1

    counter = np.zeros((countGrid.nIdx, countGrid.nIdx))
    N = bindex_init.shape[0]
    # Constructing transition matrix from bin indices
    for i in range(N):
        # Print progress
        print (f"\r Determining particle bins. {np.ceil(i/(N-1)*100)}%", end="")
        if (    bindex_final[i,0] < countGrid.flatIdx.shape[1] - 1
            and bindex_final[i,0] >= 0
            and bindex_final[i,1] < countGrid.flatIdx.shape[0]
            and bindex_final[i,1] >= 0): # moved outside domain
            sourceIdx = countGrid.flatIdx[bindex_init[i,1], bindex_init[i,0]]
            destIdx = countGrid.flatIdx[bindex_final[i,1], bindex_final[i,0]]
            counter[sourceIdx, destIdx] += 1
    return transMat(counter)
