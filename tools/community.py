# Imports
import pandas as pd
import numpy as np
import xarray as xr
import networkx as nx
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy as cart
from datetime import datetime
import warnings
import pickle
import os
try:
    import stripy
except ImportError:
    print("Stripy is not available on this machine.")

def lonlat_from_pset(pset, timedelta64=None):
    """
    Extract latitude and longitude data from particleSet.
    
    Parameters
    ----------
    pset : str
        string with path to ``parcels.ParticleSet`` output file
    timedelta64 : np.timedelta64
        relative timestamp to load data from pset at, relative to start time

    Returns
    -------
    lonlat_init
        np.array with initial longitude-latitude pairs
    lonlat_final
        np.array with final longitude-latitude pairs    
    """
    ds = xr.open_dataset(pset)
    lons = ds['lon'].data
    lats = ds['lat'].data
    ids = ds['traj'].data
    times = ds['time'].data

    if np.any(np.diff(times[:,0]).astype(bool)):
        warnings.warn("Not all starting times are equal. Behaviour may not be as expected.", Warning)
    if timedelta64:
        # Determine which trajectory idx to use for searchsorted, 
        # since it must contain timestamps in the last index.
        firstFullTrajectoryIdx = np.searchsorted(~np.isnat(times[:, -1]), True)
        # Find index at which trajectories shoudl be investigated
        final_tidx = np.searchsorted(times[firstFullTrajectoryIdx,:], 
                                     times[firstFullTrajectoryIdx,0] + timedelta64)
        if final_tidx == times.shape[1]:
            warnings.warn("`final_tidx` lies outside of time window. Choosing last index instead", Warning)
            final_tidx = times.shape[1]-1
    else:
        final_tidx = times.shape[1]-1
    lonlatInit = np.dstack((lons[:,0], lats[:,0]))
    lonlatFinal = np.dstack((lons[:,final_tidx], lats[:, final_tidx]))
    ds.close()
    return lonlatInit, lonlatFinal

class particles:
    """
    Basic instance of particles object has lists holding the latitudes and longitudes of its points.
    
    Attributes
    ----------
    lats : np.array
        list of latitudes (in degrees)
    lons : np.array
        list of longitudes (in degrees)
    lonlat : np.ndarray
        2D array holding pairs of latitude and longitude of each particle
    n : int
        number of gridpoints
    idx : np.ndarray
        index of each gridpoint
    _releaseTime : datetime
        release time of particles
    """
    def __init__(self, lons, lats, idx = None, releaseTime = None):
        assert len(lats) == len(lons), "lats and lons should be of equal size"
        self._releaseTime = releaseTime
        self.lons = lons
        self.lats = lats
        self.lonlat = np.dstack((lons, lats)) #First axis corresponds to time
        # idx should not be updated since this makes triangle points harder to track down
        if idx: 
            self.idx = idx
        else:
            self.idx = np.arange(self.n)
    
    @property
    def n(self):
        return self.lonlat.shape[1]
    
    @property
    def releaseTimes(self):
        if self._releaseTime:
            return [self._releaseTime for part in range(self.n)]
        else:
            pass
    
    @classmethod
    def from_regular_grid(cls, nlon, nlat, minLat=60., maxLat=90., minLon=-180, maxLon=180, **kwargs):
        """
        Grid construction by dividing latitude and longitude ranges into a discrete amount of points.
        
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
        """
        lonRange = np.linspace(minLon, maxLon, nlon)
        latRange = np.linspace(minLat, maxLat, nlat)
        lon2D, lat2D = np.meshgrid(lonRange, latRange)
        return cls(lon2D.flatten(), lat2D.flatten(), **kwargs)
    
    @classmethod
    def from_pickle(cls, pickFile, lonKey='lons', latKey='lats', **kwargs):
        """
        Load longitudes and latitudes of particles from pickled dictionary
        
        Parameters
        ----------
        pickFile : str
            Path to pickled dictionary
        lonKey : str
            Key for longitudes in dictionary
        latKey : str
            Key for latitudes in dictionary
        """ 
        with open(pickFile, 'rb') as pickFile:
            lonlat_dict = pickle.load(pickFile)
            return cls(lonlat_dict[lonKey], lonlat_dict[latKey], **kwargs)
    
    def remove_on_land(self, fieldset):
        """
        Uses the fieldset.landMask to remove particles that are located on land (where u, v == 0 or -1)
        
        Parameters
        ----------
        fieldset : Parcels.FieldSet
            should have a landMask attribute (created by fieldSetter)
        """
        
        nBefore = self.n
        # Load landmask and initialize mask for particles on land 
        landMask = fieldset.landMask
        try:
            landMask = landMask.compute()
        except AttributeError:
            pass
        # Use scipy.interpolate.griddata to have particles adopt value of landmask from nearest neighbor
        lonlatMask = griddata(np.dstack((fieldset.U.grid.lon.flatten(), 
                                         fieldset.U.grid.lat.flatten()))[0,:,:], 
                              landMask.flatten(), 
                              self.lonlat[0,:,:], 
                              method='nearest')
        self.lonlat = self.lonlat[:, ~lonlatMask, :]
        self.lons = self.lonlat[0, :, 0]
        self.lats = self.lonlat[0, :, 1]
        nAfter = self.n
        self.removedParticleCount = nBefore - nAfter
    
    def add_advected_from_pset(self, *args, **kwargs):
        """
        Add final particle locations by loading them from a pset. See `lonlat_from_pset()`.    
        """
        lonlatFinal = lonlat_from_pset(pset, *args, **kwargs)[1]
        self.lonlat = np.concatenate((self.lonlat, lonlat_final), axis=0)
    
    def show(self, tindex = 0, export = None, projection=None, **kwargs):
        """
        Create a plot of the particle locations in particles object.
        
        Parameters
        ----------
        tindex : int
            Index of lonlat pairs (0 is initial, 1 is final).
        export : str
            Name of exported figure. A directory 'figures' is created.
        """
        fig = plt.figure()
        if projection:
            ax = plt.axes(projection = projection)
        else:
            ax = plt.axes(projection = cart.crs.PlateCarree())
        ax.scatter(self.lonlat[tindex, :, 0], self.lonlat[tindex, :, 1], transform = cart.crs.PlateCarree(), **kwargs)
        ax.add_feature(cart.feature.COASTLINE)
        if export:
            if not os.path.exists('figures'):
                os.makedirs('figures')
            if export[-4] == '.':
                plt.savefig(f'figures/{export}', dpi=300)
            else:
                plt.savefig(f'figures/{export}.png', dpi=300)
        return ax
        
class countBins:
    """
    Bins used for counting particles.
    
    Parameters
    ----------
    binType : str
        Indicates the type of bin: `regular` or `icosahedral`
    """
    def __init__(self, bindex):
        self.bindex = bindex
        
    @property
    def n(self):
        return len(self.bindex)
    
    def load_communities(self, comFile, parser = 'clu'):
        """
        Load communities determined by a community detection algorithm on a regular grid
        
        Parameters
        ----------
        comFile : str
            Filename of community file
        parser : str
            Parser to use
        """
        #----- START PARSERS -----#
        if parser == 'legacy':
            self.communityDF = pd.read_csv(comFile,  delimiter=" ").set_index('node')
        if parser == 'clu':
            with open(comFile) as cluFile:
                clu = cluFile.read().split('\n')
            self.codelength = float(clu[0].split(' ')[3])
            header = clu[1].split(' ')[1:]
            body = [line.split(' ') for line in clu[2:] if line is not '']
            self.communityDF = pd.DataFrame(body, columns=header).astype({"node" : 'int', 
                                                                          "module" : 'int', 
                                                                          "flow" : 'float' }).set_index("node")
        if parser == 'tree':
            """
            Not yet fully impelemented. Should have the option to investigate multiple tree levels
            """
            with open(comFile) as treeFile:
                tree = treeFile.read().split('\n')
            self.codelength = float(tree[0].split(' ')[3])
            header = tree[1].split(' ')[1:]
            body = [line.split(' ') for line in tree[2:] if line is not '']
            self.communityDF = pd.DataFrame(body, columns=header).drop(columns="name").rename(columns={'physicalId': 'node'})
            self.communityDF['rank'] = self.communityDF['path'].map(lambda a: a.split(":")[-1])
            self.communityDF['module'] = self.communityDF['path'].map(lambda a: a.split(":")[-2])
            self.communityDF = self.communityDF.astype({"node" : "int",  "module" : "int", "flow" : "float"}).set_index("node")
        #------ END PARSERS ------#
        #2D
        communityID = np.zeros(self.n)
        communityID.fill(np.nan)
        for n in range(self.n):
            communityID[n] = int(self.communityDF['module'].loc[self.bindex[n]+1])
        self.communityID = communityID
        
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
            Array containing new community IDs, corresponding to different colors.
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
        maxDegree = max([d for n, d in self.communityNetwork.degree()])
        if not nx.algorithms.planarity.check_planarity(self.communityNetwork)[0]:
            print('Graph is not planar!')
            if maxDegree >= num_colors:
                num_colors = maxDegree+1
                print(f'Using {maxDegree+1} colors instead.')
        #self.colorMapping = nx.coloring.equitable_color(self.communityNetwork, num_colors=num_colors)
        self.colorMapping = nx.coloring.greedy_color(self.communityNetwork, strategy='largest_first')
        self.colorID = np.array([self.colorMapping[index] for index in self.communityID.flatten()]).reshape(self.communityID.shape)
        return self.colorID
        
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
            Array containing new community IDs, corresponding to different colors.
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
        maxDegree = max([d for n, d in self.communityNetwork.degree()])
        if not nx.algorithms.planarity.check_planarity(self.communityNetwork)[0]:
            print('Graph is not planar!')
            if maxDegree >= num_colors:
                num_colors = maxDegree+1
                print(f'Using {maxDegree+1} colors instead.')
        #self.colorMapping = nx.coloring.equitable_color(self.communityNetwork, num_colors=num_colors)
        self.colorMapping = nx.coloring.greedy_color(self.communityNetwork, strategy='largest_first')
        self.colorID = np.array([self.colorMapping[index] for index in self.communityID.flatten()]).reshape(self.communityID.shape)
        return self.colorID
    
class regularCountBins(countBins):
    def __init__(self, nlon, nlat, minLat=60., maxLat=90., minLon=-180, maxLon=180, **kwargs):
        """
        Grid construction by dividing latitude and longitude ranges into a discrete amount of points.
        
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
        """
        self.binType = 'regular'
        dlat = (maxLat - minLat)/nlat
        dlon = (maxLon - minLon)/nlon
        lonOffset = dlon/2
        latOffset = dlat/2
        self.lonBounds = np.linspace(minLon, maxLon, nlon+1)
        self.latBounds = np.linspace(minLat, maxLat, nlat+1)
        lonCenters = np.linspace(minLon + lonOffset, maxLon - lonOffset, nlon)
        latCenters = np.linspace(minLat + latOffset, maxLat - latOffset, nlat)
        self.lonCenters2D, self.latCenters2D = np.meshgrid(lonCenters, latCenters)
        self.lonIdx2D, self.latIdx2D = np.meshgrid(np.arange(nlon), np.arange(nlat))
        self.gridShape = self.lonIdx2D.shape
        super().__init__(np.arange(len(self.lonIdx2D.flatten())))
        self.bindex2D = self.bindex.reshape(self.gridShape)
        
    def particle_count(self, particles, tindex=0):
        count = np.histogram2d(particles.lonlat[tindex,:,0], particles.lonlat[tindex,:,1], bins=[self.lonBounds, self.latBounds])[0]
        if tindex == 0:
            self.initCount = count
        return count
    
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
        assert self.binType == "regular", "Bin type must be regular."
        # Construct empty adjacency dictionary
        # Using dictionary so that labels coincide labels created by InfoMap, rather than being 
        # indices, which might not coincide with labels.
        communityID2D = self.communityID.reshape(self.gridShape)
        self.adjacencyDict = {}
        # Iterate over all cells
        for i in range(self.gridShape[0]):
            for j in range(self.gridShape[1]):
                # Save current community in variable
                currentCommunity = int(communityID2D[i,j])
                # If the current community doesn't have a key and value yet, add an empty
                # set to the dictionary, with the key being the community ID.
                if currentCommunity not in self.adjacencyDict:
                    self.adjacencyDict[currentCommunity] = set()
                self.adjacencyDict[currentCommunity].add(int(communityID2D[i, j+1//self.gridShape[1]]))
                self.adjacencyDict[currentCommunity].add(int(communityID2D[i, j-1]))
                # Careful at northern and southern boundaries. 
                if i<self.gridShape[0]-1:
                    self.adjacencyDict[currentCommunity].add(int(communityID2D[i+1, j]))
                    if mode == 'Moore':
                        self.adjacencyDict[currentCommunity].add(int(communityID2D[i+1, j+1//self.gridShape[1]]))
                        self.adjacencyDict[currentCommunity].add(int(communityID2D[i+1, j-1]))
                if i>0:
                    self.adjacencyDict[currentCommunity].add(int(communityID2D[i-1, j]))
                    if mode == 'Moore':
                        self.adjacencyDict[currentCommunity].add(int(communityID2D[i-1, j+1//self.gridShape[1]]))
                        self.adjacencyDict[currentCommunity].add(int(communityID2D[i-1, j-1]))
        return self.adjacencyDict
    
class hexCountBins(countBins):
    def __init__(self, refinement):
        """
        Basic instance of hexagonal counting bins. Hexagons should be composed of 6 triangles 
        (5 in case of pentagon)
        
        Parameters
        ----------
        refinement : int
            Number of mesh refinement levels to use.
        """
        self.ico = stripy.spherical_meshes.icosahedral_mesh(refinement_levels = refinement)
        facepoints_lons, facepoints_lats = self.ico.face_midpoints()
        self.icofaces = stripy.sTriangulation(np.hstack((self.ico.lons, facepoints_lons)), 
                                              np.hstack((self.ico.lats, facepoints_lats)))
        identifier = np.ones(self.icofaces.npoints)
        identifier[self.ico.npoints:] = 0 # make index of last face 0
        self.hexIds = self.icofaces.simplices[np.where(identifier[self.icofaces.simplices] == 1.0)]
        self.hexIdx = np.arange(self.nHex)
        
    @property
    def lons(self):
        return np.degrees(self.icofaces.lons)

    @property
    def lats(self):
        return np.degrees(self.icofaces.lats)

    @property
    def simplices(self):
        return self.icofaces.simplices 
    
    @property
    def nTriangles(self):
        return self.icofaces.simplices.shape[0]
    
    @property
    def nHex(self):
        return np.unique(self.hexIds).shape[0]
    
    def particle_count(self, particles, tindex=0):
        lons = np.radians(particles.lonlat[tindex,:,0])
        lats = np.radians(particles.lonlat[tindex,:,1])
        countTri = np.zeros(self.nTriangles)
        countHex = np.zeros(self.nHex)
        in_tri = self.icofaces.containing_triangle(lons, lats)
        tri, counts = np.unique(in_tri, return_counts=True)
        for tri, n in zip(vals, counts):
            countTri[tri] = n
            hexIdx = self.hexIds[tri]
            countHex[hexIdx] += n            
        if tindex == 0:
            self.initCount = countTri
            self.initCountHex = countHex
        return countTri, countHex
    
class transMat:
    """
    Basic instance of transition matrix object
    
    Attributes
    ----------
    counter : np.array
        Square matrix with [i,j] indicating number particles from bin i to bin j
    sums : np.array
        Square tiled matrix, with all values in row i equal to the number of particles leaving bin i
    data : np.array
        Actual transition matrix, with [i,j] indicating probability for a particle
        from bin i to bin j (`counter` divided by `sums`)
    """
    def __init__(self, counter):
        """
        Initialize Transition Matrix using a counter matrix. 
        Counter is a symmetric matrix with [i,j] corresponding to particles from bin i to bin j
        
        Parameters
        ----------
        counter : np.array
            Square matrix with [i,j] indicating number particles from bin i to bin j
        """
        self.counter = counter
        self.sums = np.tile(self.counter.sum(axis=1), (self.counter.shape[1],1)).T
        self.data = np.divide(self.counter, self.sums, out=np.zeros_like(self.sums), where=self.sums!=0)

    @classmethod
    def from_pset(cls, pset, countBins, timedelta64 = None, ignore_empty = False, **kwargs):
        """
        Create transition matrix from particle trajectories (from `pset`) given a `countBins`

        Parameters
        ----------
        pset : parcels.ParticleSet
            Particle set containing particle trajectories.
        countBins : comtools.countBins
            Grid containing cells on which the transition matrix is to be created.
        timedelta64 : np.timedelta64
            Timedelta relating to the elapsed time of the particle run for which the transition 
            matrix is to be determined. Example: np.timedelta64(30,'D') for 30 days.
        ignore_empty : bool
            Only create a matrix using countBins that are not empty at the start and finish

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
        lonlatInit, lonlatFinal = lonlat_from_pset(pset, timedelta64)
        # Find initial and final counting bin index for each particle
        if countBins.binType == 'regular':
            # Search for insertion bindex for initial and final lon and lat. -1 because we are using bounds
            # so particles will be inserted on the next bindex. 
            bindexInit = np.dstack((np.searchsorted(countBins.lonBounds, lonlatInit[0,:,0]),
                                     np.searchsorted(countBins.latBounds, lonlatInit[0,:,1])))[0]-1
            bindexFinal = np.dstack((np.searchsorted(countBins.lonBounds, lonlatFinal[0,:,0]), 
                                      np.searchsorted(countBins.latBounds, lonlatFinal[0,:,1])))[0]-1
        elif countBins.binType == 'icosahedral':
            raise NotImplementedError("Transition matrices from icosahedral grids still need to be implemented")
        # `counter` counts particles from bindexInit to bindexFinal
        counter = np.zeros((countBins.n, countBins.n))
        # shape of matrix is determined by in
        N = bindexInit.shape[0]
        # Constructing transition matrix from bin indices
        for i in range(N):
            # Print progress
            inProg = np.linspace(0, N, num=100, dtype='int')
            if i in inProg or i == N-1:
                print (f"\r Determining particle bins. {int(np.ceil(i/(N-1)*100))}%", end="")
            # Only applies to regular grid
            if countBins.binType == 'regular':
                # Test if lon and lat indices are not outside of the domain. Otherwise don't include them.
                if (    bindexFinal[i,0] < countBins.gridShape[1] - 1
                    and bindexFinal[i,0] >= 0
                    and bindexFinal[i,1] < countBins.gridShape[0]
                    and bindexFinal[i,1] >= 0): 
                    sourceIdx = countBins.bindex2D[bindexInit[i,1], bindexInit[i,0]]
                    destIdx = countBins.bindex2D[bindexFinal[i,1], bindexFinal[i,0]]
                    counter[sourceIdx, destIdx] += 1
            elif countBins.binType == 'icosahedral':
                raise NotImplementedError("Transition matrices from icosahedral grids still need to be implemented")
        if ignore_empty:
            nonEmpty = np.logical_or(np.sum(counter, axis=0) > 0, np.sum(counter, axis=1) > 0)
            countBins.nonEmptyBools = nonEmpty
            countBins.nonEmptyBindex = countBins.bindex[nonEmpty]
            counter = counter[nonEmpty]
        return cls(counter, **kwargs)