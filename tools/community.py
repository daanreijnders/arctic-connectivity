# Imports
import pandas as pd
import numpy as np
import xarray as xr
import networkx as nx
from scipy.interpolate import griddata
from scipy.spatial import cKDTree, SphericalVoronoi
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
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
    
def get_cartesian(lon, lat, R=1, mode='rad', ):
    """
    Convert spherical coordinates to cartesian coordinates.
    
    Parameters
    ----------
    lon : float, np.array
        longitude coordinate(s)
    lat : float, np.array
        latitude coordinate(s)
    R : int, float
        radius of sphere
    mode : str
        Either 'rad' or 'deg', indicating whether coordinates are supplied
        in radians or degrees respectively.
        
    Returns
    -------
    x, y, z : (float, float, float)
        Cartesian coordinates
    """
    if mode=='deg':
        lat, lon = np.radians(lat), np.radians(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x, y, z

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
        if fieldset.U.grid.lon.ndim == 1:
            mesh = np.meshgrid(fieldset.U.grid.lon, fieldset.U.grid.lat)
            lonlatMask = griddata(np.dstack((mesh[0].flatten(), 
                                             mesh[1].flatten()))[0,:,:], 
                                  landMask.flatten(), 
                                  self.lonlat[0,:,:], 
                                  method='nearest')
        else: 
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
        ax.scatter(self.lonlat[tindex, :, 0], self.lonlat[tindex, :, 1], transform = cart.crs.Geodetic(), **kwargs)
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

    Attributes
    ----------
    binType : str
        Indicates the type of bin: `regular` or `hexagonal`
    """
        
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
        communityFlow = np.zeros(self.n)
        communityFlow.fill(np.nan)
        for n in range(self.n):
            # +1 because counting in DF starts at zero
            communityID[n] = int(self.communityDF['module'].loc[self.bindex[n]+1])
            communityFlow[n] = self.communityDF['flow'].loc[self.bindex[n]+1] 
        self.communityID = communityID
        self.communityFlow = communityFlow
        
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
            if maxDegree >= num_colors:
                num_colors = maxDegree+1
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
            if maxDegree >= num_colors:
                num_colors = maxDegree+1
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
        self.bindex = (np.arange(len(self.lonIdx2D.flatten())))
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
    """
    Basic instance of hexagonal counting bins. 
    Hexagons are generated from the Voronoi diagram of refined icosahedral grids.
    
    Attributes
    ----------
    bintype : str
        Type of bin ('hexagonal')
    points : np.array
        (N x 3) dimensional array containing cartesian (x, y, z)
        coordinates of the vertices of the generating triangulation.
    lons : np.array
        N longitudes (degrees) of vertices of the generating triangulation.
    lats : np.array
        N latitudes (degrees) of vertices of the generating triangulation
    vertexIndices : np.array
        N indices of the vertices
    simplices : np.array
        N dimensional array holding tuples of length 3, holding the indices
        of the vertices of each triangle in the generating triangulation
    """
    def __init__(self, points, lons, lats, vertexIndices, simplices):
        """
        Basic instance of hexagonal counting bins. Hexagons should be composed of 6 triangles 
        (5 in case of pentagon)
        
        Parameters
        ----------
        points : np.array
            (N x 3) dimensional array containing cartesian (x, y, z)
            coordinates of the vertices of the generating triangulation.
        lons : np.array
            N longitudes (degrees) of vertices of the generating triangulation.
        lats : np.array
            N latitudes (degrees) of vertices of the generating triangulation
        vertexIndices : np.array
            N indices of the vertices
        simplices : np.array
            N dimensional array holding tuples of length 3, holding the indices
            of the vertices of each triangle in the generating triangulation
        """
        self.binType = 'hexagonal'
        self.points = points
        self.lons = lons
        self.lats = lats
        self.vertexIndices = vertexIndices
        self.simplices = simplices
    
    @property
    def n(self):
        """
        Returns
        -------
        Number of points in the triangulation
        """
        return self.sv.points.shape[0]

    
    @classmethod
    def from_stripy(cls, refinement):
        """
        Create a hexCountBins` instance through a spherical 
        icosahedral mesh obtained using `stripy`.
        
        Parameters
        ----------
        refinement : int
            Refinement level of the mesh. The mesh is recursively refined through bisection
            of the edges
        
        Returns
        -------
        bins : hexCountBins
            hexCountBins instance
        """
        try:
            ico = stripy.spherical_meshes.icosahedral_mesh(refinement_levels = refinement)
            bins = cls(ico.points,
                       np.degrees(ico.lons), 
                       np.degrees(ico.lats),
                       ico._permutation,
                       ico.simplices)
            return bins
        except NameError:
            raise NameError("Has the `stripy` module been imported?")
        
    def calculate_neighbors(self):
        """
        Create a dictionary with indices of neighbors for each vertex (key)
        in the generating triangulation.
        """
        self.neighbors = {}
        for vertex in self.vertexIndices:
            self.neighbors[vertex] = set()

        # For each simplex, fill dictionary with simplex and neighbor information
        for simplex in self.simplices:
            self.neighbors[simplex[0]].add(simplex[1])
            self.neighbors[simplex[0]].add(simplex[2])
            self.neighbors[simplex[1]].add(simplex[0])
            self.neighbors[simplex[1]].add(simplex[2])
            self.neighbors[simplex[2]].add(simplex[0])
            self.neighbors[simplex[2]].add(simplex[1])
    
    def calculate_voronoi(self, mask = None, innerMaskLevel=0, outerMaskLevel = 0):
        """
        Calculate a voronoi diagram from the generating triangulation.
        Uses the `scipy.spatial.SphericalVoronoi()` function.
        
        Parameters
        ----------
        mask : hexMask
            Mask to apply to the points used for generating the diagram.
            This can significantly reduce calculation times if the generating,
            triangulation is fine, as only a small number of generating point
            can be selected
        innerMaskLevel : int
            Mask level used to calculate the Voronoi mask, which in turn is
            used to select which binCounts to return
        outerMaskLevel : int
            Mask level used for selecting generator vertices that will be 
            used in the Voronoi diagram
        """
        self.innerMaskLevel = innerMaskLevel
        self.outerMaskLevel = outerMaskLevel
        self.mask = mask
        # Calculate voronoi diagram
        if mask:
            self.sv = SphericalVoronoi(self.points[mask[outerMaskLevel]])
        else:
            self.sv = SphericalVoronoi(self.points)
        # Sort the vertices of each region so that they are clockwise with respect to the generator
        self.sv.sort_vertices_of_regions()
        assert self.sv.points.shape[0] == mask.indices[outerMaskLevel].shape[0], \
               "Voronoi should contain as many points as there are Trues in the mask."
    
        # Convert the longitudes and latitudes of the generating vertices from cartesian coordinates to spherical
        # coordinates in degrees
        svTriCenterLats, svTriCenterLons = cartesian_to_spherical(self.sv.points[:, 0], self.sv.points[:, 1], self.sv.points[:, 2])[1:]
        self.svTriCenterLats, self.svTriCenterLons = (np.degrees(svTriCenterLats.value), np.degrees(svTriCenterLons.value))
        self.svTriCenterLons = np.where(self.svTriCenterLons>180, self.svTriCenterLons-360, self.svTriCenterLons)
        
        # Convert the longitudes and latitudes of the voronoi vertices from cartesian coordinates to spherical
        # coordinates in degrees
        svVertexLats, svVertexLons = cartesian_to_spherical(self.sv.vertices[:, 0], self.sv.vertices[:, 1], self.sv.vertices[:, 2])[1:]
        self.svVertexLats, self.svVertexLons = (np.degrees(svVertexLats.value), np.degrees(svVertexLons.value))
        self.svVertexLons = np.where(self.svVertexLons>180, self.svVertexLons-360, self.svVertexLons)
        
        # Create list of voronoi simplices, based only on the generating vertices
        # (which may have been masked before using outerMaskLevel)
        # Also create a list of their longitudes and latitudes
        # by stacking the coordinates of the generator vertices on top of those
        # of the Voronoi vertices
        svSimplices = []
        self.bindex = np.arange(self.n)
        for generatorVertex in self.bindex:
            region = np.array(self.sv.regions[generatorVertex]) + self.n
            nTriangles = len(region)
            for t in range(nTriangles):
                svSimplices.append([generatorVertex, region[t], region[(t+1)%nTriangles]])
        self.svSimplices = np.array(svSimplices)
        self.svTriLons = np.hstack((self.svTriCenterLons, self.svVertexLons))
        self.svTriLats = np.hstack((self.svTriCenterLats, self.svVertexLats))
        assert np.unique(self.svSimplices).max() + 1 == self.svTriLons.shape[0]  == self.svTriLats.shape[0], \
               "Maximum element of svSimplices must correspond to the last index of svTriLons and svTriLats"
        
        # Create svMask, which is used to select which simplices of generating vertices can be used further
        # (simplices with pseudo-'infinite' coordinates can be excluded this way)
        self.svMask = mask.mask[innerMaskLevel][mask.indices[outerMaskLevel]][self.svSimplices[:, 0]]
        assert self.svMask.shape[0] == self.svSimplices.shape[0], \
               "Mask size should match svSimplices size"
    
    def calculate_subsetted_neighbors(self):
        """
        Create a dictionary with indices of neighbors for each vertex (key)
        in the subsetted triangulation.
        """
        originalIndices = self.mask.indices[self.outerMaskLevel]
        transDict = dict(zip(originalIndices, self.bindex))
        self.subsettedNeighbors = {}
        for generatorVertex in self.bindex:
            self.subsettedNeighbors[generatorVertex] = set()
            neighbors = self.neighbors[originalIndices[generatorVertex]]
            for n in neighbors:
                try:
                    self.subsettedNeighbors[generatorVertex].add(transDict[n])
                except KeyError:
                    pass
    
    def create_KDTree(self):
        """
        Create a k-dimensional tree of the (masked) generating vertices (used for interpolation),
        since interpolation in the voronoi simplices is by definition
        equivalent to finding the nearest generating vertex.
        """
        if not hasattr(self, 'sv'):
            raise RuntimeError("Cannot create KDTree before calculating the (masked) Spherical voronoi division.")
        self.tree = cKDTree(self.sv.points)
    
    def query_tree(self, points, **kwargs):
        """
        Check if a k-d tree already exist and query it.
        
        Parameters
        ----------
        points : np.array
            (m x 3) dimensional array of m points to query the tree with
        """
        if not hasattr(self, "tree"):
            self.create_KDTree()
        return self.tree.query(points)
        
    def particle_count(self, particles, tindex=0):
        """
        Create 'histogram' of particles in hexBins.
        
        Parameters
        ----------
        particles : community.particles
            Particles to create a histogram with.
        tindex : int
            Time index of particles.lonlat to determine the count for.
        
        Returns
        -------
        count : np.array
            Array containing the counts per bin index.
        """
        if not hasattr(self, "tree"):
            self.create_KDTree()
        # Convert spherical coordinates of points to cartesian coordinates
        xp, yp, zp = get_cartesian(particles.lonlat[tindex, :, 0], particles.lonlat[tindex, :, 1], mode='deg')
        # Query the tree to get the closest point
        closest = self.tree.query(np.dstack((xp, yp, zp))[0])[1]
        # Count particles per bin
        vals, counts = np.unique(closest, return_counts=True)
        transdict = dict(zip(vals, counts))
        # Arange counts in the correct order of bin indices
        if hasattr(self, "svMask"):
            count = np.array([transdict[i] if i in transdict.keys() else 0 for i in self.svSimplices[:,0][self.svMask]])
        else:
            count = np.array([transdict[i] if i in transdict.keys() else 0 for i in self.svSimplices[:,0]])
        if tindex == 0:
            self.initCount = count
        return count
    
    def find_adjacency(self):
        """
        Create an adjacency list: for each node (grid cell), determine which nodes are bordering this node.

        Returns
        -------
        dict
            Containing keys corresponding to community IDs and values being `set` objects
            containing IDs of bordering communities.
        """
        if not hasattr(self, "subsettedNeighbors"):
            self.calculate_subsetted_neighbors()
        
        self.adjacencyDict = {}
        for generatorVertex in self.bindex:   
            self.mask.indices[self.outerMaskLevel][generatorVertex]
            
        # Construct empty adjacency dictionary
        # Using dictionary so that labels coincide labels created by InfoMap, rather than being 
        # indices, which might not coincide with labels.
        self.adjacencyDict = {}
        # Iterate over all cells
        for vertex in self.bindex:
            # Save current community in variable
            currentCommunity = int(self.communityID[vertex])
            # If the current community doesn't have a key and value yet, add an empty
            # set to the dictionary, with the key being the community ID.
            if currentCommunity not in self.adjacencyDict:
                self.adjacencyDict[currentCommunity] = set()
            for neighbor in self.subsettedNeighbors[vertex]:
                self.adjacencyDict[currentCommunity].add(int(self.communityID[neighbor]))
        return self.adjacencyDict
    
    def flag_on_boundary(self):
        """
        Checks whether a cell is on a boundary between two communities.
        
        Returns
        -------
        np.array
            Array with 1 if cell is on a boundary and 0 if it is not
        """
        self.onBoundary = np.zeros(self.bindex.shape)
        for vertex in self.bindex:
            currentCommunity = int(self.communityID[vertex])
            for neighbor in self.subsettedNeighbors[vertex]:
                if int(self.communityID[neighbor]) != currentCommunity:
                            self.onBoundary[vertex] = 1
        return self.onBoundary
    
class hexMask:
    """
    Mask that can be used to determine which generating vertices in hexCountBins are kept,
    given certain constraints in latitude and longitude. Mask can 'grow', by adding neighboring
    vertices in successive levels
    
    Attributes
    ----------
    mask : dict
        For a given level (integer key), contains an array with mask
    indices : dict
        For a given level (integer key), contains an array with indices of mask that are True
    """
    def __init__(self, hexBins, minLon, maxLon, minLat, maxLat, bleed = 0):
        """
        Parameters
        ----------
        hexBins : community.hexCountBins
            Hexagonal counting bins to create the mask for
        minLon : float
            Minimum longitude of generating vertices
        maxLon : float
            Maximum longitude of generating vertices
        minLat : float
            Minimum latitude of generating vertices
        maxLat : float
            Maximum latitude of generating vertices
        bleed : float
            Increase the constraints in latitude and longitude by this margin
        """
        if hasattr(hexBins, "neighbors"):
            self.hexBins = hexBins
        else:
            hexBins.calculate_neighbors()
            self.hexBins = hexBins

        self.mask = {}
        self.indices = {}
        self.mask[0] = np.logical_and(np.logical_and(self.hexBins.lons > minLon - bleed,
                                                     self.hexBins.lons < maxLon + bleed),
                                      np.logical_and(self.hexBins.lats > minLat - bleed, 
                                                     self.hexBins.lats < maxLat + bleed))
        # Determine indices of mask
        self.indices[0] = np.array([self.hexBins.vertexIndices[i] for i in np.arange(len(self.mask[0])) if self.mask[0][i]])

    def growLevel(self):
        """
        Grow the mask by one level: mark neighboring vertices as True
        """
        currMax = max(self.mask.keys())
        self.mask[currMax + 1] = np.copy(self.mask[currMax])
        for i in self.indices[currMax]:
            for j in self.hexBins.neighbors[i]:
                self.mask[currMax + 1][j] = True
        self.indices[currMax + 1] = np.array([self.hexBins.vertexIndices[i] for i in np.arange(len(self.mask[currMax + 1])) if self.mask[currMax + 1][i]])

    def growToLevel(self, toLevel):
        """
        Grow mask to the desired level
        """
        currMax = max(self.mask.keys())
        if toLevel <= currMax:
            print("Level already reached")
        else:
            while toLevel > currMax:
                self.growLevel()
                currMax = max(self.mask.keys())

    def __getitem__(self,index):
        """
        When mask gets indexed, return the mask array for that level.
        """
        try:
            return self.mask[index]
        except:
            raise IndexError("Mask growth level not available")
    
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
        elif countBins.binType == 'hexagonal':
            if not hasattr(countBins, "tree"):
                countBins.create_KDTree()
            # Convert spherical coordinates to cartesian
            xInit, yInit, zInit = get_cartesian(lonlatInit[0,:,0], lonlatInit[0,:,1], mode='deg')
            xFinal, yFinal, zFinal = get_cartesian(lonlatFinal[0,:,0], lonlatFinal[0,:,1], mode='deg')
            # Check which indices are non NaNs (checking only for x, since NaNs in lonlat propagate in x,y,z equally)
            noNaNIndices = np.logical_and(~np.isnan(xInit), ~np.isnan(xFinal))                      
            # Find index of containing Voronoi region by querying tree
            bindexInit = countBins.tree.query(np.dstack((xInit, yInit, zInit))[0][noNaNIndices])[1]
            bindexFinal = countBins.tree.query(np.dstack((xFinal, yFinal, zFinal))[0][noNaNIndices])[1]
            
        # `counter` counts particles from bindexInit to bindexFinal. Square matrix should be of size countBins.n
        counter = np.zeros((countBins.n, countBins.n))
        # Number of particles to iterate over
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
            elif countBins.binType == 'hexagonal':
                counter[bindexInit[i], bindexFinal[i]] += 1
                
        if ignore_empty:
            nonEmpty = np.logical_or(np.sum(counter, axis=0) > 0, np.sum(counter, axis=1) > 0)
            countBins.nonEmptyBools = nonEmpty
            countBins.nonEmptyBindex = countBins.bindex[nonEmpty]
            counter = counter[nonEmpty]
        return cls(counter, **kwargs)