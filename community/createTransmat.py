import numpy as np
import xarray as xr
import pandas as pd

from datetime import timedelta as delta
from datetime import datetime

import networkx as nx

from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

import sys
sys.path.append('/home/students/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repo/tools')
import plot
import lifeline
import comtools

readdir_ice = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ice/arctic/'
readdir_ocean = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ocean/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'

fieldfile_ocean = 'daily_CESM_0.1degree_controlrun_year_300_arctic_timed_no_cord.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic.nc'
meshfile = 'POP_grid_coordinates.nc'

writedir = ''

# Move this function to a tool
def read_velocity_field(fieldfiles, meshfile=None):
    if not meshfile:
        meshfile = fieldfiles
    filenames = {'U': {'lon': meshfile,
                       'lat': meshfile,
                       'data':fieldfiles},
                 'V': {'lon': meshfile,
                       'lat': meshfile,
                       'data':fieldfiles}}

    variables = {'U': 'UVEL_5m',
                 'V': 'VVEL_5m'}

    dimensions = {'U': {'time': 'time',
                        'lat': 'ULAT',
                        'lon': 'ULON'},
                  'V': {'time': 'time',
                        'lat': 'ULAT',
                        'lon': 'ULON'}}
    fieldset = FieldSet.from_pop(filenames, variables, dimensions, allow_time_extrapolation=False)
    fieldset.U.vmax = 10;  fieldset.U.vmin = -10;  # set max of flow to 10 m/s
    fieldset.V.vmax = 10; fieldset.V.vmin = -10;
    
    fieldset.computeTimeChunk(fieldset.U.grid.time[0], 1)
    fieldset.landMask = np.logical_or(fieldset.U.data[0,:,:]==-0.01, np.abs(fieldset.U.data[0,:,:])<0.0000001)
    return fieldset
fieldset = read_velocity_field(readdir_ocean+fieldfile_ocean, meshfile=readdir_mesh+meshfile)  

countG = comtools.countGrid(240, 40)
particleG = comtools.particleGrid(3590, 590, 0)
particleG.remove_on_land(fieldset)
initCount = countG.particleCount(particleG).T

mytransmat = comtools.createTransition('/data/oceanparcels/output_data/data_Daan/pset_control_y300_P3590x590.nc', countG)
# Create network from numpy array (adjacency matrix)
G = nx.from_numpy_matrix(mytransmat.data, create_using=nx.DiGraph())
# Export to Pajek (.net) format for further manual processing
nx.write_pajek(G, 'out/graph_control_y300_P3590x590_C240x40.net')