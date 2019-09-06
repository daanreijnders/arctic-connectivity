import numpy as np
import xarray as xr

from datetime import timedelta as delta
from datetime import datetime

from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

# Specify paths to velocity field and mesh
readdir_ice = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ice/arctic/'
readdir_ocean = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ocean/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'

fieldfile_ocean = 'daily_CESM_0.1degree_controlrun_year_300_arctic_region_timed.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic.nc'
meshfile = 'POP_grid_coordinates.nc'

# Functions for creating fieldset from velocity field and adding ice fields
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
    return fieldset

def add_ice_fields(fieldset, fieldfile, iceVars=['aice', 'hisnap', 'hi'], meshfile=None):
    if not meshfile:
        meshfile = fieldfile
    for varName in iceVars:
        filenames = {'lon': meshfile,
                     'lat': meshfile,
                     'data': fieldfile}
        variable = (varName, varName)
        dimensions = {'time': 'time',
                      'lat': 'TLAT',
                      'lon': 'TLON'}
        field = Field.from_netcdf(filenames, variable, dimensions, allow_time_extrapolation=False)
        fieldset.add_fieldset(field)
        
# Execute: create fieldset and add ice fields to fieldset
fieldset = read_velocity_field(readdir_ocean+fieldfile_ocean, meshfile=readdir_mesh+meshfile)  
add_ice_fields(fieldset, readdir_ice+fieldfile_ice, meshfile=readdir_mesh+meshfile) # BUG APPEARS HERE