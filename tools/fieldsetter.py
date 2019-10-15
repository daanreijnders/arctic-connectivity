"""
@author: Daan Reijnders
Function for easily setting fieldsets in Parcels, including CLI
"""

import numpy as np
import xarray as xr
import pandas as pd
from parcels import (grid, Field, FieldSet, VectorField, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4, ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

def read_velocity_field(fieldfiles, meshfile=None, antiBeach=None, mode='pop', tindex='time', **kwargs):
    """
    Creates a parcels.FieldSet object from hydrodynamic data in a netCDF file.
    
    Parameters
    ----------
    fieldfiles : str
        String pointing to hydrodynamic data.
    
    meshfile : str
        String pointing to grid data. Use 'None' if grid data is stored in fieldfiles.
        
    antiBeach : str
        String pointing to file with antiBeach velocities. If `None`, no antiBeach velocity will be saved.
        
    mode : str
        String indicating way of loading data: 'pop' for B-grid, 'netcdf' for A-grid.
    
    tindex : str
        String indicating the name of the time dimension. Workaround for RCP8.5 files.
    
    Returns
    ----------
    parcels.FieldSet    
    """
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

    dimensions = {'U': {'time': tindex,
                        'lat': 'ULAT',
                        'lon': 'ULON'},
                  'V': {'time': tindex,
                        'lat': 'ULAT',
                        'lon': 'ULON'}}
    if mode == 'pop':
        fieldset = FieldSet.from_pop(filenames, variables, dimensions, allow_time_extrapolation=False, **kwargs)
    if mode == 'netcdf':
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, allow_time_extrapolation=False, **kwargs)
    fieldset.U.vmax = 10;  fieldset.U.vmin = -10;  # set max of flow to 10 m/s
    fieldset.V.vmax = 10; fieldset.V.vmin = -10;
    
    fieldset.computeTimeChunk(fieldset.U.grid.time[0], 1)
    fieldset.landMask = np.logical_or(fieldset.U.data[0,:,:]==-0.01, np.abs(fieldset.U.data[0,:,:])<0.0000001)
    
    if antiBeach:
        dimensions = {'lon': 'ULON', 'lat': 'ULAT'}
        U_unbeach = Field.from_netcdf(antiBeach, ('U_unbeach', 'unBeachU'), dimensions, fieldtype='U')
        V_unbeach = Field.from_netcdf(antiBeach, ('V_unbeach', 'unBeachV'), dimensions, fieldtype='V')
        fieldset.add_field(U_unbeach)
        fieldset.add_field(V_unbeach)
        UVunbeach = VectorField('UVunbeach', fieldset.U_unbeach, fieldset.V_unbeach)
        fieldset.add_vector_field(UVunbeach)
    return fieldset

# def add_ice_fields(fieldset, fieldfile, iceVars=['aice', 'hisnap', 'hi'], meshfile=None):
#     """Not working for now"""
#     if not meshfile:
#         meshfile = fieldfiles
#     for varName in iceVars:
#         filenames = {'lon': [meshfile],
#                      'lat': [meshfile],
#                      'data': [fieldfile]}
#         variable = (varName, varName)
#         dimensions = {'time': 'time',
#                       'lat': 'TLAT',
#                       'lon': 'TLON'}
#         field = Field.from_netcdf(filenames, variable, dimensions, allow_time_extrapolation=False)
#         fieldset.add_field(field)