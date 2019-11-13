"""
@author: Daan Reijnders
Function for easily setting fieldsets from Global Ocean Physical Reanalysis data in Parcels, including CLI.
"""
import numpy as np
import xarray as xr
from glob import glob
from parcels import (grid, Field, FieldSet, VectorField, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4, ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

def create(startDate, days, lonRange=(-180, 180), latRange=(55,90), antiBeach=True, halo=False, **kwargs):
    """
    Creates a parcels.FieldSet object from hydrodynamic data in a netCDF file.
    
    Parameters
    ----------
    startDate : str, int
        String or int indicating the first date to load (YYYYMMDD)
    days : int
        Number of days to load    
    antiBeach : bool
        Load anti-beach field   
    halo : bool
        Add a periodic halo
        
    Returns
    ----------
    parcels.FieldSet    
    """
    readDir = "/data/oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/"
    fieldFiles = sorted(glob(readDir + "mercatorglorys12v1_gl12_mean_*.nc"))
    unBeachFile = "mercatorglorys12v1_gl12_unbeaching_vel.nc"
    startFile = glob(readDir + f"mercatorglorys12v1_gl12_mean_{startDate}_*.nc")
    assert len(startFile) == 1, "No file found for this `start_date`."
    startFileIndex = fieldFiles.index(startFile[0])
    endFileIndex = startFileIndex + days
    if endFileIndex >= len(fieldFiles) - 1:
        days = len(fieldFiles) - startFileIndex -1
        endFileIndex = len(fieldFiles) - 1
        warnings.warn("\n Timespan of simulation exceeds the amount of data that is available. " \
                     +"Reducing the amount of `days` to " + str(days) +".")
    selectedFiles = fieldFiles[startFileIndex:endFileIndex]

    variables = {'U' : 'uo',
                 'V' : 'vo'}
    dimensions = {'U': {'time' : 'time',
                        'lat' : 'latitude',
                        'lon' : 'longitude'},
                  'V': {'time' : 'time',
                        'lat' : 'latitude',
                        'lon' : 'longitude'}}
    mesh = fieldFiles[0]
    filenames = {'U' : {'lon' : mesh, 
                        'lat' : mesh, 
                        'data' : selectedFiles},
                 'V' : {'lon' : mesh, 
                        'lat' : mesh, 
                        'data' : selectedFiles}}  
    
    ds = xr.open_dataset(fieldFiles[0])
    
    minLonIdx = np.searchsorted(ds.longitude, lonRange[0]) 
    maxLonIdx = np.searchsorted(ds.longitude, lonRange[1])
    minLatIdx = np.searchsorted(ds.latitude, latRange[0]) 
    maxLatIdx = np.searchsorted(ds.latitude, latRange[1])
    
    indices = {'lon' : range(minLonIdx, maxLonIdx),
               'lat' : range(minLatIdx, maxLatIdx)}

    fieldset = FieldSet.from_netcdf(selectedFiles, 
                                    variables, 
                                    dimensions, 
                                    indices = indices,
                                    allow_time_extrapolation = False,
                                   )
    
    if antiBeach:
        dimensions = {'lon': 'longitude', 'lat': 'latitude'}
        U_unbeach = Field.from_netcdf(readDir + unBeachFile, ('U_unbeach', 'unBeachU'), dimensions, fieldtype='U')
        V_unbeach = Field.from_netcdf(readDir + unBeachFile, ('V_unbeach', 'unBeachV'), dimensions, fieldtype='V')
        fieldset.add_field(U_unbeach)
        fieldset.add_field(V_unbeach)
        UVunbeach = VectorField('UVunbeach', fieldset.U_unbeach, fieldset.V_unbeach)
        fieldset.add_vector_field(UVunbeach)
        
    if halo:
        fieldset.add_periodic_halo(zonal = True)
    
    fieldset.computeTimeChunk(fieldset.U.grid.time[0], 1)
    
    fieldset.landMask = np.isnan(ds.uo[0, 0, minLatIdx:maxLatIdx, minLonIdx:maxLonIdx].data)
    ds.close()
    return fieldset