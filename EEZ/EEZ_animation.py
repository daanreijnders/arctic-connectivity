import numpy as np
import xarray as xr
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import cartopy as cart

from glob import glob
from datetime import timedelta as delta
from datetime import datetime

from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

import sys
sys.path.append('/home/students/4302001/arctic-connectivity/tools')
import plot

# Specify paths to velocity field and mesh
#readdir_ocean = '/Users/daanreijnders/Datasets/'
#readdir_ice = '/Users/daanreijnders/Datasets/'

readdir_ice = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ice/arctic/'
readdir_ocean = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ocean/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'

fieldfile_ocean = 'daily_CESM_0.1degree_controlrun_year_300_arctic_timed_no_cord.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic'
meshfile = 'POP_grid_lat1800plus.nc'

writedir = ''

def read_velocity_field(fieldfiles, meshfiles=None):
    if not meshfiles:
        meshfiles = fieldfiles
    filenames = {'U': {'lon': meshfiles,
                       'lat': meshfiles,
                       'data':fieldfiles},
                 'V': {'lon': meshfiles,
                       'lat': meshfiles,
                       'data':fieldfiles}}

    variables = {'U': 'UVEL_5m',
                 'V': 'VVEL_5m'}

    dimensions = {'U': {'time': 'time',
                        'lat': 'U_LAT_2D',
                        'lon': 'U_LON_2D'},
                  'V': {'time': 'time',
                        'lat': 'U_LAT_2D',
                        'lon': 'U_LON_2D'}}
    fieldset = FieldSet.from_pop(filenames, variables, dimensions, allow_time_extrapolation=False)
    fieldset.U.vmax = 10;  fieldset.U.vmin = -10;  # set max of flow to 10 m/s
    fieldset.V.vmax = 10; fieldset.V.vmin = -10;
    return fieldset

fieldset = read_velocity_field(readdir_ocean+fieldfile_ocean, readdir_mesh+meshfile)    

# Load and add EEZ field
EEZ_ds = xr.open_dataset('EEZ.nc')
EEZ = EEZ_ds['EEZ'][:,:]
EEZ_lats = EEZ_ds['latitude'].data
EEZ_lons = EEZ_ds['longitude'].data

EEZ_field = Field(name="EEZ", data=EEZ.data[0], mesh='spherical', lon=EEZ_lons, lat=EEZ_lats, allow_time_extrapolation=True)
fieldset.add_field(EEZ_field)

plot.show(fieldset.EEZ)

# Kernels and particleclasses
class myParticle(JITParticle):
        EEZ = Variable('EEZ', dtype=np.int32, to_write=True)

def EEZ_sampler(particle, fieldset, time):
    particle.EEZ = fieldset.EEZ[time, 0., particle.lat, particle.lon]
        
def wrapLon(particle, fieldset, time):
    if particle.lon > 180.:
        particle.lon = particle.lon - 360.
    if particle.lon < -180.:
        particle.lon = particle.lon + 360.

def deleteParticle(particle, fieldset, time):
    particle.delete()
    
# Execution 
def execute_particles(fieldset, \
                      experiment_name='', \
                      npart=10, \
                      runtime=delta(days=200), \
                      dt = delta(minutes=5), \
                      outputdt = delta(hours = 6), \
                      start=(3,75), \
                      finish=(4,76), \
                      show_pset=False):
    if start[0] > 180 or finish[0] > 180:
        raise ValueError("Longitude bounds must be within range (-180,180)")
        
    startLons = np.ones(npart) * start[0]
    startLats = np.ones(npart) * start[1]
    startTime = np.array([datetime(2000, 1, 1) + delta(days = i) for i in range(npart)]) 
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=myParticle, lon=startLons.tolist(), lat=startLats.tolist(), \
                       time = startTime)
    if show_pset:
        edge = 5 # Degrees around initial distribution
        pset.show()

    kernels = pset.Kernel(AdvectionRK4) + EEZ_sampler + wrapLon
    pfile = pset.ParticleFile(name = writedir+experiment_name+"_particles", outputdt=outputdt)
    print(f"Run: Advecting {npart} particles for {runtime}")

    pset.execute(kernels, \
                 runtime = runtime, \
                 dt = dt, \
                 output_file = pfile, \
                 recovery = {ErrorCode.ErrorOutOfBounds: deleteParticle})
    return pset
# EXECUTE
pset_loc1 = execute_particles(fieldset, experiment_name='pset_EEZ_loc1', start=(-50, 58), show_pset=True, npart = 1000)
pset_loc2 = execute_particles(fieldset, experiment_name='pset_EEZ_loc2', start=(5.15,58.85), show_pset=True, npart = 1000)
# Investigate data. Which EEZs are visited?
pfile_loc1 = xr.open_dataset('pset_EEZ_loc1_particles.nc', decode_cf=True)
pfile_loc2 = xr.open_dataset('pset_EEZ_loc2_particles.nc', decode_cf=True)

plot.particleAnimation.create(pfile_loc1, fieldset.EEZ, nbar=6, EEZ_mapping=f'EEZ_mapping.json', mask=False, titleAttribute='MidNorth')
plot.particleAnimation.create(pfile_loc2, fieldset.U, mask=False, titleAttribute='NorwayBay')

pset_loct_2 = execute_particles(fieldset, experiment_name='pset_EEZ_loct_2', start=(9, 61.5), show_pset=True, npart = 1000, runtime=delta(days=200) )
pfile_loct_2 = xr.open_dataset('pset_EEZ_loct_2_particles.nc', decode_cf=True)
plot.particleAnimation.create(pfile_loct_2, fieldset.U, mask=False, titleAttribute='Test_2')
pfile_loct_2.close()