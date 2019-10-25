# Imports
import numpy as np

from datetime import timedelta as delta
from datetime import datetime

import pickle
import sys
import os.path
import warnings
from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

sys.path.append('/home/students/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repo/tools')

import fieldsetter
import kernelCollection
import community
import advectParticles

readdir_ocean_rcp85 = '/data/oceanparcels/input_data/CESM/0.1_deg/rcp8.5/ocean/arctic/'
readdir_ocean_control = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ocean/arctic/'
readdir_ice = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ice/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'

fieldfile_ocean_rcp85 = 'daily_CESM_0.1degree_rcp8.5run_years_2000-2010_arctic.nc'
fieldfile_ocean_control = 'daily_CESM_0.1degree_controlrun_year_300_arctic_timed_no_cord.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic.nc'
meshfile = 'POP_grid_coordinates.nc'
fieldfile_antibeach = "CESM_POP_0.1degree_rcp8.5run_j1800_antibeach_vel.nc"

writedir = '/scratch/DaanR/psets/'
    
# ––––– SETTINGS –––––––––– #

runtime = delta(days = 90)
dt = delta(minutes = 20)
outputdt = delta(hours = 24)
timestamps = [[np.datetime64('2000-01-09', 'D') + np.timedelta64(day, 'D') for day in range(4007)]]
releaseTime = datetime(2000, 1, 9)
#refinement = 11
nlon = 2200
nlat = 500
minLat = 60, 
maxLat = 85, 
minLon = -45, 
maxLon = 65,
experiment_name = f"Rrcp85_P{nlon}x{nlat}_S{str(datetime(2000, 1, 9))[:10]}_D{str(runtime)[:2]}_DT{str(dt)[:2]}_ODT{str(outputdt)[:2]}_LAT{str(minLat)}-{str(maxLat)}_LON{str(minLon)}-{str(maxLon)}"
# ––––––––––––––––––––––––– #

fieldset = fieldsetter.read_velocity_field(readdir_ocean_rcp85 + fieldfile_ocean_rcp85, 
                                           meshfile = readdir_mesh + meshfile, 
                                           antiBeach = readdir_ocean_rcp85 + fieldfile_antibeach,
                                           timestamps = timestamps,
                                           tindex = 'record'
                                          ) 

particles = community.particles.from_regular_grid(nlon, nlat, 
                                                  minLat=minLat, 
                                                  maxLat=maxLat, 
                                                  minLon=minLon, 
                                                  maxLon=maxLon, 
                                                  releaseTime = releaseTime)
particles.remove_on_land(fieldset)

pset = advectParticles.advection(fieldset,
                                 particles,
                                 experiment_name = experiment_name,
                                 runtime = runtime,
                                 dt = dt,
                                 outputdt = outputdt,
                                 overwrite = False,
                                 simple = False
                                )
