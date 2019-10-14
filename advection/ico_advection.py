# Imports
import numpy as np
from scipy.interpolate import griddata

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
experiment_name = '',
runtime = delta(days = 90)
dt = delta(minutes = 20)
outputdt = delta(hours = 24)
# timestamps = [[np.datetime64('2000-01-09', 'D') + np.timedelta64(day, 'D') for day in range(4007)]]
# release_time = datetime(2000, 1, 9)
release_time = datetime(2000, 1, 1)
refinement = 11
# ––––––––––––––––––––––––– #

# fieldset = fieldsetter.read_velocity_field(readdir_ocean_rcp85+fieldfile_ocean_rcp85, 
#                                            meshfile = readdir_mesh+meshfile, 
#                                            tindex = 'record', 
#                                            timestamps=timestamps
#                                           ) 
fieldset = fieldsetter.read_velocity_field(readdir_ocean_control+fieldfile_ocean_control, 
                                           meshfile = readdir_mesh+meshfile, 
                                           antiBeach = readdir_ocean_rcp85 + fieldfile_antibeach
                                          ) 

with open(f'/scratch/DaanR/meshes/ico_mesh_parts_deg_arctic_r{refinement}.pickle', 'rb') as meshf:
    mesh = pickle.load(meshf)

particles = community.particles(mesh['lons'],
                                mesh['lats'],
                                releaseTime = datetime(2000, 1, 9),
                                )
particles.remove_on_land(fieldset)

pset = advectParticles.advection(fieldset,
                                 particles,
                                 experiment_name = f"pset_Rcontroly300_nobeach_Piconorth{refinement}_S2000-01-1_D90_DT20_ODT24_LAT60-90_LON-180-180",
                                 runtime = runtime,
                                 dt = dt,
                                 outputdt = outputdt,
                                 overwrite = False,
                                 antiBeach = True)
pset.close()