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
runtime = delta(days = 90)
dt = delta(minutes = 20)
outputdt = delta(hours = 24)
timestamps = [[np.datetime64('2000-01-09', 'D') + np.timedelta64(day, 'D') for day in range(4007)]]
release_time = datetime(2000, 1, 9)
refinement = 11

# ––––––––––––––––––––––––– #

if __name__ == '__main__':
    # Argument parsing
    # """ONLY USE THIS FOR CONTROL RUN OR FIRST TEN YEARS OF RCP AND DON'T USE IT FOR ICOSAHEDRAL GRIDS"""
    parser = argparse.ArgumentParser(description="Advect particles on a rectilinear grid.")
    parser.add_argument('start_date', type=str, help="Particle initialization time. Must be formatted as YYYY-MM-DD.")
    parser.add_argument('-n', '--name', default='', type=str, help='Experiment name to save pset with.')
    parser.add_argument('-d', '--days', default=30, type=int, help='Number of days used for advection.')
    parser.add_argument('-dt', '--advectdt', default=5, type=int, help='Timestep for advection in minutes')
    parser.add_argument('-odt', '--outputdt', default=12, type=int, help='Output timestep in hours')
    parser.add_argument('--minlat', default=60.1, type=float, help='Minimum latitude for rectilinear particle initialization.')
    parser.add_argument('--maxlat', default=89.9, type=float, help='Maximum latitude for rectilinear particle initialization')
    parser.add_argument('--minlon', default=-179.9, type=float, help='Minimum longitude for rectilinear particle initialization')
    parser.add_argument('--maxlon', default=179.9, type=float, help='Maximum latitude for rectilinear particle initialization.')
    parser.add_argument('--nodland', action='store_true', help='Do not remove particles on land.')
    parser.add_argument('--letbeach', action='store_true', help='Let particles beach.')

    args = parser.parse_args()
    
    experiment_name = f"pset_Rrcp_Pico{refinement}_S{str(datetime(2000, 1, 9))[:10]}_D{str(runtime)[:2]}_DT{dt.seconds//60}_ODT{outputdt.days*24 + outputdt.seconds//3600}_wedgedomain"
    
    
    
    fieldset = fieldsetter.read_velocity_field(readdir_ocean_rcp85 + fieldfile_ocean_rcp85, 
                                               meshfile = readdir_mesh + meshfile, 
                                               antiBeach = readdir_ocean_rcp85 + fieldfile_antibeach,
                                               timestamps=timestamps,
                                               tindex = 'record'
                                              ) 

    with open(f'/scratch/DaanR/meshes/ico_mesh_parts_deg_arctic_r{refinement}.pickle', 'rb') as meshf:
        mesh = pickle.load(meshf)

    particles = community.particles(mesh['lons'],
                                    mesh['lats'],
                                    releaseTime = datetime(2000, 1, 9),
                                    )
    domainMask = np.logical_and(np.logical_and(np.degrees(testParticles.lats) > 60, 
                                               np.degrees(testParticles.lats) < 85),
                                np.logical_and(np.degrees(testParticles.lons) < 65,
                                               np.degrees(testParticles.lons) > -45))
    pLons = np.degrees(particles.lons)[domainMask]
    pLats = np.degrees(particles.lats)[domainMask]
    particles = particles(pLons, pLats)
    particles.remove_on_land(fieldset)

    pset = advectParticles.advection(fieldset,
                                     particles,
                                     experiment_name = experiment_name,
                                     runtime = runtime,
                                     dt = dt,
                                     outputdt = outputdt,
                                     overwrite = False,
                                     antiBeach = True
                                    )
