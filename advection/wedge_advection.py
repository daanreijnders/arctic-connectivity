# Imports
import numpy as np

from datetime import timedelta as delta
from datetime import datetime

import argparse
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
    
if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Advect particles on an icosahedral grid within the wedge domain (60 < latitude < 85, -45 < longitude < 65).")
    parser.add_argument('start_date', type=str, help="Particle initialization time. Must be formatted as YYYY-MM-DD.")
    parser.add_argument('-d', '--days', default=90, type=int, help='Number of days used for advection.')
    parser.add_argument('-dt', '--advectdt', default=20, type=int, help='Timestep for advection in minutes')
    parser.add_argument('-odt', '--outputdt', default=24, type=int, help='Output timestep in hours')
    parser.add_argument('-r', '--refinement', default=11, type=int, help='Refinement level of the icosahedral mesh')
    
    args = parser.parse_args()

    # ––––– Load settings ––––– #
    runtime = delta(days=args.days)
    dt = delta(minutes=args.advectdt)
    outputdt = delta(hours=args.outputdt)
    start_year = int(args.start_date[0:4])
    start_month = int(args.start_date[4:6])
    start_day = int(args.start_date[6:8])
    releaseTime = datetime(start_year, start_month, start_day)
    refinement = args.refinement
    experiment_name = f"Rrcp85_Pico{refinement}_S{start_year}-{start_month}-{start_day}_D{args.days}_DT{args.advectdt}_ODT{args.outputdt}_wedge"
    # ––––––––––––––––––––––––– #
    
    # Fieldset
    timestamps = [[np.datetime64('2000-01-09', 'D') + np.timedelta64(day, 'D') for day in range(4007)]]
    fieldset = fieldsetter.read_velocity_field(readdir_ocean_rcp85 + fieldfile_ocean_rcp85, 
                                               meshfile = readdir_mesh + meshfile, 
                                               antiBeach = readdir_ocean_rcp85 + fieldfile_antibeach,
                                               timestamps = timestamps,
                                               tindex = 'record'
                                              ) 
    # Particles
    with open(f'/scratch/DaanR/meshes/ico_mesh_parts_deg_arctic_r{refinement}.pickle', 'rb') as meshf:
        mesh = pickle.load(meshf)
    tempParticles = community.particles(mesh['lons'],
                                        mesh['lats'],
                                        )
    domainMask = np.logical_and(np.logical_and(tempParticles.lats > 60, 
                                               tempParticles.lats < 85),
                                np.logical_and(tempParticles.lons < 65,
                                               tempParticles.lons > -45))
    pLons = tempParticles.lons[domainMask]
    pLats = tempParticles.lats[domainMask]
    particles = community.particles(pLons, pLats, releaseTime = releaseTime)
    del tempParticles
    particles.remove_on_land(fieldset)
    
    # Pset
    pset = advectParticles.advection(fieldset,
                                     particles,
                                     experiment_name = experiment_name,
                                     runtime = runtime,
                                     dt = dt,
                                     outputdt = outputdt,
                                     overwrite = False,
                                     simple = False
                                    )
