import numpy as np
from scipy.interpolate import griddata
import xarray as xr
import pandas as pd

from datetime import timedelta as delta
from datetime import datetime
import sys
import argparse

from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

sys.path.append('/home/students/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repo/tools')
import comtools
import fieldsetter

# Specify paths to velocity field and mesh
# readdir_ocean = '/Users/daanreijnders/Datasets/'
# readdir_ice = '/Users/daanreijnders/Datasets/'
# readdir_mesh = '/Users/daanreijnders/Datasets/'

readdir_ice = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ice/arctic/'
readdir_ocean = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ocean/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'

fieldfile_ocean = 'daily_CESM_0.1degree_controlrun_year_300_arctic_timed_no_cord.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic.nc'
meshfile = 'POP_grid_coordinates.nc'

writedir = '/scratch/DaanR/psets/'

# Kernels for circular boundary and for deleting particles as fallback.
def wrapLon(particle, fieldset, time):
    if particle.lon > 180.:
        particle.lon = particle.lon - 360.
    if particle.lon < -180.:
        particle.lon = particle.lon + 360.

def deleteParticle(particle, fieldset, time):
    particle.delete()
    
# Particle execution function
def gridAdvection(fieldset, \
#                   countGrid, \
                  particleGrid, \
                  experiment_name='', \
                  runtime=delta(days=30), \
                  dt = delta(minutes=5), \
                  outputdt = delta(hours = 12)):
    pset = ParticleSet.from_list(fieldset, JITParticle, particleGrid.lonlat[0,:,0], particleGrid.lonlat[0,:,1])
    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(wrapLon)
    pfile = pset.ParticleFile(name = writedir+"pset_"+experiment_name, outputdt=outputdt)
    print(f"Run: Advecting particles for {runtime}")
    pset.execute(kernels, \
                 runtime = runtime, \
                 dt = dt, \
                 output_file = pfile, \
                 recovery = {ErrorCode.ErrorOutOfBounds: deleteParticle})
    return pset

if __name__ == '__main__':
    # For now, using control_y300
    fieldset = fieldsetter.read_velocity_field(readdir_ocean+fieldfile_ocean, meshfile=readdir_mesh+meshfile)  
    parser = argparse.ArgumentParser(description="Advect particles on a rectilinear grid.")
    parser.add_argument('plon', type=int, help='Number of particles spaced over longitudes.')
    parser.add_argument('plat', type=int, help='Number of particles spaced over latitudes.')
    parser.add_argument('-n', '--name', default='', type=str, help='Experiment name to save pset with.')
    parser.add_argument('-d', '--days', default=30, type=int, help='Number of days used for advection.')
    parser.add_argument('-odt', '--outputdt', default=12, type=int, help='Output timestep in hours')
    parser.add_argument('--minlat', default=60.5, type=float, help='Minimum latitude for rectilinear particle initialization.')
    parser.add_argument('--maxlat', default=89.5, type=float, help='Maximum latitude for rectilinear particle initialization')
    parser.add_argument('--minlon', default=-179.5, type=float, help='Minimum longitude for rectilinear particle initialization')
    parser.add_argument('--maxlon', default=179.5, type=float, help='Maximum latitude for rectilinear particle initialization.')
    args = parser.parse_args()
    if args.name:
        name = args.name
    else:
        name = ''
        
    particleG = comtools.particleGrid(args.plon,\
                                      args.plat,\
                                      0,\
                                      minLat=args.minlat,\
                                      maxLat=args.maxlat,\
                                      minLon=args.minlon,\
                                      maxLon=args.maxlon)
    psetTest = gridAdvection(fieldset,\
                             particleG,\
                             runtime=delta(days=args.days),\
                             outputdt = delta(hours=args.outputdt)
                             experiment_name=f"{name}_P{args.plon}x{args.plat}_D{args.days}_ODT{args.outputdt}_LAT{args.minlat}-{args.maxlat}_LON{args.minlon}-{args.maxlon}")