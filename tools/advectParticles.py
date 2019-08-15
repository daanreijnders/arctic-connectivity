# Imports
from datetime import timedelta as delta
from datetime import datetime
import sys
import os.path
import argparse
import warnings
from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

sys.path.append('/home/students/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repo/tools')
import comtools
import fieldsetter
import kernelCollection

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
    
# Particle execution function
def gridAdvection(fieldset, \
                  particleGrid, \
                  experiment_name = '', \
                  runtime = delta(days = 30), \
                  dt = delta(minutes = 5), \
                  outputdt = delta(hours = 12)):
    """
    Advect particles on a `fieldset`. Original particle locations are stored in `particleGrid` object.
    
    Parameters
    ----------
    fieldset : parcels.fieldset
        Fieldset used for advecting particles
    particleGrid: comtools.particleGrid
        Grid containing initial distribution of particles
    experiment_name : str
        Name to label experiment. Saved in filename of the `ParticleSet` output.
    runtime : datetime.timedelta
        Amount of time used for advection
    dt : datetime.timedelta
        Timestep of advection
    outputdt : datetime.timedelta
        Timestep between saved output.
        
    Returns
    -------
    parcels.ParticleSet
        Contains particle trajectories.
    """
    pset = ParticleSet.from_list(fieldset, \
                                 JITParticle, \
                                 particleGrid.lonlat[0,:,0], \
                                 particleGrid.lonlat[0,:,1], \
                                 time = particleGrid.release_times)
    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(kernelCollection.wrapLon)
    if os.path.exists(writedir+"pset_"+experiment_name+".nc"):
        warnings.warn("File already exists!")
    pfile = pset.ParticleFile(name = writedir+"pset_"+experiment_name, outputdt=outputdt)
    pfile.add_metadata("dt", str(dt))
    pfile.add_metadata("Output dt", str(outputdt))
    pfile.add_metadata("Runtime", str(runtime))
    pfile.add_metadata("Release time of first particle", str(particleGrid.release_times[0]))
    print(f"Run: Advecting particles for {runtime}")
    pset.execute(kernels, \
                 runtime = runtime, \
                 dt = dt, \
                 output_file = pfile, \
                 recovery = {ErrorCode.ErrorOutOfBounds: kernelCollection.deleteParticle})
    return pset

# Argument parsing to execute from command line
if __name__ == '__main__':
    # For now, using control_y300
    fieldset = fieldsetter.read_velocity_field(readdir_ocean+fieldfile_ocean, meshfile=readdir_mesh+meshfile) 
    
    parser = argparse.ArgumentParser(description="Advect particles on a rectilinear grid.")
    
    parser.add_argument('plon', type=int, help='Number of particles spaced over longitudes.')
    parser.add_argument('plat', type=int, help='Number of particles spaced over latitudes.')
    parser.add_argument('start_date', type=str, help="Particle initialization time. Must be formatted as YYYY-MM-DD.")
    parser.add_argument('-n', '--name', default='', type=str, help='Experiment name to save pset with.')
    parser.add_argument('-d', '--days', default=30, type=int, help='Number of days used for advection.')
    parser.add_argument('-dt', '--advectdt', default=5, type=int, help='Timestep for advection in minutes')
    parser.add_argument('-odt', '--outputdt', default=12, type=int, help='Output timestep in hours')
    parser.add_argument('--minlat', default=60.1, type=float, help='Minimum latitude for rectilinear particle initialization.')
    parser.add_argument('--maxlat', default=89.9, type=float, help='Maximum latitude for rectilinear particle initialization')
    parser.add_argument('--minlon', default=-179.9, type=float, help='Minimum longitude for rectilinear particle initialization')
    parser.add_argument('--maxlon', default=179.9, type=float, help='Maximum latitude for rectilinear particle initialization.')
    args = parser.parse_args()
    if args.name:
        name = args.name
    else:
        name = ''
    start_year = int(args.start_date[0:4])
    start_month = int(args.start_date[4:6])
    start_day = int(args.start_date[6:8])
    
    particleG = comtools.particleGrid(args.plon,\
                                      args.plat,\
                                      datetime(start_year, start_month, start_day),\
                                      minLat=args.minlat,\
                                      maxLat=args.maxlat,\
                                      minLon=args.minlon,\
                                      maxLon=args.maxlon)
    pset_out = gridAdvection(fieldset,\
                             particleG,\
                             runtime=delta(days=args.days),\
                             dt = delta(minutes=args.advectdt),\
                             outputdt = delta(hours=args.outputdt),\
                             experiment_name=f"{name}_P{args.plon}x{args.plat}_S{start_year}-{start_month}-{start_day}_D{args.days}_DT{args.advectdt}_ODT{args.outputdt}_LAT{args.minlat}-{args.maxlat}_LON{args.minlon}-{args.maxlon}")
