# Imports
import numpy as np
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
import community
import fieldsetter
import kernelCollection

# Specify paths to velocity field and mesh
# readdir_ocean = '/Users/daanreijnders/Datasets/'
# readdir_ice = '/Users/daanreijnders/Datasets/'
# readdir_mesh = '/Users/daanreijnders/Datasets/'

readdir_ocean_control = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ocean/arctic/'
readdir_ocean_rcp85 = '/data/oceanparcels/input_data/CESM/0.1_deg/rcp8.5/ocean/arctic/'
readdir_ice = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ice/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'

fieldfile_ocean_control = 'daily_CESM_0.1degree_controlrun_year_300_arctic_timed_no_cord.nc'
fieldfile_ocean_rcp85 = 'daily_CESM_0.1degree_rcp8.5run_years_2000-2010_arctic.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic.nc'
meshfile = 'POP_grid_coordinates.nc'

writedir = '/scratch/DaanR/psets/'
    
# Particle execution function
def advection(fieldset,
              particles,
              experiment_name = '',
              runtime = delta(days = 30),
              dt = delta(minutes = 5),
              outputdt = delta(hours = 12),
              overwrite = False,
              simple = False):
    """
    Advect particles on a `fieldset`. Original particle locations are stored in `particleGrid` object.
    
    Parameters
    ----------
    fieldset : parcels.fieldset
        Fieldset used for advecting particles
    particle: community.particles
        Object containing initial distribution of particles
    experiment_name : str
        Name to label experiment. Saved in filename of the `ParticleSet` output.
    runtime : datetime.timedelta
        Amount of time used for advection
    dt : datetime.timedelta
        Timestep of advection
    outputdt : datetime.timedelta
        Timestep between saved output.
    overwrite : bool
        Overwrite existing psets?
    simple : bool
        Don't use anti-beaching kernel and don't freeze out-of-bound particles
        
    Returns
    -------
    parcels.ParticleSet
        Contains particle trajectories.
    """
    if simple:
        pclass = JITParticle
    else:
        assert hasattr(fieldset, 'UVunbeach'), "To unbeach, UVunbeach must be in fieldset!"
        pclass = kernelCollection.unbeachableBoundedParticle
    pset = ParticleSet.from_list(fieldset = fieldset,
                                 pclass = pclass,
                                 lon = particles.lonlat[0,:,0],
                                 lat = particles.lonlat[0,:,1],
                                 time = particles.releaseTimes,
                                 lonlatdepth_dtype = np.float64)
    
    # Set kernels
    if simple:
        kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(kernelCollection.wrapLon)
    else:
        kernels = pset.Kernel(kernelCollection.advectionRK4) \
                  + pset.Kernel(kernelCollection.beachTesting) \
                  + pset.Kernel(kernelCollection.freezeOutOfBounds) \
                  + pset.Kernel(kernelCollection.unBeaching)
    
    # Check if file exists    
    if os.path.exists(writedir+"pset_"+experiment_name+".nc"):
        if overwrite == True:
            warnings.warn("File already exists!")
        else:
            raise Exception('File already exists') 
    
    # Set ParticleFile
    pfile = pset.ParticleFile(name = writedir+"pset_"+experiment_name, outputdt=outputdt)
    pfile.add_metadata("dt", str(dt))
    pfile.add_metadata("Output dt", str(outputdt))
    pfile.add_metadata("Runtime", str(runtime))
    pfile.add_metadata("Release time of first particle", str(particles.releaseTimes[0]))
    print(f"Run: Advecting particles for {runtime}")
    
    # Run
    pset.execute(kernels,
                 runtime = runtime,
                 dt = dt,
                 output_file = pfile,
                 recovery = {ErrorCode.ErrorOutOfBounds: kernelCollection.deleteParticle})
    pfile.close()
    return pset

if __name__ == '__main__':
    # Argument parsing
    # """ONLY USE THIS FOR CONTROL RUN OR FIRST TEN YEARS OF RCP AND DON'T USE IT FOR ICOSAHEDRAL GRIDS"""
    parser = argparse.ArgumentParser(description="Advect particles on a rectilinear grid.")
    parser.add_argument('run', type=str, help='Select which run to use. Either `rcp85` or `control`')
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
    parser.add_argument('--nodland', action='store_true', help='Do not remove particles on land.')
    parser.add_argument('--letbeach', action='store_true', help='Let particles beach.')

    args = parser.parse_args()
    if args.name:
        name = args.name
    else:
        name = ''
    if args.run == 'rcp85':
        timestamps = [[np.datetime64('2000-01-09', 'D') + np.timedelta64(day, 'D') for day in range(4007)]]
        readdir_ocean = readdir_ocean_rcp85
        fieldfile_ocean = fieldfile_ocean_rcp85
        tindex = 'record'
    elif args.run == 'control':
        timestamps = None
        readdir_ocean = readdir_ocean_control
        fieldfile_ocean = fieldfile_ocean_control
        tindex = 'time'
    # Read field
    if not letbeach:
        fieldset = fieldsetter.read_velocity_field(readdir_ocean+fieldfile_ocean, 
                                                   meshfile=readdir_mesh+meshfile, 
                                                   tindex=tindex,
                                                   antiBeach = readdir_ocean_rcp85 + fieldfile_antibeach,
                                                   timestamps=timestamps)
    else:
        fieldset = fieldsetter.read_velocity_field(readdir_ocean+fieldfile_ocean, 
                                                   meshfile=readdir_mesh+meshfile, 
                                                   tindex=tindex, 
                                                   timestamps=timestamps)     
    # Read start date   
    start_year = int(args.start_date[0:4])
    start_month = int(args.start_date[4:6])
    start_day = int(args.start_date[6:8])
    
    # Create particle grid
    particles = community.particles.from_regular_grid(args.plon,\
                                                      args.plat,\
                                                      releaseTime = datetime(start_year, start_month, start_day),\
                                                      minLat=args.minlat,\
                                                      maxLat=args.maxlat,\
                                                      minLon=args.minlon,\
                                                      maxLon=args.maxlon)
    
    # Check whether land particles need to be removed
    if not args.nodland:
        particles.remove_on_land(fieldset)
    # Name parsing
    if len(name) > 0:
        if name[-1] != '_':
            name = name + '_'
    if args.nodland:
        name = name + 'nodelete_'
    if args.letbeach:
        name = name + 'beach'
        
    # Run
    pset_out = advection(fieldset,
                         particles,\
                         runtime=delta(days=args.days),\
                         dt = delta(minutes=args.advectdt),\
                         outputdt = delta(hours=args.outputdt),\
                         experiment_name=f"{name}R{args.run}_P{args.plon}x{args.plat}_S{start_year}-{start_month}-{start_day}_D{args.days}_DT{args.advectdt}_ODT{args.outputdt}_LAT{args.minlat}-{args.maxlat}_LON{args.minlon}-{args.maxlon}")
