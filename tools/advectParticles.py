"""Advect particles in community.particles object using parcels. Includes CLI"""
# Imports
import numpy as np

from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

from datetime import timedelta as delta
from datetime import datetime

import argparse
import pickle
import os.path
import warnings

import sys
sys.path.append('/science/users/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repo/tools')
import fieldsetter_cmems
import kernelCollection
import community

writedir = '/scratch/DaanR/psets/'

# Particle execution function
def advection(fieldset,
              particles,
              experiment_name = '',
              runtime = delta(days = 30),
              dt = delta(minutes = 5),
              outputdt = delta(hours = 12),
              overwrite = False,
              freeze = True,
              unBeach = False
             ):
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
        If True, overwrite existing psets.
    freeze : bool
        If True, freeze particles leaving the boundary
    unBeach : bool
        If True, use anti-beaching kernel
        
    Returns
    -------
    parcels.ParticleSet
        Contains particle trajectories.
    """
    if unBeach:
        assert hasattr(fieldset, 'UVunbeach'), "To unbeach, UVunbeach must be in fieldset!"
    if freeze and unBeach:
        pclass = kernelCollection.unbeachableBoundedParticle
    elif freeze:
        pclass = kernelCollection.boundedParticle
    elif unBeach:
        pclass = kernelCollection.unbeachableParticle
    else: 
        pclass = JITParticle
    pset = ParticleSet.from_list(fieldset = fieldset,
                                 pclass = pclass,
                                 lon = particles.lonlat[0,:,0],
                                 lat = particles.lonlat[0,:,1],
                                 time = particles.releaseTimes)
    
    # Set kernels
    kernels = pset.Kernel(kernelCollection.northPolePushBack)
    if freeze and unBeach:
        kernels += kernelCollection.UnbeachBoundedAdvectionRK4
        print("Particles will be unbeached and will be frozen upon leaving the domain.")
    elif freeze:
        kernels += kernelCollection.BoundedAdvectionRK4
        print("Particles will not be unbeached and will be frozen upon leaving the domain.")
    elif unBeach: 
        kernels += kernelCollection.UnbeachAdvectionRK4
        print("Particles will be unbeached and are allowed to leave the domain.")
    else:
        kernels += pset.Kernel(AdvectionRK4)
        print("Particles will not be unbeached and are allowed to leave the domain.")

    kernels += pset.Kernel(kernelCollection.wrapLon)
    if freeze:
        kernels += pset.Kernel(kernelCollection.freezeOutOfBoundsArctic)
    if unBeach:
        kernels += pset.Kernel(kernelCollection.beachTesting) + pset.Kernel(kernelCollection.unBeaching)
        
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
    parser = argparse.ArgumentParser(description="Advect particles on an icosahedral grid in the Arctic (>60N)")
    parser.add_argument('start_date', type=str, help="Particle initialization time. Must be formatted as YYYY-MM-DD.")
    parser.add_argument('-d', '--days', default=90, type=int, help='Number of days used for advection.')
    parser.add_argument('-dt', '--advectdt', default=20, type=int, help='Timestep for advection in minutes')
    parser.add_argument('-odt', '--outputdt', default=24, type=int, help='Output timestep in hours')
    parser.add_argument('-r', '--refinement', default=11, type=int, help='Refinement level of the icosahedral mesh')
    parser.add_argument('--noLandDelete', action='store_true', help='Do not remove particles initialized on land.')
    parser.add_argument('--noFreeze', action='store_true', help='Let particles leave the domain.')
    parser.add_argument('--unBeach', action='store_true', help='Unbeach particles')
   
    args = parser.parse_args()
    
    # Load Settings
    runtime = delta(days=args.days)
    dt = delta(minutes=args.advectdt)
    outputdt = delta(hours=args.outputdt)
    start_year = int(args.start_date[0:4])
    start_month = int(args.start_date[4:6])
    start_day = int(args.start_date[6:8])
    releaseTime = datetime(start_year, start_month, start_day, 12)
    refinement = args.refinement
    experiment_name = f"Rcmems_Pico{refinement}_S{start_year}-{start_month}-{start_day}_D{args.days}_DT{args.advectdt}_ODT{args.outputdt}"
    if args.unBeach:
        experiment_name += '_unBeach'
    if args.noFreeze:
        experiment_name += '_noFreeze'
    
    fieldset = fieldsetter_cmems.create(args.start_date, args.days+2, antiBeach = args.unBeach, halo=True)
    
    # Read start date   
    start_year = int(args.start_date[0:4])
    start_month = int(args.start_date[4:6])
    start_day = int(args.start_date[6:8])
    
    with open(f'/scratch/DaanR/meshes/ico_mesh_parts_deg_arctic_r{refinement}_delland.pickle', 'rb') as meshf:
            mesh = pickle.load(meshf)
    particles = community.particles(mesh['lons'],
                                    mesh['lats'],
                                    releaseTime = releaseTime
                                    )
    del mesh
   
    Check whether land particles need to be removed
    if not args.noLandDelete:
        particles.remove_on_land(fieldset)
    
    # Run
    pset_out = advection(fieldset,
                         particles,\
                         runtime = runtime,\
                         dt = dt,\
                         outputdt = outputdt,\
                         experiment_name = experiment_name,
                         freeze = not args.noFreeze,
                         unBeach = args.unBeach
                         )
