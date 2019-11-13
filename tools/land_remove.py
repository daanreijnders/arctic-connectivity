import numpy as np
import pickle

from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

import sys
sys.path.append('/science/users/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repo/tools')
import fieldsetter_cmems
import community

fieldset = fieldsetter_cmems.create(20000101, 3, antiBeach=False)

for refinement in [10, 11, 12]:
    with open(f'/scratch/DaanR/meshes/ico_mesh_parts_deg_arctic_r{refinement}.pickle', 'rb') as meshf:
        mesh = pickle.load(meshf)
    particles = community.particles(mesh['lons'],
                                    mesh['lats'],
                                    )
    del mesh

    # Check whether land particles need to be removed
    particles.remove_on_land(fieldset)

    newmesh = {'lons' : particles.lons,
               'lats' : particles.lats}
    with open(f'/scratch/DaanR/meshes/ico_mesh_parts_deg_arctic_r{refinement}_delland.pickle', 'wb') as meshf:
        pickle.dump(newmesh, meshf)