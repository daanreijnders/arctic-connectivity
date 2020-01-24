"""Creates pickle files with the lats and lons of icosahedral meshes at a given grid refinement."""
import stripy as stripy
import pickle

def save_ico_mesh_particles(refinement):
    icoMesh = {}
    ico = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=refinement)
    icoMesh['lats'] = ico.lats
    icoMesh['lons'] = ico.lons
    with open('ico_mesh_parts_r{}_noF'.format(refinement), 'wb') as dumpFile:
        pickle.dump(icoMesh, dumpFile)