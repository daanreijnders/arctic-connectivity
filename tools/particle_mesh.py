import stripy as stripy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path as mpath
import cartopy as cart
import cartopy.crs as ccrs
import pickle

def save_ico_mesh_particles(refinement):
    icoMesh = {}
    ico = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=refinement)
    icoMesh['lats'] = ico.lats
    icoMesh['lons'] = ico.lons
    with open('ico_mesh_parts_r{}_noF'.format(refinement), 'wb') as dumpFile:
        pickle.dump(icoMesh, dumpFile)
save_ico_mesh(10)
#save_ico_mesh(11)
#save_ico_mesh(12)
