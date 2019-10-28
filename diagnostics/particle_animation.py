import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean

import sys
sys.path.append('/home/students/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repo/tools')
import plot
import lifeline
import community
import fieldsetter
import advectParticles

readdir_ocean = '/data/oceanparcels/input_data/CESM/0.1_deg/rcp8.5/ocean/arctic/'
readdir_ice = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ice/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'

fieldfile_ocean = 'daily_CESM_0.1degree_rcp8.5run_years_2000-2010_arctic.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic.nc'
meshfile = 'POP_grid_coordinates.nc'

writedir = '/scratch/DaanR/psets/'
psetdir = '/data/oceanparcels/output_data/data_Daan/psets/'
# timestamps = [[np.datetime64('2000-01-09', 'D') + np.timedelta64(day, 'D') for day in range(4007)]]
# fieldset = fieldsetter.read_velocity_field(readdir_ocean+fieldfile_ocean, 
#                                            meshfile = readdir_mesh+meshfile,
#                                            tindex = 'record',
#                                            timestamps = timestamps)

#pfile = xr.open_dataset('/data/oceanparcels/output_data/data_Daan/psets/pset_Rrcp85_Piconorth11_S2000-01-9_D90_DT20_ODT24_LAT60-90_LON-180-180.nc', decode_cf=True)
# pfile = xr.open_dataset('/data/oceanparcels/output_data/data_Daan/psets/pset_Rcontrol_y300_DT5C_P3590x590_S2000-1-1_D30_DT5_ODT12_LAT60.5-89.5_LON-179.5-179.5.nc', decode_cf=True)
psetname = '/data/oceanparcels/output_data/data_Daan/psets/pset_Rrcp85_P2200x500_S2000-01-09_D90_DT0:_ODT1 _LAT(60,)-(85,)_LON(-45,)-(65,).nc'

pfile = xr.open_dataset(psetdir + psetname, decode_cf=True)
plot.particleAnimation.create(pfile, particle_subsample=1, times='flat', fps=6, extent=(-180,180,60,90), polar=True, mask=False, titleAttribute=psetname)
pfile.close()
