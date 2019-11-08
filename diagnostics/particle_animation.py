import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean

import sys
sys.path.append('/science/users/4302001/arctic-connectivity/tools')
sys.path.append('/Users/daanreijnders/surfdrive/Thesis/repo/tools')
import plot
import lifeline
import community
import fieldsetter_cmems
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

# pfile = xr.open_dataset('/data/oceanparcels/output_data/data_Daan/psets/pset_Rrcp85_Piconorth11_S2000-01-9_D90_DT20_ODT24_LAT60-90_LON-180-180.nc', decode_cf=True)
# pfile = xr.open_dataset('/data/oceanparcels/output_data/data_Daan/psets/pset_Rcontrol_y300_DT5C_P3590x590_S2000-1-1_D30_DT5_ODT12_LAT60.5-89.5_LON-179.5-179.5.nc', decode_cf=True)
# psetname = 'pset_Rrcp85_Pico11_S2001-12-1_D90_DT20_ODT24_wedge.nc'
# psetname = 'pset_Rrcp85_Pico11_S2051-12-1_D90_DT20_ODT24_wedge.nc'
psetname = 'pset_Rcmems_Pico11_S2001-1-1_D90_DT20_ODT24.nc'

pfile = xr.open_dataset(psetdir + psetname, decode_cf=True)
plot.particleAnimation.create(pfile, 
                              lonRange = (-180, 180), 
                              latRange = (59, 90),
                              cbar = True,
                              coast = True,
                              wedge = False,
                              polar = True,
                              times = 'flat',
                              title = psetname,
                              units = 'Initial Latitude',
                              fps = 6,
                              size = (8,8))
pfile.close()
