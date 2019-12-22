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

psetname = 'pset_Rcmems_Pico11_S2018-3-1_D90_DT20_ODT24.nc'
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

psetname = 'pset_Rcmems_Pico11_S2018-9-1_D90_DT20_ODT24.nc'
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

psetname = 'pset_Rcmems_Pico11_S1993-3-1_D90_DT20_ODT24.nc'
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

psetname = 'pset_Rcmems_Pico11_S1993-9-1_D90_DT20_ODT24.nc'
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



