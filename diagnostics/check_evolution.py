#!/usr/bin/env python
# coding: utf-8

# In[8]:


import xarray as xr

from importlib import reload
import sys
sys.path.append('/home/students/4302001/arctic-connectivity/tools')
import plot
import lifeline


# In[9]:


# Run in case you're debugging
reload(plot)
reload(lifeline)


# In[10]:


# Specify paths to velocity field and mesh
#readdir_ocean = '/Users/daanreijnders/Datasets/'
#readdir_ice = '/Users/daanreijnders/Datasets/'

readdir_ice = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ice/arctic/'
readdir_ocean = '/data/oceanparcels/input_data/CESM/0.1_deg/control/ocean/arctic/'
readdir_mesh = '/scratch/DaanR/fields/'

fieldfile_ocean = 'daily_CESM_0.1degree_controlrun_year_300_arctic_region_timed.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic'
meshfile = 'POP_grid_lat1800plus.nc'

writedir = ''


# In[11]:


U_field = xr.open_dataset(readdir_ocean+fieldfile_ocean)['UVEL_5m']
V_field = xr.open_dataset(readdir_ocean+fieldfile_ocean)['VVEL_5m']
POP_mesh = xr.open_dataset(readdir_mesh+meshfile)
ULAT = POP_mesh['U_LAT_2D']
ULON = POP_mesh['U_LON_2D']


# In[12]:


U_field['time'][115] # Check timestamp associated with field at index 115


# In[13]:


U_field.shape # Check if shape makes sense


# In[ ]:


plot.fromDataset(ULON, ULAT, U_field[0,:,:], polar=True, latRange=(60,90),                 units='cm/s', export="POP_field_drift_world_polar_U0", title=f"U at index 0 ({str(U_field['time'][0].data)[:10]})")
plot.fromDataset(ULON, ULAT, U_field[50,:,:], polar=True, latRange=(60,90),                 units='cm/s', export="POP_field_drift_world_polar_U50", title=f"U at index 50 ({str(U_field['time'][50].data)[:10]})")
plot.fromDataset(ULON, ULAT, U_field[300,:,:], polar=True, latRange=(60,90),                 units='cm/s', export="POP_field_drift_world_polar_U300", title=f"U at index 300 ({str(U_field['time'][300].data)[:10]})")
plot.fromDataset(ULON, ULAT, V_field[0,:,:], polar=True, latRange=(60,90),                 units='cm/s', export="POP_field_drift_world_polar_V0", title=f"U at index 0 ({str(U_field['time'][0].data)[:10]})")
plot.fromDataset(ULON, ULAT, V_field[50,:,:], polar=True, latRange=(60,90),                 units='cm/s', export="POP_field_drift_world_polar_V50", title=f"U at index 50 ({str(U_field['time'][50].data)[:10]})")
plot.fromDataset(ULON, ULAT, V_field[300,:,:], polar=True, latRange=(60,90),                 units='cm/s', export="POP_field_drift_world_polar_V300", title=f"U at index 300 ({str(U_field['time'][300].data)[:10]})")

