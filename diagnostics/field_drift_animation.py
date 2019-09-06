import numpy as np
import xarray as xr
import cartopy as cart
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Open datasets
U_field = xr.open_dataset(readdir_ocean+fieldfile_ocean)['UVEL_5m']
POP_mesh = xr.open_dataset(readdir_mesh+meshfile)
ULAT = POP_mesh['U_LAT_2D']
ULON = POP_mesh['U_LON_2D']

# Initalize plots
#map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
map_crs = ccrs.PlateCarree()
fig = plt.figure(figsize=(14,6))
ax = plt.axes(projection=map_crs)
# Set view
ax.set_extent((0,30,40,70), crs=ccrs.PlateCarree())
ax.coastlines()
gl      = ax.gridlines()

# Plot first field
pcmesh = ax.pcolormesh(ULON, ULAT, U_field[0,:,:], transform=map_crs)
#Animate
def animate(t):
    pcmesh.set_array(U_field[t,:,:].data.ravel())
#anim = animation.FuncAnimation(fig, animate, frames=3, blit=False)
anim = animation.FuncAnimation(fig, animate, frames=U_field.shape[0], blit=False)
plt.draw()
anim.save(f'driftingFields.mp4', fps=10, metadata={'artist':'Daan', 'title':f'Evolution of velocity field'}, extra_args=['-vcodec', 'libx264'])

plt.show()
plt.close()