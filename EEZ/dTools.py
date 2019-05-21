"""This package contains specific tools I use for my research project."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parcels import (grid, Field, FieldSet, ParticleSet,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

def show(inputfield, trajectoryFile=None, particleDensity=False, \
         binGridWidth=1, latrange=(-90, 90), lonrange=(-180, 180), \
         land=True, export=None, vectorField=False, t_end=None, titleAttribute=""):
    """This function creates a cartopy plot of the input field.
    ADD DOCUMENTATION. ADD TIMESTAMP
    """
    if not isinstance(inputfield, Field): raise TypeError("field is not a parcels fieldobject")
    if inputfield.grid.defer_load:
        inputfield.fieldset.computeTimeChunk(inputfield.grid.time[0], 1)    
    minlat, maxlat = latrange
    minlon, maxlon = lonrange
    lons    = inputfield.grid.lon
    lats    = inputfield.grid.lat
    fig     = plt.figure()
    ax      = plt.axes(projection=ccrs.PlateCarree())
    
    # Determine boundaries and add land mask
    ax.set_extent((minlon,maxlon,minlat,maxlat), crs=ccrs.PlateCarree())
    if land:
        ax.coastlines()
        ax.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
    
    # Add gridlines
    gl      = ax.gridlines(crs=ccrs.PlateCarree(), linestyle='--', draw_labels = True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Trajectories
    if trajectoryFile != None:
        try:
            pfile = xr.open_dataset(str(trajectoryFile), decode_cf=True)
        except:
            pfile = xr.open_dataset(str(trajectoryFile), decode_cf=False)
        
        lon = np.ma.filled(pfile.variables['lon'], np.nan)
        T = lon.shape[1]
        if t_end == None:
            t_end = T
        lon = lon[:,:t_end]
        lat = np.ma.filled(pfile.variables['lat'], np.nan)[:,:t_end]
        time = np.ma.filled(pfile.variables['time'], np.nan)[:,:t_end]
        z = np.ma.filled(pfile.variables['z'], np.nan)[:,:t_end]
        mesh = pfile.attrs['parcels_mesh'] if 'parcels_mesh' in pfile.attrs else 'spherical'
        pfile.close()
        nPart = lon.shape[0]
        for p in range(lon.shape[0]):
            lon[p, :] = [ln if ln < 180 else ln - 360 for ln in lon[p, :]]
        if not particleDensity:
            if nPart > 25: # More than 25 particles? Plot trajectories transparently
                ax.plot(np.transpose(lon), np.transpose(lat), color='black', alpha=0.1, transform=ccrs.Geodetic(), zorder=10, linewidth=0.5)
            else:
                ax.plot(np.transpose(lon), np.transpose(lat), '.-', transform=ccrs.Geodetic(), zorder=10)
    
    # Plot field
    if particleDensity:
        densLats = np.arange(-90, 90, binGridWidth)
        densLons = np.arange(-180, 180, binGridWidth)
        density = np.zeros((len(densLats), len(densLons)))
        for i in range(nPart):
#             if particle.lon > 180:
#                 offset = 360
#             else:
#                 offset = 0
            latsIdx = bisect_left(densLats, lat[i, -1] )
            lonsIdx = bisect_left(densLons, lon[i, -1] )#- offset)
            density[latsIdx-1, lonsIdx-1] += 1
        maxDens = np.max(density)
        plotfield = ax.pcolormesh(densLons, densLats, density, transform=ccrs.PlateCarree(), zorder=1)
    elif vectorField:
        U = inputfield.fieldset.U.data[0,:,:]
        V = inputfield.fieldset.V.data[0,:,:]
        magnitude = np.sqrt(U**2 + V**2)
        plotfield = ax.quiver(lons, lats, U, V, magnitude, alpha=.5)
    else:
        plotfield = ax.pcolormesh(lons, lats, inputfield.data[0,:,:], transform=ccrs.PlateCarree(), zorder=1)
    
    # Colorbar
    divider = make_axes_locatable(ax)
    ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cbar    = plt.colorbar(plotfield, cax=ax_cb)
    if particleDensity:
        units = 'number of particles'
    elif inputfield.name == 'U':
        units = 'm/s'
    elif inputfield.name == 'V':
        units = 'm/s'
    elif vectorField:
        units = 'm/s'
    elif inputfield.name == 'Vh':
        units = '$m^2/s$'
    else:
        units = ''
    cbar.ax.set_ylabel(f'({units})')
    
    # Title
    if particleDensity:
        titlestring = f"Particle distributions {titleAttribute}"
    elif trajectoryFile != None:
        titlestring = f"Particle trajectories {titleAttribute}"
    else:
        titlestring = inputfield.name
    ax.set_title(titlestring)
    if export != None:
        plt.savefig(f'figures/{export}.png', dpi=300)
    plt.show()

