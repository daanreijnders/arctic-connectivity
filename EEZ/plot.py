"""This package contains specific tools I use for my research project."""
import numpy as np
import xarray as xr
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable

from parcels import (grid, Field, FieldSet, ParticleSet,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

def show(inputfield, trajectoryFile=None, particleDensity=False, \
         binGridWidth=1, latrange=(-90, 90), lonrange=(-180, 180), \
         coast=True, land=True, polar=False, \
         vectorField=False, export=None, t_end=None, titleAttribute=""):
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
    if polar:
        map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
    else:
        map_crs = ccrs.PlateCarree()
    ax = plt.axes(projection=map_crs)
    
    # Determine boundaries and add land mask
    ax.set_extent((minlon,maxlon,minlat,maxlat), crs=ccrs.PlateCarree())
    if coast:
        ax.coastlines()
    if land:
        ax.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
    
    # Add gridlines
    if polar:
        gl      = ax.gridlines()
    else: 
        gl      = ax.gridlines(crs=map_crs, linestyle='--', draw_labels = True)
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
        plotfield = ax.pcolormesh(densLons, densLats, density, transform=map_crs, zorder=1)
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
    unitBrackets = True
    if particleDensity:
        units = '(number of particles)'
    elif inputfield.name == 'U':
        units = '(m/s)'
    elif inputfield.name == '(V)':
        units = '(m/s)'
    elif vectorField:
        units = '(m/s)'
    elif inputfield.name == 'Vh':
        units = '($m^2/s$)'
    else:
        units = ''
    cbar.ax.set_ylabel(f'{units}')
    
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


class particleAnimation:
    def EEZ_particles(pfile, EEZ_ds, bleed=3, polar=False, nbar=False, eezIDmap=None, barLength=100, fps=24):
        # Load arrays from file
        lon = np.ma.filled(pfile.variables['lon'], np.nan)
        lat = np.ma.filled(pfile.variables['lat'], np.nan)
        time = np.ma.filled(pfile.variables['time'], np.nan)
        EEZ_evol = np.ma.filled(pfile.variables['z'], np.nan)
        mesh = pfile.attrs['parcels_mesh'] if 'parcels_mesh' in pfile.attrs else 'spherical'
        
        minlon = np.amin(pfile.variables['lon'])
        minlat = np.amin(pfile.variables['lat'])
        maxlon = np.amax(pfile.variables['lon'])
        maxlat = np.amax(pfile.variables['lat'])
        if polar:
            extent = (0, 360, minlat-bleed, 90)
        else:
            extent = (max(minlon-bleed, -180), min(maxlon+bleed, 180), max(minlat-bleed, -90), min(maxlat+bleed, 90))
        
        if polar:
            map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
        else:
            map_crs = ccrs.PlateCarree()
        # Create figure
        fig     = plt.figure(figsize=(9,5))
        particle_map = plt.subplot(projection=map_crs)   

        # Add background for land plot
        particle_map.set_extent(extent, crs=map_crs)
        particle_map.coastlines()
        particle_map.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
        
        EEZ = EEZ_ds['EEZ'][:,:][0]
        EEZ_lats = EEZ_ds['latitude'].data
        EEZ_lons = EEZ_ds['longitude'].data
        
        particle_map.pcolormesh(EEZ_lons, EEZ_lats, EEZ, transform=map_crs, cmap='Set3', zorder=1)
        gl = particle_map.gridlines(crs=map_crs, linestyle='--', draw_labels = True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        plottimes = np.unique(time)
        if isinstance(plottimes[0], (np.datetime64, np.timedelta64)):
            plottimes = plottimes[~np.isnat(plottimes)]
        else:
            try:
                plottimes = plottimes[~np.isnan(plottimes)]
            except:
                pass
        print('Start and end plottimes:', plottimes[0], plottimes[-1])
        currtime = time == plottimes[0]

        scat   = particle_map.scatter(lon[currtime], lat[currtime], s=20, color='k', transform=ccrs.Geodetic())
        
        if nbar:
            uniques, counts = np.unique(np.where(np.isnan(pfile['EEZ'].data), -1, pfile['EEZ'].data), return_counts=True) # NaNs become -1
            EEZoccurenceDict = dict(zip(uniques, counts))
            EEZoccurenceDict.pop(-1)
            plotEEZbars = []
            while len(EEZoccurenceDict) >= nbar and len(plotEEZbars) < nbar:
                plotEEZbars.append(max(EEZoccurenceDict, key=EEZoccurenceDict.get))
                popVal = EEZoccurenceDict.pop(max(EEZoccurenceDict, key=EEZoccurenceDict.get))
            # Count instances of particles at EEZ in target list at a certain timestep
            def EEZ_counter(timestep):
                currEEZ = np.where(np.isnan(pfile['EEZ'].data[timestep]), -1, pfile['EEZ'].data[timestep])
                currCount = np.zeros(len(plotEEZbars))
                for el in currEEZ:
                    try:
                        idx = plotEEZbars.index(el)
                        currCount[idx] += 1
                    except ValueError:
                        pass
                return currCount

            divider = make_axes_locatable(particle_map)
        
            barplot = divider.append_axes("right", size=2, pad=1, axes_class=plt.Axes)

            bar    = barplot.barh(np.arange(nbar), EEZ_counter(currtime))
            barplot.set_xlim(0, barLength)
            barplot.set_yticks(np.arange(nbar))
            if eezIDmap is not None:
                EEZ_df = pickle.load(eezIDmap)['df']
                barplot.set_yticklabels([EEZ_df[EEZ_df['ID'] == ID]['ISO'].values[0] for ID in plotEEZbars])
            else: 
                barplot.set_yticklabels([str(label) for label in plotEEZbars])
            barplot.set_ylabel('EEZ ID')
            barplot.set_xlabel('Count')
            barplot.invert_yaxis()
        
        title = fig.suptitle('Particles at time ' + str(plottimes[0])[:13])
        frames = np.arange(0, len(plottimes))

        def animate(t):
            currtime = time == plottimes[t]
            scat.set_offsets(np.vstack((lon[currtime], lat[currtime])).transpose())
            title.set_text('Particles in different EEZs at time ' + str(plottimes[t])[:13])
            for rect, width in zip(bar, EEZ_counter(currtime)):
                rect.set_width(width)
            return scat,
        anim = animation.FuncAnimation(fig, animate, frames=len(plottimes), blit=False)
        anim.save('particle_EEZ_evo.mp4', fps=fps, metadata={'artist':'Daan', 'title':'Particles in EEZs'}, extra_args=['-vcodec', 'libx264'])

        plt.show()
        plt.close()