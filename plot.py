"""Plotting and animation tools"""
import numpy as np
import xarray as xr
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable

from parcels import (grid, Field, FieldSet, ParticleSet,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

#########################################


def show(inputfield, trajectoryFile=None, particleDensity=False, binGridWidth=1, latRange=(-90, 90), lonRange=(-180, 180), coast=True, t_index=0,land=True, polar=False, vectorField=False, export=None, t_end=None, titleAttribute=""):
    """This function creates a cartopy plot of the input field.
    
    :param inputfield: field to plot
    :param trajectoryFile: file containing particletrajectories
    :param particleDensity: Boolean to specify whether to create a 2D histogram
    :param binGridWidth: if particleDensity == True, specify width (in degrees) of histogram bins
    :param latRange: tuple to specify latitudinal extent of plot (minLat, maxLat)
    :param lonRange: tuple to specify longitudinal extent of plot (minLon, maxLon)
    :param coast: boolean to specify whether to plot coast
    :param land: boolean to specify whether to plot land mask
    :param polar: boolean to specify plot should be NorthPolarStereo
    :param vectorfield: boolean to plot velocity field as vectors (using quivers)
    :param export: name for .png export. If None, won't export
    :param t_end: if trajectory field is plotted, index to specify until which timestep particle trajectories are plotted
    :param t_index: index to obtain field from
    :param titleAttribute: string to extend the title of the plot with
    """
    
    if not isinstance(inputfield, Field): raise TypeError("field is not a parcels fieldobject")
    if inputfield.grid.defer_load:
        inputfield.fieldset.computeTimeChunk(inputfield.grid.time[t_index], 1)    
    minLat, maxLat = latRange
    minLon, maxLon = lonRange
    lons    = inputfield.grid.lon
    lats    = inputfield.grid.lat
    fig     = plt.figure()
    if polar:
        map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
    else:
        map_crs = ccrs.PlateCarree()
    ax = plt.axes(projection=map_crs)
    
    # Determine boundaries and add land mask
    ax.set_extent((minLon,maxLon,minLat,maxLat), crs=ccrs.PlateCarree())
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
        inputfield.fieldset.computeTimeChunk(inputfield.grid.time[t_index], 1)
        U = inputfield.fieldset.U.data[t_index,:,:]
        V = inputfield.fieldset.V.data[t_index,:,:]
        magnitude = np.sqrt(U**2 + V**2)
        plotfield = ax.quiver(lons, lats, U, V, magnitude, alpha=.5)
    else:
        inputfield.fieldset.computeTimeChunk(inputfield.grid.time[t_index], 1)
        plotfield = ax.pcolormesh(lons, lats, inputfield.data[t_index,:,:], transform=ccrs.PlateCarree(), zorder=1)
    
    # Colorbar
    divider = make_axes_locatable(ax)
    ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cbar    = plt.colorbar(plotfield, cax=ax_cb)
    unitBrackets = True
    # Set units
    if particleDensity:
        units = '(number of particles)'
    elif inputfield.name == 'U':
        units = '(m/s)'
    elif inputfield.name == 'V':
        units = '(m/s)'
    elif vectorField:
        units = '(m/s)'
    elif inputfield.name == 'Vh':
        units = '($m^2/s$)'
    else:
        units = ''
    cbar.ax.set_ylabel(f'{units}')
    
    # Set title of plot
    if particleDensity:
        titlestring = f"Particle distributions {titleAttribute}"
    elif trajectoryFile != None:
        titlestring = f"Particle trajectories {titleAttribute}"
    else:
        titlestring = inputfield.name + 'at' + str(inputfield.grid.time[t_index])
    ax.set_title(titlestring)
    # Export as figure
    if export:
        if export[-4] == '.':
            plt.savefig(f'figures/{export}', dpi=300)
        else:
            plt.savefig(f'figures/{export}.png', dpi=300)
    plt.show()


class particleAnimation:
    def create(pfile, field=None, margin=3, polar=False, nbar=False, EEZ_mapping=None, barLength=100, titleAttribute='', exportFolder='', mask=True, fps=24):
        """Create particle animations
        
        :param pfile: particleset.nc file
        :param field: field to plot animation on
        :param margin: number of degrees of margin around maximum extent that particles have travelled
        :param polar: boolean to specify plot should be NorthPolarStereo
        :param nbar: specify number of bars in barchart. If False, no barchart is produced
        :param eezIDmap: dataframe linking EEZ ID's to name
        :param barLength: integer specifying maximum y-extent of barchart
        :param fps: frames per second of animation
        :param titleAttribute: string to extend the title of the plot with
        """
        # Load arrays from file
        lon = np.ma.filled(pfile.variables['lon'], np.nan)
        lat = np.ma.filled(pfile.variables['lat'], np.nan)
        time = np.ma.filled(pfile.variables['time'], np.nan)
        EEZ_evol = np.ma.filled(pfile.variables['z'], np.nan)
        mesh = pfile.attrs['parcels_mesh'] if 'parcels_mesh' in pfile.attrs else 'spherical'
        
        # Set projection
        if polar:
            map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
        else:
            map_crs = ccrs.PlateCarree()
        
        # Create figure
        fig     = plt.figure(figsize=(9,5))
        particle_map = plt.subplot(projection=map_crs)  
        
        # Find extent for plot
        minlon = np.amin(pfile.variables['lon'])
        minlat = np.amin(pfile.variables['lat'])
        maxlon = np.amax(pfile.variables['lon'])
        maxlat = np.amax(pfile.variables['lat'])
        if polar:
            extent = (0, 360, minlat-margin, 90)
        else:
            extent = (max(minlon-margin, -180), min(maxlon+margin, 180), max(minlat-margin, -90), min(maxlat+margin, 90))
        particle_map.set_extent(extent, crs=map_crs)
        
        # Add coastlines and land mask
        particle_map.coastlines()
        if mask:
            particle_map.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
        
        if field:
            fieldName = field.name
            if fieldName == 'EEZ':
                colormap = 'Set3'
            elif fieldName == 'U' or 'V':
                colormap = 'viridis'
            else:
                colormap = 'viridis'
            field.fieldset.computeTimeChunk(field.grid.time[0], 1)
            particle_map.pcolormesh(field.lon, field.lat, field.data[0,:,:], transform=map_crs, cmap=colormap, zorder=1)
        else:
            fieldName = 'noField'
            
        # Draw gridlines
        gl = particle_map.gridlines(crs=map_crs, linestyle='--', draw_labels = True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Determine plotting time indices
        plottimes = np.unique(time)
        if isinstance(plottimes[0], (np.datetime64, np.timedelta64)):
            plottimes = plottimes[~np.isnat(plottimes)]
        else:
            try:
                plottimes = plottimes[~np.isnan(plottimes)]
            except:
                pass
        currtime = time == plottimes[0]
    
        # Create initial scatter plot of particles
        scat   = particle_map.scatter(lon[currtime], lat[currtime], s=20, color='k', transform=ccrs.Geodetic(), zorder=10)
        
        # Add bar chart
        if nbar and fieldName=='EEZ':
            # Determine EEZs that are most visited in pfile
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
            # Draw bar chart
            divider = make_axes_locatable(particle_map)
            barplot = divider.append_axes("right", size=2, pad=1, axes_class=plt.Axes)
            bar    = barplot.barh(np.arange(nbar), EEZ_counter(currtime))
            barplot.set_xlim(0, barLength)
            barplot.set_yticks(np.arange(nbar))
            if EEZ_mapping is not None:
                # Map EEZ IDs to ISO country codes (ROOM FOR IMPROVEMENT ON HANDLING THIS MAPPING PROCESS)
                EEZ_df = pd.read_json(EEZ_mapping)
                barplot.set_yticklabels([EEZ_df[EEZ_df['ID'] == ID]['ISO'].values[0] for ID in plotEEZbars])
            else: 
                barplot.set_yticklabels([str(label) for label in plotEEZbars])
            barplot.set_ylabel('EEZ ID')
            barplot.set_xlabel('Count')
            barplot.invert_yaxis()
        
        title = fig.suptitle('Particles at time ' + str(plottimes[0])[:13])
        frames = np.arange(0, len(plottimes))
        
        # Animation
        def animate(t):
            currtime = time == plottimes[t]
            scat.set_offsets(np.vstack((lon[currtime], lat[currtime])).transpose())
            title.set_text('Particles in different EEZs at time ' + str(plottimes[t])[:13])
            if nbar:
                for rect, width in zip(bar, EEZ_counter(currtime)):
                    rect.set_width(width)
            return scat,
        anim = animation.FuncAnimation(fig, animate, frames=len(plottimes), blit=True)
        if exportFolder != '':
            exportFolder = exportFolder + '/'
        anim.save(f'{exportFolder}particle_evolution_{fieldName}_{titleAttribute}.mp4', fps=fps, metadata={'artist':'Daan', 'title':f'Particles on {fieldName} - {titleAttribute}'}, extra_args=['-vcodec', 'libx264'])

        plt.show()
        plt.close()
