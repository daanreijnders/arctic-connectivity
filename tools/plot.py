"""Plotting and animation tools"""
import numpy as np
import xarray as xr
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.animation as animation

import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable

from parcels import (grid, Field, FieldSet, ParticleSet,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

import os

def set_circular_boundary(ax):
    theta = np.linspace(0, 2*np.pi, 400)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circlePath = mpath.Path(verts * radius + center)
    ax.set_boundary(circlePath, transform=ax.transAxes)
    return circlePath

def set_wedge_boundary(ax, minLon, maxLon, minLat, maxLat):
    wedgeLons = np.concatenate((np.linspace(minLon, maxLon, 50),
                                np.linspace(maxLon, maxLon, 50),
                                np.linspace(maxLon, minLon, 50),
                                np.linspace(minLon, minLon, 50)))
    wedgeLats = np.concatenate((np.linspace(minLat, minLat, 50),
                                np.linspace(minLat, maxLat, 50),
                                np.linspace(maxLat, maxLat, 50),
                                np.linspace(maxLat, minLat, 50)))
    wedgePath = mpath.Path(np.dstack((wedgeLons, wedgeLats))[0])
    ax.set_boundary(wedgePath, transform=ccrs.PlateCarree())
    return wedgePath

########################################################################################
def field_from_dataset(lons, lats, data, latRange=(-90, 90), lonRange=(-180, 180), \
                    coast=True, land=False, projection=False, polar=False, wedge=False, export=None, \
                    units=None, t_end=None, title="", colormap=None, size=None, cbar=True, cbextend='neither', **kwargs):
    # Extract Options
    minLat, maxLat = latRange
    minLon, maxLon = lonRange
    
    if projection:
        map_crs = projection
    else:
        if polar:
            map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
        elif wedge: 
            map_crs = ccrs.Stereographic(central_latitude = minLat+(maxLat-minLat)/2, central_longitude=minLon+(maxLon-minLon)/2)

        else:
            map_crs = ccrs.PlateCarree()
    
    # Build axes
    if size:
        fig     = plt.figure(figsize=size)
    else:
        fig     = plt.figure()
    ax      = plt.axes(projection=map_crs)

    ax.set_extent((minLon,maxLon,minLat,maxLat), crs=ccrs.PlateCarree())
    
    # Set masks
    if coast:
        ax.coastlines()
    if land:
        ax.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
    
    # Add gridlines
    if projection or polar or wedge:
        gl = ax.gridlines(linestyle='--', alpha=0.75, linewidth=0.5)
    else: 
        gl = ax.gridlines(crs=map_crs, linestyle='--', alpha=0.75, linewidth=0.5, draw_labels = True)
        gl.xlabels_top   = False
        gl.ylabels_right = False
        gl.xformatter    = LONGITUDE_FORMATTER
        gl.yformatter    = LATITUDE_FORMATTER
        
    # Circular clipping
    if polar:
        circle_clip = set_circular_boundary(ax)
    if wedge:
        wedge_clip = set_wedge_boundary(ax, minLon, maxLon, minLat, maxLat)
    
    if not colormap:
        colormap = 'viridis'
    # Plot field
    if polar: 
        plotfield = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), clip_path=(circle_clip, ax.transAxes), cmap=colormap, **kwargs)
    else:
        plotfield = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), cmap=colormap, **kwargs)
    
    # Colorbar
    if cbar:
        divider = make_axes_locatable(ax)
        if wedge:
            ax_cb = divider.new_vertical(size="5%", pad=0.1, axes_class=plt.Axes, pack_start=True)
            fig.add_axes(ax_cb)
            cbar = plt.colorbar(plotfield, cax=ax_cb, orientation='horizontal', extend=cbextend)
        else:
            ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
            fig.add_axes(ax_cb)
            cbar = plt.colorbar(plotfield, cax=ax_cb, extend=cbextend)
        
        # Set units
        if units:
            if wedge:
                cbar.ax.set_xlabel(f"{str(units)}")
            else:
                cbar.ax.set_ylabel(f"{str(units)}")
    
    ax.set_title(title)
    # Export as figure
    if export:
        if not os.path.exists('figures'):
            os.makedirs('figures')
        if export[-4] == '.':
            plt.savefig(f'figures/{export}', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'figures/{export}.png', dpi=300, bbox_inches='tight')
    return fig, ax

########################################################################################

def triangular_field_from_dataset(lons, lats, triangles, data, latRange=(-90, 90), lonRange=(-180, 180), \
                    coast=True, land=False, projection=False, polar=False, wedge=False, export=None, \
                    units=None, t_end=None, title="", colormap=None, size=None, cbar=True, cbextend='neither', **kwargs):
    # Extract Options
    minLat, maxLat = latRange
    minLon, maxLon = lonRange
    
    if projection:
        map_crs = projection
    else:
        if polar:
            map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
        elif wedge: 
            map_crs = ccrs.Stereographic(central_latitude = minLat+(maxLat-minLat)/2, central_longitude=minLon+(maxLon-minLon)/2)

        else:
            map_crs = ccrs.PlateCarree()
    
    # Build axes
    if size:
        fig     = plt.figure(figsize=size)
    else:
        fig     = plt.figure()
    ax      = plt.axes(projection=map_crs)

    ax.set_extent((minLon,maxLon,minLat,maxLat), crs=ccrs.PlateCarree())
    
    # Set masks
    if coast:
        ax.coastlines()
    if land:
        ax.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
    
    # Add gridlines
    if projection or polar or wedge:
        gl = ax.gridlines(linestyle='--', alpha=0.75, linewidth=0.5)
    else: 
        gl = ax.gridlines(crs=map_crs, linestyle='--', alpha=0.75, linewidth=0.5, draw_labels = True)
        gl.xlabels_top   = False
        gl.ylabels_right = False
        gl.xformatter    = LONGITUDE_FORMATTER
        gl.yformatter    = LATITUDE_FORMATTER
    # Circular clipping
    if polar:
        circle_clip = set_circular_boundary(ax)
    if wedge:
        wedge_clip = set_wedge_boundary(ax, minLon, maxLon, minLat, maxLat)
    
    if not colormap:
        colormap = 'viridis'
        
    # Plot field
    if polar: 
        plotfield = ax.tripcolor(lons, lats, triangles=triangles, facecolors=data, transform=ccrs.Geodetic(), clip_path=(circle_clip, ax.transAxes), cmap=colormap, **kwargs)
    else:
        plotfield = ax.tripcolor(lons, lats, triangles=triangles, facecolors=data, transform=ccrs.Geodetic(), cmap=colormap, **kwargs)
    
    # Colorbar
    if cbar:
        divider = make_axes_locatable(ax)
        if wedge:
            ax_cb = divider.new_vertical(size="5%", pad=0.1, axes_class=plt.Axes, pack_start=True)
            fig.add_axes(ax_cb)
            cbar = plt.colorbar(plotfield, cax=ax_cb, orientation='horizontal', extend=cbextend)
        else:
            ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
            fig.add_axes(ax_cb)
            cbar = plt.colorbar(plotfield, cax=ax_cb, extend=cbextend)
        
        # Set units
        if units:
            if wedge:
                cbar.ax.set_xlabel(f"{str(units)}")
            else:
                cbar.ax.set_ylabel(f"{str(units)}")
    
    ax.set_title(title)
    # Export as figure
    if export:
        if not os.path.exists('figures'):
            os.makedirs('figures')
        if export[-4] == '.':
            plt.savefig(f'figures/{export}', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'figures/{export}.png', dpi=300, bbox_inches='tight')
    return fig, ax

########################################################################################

def scatter_from_dataset(lons, lats, latRange=(-90, 90), lonRange=(-180, 180), \
                         coast=True, land=False, projection=False, polar=False, wedge=False, export=None, \
                         title="", colormap=None, size=None, **kwargs):
    # Extract Options
    minLat, maxLat = latRange
    minLon, maxLon = lonRange
    
    if projection:
        map_crs = projection
    else:
        if polar:
            map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
        elif wedge: 
            map_crs = ccrs.Stereographic(central_latitude = minLat+(maxLat-minLat)/2, central_longitude=minLon+(maxLon-minLon)/2)

        else:
            map_crs = ccrs.PlateCarree()
    
    # Build axes
    if size:
        fig     = plt.figure(figsize=size)
    else:
        fig     = plt.figure()
    ax      = plt.axes(projection=map_crs)

    ax.set_extent((minLon,maxLon,minLat,maxLat), crs=ccrs.PlateCarree())
    
    # Set masks
    if coast:
        ax.coastlines()
    if land:
        ax.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
    
    # Add gridlines
    if projection or polar or wedge:
        gl = ax.gridlines(linestyle='--', alpha=0.75, linewidth=0.5)
    else: 
        gl = ax.gridlines(crs=map_crs, linestyle='--', alpha=0.75, linewidth=0.5, draw_labels = True)
        gl.xlabels_top   = False
        gl.ylabels_right = False
        gl.xformatter    = LONGITUDE_FORMATTER
        gl.yformatter    = LATITUDE_FORMATTER
    # Circular clipping
    if polar:
        circle_clip = set_circular_boundary(ax)
    if wedge:
        wedge_clip = set_wedge_boundary(ax, minLon, maxLon, minLat, maxLat)
        
        
    if polar: 
        plotfield = ax.scatter(lons, lats, transform=ccrs.PlateCarree(), clip_path=(circle_clip, ax.transAxes), cmap=colormap, **kwargs)
    else:
        plotfield = ax.scatter(lons, lats, transform=ccrs.PlateCarree(), cmap=colormap, **kwargs)
    
    ax.set_title(title)
    # Export as figure
    if export:
        if not os.path.exists('figures'):
            os.makedirs('figures')
        if export[-4] == '.':
            plt.savefig(f'figures/{export}', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'figures/{export}.png', dpi=300, bbox_inches='tight')
    return fig, ax

########################################################################################

def from_field(inputfield, trajectoryFile=None, particleDensity=False, binGridWidth=1, latRange=(-90, 90), lonRange=(-180, 180), coast=True, wedge=False, t_index=0, land=True, projection=False, polar=False, vectorField=False, export=None, t_end=None, titleAttribute="", size=None, colormap=None, cbextend='neither'):
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
    if size:
        fig     = plt.figure(figsize=size)
    else:
        fig     = plt.figure()
    if projection:
        map_crs = projection
    else:
        if polar:
            map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
        else:
            map_crs = ccrs.PlateCarree()
            
    ax = plt.axes(projection=map_crs)
    
    # Determine boundaries and add land mask
    if wedge:
        ax.set_extent((-50,70,57.5,90), crs=ccrs.PlateCarree())
    else:
        ax.set_extent((minLon,maxLon,minLat,maxLat), crs=ccrs.PlateCarree())
    if coast:
        ax.coastlines()
    if land:
        ax.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
    
    # Add gridlines
    if polar or projection:
        gl = ax.gridlines(linestyle='--', alpha=0.75, linewidth=0.5)
    else: 
        gl = ax.gridlines(crs=map_crs, linestyle='--', alpha=0.75, linewidth=0.5, draw_labels = True)
        gl.xlabels_top   = False
        gl.ylabels_right = False
        gl.xformatter    = LONGITUDE_FORMATTER
        gl.yformatter    = LATITUDE_FORMATTER
    # Circular clipping
    if polar:
        circle_clip = set_circular_boundary(ax)
    
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
    if not colormap:
        colormap = 'viridis'
    # Plot field
    if particleDensity:
        densLats = np.arange(minLat, maxLat, binGridWidth)
        densLons = np.arange(minLon, maxLon, binGridWidth)
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
        # Base case: pColormesh
        inputfield.fieldset.computeTimeChunk(inputfield.grid.time[t_index], 1)
        if polar:
            plotfield = ax.pcolormesh(lons, lats, inputfield.data[t_index,:,:], transform=ccrs.PlateCarree(), zorder=1, clip_path=(circle_clip, ax.transAxes), cmap=colormap)
        else:
            plotfield = ax.pcolormesh(lons, lats, inputfield.data[t_index,:,:], transform=ccrs.PlateCarree(), zorder=1, cmap=colormap)
    
    # Colorbar
    divider = make_axes_locatable(ax)
    ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cbar    = plt.colorbar(plotfield, cax=ax_cb, extend=cbextend)
    
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
        if hasattr(inputfield.grid, 'timeslices'):
            titlestring = inputfield.name + ' at ' + str(inputfield.grid.timeslices.flatten()[t_index])[0:16]
        else:
            titlestring = inputfield.name
    ax.set_title(titlestring)
    # Export as figure
    if export:
        if not os.path.exists('figures'):
            os.makedirs('figures')
        if export[-4] == '.':
            plt.savefig(f'figures/{export}', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'figures/{export}.png', dpi=300, bbox_inches='tight')
    return fig, ax


class particleAnimation:
    def create(pfile, field=None, lonRange=None, latRange=None, coast=True, land=False, projection=False, polar=False, wedge=False, times='flat', particle_subsample=1, title="", fps=24, colormap=None, size=None, cbar=True, cbextend='neither', units=None, s=0.01, **kwargs):
        """
        Create particle animations
        """
        
        # Load arrays from file
        lon = np.ma.filled(pfile.variables['lon'][::particle_subsample], np.nan)
        lat = np.ma.filled(pfile.variables['lat'][::particle_subsample], np.nan)
        time = np.ma.filled(pfile.variables['time'][::particle_subsample], np.nan)
        mesh = pfile.attrs['parcels_mesh'] if 'parcels_mesh' in pfile.attrs else 'spherical'
        
        # Range
        if lonRange:
            minLon, maxLon = lonRange
        else: 
            minlon = np.amin(pfile.variables['lon']) - margin
            maxlon = np.amax(pfile.variables['lon']) + margin
        if latRange:
            minLat, maxLat = latRange
        else:
            minlat = np.amin(pfile.variables['lat']) - margin
            maxlat = np.amax(pfile.variables['lat']) + margin
        
        # Set projection
        if projection:
            map_crs = projection
        else:
            if polar:
                map_crs = ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
            elif wedge: 
                map_crs = ccrs.Stereographic(central_latitude = minLat+(maxLat-minLat)/2, central_longitude=minLon+(maxLon-minLon)/2)
            else:
                map_crs = ccrs.PlateCarree()
        
        if size:
            fig     = plt.figure(figsize=size)
        else:
            fig     = plt.figure()
        ax      = plt.axes(projection=map_crs)
        ax.set_extent((minLon,maxLon,minLat,maxLat), crs=ccrs.PlateCarree())
        
        # Set masks
        if coast:
            ax.coastlines()
        if land:
            ax.add_feature(cart.feature.LAND, zorder=5, edgecolor='k')
        
            # Add gridlines
        if projection or polar or wedge:
            gl = ax.gridlines(linestyle='--', alpha=0.75, linewidth=0.5)
        else: 
            gl = ax.gridlines(crs=map_crs, linestyle='--', alpha=0.75, linewidth=0.5, draw_labels = True)
            gl.xlabels_top   = False
            gl.ylabels_right = False
            gl.xformatter    = LONGITUDE_FORMATTER
            gl.yformatter    = LATITUDE_FORMATTER

        # Circular clipping
        if polar:
            circle_clip = set_circular_boundary(ax)
        if wedge:
            wedge_clip = set_wedge_boundary(ax, minLon, maxLon, minLat, maxLat)
            
        if field:
            fieldName = field.name
            field.fieldset.computeTimeChunk(field.grid.time[0], 1)
            if not colormap:
                colormap = 'viridis'
            ax.pcolormesh(field.lon, field.lat, field.data[0,:,:], transform=map_crs, cmap=colormap, zorder=1)
        else:
            fieldName = 'noField'

        # Determine plotting time indices
        if times == 'flat':
            firstFullTrajectoryIdx = np.searchsorted(~np.isnat(time[:, -1]), True)
            plottimes = time[firstFullTrajectoryIdx,:]
        else:
            plottimes = np.unique(time)
            if isinstance(plottimes[0], (np.datetime64, np.timedelta64)):
                plottimes = plottimes[~np.isnat(plottimes)]
            else:
                try:
                    plottimes = plottimes[~np.isnan(plottimes)]
                except:
                    pass
    
        # Create initial scatter plot of particles
        if times == 'flat':
            scat   = ax.scatter(lon[:,0], lat[:,0], c=lat[:,0], s=s, transform=ccrs.Geodetic(), zorder=10)
        else:
            currtime = time == plottimes[0]
            scat   = ax.scatter(lon[currtime], lat[currtime], c=lat[currtime], s=s, transform=ccrs.Geodetic(), zorder=10)
        
        # Colorbar
        if cbar:
            divider = make_axes_locatable(ax)
            if wedge:
                ax_cb = divider.new_vertical(size="5%", pad=0.1, axes_class=plt.Axes, pack_start=True)
                fig.add_axes(ax_cb)
                cbar = plt.colorbar(scat, cax=ax_cb, orientation='horizontal', extend=cbextend)
            else:
                ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
                fig.add_axes(ax_cb)
                cbar = plt.colorbar(scat, cax=ax_cb, extend=cbextend)

            # Set units
            if units:
                if wedge:
                    cbar.ax.set_xlabel(f"{str(units)}")
                else:
                    cbar.ax.set_ylabel(f"{str(units)}")
        
        head = fig.suptitle('Particles at time ' + str(plottimes[0])[:13])
        frames = np.arange(0, len(plottimes))
        
        # Animation
        def animate(t):
            if times == 'flat':
                scat.set_offsets(np.vstack((lon[:, t], lat[:, t])).transpose())
            else:
                currtime = time == plottimes[t]
                scat.set_offsets(np.vstack((lon[currtime], lat[currtime])).transpose())
            scat.set_color
            head.set_text('Particles at time ' + str(plottimes[t])[:13])
            return scat,
        
        anim = animation.FuncAnimation(fig, 
                                       animate, 
                                       frames=len(plottimes),
                                       blit=True)
        if not os.path.exists('animations'):
            os.makedirs('animations')
        anim.save(f'animations/particle_evolution_{fieldName}_{title}.mp4', fps=fps, metadata={'artist':'Daan', 'title':f'Particles on {fieldName} - {title}'}, extra_args=['-vcodec', 'libx264'])
        plt.show()
        plt.close()

