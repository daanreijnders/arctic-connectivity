import numpy as np
from datetime import timedelta as delta
from datetime import datetime
import sys
import os.path
from parcels import (grid, Field, FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4,
                     ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)

sys.path.append('/daanreijnders/surfdrive/thesis/repository/tools/')
import comtools
import fieldsetter

readdir_ice = '/daanreijnders/datasets/'
readdir_ocean = '/daanreijnders/datasets/'
readdir_mesh = '/daanreijnders/datasets/'

fieldfile_ocean = 'daily_CESM_0.1degree_controlrun_year_300_arctic_timed_no_cord.nc'
fieldfile_ice = 'monthly_icefields_CESM_0.1degree_controlrun_year_300_arctic.nc'
meshfile = 'POP_grid_coordinates.nc'

writedir = '/daanreijnders/datasets/out/'

#### SETTINGS #####
plon = 3590 # Number of particles per longitude band
plat = 590 # Number of particles per latitude band
start_year = '2000'
start_month = '01'
start_day = '01'
name = 'control_run_year_300'
ndays = 30 # Days of advection
advectdt = 60 # minutes
outputdt = 12 # hours
minlat = 60.5
maxlat = 89.5
minlon = -179.5
maxlon = 179.5
####################

particleG = comtools.particleGrid(plat, 
                                  plon, 
                                  release_time=datetime(int(start_year), int(start_month), int(start_day)),
                                  minLat=minlat,
                                  maxLat=maxlat,
                                  minLon=minlon,
                                  maxLon=maxlon)
#particleG.remove_on_land(fieldset)
lonList = particleG.lonlat[0,:,0]
latList = particleG.lonlat[0,:,1]
timeList = particleG.release_times
fieldset = fieldsetter.read_velocity_field(readdir_ocean+fieldfile_ocean, meshfile=readdir_mesh+meshfile)

experiment_name = f"{name}_P{plon}x{plat}_S{start_year}-{start_month}-{start_day}_D{ndays}_DT{advectdt}_ODT{outputdt}_LAT{minlat}-{maxlat}_LON{minlon}-{maxlon}"

pset = ParticleSet.from_list(fieldset,
                             JITParticle,
                             lonList,
                             latList,
                             time = timeList,
                             lonlatdepth_dtype = np.float64)

# Kernels for circular boundary and for deleting particles as fallback.
def wrapLon(particle, fieldset, time):
    if particle.lon > 180.:
        particle.lon = particle.lon - 360.
    if particle.lon < -180.:
        particle.lon = particle.lon + 360.

def deleteParticle(particle, fieldset, time):
    particle.delete()

kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(wrapLon)
pfile = pset.ParticleFile(name = writedir+experiment_name, 
                          outputdt=delta(hours=outputdt))
pfile.add_metadata("dt", str(advectdt))
pfile.add_metadata("Output dt", str(outputdt))
pfile.add_metadata("Runtime", str(ndays))
pfile.add_metadata("Release time of first particle", str(particleG.release_times[0]))
print(f"Run: Advecting particles for {ndays}")
pset.execute(kernels, \
             runtime = delta(days = ndays), \
             dt = delta(minutes = advectdt), \
             output_file = pfile, \
             recovery = {ErrorCode.ErrorOutOfBounds: deleteParticle})