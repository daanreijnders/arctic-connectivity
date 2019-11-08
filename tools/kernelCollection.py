from parcels import (JITParticle, Variable)
import numpy as np

class unbeachableBoundedParticle(JITParticle):
        # Beaching dynamics from https://github.com/OceanParcels/Parcelsv2.0PaperNorthSeaScripts        
        # beached : 0 sea, 1 beached, 2 after non-beach dyn, 3 after beach dyn, 4 please unbeach
        beached = Variable('beached', dtype=np.int32, initial=0.)
        unbeachCount = Variable('unbeachCount', dtype=np.int32, initial=0.)
        # inBounds : 1 yes, 0 no
        inBounds = Variable('inBounds', dtype=np.int32, initial=1.)
        
# Kernels for circular boundary
def wrapLon(particle, fieldset, time):
    if particle.lon > 180.:
        particle.lon = particle.lon - 360.
    if particle.lon < -180.:
        particle.lon = particle.lon + 360.

def northPolePushBack(particle, fieldset, time):
    if particle.lat > 89.915:
        particle.lat = 89.915
        
# Freeze particles that get out of bounds
def freezeOutOfBoundsWedge(particle, fieldset, time):
    lon, lat = (particle.lon, particle.lat)
    if lon > 65. or lon < -45. or lat > 85. or lat < 60.:
        particle.inBounds = 0

# Freeze particles that get out of bounds
def freezeOutOfBoundsArctic(particle, fieldset, time):
    lon, lat = (particle.lon, particle.lat)
    if lat < 60.:
        particle.inBounds = 0

# Advection kernel. Checks first whether a particle is within bounds and whether it is not beached.        
def advectionRK4(particle, fieldset, time):
    if particle.inBounds == 1:
        if particle.beached == 0:        
            (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
            lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)

            (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
            lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)

            (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
            lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)

            (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
            particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
            particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
            particle.beached = 2
        
def deleteParticle(particle, fieldset, time):
    print(f"Particle {particle.id} deleted: ({particle.lon}, {particle.lat} at {particle.time})")
    particle.delete()
    
# Beaching dynamics from https://github.com/OceanParcels/Parcelsv2.0PaperNorthSeaScripts        
def beachTesting(particle, fieldset, time):
    if particle.inBounds == 1:
        if particle.beached == 2 or particle.beached == 3:
            (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
            if fabs(u) < 1e-14 and fabs(v) < 1e-14:
                if particle.beached == 2:
                    particle.beached = 4
                else:
                    particle.beached = 1
            else:
                particle.beached = 0
                
# Beaching dynamics from https://github.com/OceanParcels/Parcelsv2.0PaperNorthSeaScripts        
def unBeaching(particle, fieldset, time):
    if particle.inBounds == 1:
        if particle.beached == 4:
            (ub, vb) = fieldset.UVunbeach[time, particle.depth, particle.lat, particle.lon]
            particle.lon += ub * particle.dt
            particle.lat += vb * particle.dt
            particle.beached = 0
            particle.unbeachCount += 1