# Kernels for circular boundary and for deleting particles as fallback.
def wrapLon(particle, fieldset, time):
    if particle.lon > 180.:
        particle.lon = particle.lon - 360.
    if particle.lon < -180.:
        particle.lon = particle.lon + 360.

def deleteParticle(particle, fieldset, time):
    particle.delete()
    
    
# From
def BeachTesting_2D(particle, fieldset, time):
    if particle.beached == 2 or particle.beached == 3:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if fabs(u) < 1e-14 and fabs(v) < 1e-14:
            if particle.beached == 2:
                particle.beached = 4
            else:
                particle.beached = 1
        else:
            particle.beached = 0

def UnBeaching(particle, fieldset, time):
    if particle.beached == 4:
        (ub, vb) = fieldset.UVunbeach[time, particle.depth, particle.lat, particle.lon]
        particle.lon += ub * particle.dt
        particle.lat += vb * particle.dt
        particle.beached = 0
        particle.unbeachCount += 1