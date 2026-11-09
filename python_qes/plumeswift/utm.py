"""QES GIStool UTM projection on its Clarke 1866 ellipsoid."""
import math

import numpy as np


_A, _B, _SCALE = map(float, map(np.float32, (6378206.4, 6356583.8, .9996)))
_E2 = float(np.float32(1-float(np.float32(_B/_A))**2))
_E4, _E6 = _E2**2, _E2**3
_EP = _E2/(1-_E2)
_M = 1-_E2/4-3*_E4/64-5*_E6/256


def to_utm(longitude, latitude, *, wrf_legacy=False, northern_hemisphere=True, native_zone=False):
    longitude, latitude = np.broadcast_arrays(np.asarray(longitude, dtype=np.float32),
                                               np.asarray(latitude, dtype=np.float32))
    # GIStool's float longitude rounds both operations before applying ceil.
    zone_number = (longitude+np.float32(180))/np.float32(6) if native_zone else (longitude.astype(float)+180)/6
    zone = np.clip((np.floor(zone_number)+1 if wrf_legacy else np.ceil(zone_number)).astype(int), 1, 60)
    # QES rounds assigned intermediates to float, while expressions with
    # double literals retain double precision until assignment.
    def f32(value):
        return np.asarray(value, dtype=np.float32)

    degrees = np.float32(float(np.float32(np.pi))/180)
    phi = (degrees*latitude).astype(float)
    delta = (degrees*f32(longitude-(6*zone-183))).astype(float)
    e4 = float(f32(_E2**2))
    e6 = float(f32(np.float32(_E2)*np.float32(e4)))
    ep = float(f32(_EP))
    arc1 = f32((1-_E2/4-3*e4/64-5*e6/256)*phi)
    arc2 = f32(float(f32(3*_E2/8+3*e4/32+45*e6/1024))*np.sin(2*phi))
    # The separate WRF converter multiplies these coefficients.
    third = (15*e4/256)*(45*e6/1024) if wrf_legacy else 15*e4/256+45*e6/1024
    arc3 = f32(float(f32(third))*np.sin(4*phi))
    arc4 = f32(float(f32(35*e6/3072))*np.sin(6*phi))
    arc = (np.float32(_A)*(arc1-arc2+arc3-arc4)).astype(float)
    sn, cs = np.sin(phi), np.cos(phi)
    radius = f32(_A/np.sqrt(1-_E2*sn**2))
    t, c, a = (f32(value).astype(float) for value in (np.tan(phi)**2, ep*cs**2, cs*delta))
    east3 = f32((1-t+c)*a**3/6)
    east5 = f32(f32(5-18*t+t*t+72*c-58*ep)*a**5/120)
    x = np.float32(500000)+np.float32(_SCALE)*radius*(f32(a)+east3+east5)
    north2 = f32(a*a/2)
    north4 = f32(f32(5-t+9*c+4*c*c)*a**4/24)
    north6 = f32(f32(61-58*t+t*t+600*c-330*ep)*a**6/720)
    y = f32(_SCALE*(arc+radius.astype(float)*np.tan(phi)*(north2+north4+north6)))
    if not northern_hemisphere:
        y += np.float32(10_000_000)
    return x, y, zone


def from_utm(x, y, zone, *, northern_hemisphere=True):
    if not northern_hemisphere:
        y = np.asarray(y)-10_000_000
    e = (1-np.sqrt(1-_E2))/(1+np.sqrt(1-_E2))
    u = y/(_SCALE*_A*_M)
    phi = u+(1.5*e-27*e**3/32)*np.sin(2*u)+(21*e**2/16-55*e**4/32)*np.sin(4*u)+151*e**3/96*np.sin(6*u)
    f = 1-_E2*np.sin(phi)**2
    rn, r = _A/np.sqrt(f), _A*(1-_E2)/f**1.5
    t, c = np.tan(phi)**2, _EP*np.cos(phi)**2
    d = (x-500000)/(rn*_SCALE)
    lat = phi-rn*np.tan(phi)/r*(d*d/2-(5+3*t+10*c-4*c*c-9*_EP)*d**4/24
                                 +(61+90*t+298*c+45*t*t-252*_EP-3*c*c)*d**6/720)
    delta = (d-(1+2*t+c)*d**3/6+(5-2*c+28*t-3*c*c+8*_EP+24*t*t)*d**5/120)/np.cos(phi)
    return np.rad2deg(delta)+(6*zone-183), np.rad2deg(lat)


def from_wrf_utm(x, y, zone):
    """Invert the signed-northing UTM coordinates used by WRFInput::UTMConv."""
    def single(value):
        return float(np.float32(value))

    pi = single(math.pi)
    degrees = single(180/pi)
    meridian = single((6*zone-183)*single(pi/180))
    e4 = single(_E2**2)
    e6 = single(_E2*e4)
    ep = single(_EP)
    e = single(math.sqrt(1-_E2))
    e = single((1-e)/(1+e))
    meridional = single(single(y)/_SCALE)
    scale = single(1-_E2/4-3*e4/64-5*e6/256)
    u = single(meridional/single(_A*scale))
    first = single(single(3*e/2-27*e**3/32)*math.sin(2*u))
    second = single(single(21*e**2/16-55*e**4/32)*math.sin(4*u))
    third = single(single(151*e**3/96)*math.sin(6*u))
    phi = single(single(single(u+first)+second)+third)
    if abs(single(phi*degrees)) >= 90:
        return float(6*zone-183), 0.
    c = single(ep*math.cos(phi)**2)
    t = single(math.tan(phi)**2)
    f = single(1-_E2*math.sin(phi)**2)
    rn = single(_A/math.sqrt(f))
    r = single(_A*(1-_E2)/math.sqrt(f**3))
    d = single(single(single(x)-500000)/single(rn*_SCALE))
    first = single(rn*math.tan(phi)/r)
    second = single(d**2/2)
    # The WRF converter uses a product in this coefficient.
    third = single(single(15*t+10*c-4*c**2-9*ep)*d**4/24)
    fourth = single(single(61+90*t+298*c+45*t**2-252*ep-3*c**2)*d**6/720)
    latitude_radians = single(phi-single(first*single(single(second-third)+fourth)))
    lat = single(latitude_radians*degrees)
    first = single(single(1+2*t+c)*d**3/6)
    second = single(single(5-2*c+28*t-3*c**2+8*ep+24*t**2)*d**5/120)
    lon = single((meridian+single(single(d-first)+second)/math.cos(phi))*degrees)
    if lon < -180:
        lon = single(lon+360)
    elif lon > 180:
        lon = single(lon-360)
    return lon, lat


def grid_convergence(longitude, latitude, zone):
    """Return QES GIStool's convergence angle in degrees."""
    longitude, latitude = (np.asarray(value, dtype=np.float32).astype(float)
                           for value in (longitude, latitude))
    radians = 3.14159265359/180
    delta = np.asarray(6.*np.asarray(zone)-183.-longitude, dtype=np.float32).astype(float)
    with np.errstate(invalid='ignore'):
        angle = np.arctan(np.arctan(delta*radians)*np.arcsin(latitude*radians))/radians
    return np.asarray(angle, dtype=np.float32)
