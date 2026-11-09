"""QES-Fire Rothermel seed and iterative Balbi spread model (GPL-3)."""
from dataclasses import dataclass, fields
from math import pi

import numpy as np

from .fire_fuels import Fuel


@dataclass(frozen=True)
class FireProperties:
    updraft: np.ndarray
    flame_height: np.ndarray
    flame_depth: np.ndarray
    spread_rate: np.ndarray
    temperature: np.ndarray
    residence_time: np.ndarray
    drag: np.ndarray
    heat_release: np.ndarray


def rothermel(fuel: Fuel, moisture):
    """Native no-wind/no-slope R0 in m/s for combustible fuel."""
    fine_load = fuel.one_hour+fuel.live_herb+fuel.live_woody
    total_load = fine_load+fuel.ten_hour+fuel.hundred_hour
    if np.any(np.asarray(fine_load) <= 0):
        raise ValueError('Rothermel seed requires positive fine fuel load')
    sav = (fuel.sav_one_hour*fuel.one_hour+109*fuel.ten_hour+30*fuel.hundred_hour+
           fuel.sav_herb*fuel.live_herb+fuel.sav_woody*fuel.live_woody)/total_load
    moisture = np.asarray(moisture)
    load = (1-moisture/(1+moisture))*fine_load*.2471*.3048**2*2.205
    packing = load/(fuel.depth*fuel.density)
    ratio = packing/(3.348*sav**-.8189)
    preignition = 250+1116*moisture
    bulk_density = load/fuel.depth
    maximum = sav**1.5/(495+.0594*sav**1.5)
    exponent = 1/(4.774*sav**.1-7.27)
    reaction_velocity = maximum*ratio**exponent*np.exp(exponent*(1-ratio))
    relative_moisture = moisture/(fuel.extinction_moisture/100)
    moisture_damping = 1-2.59*relative_moisture+5.11*relative_moisture**2-3.52*relative_moisture**3
    intensity = reaction_velocity*(load/1.0555)*fuel.heat_content*moisture_damping*.174*.01**-.19
    flux = np.exp((.792+.681*np.sqrt(sav))*(packing+.1))/(192+.2595*sav)
    return .005080*intensity*flux/(bulk_density*np.exp(-138/sav)*preignition)


def _relative_angle(x, y, normal_x, normal_y):
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine = (x*normal_x+y*normal_y)/(np.hypot(x, y)*np.hypot(normal_x, normal_y))
        angle = np.arccos(np.clip(cosine, -1., 1.))
    small = ((abs(x) < .001) & (abs(y) < .001)) | ((abs(normal_x) < .001) & (abs(normal_y) < .001))
    return np.where(small | np.isnan(angle), 0., angle)


def balbi(fuel: Fuel, wind_u, wind_v, normal_x, normal_y, slope_x, slope_y, moisture, *, cure, dx, dy):
    """Broadcast the native Balbi model over fuel and wind arrays."""
    if not np.isfinite([cure, dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError('Invalid Balbi cure or grid spacing')
    cure = np.clip(cure, .30, 1.20)
    values = np.broadcast_arrays(*(np.asarray(getattr(fuel, f.name), dtype=float) for f in fields(Fuel)),
                                 wind_u, wind_v, normal_x, normal_y, slope_x, slope_y, moisture)
    shape = values[0].shape
    fgi = (values[0]+values[3]+values[4])*.2471
    active = fgi >= 1e-5
    result = [np.zeros(shape) for _ in fields(FireProperties)]
    result[5].fill(5000.)
    if not np.any(active):
        return FireProperties(*result)
    fuel = Fuel(*(value[active] for value in values[:12]))
    u, v, nx, ny, sx, sy, moisture = (value[active] for value in values[12:])
    if not all(np.isfinite(value).all() for value in (u, v, sx, sy, moisture)) or np.any(moisture < 0):
        raise ValueError('Invalid wind, slope or moisture for Balbi model')
    fine_load = fuel.one_hour+fuel.live_herb+fuel.live_woody
    total_load = fine_load+fuel.ten_hour+fuel.hundred_hour
    fine_sav_sum = fuel.sav_one_hour*fuel.one_hour+fuel.sav_herb*fuel.live_herb+fuel.sav_woody*fuel.live_woody
    sav = fine_sav_sum/fine_load/.3048
    total_sav = (fine_sav_sum+109*fuel.ten_hour+30*fuel.hundred_hour)/total_load
    tau0, air_temperature, air_heat_capacity = 75591., 289.15, 1150.
    residence_time = tau0/(total_sav/.3048)+300
    fuel_depth = fuel.depth*.3048
    heat_content = fuel.heat_content*2326
    density = fuel.density*16.0185
    dead_load = (fuel.one_hour+fuel.live_herb*(1.20-cure)/.90)*.2471
    beta = dead_load/(fuel_depth*density)
    total_beta = fgi[active]/(fuel_depth*density)
    if np.any(beta <= 0):
        raise FloatingPointError('Native Balbi equations are singular at zero dead fine fuel')
    absorption = np.fmin(sav*fuel_depth*beta, 2*pi*beta/total_beta)
    phi = _relative_angle(sx, sy, nx, ny)
    cos_psi = np.cos(_relative_angle(u, v, nx, ny))
    alpha = np.where(phi > .785, -1., 1.)*np.hypot(sx, sy)
    drag = 100*total_beta
    activation = 2000*(600-air_temperature)+moisture*2.257e6
    radiant = np.fmin(sav/(2*pi), beta/total_beta)*.3*heat_content/(4*activation)
    speed = np.hypot(u, v)
    rate = rothermel(fuel, moisture)
    tilt = alpha.copy()
    running = np.ones(rate.shape, dtype=bool)
    chi = np.zeros_like(rate)
    temperature = np.zeros_like(rate)
    updraft = np.zeros_like(rate)
    height = np.zeros_like(rate)
    for _ in range(99):
        cosine = np.cos(tilt)
        step_chi = .3/(1+rate*cosine/(sav*2.5e-5))
        step_temp = air_temperature+heat_content*(1-step_chi)/(18*air_heat_capacity)
        step_up = 2*absorption*(18/tau0)*(density/1.125)*(step_temp/air_temperature)
        step_tilt = np.arctan(np.tan(alpha)*np.cos(phi)+speed*cos_psi/step_up)
        step_height = step_up**2/(9.81*(step_temp/air_temperature-1)*np.cos(alpha)**2)
        convection = 2.257e6*absorption*(17/30)/(activation*tau0*step_up*total_beta)
        base = np.fmin(sav*fuel_depth*total_beta/pi, 1)*(beta/total_beta)**2*(5.67e-8*step_temp**4)/(beta*density*activation)
        flame = radiant*rate*(1+np.sin(step_tilt)-np.cos(step_tilt))/(1+rate*np.cos(step_tilt)/(sav*2.5e-5))
        convective = convection*(np.tan(alpha)+2*speed/step_up*np.exp(-drag*rate))*cos_psi
        step_rate = np.maximum(base+flame+convective, base)
        error = np.abs(step_rate-rate)
        for previous, new in ((rate, step_rate), (tilt, step_tilt), (chi, step_chi),
                              (temperature, step_temp), (updraft, step_up), (height, step_height)):
            np.copyto(previous, new, where=running)
        running &= error > 1e-5
        if not np.any(running):
            break
    depth = rate*residence_time
    heat = (1-chi)*heat_content*fgi[active]*(dx+dy)/2*np.minimum(depth, dx)/residence_time
    for output, value in zip(result, (updraft, height, depth, rate, temperature, residence_time, drag, heat)):
        if not np.isfinite(value).all():
            raise FloatingPointError('Non-finite native Balbi model result')
        output[active] = value
    return FireProperties(*result)
