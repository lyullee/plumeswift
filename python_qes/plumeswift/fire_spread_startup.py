"""Native QES-Fire arithmetic for the calm, level startup closure (GPL-3)."""
import math

import numpy as np

from .fire_spread import FireProperties

_f32 = np.float32


def _rothermel(fuel, moisture, fine_mass, sav):
    """Retain the float storage points of Rothermel.cpp."""
    depth, density = _f32(fuel.depth), _f32(fuel.density)
    extinction = _f32(fuel.extinction_moisture)/100
    load_moisture = moisture/(1+moisture)
    load_m = _f32((1.-float(load_moisture))*float(fine_mass))
    load = _f32(float(load_m)*.3048**2*2.205)
    packing = load/(depth*density)
    optimum = _f32(3.348*float(sav)**-.8189)
    ignition_heat = _f32(250.+1116.*float(moisture))
    epsilon = _f32(math.exp(-138./float(sav)))
    bulk_density = load/depth
    power = _f32(float(sav)**1.5)
    maximum = _f32(float(power)/(495.+.0594*float(power)))
    exponent = _f32(1./(4.774*float(sav)**.1-7.27))
    ratio = packing/optimum
    ratio_power = _f32(float(ratio)**float(exponent))
    reaction = _f32(float(maximum)*float(ratio_power)*math.exp(float(exponent)*(1.-float(ratio))))
    net_load = load/(1+_f32(.0555))
    relative = float(moisture/extinction)
    damping = _f32(1.-2.59*relative+5.11*relative**2-3.52*relative**3)
    mineral = _f32(.174*float(_f32(.0100))**-.19)
    intensity = reaction*net_load*_f32(fuel.heat_content)*damping*mineral
    flux = _f32(math.exp((.792+.681*math.sqrt(float(sav)))*(float(packing)+.1))/(192.+.2595*float(sav)))
    return _f32(float(intensity*flux/(bulk_density*epsilon*ignition_heat))*.005080)


def calm_startup(fuel, normal_x, normal_y, slope_x, slope_y, moisture, *, cure, dx, dy):
    """Evaluate the original zero-wind branch at a single fuel bed."""
    one, ten, hundred, herb, woody = map(_f32, (fuel.one_hour, fuel.ten_hour, fuel.hundred_hour,
                                              fuel.live_herb, fuel.live_woody))
    fine_mass = _f32(float(one+herb+woody)*.2471)
    if fine_mass < .00001:
        zero = np.zeros_like(slope_x, dtype=_f32)
        return FireProperties(zero, zero, zero, zero, zero, zero+5000, zero, zero)
    one_sav, herb_sav, woody_sav = map(_f32, (fuel.sav_one_hour, fuel.sav_herb, fuel.sav_woody))
    sav = _f32(float((one_sav*one+herb_sav*herb+woody_sav*woody)/(one+herb+woody))/.3048)
    total_sav = (one_sav*one+109*ten+30*hundred+herb_sav*herb+woody_sav*woody)/(one+ten+hundred+herb+woody)
    moisture, cure, dx, dy = map(_f32, (moisture, np.clip(cure, .3, 1.2), dx, dy))
    depth = _f32(float(_f32(fuel.depth))*.3048)
    heat = _f32(fuel.heat_content)*2326
    density = _f32(float(_f32(fuel.density))*16.0185)
    transfer = _f32(float(herb)*(1.2-float(cure))/.90)
    dead_mass = _f32(float(one+transfer)*.2471)
    beta, total_beta = dead_mass/(depth*density), fine_mass/(depth*density)
    pi = _f32(math.pi)
    absorption = np.fmin(sav*depth*beta, 2*pi*beta/total_beta)
    tau0, air_temperature = _f32(75591), _f32(289.15)
    tau = _f32(float(tau0)/(float(total_sav)/.3048)+300)
    drag = 100*total_beta
    activation = 2000*(600-air_temperature)+moisture*_f32(2.257e6)
    radiant = np.fmin(sav/(2*pi), beta/total_beta)*_f32(.3)*heat/(4*activation)
    normal_x, normal_y, slope_x, slope_y = (np.asarray(a, dtype=_f32)
                                           for a in (normal_x, normal_y, slope_x, slope_y))
    alpha = np.sqrt(slope_x*slope_x+slope_y*slope_y)
    with np.errstate(divide='ignore', invalid='ignore'):
        angle = np.arccos(((slope_x*normal_x+slope_y*normal_y)/
                           (alpha*np.sqrt(normal_x*normal_x+normal_y*normal_y))).astype(float)).astype(_f32)
    small = ((abs(slope_x.astype(float)) < .001) & (abs(slope_y.astype(float)) < .001) |
             (abs(normal_x.astype(float)) < .001) & (abs(normal_y.astype(float)) < .001))
    phi = np.where(small | np.isnan(angle), _f32(0), angle)
    alpha = np.where(phi.astype(float) > .785, -alpha, alpha)
    alpha_cos = np.cos(alpha.astype(float)).astype(_f32)
    alpha_tan = np.tan(alpha.astype(float)).astype(_f32)
    tilt_argument = alpha_tan*np.cos(phi.astype(float)).astype(_f32)
    tilt = alpha
    rate = np.full_like(alpha, _rothermel(fuel, moisture, fine_mass, total_sav))
    running = np.ones_like(rate, dtype=bool)
    chi = np.empty_like(rate)
    temperature, updraft, height = (np.empty_like(rate) for _ in range(3))
    for _ in range(99):
        cosine = np.cos(tilt.astype(float)).astype(_f32)
        step_chi = _f32(.3)/(1+rate*cosine/(sav*_f32(2.5e-5)))
        step_temp = air_temperature+heat*(1-step_chi)/(18*1150)
        step_up = 2*absorption*(_f32(18)/tau0)*(density/_f32(1.125))*(step_temp/air_temperature)
        step_tilt = np.arctan(tilt_argument.astype(float)).astype(_f32)
        step_height = step_up*step_up/(_f32(9.81)*(step_temp/air_temperature-1)*alpha_cos*alpha_cos)
        convection = 1/(activation*tau0*step_up*total_beta)*_f32(2.257e6)*absorption
        convection = (convection.astype(float)*float(_f32(17)/30)).astype(_f32)
        thermal = _f32(5.67e-8)*step_temp*step_temp*step_temp*step_temp
        packing_ratio = float(beta/total_beta)
        base = (min(float(sav*depth*total_beta/pi), 1.)*packing_ratio*packing_ratio*thermal.astype(float)
                /float(beta*density*activation)).astype(_f32)
        sine = np.sin(step_tilt.astype(float)).astype(_f32)
        cosine = np.cos(step_tilt.astype(float)).astype(_f32)
        flame = radiant*rate*(1+sine-cosine)/(1+rate*cosine/(sav*_f32(2.5e-5)))
        step_rate = np.maximum(base+flame+convection*alpha_tan, base)
        error = np.abs(step_rate-rate)
        for value, update in ((rate, step_rate), (chi, step_chi), (temperature, step_temp),
                              (updraft, step_up), (height, step_height)):
            np.copyto(value, update, where=running)
        tilt = step_tilt
        running &= error > _f32(1e-5)
        if not running.any():
            break
    length = rate*tau
    release = heat*fine_mass*((dx+dy)/2)*np.minimum(length, dx)/tau
    release = ((1.-chi.astype(float))*release).astype(_f32)
    return FireProperties(updraft, height, length, rate, temperature, np.broadcast_to(tau, rate.shape),
                          np.broadcast_to(drag, rate.shape), release)
