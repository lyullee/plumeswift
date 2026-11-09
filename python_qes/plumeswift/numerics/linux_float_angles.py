# SPDX-License-Identifier: LGPL-2.1-or-later
"""Single-precision angle arithmetic for the Linux QES reference.

The atan range reductions and coefficients are from Sun Microsystems' fdlibm
implementation (Copyright 1993 Sun Microsystems, Inc. All rights reserved).
Developed at SunPro, a Sun Microsystems, Inc. business. Permission to use,
copy, modify, and distribute this software is freely granted, provided that
this notice is preserved.

The bounded sine/cosine reduction and coefficients are adapted from glibc 2.35
(Copyright 2018–2022 Free Software Foundation, Inc., LGPL-2.1-or-later).
Only the finite angle range used by QES rotation is needed here.
"""

import numpy as np


_ATAN_HIGH = np.array([.46364760399, .78539812565, .98279368877, 1.5707962513], dtype=np.float32)
_ATAN_LOW = np.array([5.0121582440e-9, 3.7748947079e-8, 3.4473217170e-8, 7.5497894159e-8], dtype=np.float32)
_ATAN_COEFF = np.array([.33333334327, -.20000000298, .14285714924, -.11111110449,
                        .090908870101, -.076918758452, .066610731184, -.058335702866,
                        .049768779427, -.036531571299, .016285819933], dtype=np.float32)
_COS_COEFF = tuple(map(float.fromhex, ('-0x1.ffffffd0c621cp-2', '0x1.55553e1068f19p-5',
                                      '-0x1.6c087e89a359dp-10', '0x1.99343027bf8c3p-16')))
_SIN_COEFF = tuple(map(float.fromhex, ('-0x1.555545995a603p-3', '0x1.1107605230bc4p-7', '-0x1.994eb3774cf24p-13')))


def _horner(x, coefficients):
    value = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        value = coefficient+x*value
    return value


def _atan_positive(value):
    interval = np.searchsorted(np.array([.4375, .6875, 1.1875, 2.4375], dtype=np.float32), value, side='right')
    reduced = value.copy()
    for number in range(1, 5):
        mask = interval == number
        x = value[mask]
        if number == 1:
            reduced[mask] = (2*x-1)/(2+x)
        elif number == 2:
            reduced[mask] = (x-1)/(x+1)
        elif number == 3:
            reduced[mask] = (x-1.5)/(1+1.5*x)
        else:
            reduced[mask] = -1/x
    square = reduced*reduced
    fourth = square*square
    correction = reduced*(square*_horner(fourth, _ATAN_COEFF[::2])+fourth*_horner(fourth, _ATAN_COEFF[1::2]))
    result = reduced-correction
    shifted = interval > 0
    anchor = interval[shifted]-1
    result[shifted] = _ATAN_HIGH[anchor]-((correction[shifted]-_ATAN_LOW[anchor])-reduced[shifted])
    result[value >= 2.**25] = _ATAN_HIGH[3]+_ATAN_LOW[3]
    return result


def atanf(value):
    """Evaluate the Linux reference float32 arctangent."""
    value = np.asarray(value, dtype=np.float32)
    angles = _atan_positive(np.abs(value).ravel()).reshape(value.shape)
    return np.copysign(angles, value)


def atan2f(y, x):
    """Evaluate the reference float32 atan2 for wind vectors and polygon edges."""
    y, x = np.broadcast_arrays(np.asarray(y, dtype=np.float32), np.asarray(x, dtype=np.float32))
    result = np.array(np.arctan2(y.astype(float), x.astype(float)), dtype=np.float32)
    valid = (x != 0) & (y != 0) & np.isfinite(x) & np.isfinite(y)
    ax, ay = np.abs(x[valid]), np.abs(y[valid])
    shift = (ay.view(np.uint32).astype(np.int64)-ax.view(np.uint32).astype(np.int64)) >> 23
    negative_x = np.signbit(x[valid])
    regular = (shift <= 60) & (~negative_x | (shift >= -60))
    quotient = np.divide(ay, ax, out=np.zeros_like(ax), where=regular)
    angle = _atan_positive(quotient)
    pi, pi_low = np.float32(np.pi), np.float32(-8.7422776573e-8)
    angle[shift > 60] = np.float32(np.pi/2)+np.float32(.5)*pi_low
    angle[negative_x] = pi-(angle[negative_x]-pi_low)
    result[valid] = np.copysign(angle, y[valid])
    return result


def sincosf(angle):
    """Return float32 sine and cosine for angles in the QES rotation range."""
    value = np.asarray(angle, dtype=np.float32)
    if not np.isfinite(value).all() or np.any(abs(value) >= 120):
        raise ValueError('Reference rotation requires finite angles within 120 radians')
    x = value.astype(float)
    quadrant = np.floor(x*(2/np.pi)+.5).astype(int)
    x = x-quadrant*(np.pi/2)
    square = x*x
    fourth = square*square
    s1, s2, s3 = _SIN_COEFF
    c1, c2, c3, c4 = _COS_COEFF
    cube = x*square
    sine = (x+cube*s1)+(cube*square)*(s2+square*s3)
    cosine = ((1+square*c1)+fourth*c2)+(fourth*square)*(c3+square*c4)
    q = quadrant & 3
    sin_result = np.select((q == 0, q == 1, q == 2), (sine, cosine, -sine), default=-cosine)
    cos_result = np.select((q == 0, q == 1, q == 2), (cosine, -sine, -cosine), default=sine)
    tiny = abs(value) < 2.**-12
    return np.where(tiny, value, sin_result).astype(np.float32), np.where(tiny, 1., cos_result).astype(np.float32)
