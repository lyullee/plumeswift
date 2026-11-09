# SPDX-License-Identifier: LGPL-2.1-or-later
"""Float32 log and exponential using the glibc 2.35 range reductions.

Adapted from glibc e_logf.c, e_logf_data.c, e_expf.c and e_exp2f_data.c.
Copyright 2017–2022 Free Software Foundation, Inc., LGPL-2.1-or-later.
"""
import math

import numpy as np

from .linux_float_power import _EXP2_POLY, _EXP2_TABLE, _FLOAT32, _FLOAT64, _LOG2_CENTERS, _UINT32, _UINT64


_LOG_CENTERS = tuple(map(float.fromhex, (
    '-0x1.57bf7808caadep-2', '-0x1.2bef0a7c06ddbp-2', '-0x1.01eae7f513a67p-2',
    '-0x1.b31d8a68224e9p-3', '-0x1.6574f0ac07758p-3', '-0x1.1aa2bc79c81p-3',
    '-0x1.a4e76ce8c0e5ep-4', '-0x1.1973c5a611cccp-4', '-0x1.252f438e10c1ep-5',
    '0x0p+0', '0x1.aa5aa5df25984p-5', '0x1.c5e53aa362eb4p-4', '0x1.526e57720db08p-3',
    '0x1.bc2860d22477p-3', '0x1.1058bc8a07ee1p-2', '0x1.4043057b6ee09p-2',
)))
_LOG_POLY = tuple(map(float.fromhex, (
    '-0x1.00ea348b88334p-2', '0x1.5575b0be00b6ap-2', '-0x1.ffffef20a4123p-2',
)))
_LN2 = float.fromhex('0x1.62e42fefa39efp-1')
_INV_LN2_32 = float.fromhex('0x1.71547652b82fep+0')*32
_EXP_POLY = tuple(value/32**order for value, order in zip(_EXP2_POLY, (3, 2, 1)))
_EXP_MAX = float.fromhex('0x1.62e42ep6')
_EXP_MIN = -float.fromhex('0x1.9fe368p6')


def logf(value):
    """Evaluate the Linux reference logarithm with round-to-nearest input and output."""
    x = float(np.float32(value))
    if x == 0.:
        return np.float32(-math.inf)
    if x < 0. or math.isnan(x):
        return np.float32(math.nan)
    if math.isinf(x) or x == 1.:
        return np.float32(x if math.isinf(x) else 0.)

    bits = _UINT32.unpack(_FLOAT32.pack(x))[0]
    if bits < 0x00800000:
        bits = _UINT32.unpack(_FLOAT32.pack(x*2**23))[0]-(23 << 23)
    offset = bits-0x3f330000
    index = (offset >> 19) & 15
    exponent = offset >> 23
    reduced = _FLOAT32.unpack(_UINT32.pack(bits-(exponent << 23)))[0]
    residual = reduced*_LOG2_CENTERS[index][0]-1.
    squared = residual*residual
    origin = _LOG_CENTERS[index]+exponent*_LN2
    a, b, c = _LOG_POLY
    polynomial = a*squared+(b*residual+c)
    return np.float32(polynomial*squared+(origin+residual))


def expf(value):
    """Evaluate the Linux reference exponential with round-to-nearest input and output."""
    x = float(np.float32(value))
    if math.isnan(x) or x > _EXP_MAX:
        return np.float32(x if math.isnan(x) else math.inf)
    if x < _EXP_MIN:
        return np.float32(0.)

    scaled = x*_INV_LN2_32
    exponent = round(scaled)
    residual = scaled-exponent
    scale_bits = _EXP2_TABLE[exponent % 32]+(exponent << 47)
    scale = _FLOAT64.unpack(_UINT64.pack(scale_bits))[0]
    a, b, c = _EXP_POLY
    polynomial = (a*residual+b)*(residual*residual)+(c*residual+1.)
    return np.float32(polynomial*scale)
