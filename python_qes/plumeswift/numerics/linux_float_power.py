# SPDX-License-Identifier: LGPL-2.1-or-later
"""Linux float32 wake exponent, adapted from glibc 2.35 powf.

The range reduction, coefficient tables and polynomials are from glibc's
e_powf.c, e_powf_log2_data.c and e_exp2f_data.c (Copyright 2017–2022 Free
Software Foundation, Inc., LGPL-2.1-or-later).
"""

from struct import Struct

import numpy as np


_FLOAT32 = Struct('<f')
_UINT32 = Struct('<I')
_FLOAT64 = Struct('<d')
_UINT64 = Struct('<Q')

_LOG2_CENTERS = tuple(tuple(map(float.fromhex, pair)) for pair in (
    ('0x1.661ec79f8f3bep+0', '-0x1.efec65b963019p-2'),
    ('0x1.571ed4aaf883dp+0', '-0x1.b0b6832d4fca4p-2'),
    ('0x1.49539f0f010bp+0', '-0x1.7418b0a1fb77bp-2'),
    ('0x1.3c995b0b80385p+0', '-0x1.39de91a6dcf7bp-2'),
    ('0x1.30d190c8864a5p+0', '-0x1.01d9bf3f2b631p-2'),
    ('0x1.25e227b0b8eap+0', '-0x1.97c1d1b3b7afp-3'),
    ('0x1.1bb4a4a1a343fp+0', '-0x1.2f9e393af3c9fp-3'),
    ('0x1.12358f08ae5bap+0', '-0x1.960cbbf788d5cp-4'),
    ('0x1.0953f419900a7p+0', '-0x1.a6f9db6475fcep-5'),
    ('0x1p+0', '0x0p+0'),
    ('0x1.e608cfd9a47acp-1', '0x1.338ca9f24f53dp-4'),
    ('0x1.ca4b31f026aap-1', '0x1.476a9543891bap-3'),
    ('0x1.b2036576afce6p-1', '0x1.e840b4ac4e4d2p-3'),
    ('0x1.9c2d163a1aa2dp-1', '0x1.40645f0c6651cp-2'),
    ('0x1.886e6037841edp-1', '0x1.88e9c2c1b9ff8p-2'),
    ('0x1.767dcf5534862p-1', '0x1.ce0a44eb17bccp-2'),
))
_LOG2_POLY = tuple(map(float.fromhex, (
    '0x1.27616c9496e0bp-2', '-0x1.71969a075c67ap-2', '0x1.ec70a6ca7baddp-2',
    '-0x1.7154748bef6c8p-1', '0x1.71547652ab82bp+0',
)))
# Encoded 2**(i/32), with the index contribution removed from the exponent bits.
_EXP2_TABLE = (
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
)
_EXP2_POLY = tuple(map(float.fromhex, ('0x1.c6af84b912394p-5', '0x1.ebfce50fac4f3p-3',
                                       '0x1.62e42ff0c52d6p-1')))


def powf_three_halves(ratio):
    """Evaluate native powf(ratio, 1.5) for a wake ratio in [0, 1]."""
    bits = _UINT32.unpack(_FLOAT32.pack(ratio))[0]
    if bits < 0x00800000:
        return np.float32(0.)  # Zero and float32 subnormals round to zero after the power.

    # x = 2**exponent * reduced, with the interval determined by float32 bits.
    offset = bits-0x3f330000
    index = (offset >> 19) & 15
    exponent = offset >> 23
    reduced = _FLOAT32.unpack(_UINT32.pack(bits-(exponent << 23)))[0]
    inverse_center, log_center = _LOG2_CENTERS[index]
    delta = reduced*inverse_center-1.
    delta_squared = delta*delta
    a0, a1, a2, a3, a4 = _LOG2_POLY
    log_linear = a4*delta+(log_center+exponent)
    log_quadratic = (a2*delta+a3)*delta_squared+log_linear
    log2_ratio = (a0*delta+a1)*(delta_squared*delta_squared)+log_quadratic

    power_log = 1.5*log2_ratio
    step = round(32*power_log)
    remainder = power_log-step/32
    scale = _FLOAT64.unpack(_UINT64.pack(_EXP2_TABLE[step % 32]+(step << 47)))[0]
    c0, c1, c2 = _EXP2_POLY
    correction = (c0*remainder+c1)*(remainder*remainder)+(c2*remainder+1.)
    return np.float32(correction*scale)
