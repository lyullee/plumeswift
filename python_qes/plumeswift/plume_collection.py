"""CPU concentration sampling and accumulation for QES plumes (GPL-3.0)."""

import numpy as np


def native_box_geometry(config):
    lower = np.asarray(config.lower, dtype=np.float32)
    upper = np.asarray(config.upper, dtype=np.float32)
    return lower, (upper-lower)/np.asarray(config.boxes, dtype=np.float32)


def native_box_centers(config):
    lower, widths = native_box_geometry(config)
    return {axis: (lo+np.arange(count, dtype=np.float32)*width)+width/np.float32(2)
            for axis, lo, width, count in zip('xyz', lower, widths, config.boxes, strict=True)}


class PlumeCollection:
    def __init__(self, config, *, native_math=False, allow_zero_volume=False):
        self.boxes = np.asarray(config.boxes)
        self.native_math = native_math
        self.p_count = np.zeros(tuple(reversed(config.boxes)), dtype=np.int32)
        if native_math:
            self.lower, self.widths = native_box_geometry(config)
            self.volume = self.widths[0]*self.widths[1]*self.widths[2]
            if (np.any(self.widths < 0) or not np.isfinite(self.volume) or
                    (self.volume == 0 and not allow_zero_volume)):
                raise ValueError('Invalid collection volume in float32')
            # The unsuffixed 1e-9 in Concentration.h promotes the divisor.
            self.divisor = self.widths.astype(np.float64)+1e-9
            self.mass_time = np.zeros(self.p_count.shape, dtype=np.float32)
            self.averaging_s = np.float32(0.)
        else:
            self.lower = np.asarray(config.lower)
            self.widths = (np.asarray(config.upper)-self.lower)/self.boxes
            self.volume = np.prod(self.widths)
            self.divisor = self.widths+1e-9
            self.mass_time = np.zeros(self.p_count.shape)
            self.averaging_s = 0.

    def collect(self, positions, mass_kg, dt):
        if self.native_math:
            positions = positions.astype(np.float32, copy=False)
            dt = np.float32(dt)
        indices = np.floor((positions-self.lower)/self.divisor).astype(int)
        valid = np.all((indices >= 0) & (indices < self.boxes), axis=1)
        i, j, k = indices[valid].T
        mass = mass_kg[valid]
        if self.native_math:
            # Native stores grams in ParticleCore. The Python batch stores kg.
            mass = (1000*mass).astype(np.float32)
        np.add.at(self.p_count, (k, j, i), 1)
        np.add.at(self.mass_time, (k, j, i), mass*dt)
        self.averaging_s += dt

    def finalize(self):
        mass_time_g = self.mass_time if self.native_math else 1000*self.mass_time
        # Native collection allows equal bounds and writes IEEE NaN or infinity for zero volume.
        with np.errstate(divide='ignore', invalid='ignore'):
            concentration = mass_time_g/(self.averaging_s*self.volume)
        return dict(p_count=self.p_count.copy(), c=concentration, t_avg=self.averaging_s)

    def reset(self):
        self.p_count.fill(0)
        self.mass_time.fill(0)
        self.averaging_s = self.mass_time.dtype.type(0.)

    def take(self):
        fields = self.finalize()
        self.reset()
        return fields
