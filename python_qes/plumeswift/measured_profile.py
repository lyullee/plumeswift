"""Native multi-height sensor profile flag 4 (WindProfilerSensorType, GPL-3).

Preserves native segment advancement, direction-wrap slopes and polynomial/log
branch arithmetic. It does not silently substitute standard spline interpolation.
"""
from dataclasses import InitVar, dataclass, field
import numpy as np


@dataclass(frozen=True)
class MeasuredProfile:
    heights: tuple[float, ...]
    speeds: tuple[float, ...]
    directions: tuple[float, ...]
    roughness: float = .02
    reciprocal_length: float = 0.
    kind: int = field(default=4, init=False)
    allow_negative_heights: InitVar[bool] = field(default=False, kw_only=True)

    @property
    def speed(self): return self.speeds[0]
    @property
    def direction(self): return self.directions[0]
    @property
    def reference_height(self): return self.heights[0]

    def __post_init__(self, allow_negative_heights):
        h, u, d = [np.asarray(values, dtype=float) for values in [self.heights, self.speeds, self.directions]]
        if h.ndim != 1 or len(h) < 1 or u.shape != h.shape or d.shape != h.shape:
            raise ValueError('Measured heights, speeds and directions must have equal lengths')
        if (not all(np.isfinite(values).all() for values in (h, u, d)) or np.any(np.diff(h) <= 0)
                or (h[0] < 0 and not allow_negative_heights) or (len(h) == 1 and h[0] <= 0)
                or np.any(u < 0) or self.roughness <= 0):
            raise ValueError('Invalid measured sensor profile')

    def evaluate(self, heights, *, native_math=False):
        scalar = np.float32 if native_math else np.float64
        z = np.asarray(heights, dtype=scalar)
        if z.ndim != 1 or np.any(np.diff(z) < 0) or not np.isfinite(z).all() or np.any(z < 0):
            raise ValueError('Native measured-profile evaluation needs ascending nonnegative heights')
        h, u, direction = (np.asarray(values, dtype=scalar)
                           for values in (self.heights, self.speeds, self.directions))
        if np.any(np.diff(h) <= 0):
            raise ValueError('Measured heights must remain distinct at the selected precision')
        squared = h.astype(float)**2
        roughness, vk = scalar(self.roughness), scalar(.4)
        result = np.zeros((len(z), 2), dtype=scalar)
        segment = -1
        theta = scalar((270.-float(direction[0]))*np.pi/180.)
        log_flag = False
        a1 = a2 = a3 = slope = ustar = scalar(0)
        z0_new = None

        def log_height(height, z0):
            return scalar(np.log(float((height+z0)/z0)))

        def direction_components(angle):
            return np.array([np.cos(float(angle)), np.sin(float(angle))], dtype=scalar)

        for k, height in enumerate(z):
            if height < h[0] or len(h) == 1:
                reference = log_height(h[0], roughness)
                local = log_height(height, roughness)
                if native_math:
                    result[k] = (u[0]*direction_components(theta)/reference)*local
                    continue
                magnitude = u[0]/reference*local
            else:
                if segment < len(h)-2 and height >= h[segment+1]:
                    segment += 1; i = segment
                    delta = h[i+1]-h[i]
                    diff = direction[i+1]-direction[i]
                    # These unusual ±360/dz slopes are the actual source.
                    slope = scalar((-360. if diff > 0 else 360.)/float(delta)) if abs(diff) > 180 else diff/delta
                    # At nonpositive lower height the native log-fit bounds contain
                    # NaN or infinity, so their test selects the polynomial.
                    log_flag = False
                    if h[i] > 0:
                        high, low = scalar(20), scalar(1e-9)
                        ustar = vk*u[i]/log_height(h[i], high)
                        u_high = ustar/vk*log_height(h[i], high)
                        ustar = vk*u[i]/log_height(h[i], low)
                        u_low = ustar/vk*log_height(h[i+1], low)
                        log_flag = u[i+1] > u_low and u[i+1] < u_high
                    if log_flag:
                        ustar = vk*u[i]/log_height(h[i], roughness)
                        u_new = ustar/vk*log_height(h[i+1], roughness)
                        for _ in range(200):
                            if abs(float(u_new-u[i])) <= .0001*float(u[i]):
                                break
                            z0_new = scalar(.5*float(low+high))
                            ustar = vk*u[i]/log_height(h[i], z0_new)
                            u_new = ustar/vk*log_height(h[i+1], z0_new)
                            if u_new > h[i+1]: high = z0_new
                            else: low = z0_new
                    else:
                        a1 = scalar(0)
                        if i < len(h)-2:
                            numerator = delta*(u[i+2]-u[i])+(h[i]-h[i+2])*(u[i+1]-u[i])
                            denominator = (float(delta)*(squared[i+2]-squared[i])
                                           +(squared[i+1]-squared[i])*float(h[i]-h[i+2]))
                            a1 = scalar(float(numerator)/denominator)
                        a2 = scalar((float(u[i+1]-u[i])-float(a1)*(squared[i+1]-squared[i]))/float(delta))
                        a3 = scalar(float(u[i])-float(a1)*squared[i]-float(a2*h[i]))
                if log_flag:
                    if z0_new is None:
                        raise FloatingPointError('Native flag-4 log branch leaves z0_new uninitialized')
                    magnitude = ustar/vk*log_height(height, z0_new)
                else:
                    magnitude = scalar(float(a1)*float(height)**2+float(a2*height)+float(a3))
                angle = direction[segment]+slope*(height-h[segment])
                theta = scalar((270.-float(angle))*np.pi/180.)
            result[k] = magnitude*direction_components(theta)
        if not np.isfinite(result).all():
            raise FloatingPointError('Non-finite measured wind profile')
        return result[:, 0], result[:, 1]
