"""Native analytic power-law test interpolation (InterpPowerLaw.cpp, GPL-3.0).

Upstream's mat3sym aggregate is ordered xx,xy,xz,yy,yz,zz. Its initializer
therefore creates an indefinite tensor. That observable source behavior is
preserved; this diagnostic/test model is not a recommended ambient turbulence
closure and is not silently replaced with an intended diagonal tensor.
"""
import numpy as np
from .interpolation import FieldSampler


class PowerLawSampler(FieldSampler):
    _allow_nonpositive_height = False

    def _height(self, positions):
        p = np.asarray(positions, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError('Expected finite Nx3 coordinates')
        z = .1*np.ceil(10*p[:, 2])
        if not self._allow_nonpositive_height and np.any(z <= 0):
            raise ValueError('Analytic power-law turbulence is singular at nonpositive height')
        return z

    def max_fluctuation(self):
        return 10*2.5*(.4*.15*4.8*self.zf[-2]**.15)

    def sample_wind(self, positions):
        z = self._height(positions)
        return np.c_[4.8*z**.15, np.zeros((len(z), 2))]

    def sample_initial_turbulence(self, positions):
        tau = self.sample_turbulence_and_distance(positions)[0]
        us = .4*.15*4.8*self._height(positions)**.15
        return tau, us[:, None]*np.array([2.5, 2.3, 1.3])

    def sample_turbulence_and_distance(self, positions):
        z = self._height(positions); p = .15; a = 4.8
        us = .4*p*a*z**p
        ce = 5.7*us**3/(.4*z)
        tau = np.zeros((len(z), 3, 3))
        tau[:, 0, 0] = (2.5*us)**2
        tau[:, 0, 1] = tau[:, 1, 0] = (2.3*us)**2
        tau[:, 0, 2] = tau[:, 2, 0] = (1.3*us)**2
        tau[:, 2, 2] = -us**2
        flux = np.zeros((len(z), 3))
        flux[:, 0] = -2*p*(.4*p*a)**2*z**(2*p-1)
        flux[:, 2] = 2*p*(1.3*.4*p*a)**2*z**(2*p-1)
        distance = self.sample_distance(positions)
        return tau, flux, ce, None, distance


class NativePowerLawSampler(PowerLawSampler):
    """Float32 storage and the original mixed-precision CPU expressions."""

    _a = np.float32(4.8)
    _p = np.float32(.15)
    _initial_scale = np.array([2.5, 2.3, 1.3], dtype=np.float32)
    _allow_nonpositive_height = True
    # A negative sample also gives NaN mean wind. The native GLE rejects
    # the particle after consuming its draws, before any position update.
    defer_wind_validation = True

    def _height(self, positions):
        return super()._height(np.asarray(positions, dtype=np.float32)).astype(np.float32)

    @staticmethod
    def _powerf(base, exponent):
        # Evaluate in double and round at the original powf storage point.
        with np.errstate(invalid='ignore'):
            return np.power(np.asarray(base, dtype=np.float64), float(exponent)).astype(np.float32)

    def _friction_velocity(self, height, *, single_power):
        with np.errstate(invalid='ignore'):
            power = (self._powerf(height, self._p) if single_power
                     else np.power(height.astype(np.float64), float(self._p)))
        return (.4*float(self._p)*float(self._a)*power.astype(np.float64)).astype(np.float32)

    def max_fluctuation(self):
        velocity = self._friction_velocity(np.asarray(self.zf[-2], dtype=np.float32), single_power=False)
        return float(np.float32(25.*float(velocity)))

    def sample_wind(self, positions):
        height = self._height(positions)
        speed = self._a*self._powerf(height, self._p)
        return np.column_stack((speed, np.zeros((len(height), 2), dtype=np.float32)))

    def _stress(self, velocity, sigma):
        tau = np.zeros((len(velocity), 3, 3), dtype=np.float32)
        xx, xy, xz = (sigma*sigma).T
        tau[:, 0, 0] = xx
        tau[:, 0, 1] = tau[:, 1, 0] = xy
        tau[:, 0, 2] = tau[:, 2, 0] = xz
        tau[:, 2, 2] = -(velocity*velocity)
        return tau

    def sample_initial_turbulence(self, positions):
        velocity = self._friction_velocity(self._height(positions), single_power=True)
        sigma = velocity[:, None]*self._initial_scale
        return self._stress(velocity, sigma), sigma

    def sample_turbulence_and_distance(self, positions):
        height = self._height(positions)
        p, a = self._p, self._a
        velocity = self._friction_velocity(height, single_power=False)
        # The native ground sample has zero stress. Preserve its IEEE
        # dissipation/divergence so the GLE rejects just that particle.
        with np.errstate(divide='ignore', invalid='ignore'):
            eps = (np.float32(5.7)*velocity*velocity*velocity)/(np.float32(.4)*height)
            derivative = self._powerf(height, np.float32(2)*p-np.float32(1))
        scale_x = np.float32(.4)*p*a
        scale_z = np.float32(1.3)*np.float32(.4)*p*a
        flux = np.zeros((len(height), 3), dtype=np.float32)
        flux[:, 0] = -np.float32(2)*p*(scale_x*scale_x)*derivative
        flux[:, 2] = np.float32(2)*p*(scale_z*scale_z)*derivative
        tau = self._stress(velocity, velocity[:, None]*self._initial_scale)
        return tau, flux, eps, None, self.sample_distance(positions)
