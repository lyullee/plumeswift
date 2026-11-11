"""Source-scale and measured-wind sensitivity inputs for the EEC trials."""

from dataclasses import dataclass

import numpy as np

from plumeswift.interpolation import Trilinear
from plumeswift.meteorology import SensorProfile


@dataclass(frozen=True)
class JetDragLength:
    source: np.ndarray
    axis: np.ndarray
    diameter: float
    far_length: float = 1.

    def __call__(self, positions):
        axial = np.maximum((positions-self.source) @ self.axis, 0.)
        return np.minimum(self.far_length, self.diameter + .2*axial)


@dataclass(frozen=True)
class ApproachWindBlend:
    mast: SensorProfile
    approach: SensorProfile
    mast_y: float
    approach_y: float

    def apply(self, workspace, buildings, context):
        coords = workspace.coordinates
        heights = np.maximum(coords['z'], 0.)
        for axis, far, near in zip('uv', self.mast.evaluate(heights), self.approach.evaluate(heights), strict=True):
            y = coords['y_face'] if axis == 'v' else coords['y']
            weight = np.clip((self.mast_y-y)/(self.mast_y-self.approach_y), 0., 1.)
            field = workspace.arrays[axis+'0']
            field[:len(heights), :len(y), :len(coords['x'])] += (near-far)[:, None, None]*weight[None, :, None]
            if axis == 'u':
                field[:len(heights), :len(y), len(coords['x'])] += (near-far)[:, None]*weight


@dataclass(frozen=True)
class WakeWindObservation:
    location: tuple[float, float, float]
    components: tuple[float, float]
    scale: tuple[float, float, float]

    def apply(self, workspace, buildings, context):
        c = workspace.coordinates
        point = np.asarray(self.location)
        for key, target in zip('uv', self.components, strict=True):
            axes = [c[axis+'_face'] if axis == ('x' if key == 'u' else 'y') else c[axis] for axis in 'xyz']
            field = workspace.arrays[key+'0']
            view = field[:len(axes[2]), :len(axes[1]), :len(axes[0])]
            background = Trilinear(*axes)(view, point[None])[0]
            kernels = [np.maximum(1.-((coord-centre)/width)**2, 0.)**2
                       for coord, centre, width in zip(axes, point, self.scale, strict=True)]
            view += (target-background)*kernels[2][:, None, None]*kernels[1][None, :, None]*kernels[0][None, None, :]
