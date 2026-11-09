"""Independent observers: exact box counting and time-window accounting."""
from dataclasses import dataclass
import numpy as np


class StatisticsDirector:
    """Schedule named plume statistics on the shared integer-millisecond clock."""

    def __init__(self, start_ms, period_ms):
        self.start_ms = start_ms
        self.period_ms = period_ms
        self.next_output_ms = start_ms+period_ms
        self._sources = {}

    def attach(self, key, collect, finalize):
        """Register a collector and a callback that returns and resets its window."""
        if key in self._sources:
            raise ValueError(f'Duplicate plume statistics source {key!r}')
        self._sources[key] = (collect, finalize)
        self._sources = dict(sorted(self._sources.items()))

    def compute(self, time_ms, step_s):
        if time_ms <= self.start_ms:
            return None
        for collect, _ in self._sources.values():
            collect(time_ms, step_s)
        if time_ms < self.next_output_ms:
            return None
        result = {key: finalize(time_ms) for key, (_, finalize) in self._sources.items()}
        # Native advances one period per output, even after a step spans several periods.
        self.next_output_ms += self.period_ms
        return result


@dataclass
class SensorBoxes:
    centres:np.ndarray
    widths:tuple[float,float,float]=(2.,2.,.2)
    def __post_init__(self):
        self.centres=np.asarray(self.centres,dtype=float)
        if self.centres.ndim!=2 or self.centres.shape[1]!=3 or min(self.widths)<=0:raise ValueError("Invalid sensor boxes")
    def concentrations(self,positions,masses):
        positions = np.asarray(positions)
        positions = positions.astype(np.result_type(positions.dtype, self.centres.dtype), copy=False)
        values = np.empty(len(self.centres))
        volume = np.prod(self.widths)
        half_width = np.asarray(self.widths)/2
        heights, groups = np.unique(self.centres[:, 2], return_inverse=True)
        for group, height in enumerate(heights):
            at_height = np.abs(positions[:, 2]-height) < half_width[2:3]
            horizontal = positions[at_height, :2]
            layer_mass = masses[at_height]
            for j in np.flatnonzero(groups == group):
                x, y, _ = self.centres[j]
                inside = ((np.abs(horizontal[:, 0]-x) < half_width[0:1])
                          & (np.abs(horizontal[:, 1]-y) < half_width[1:2]))
                values[j] = np.sum(layer_mass[inside], dtype=float)/volume
        return values

class TimeAveragedSensors:
    def __init__(self,boxes,start_s,end_s,thermo=None):
        if start_s<0 or end_s<=start_s:raise ValueError("Invalid averaging interval")
        self.boxes=boxes;self.start=start_s;self.end=end_s;self.thermo=thermo
        self.integral=np.zeros(len(boxes.centres));self.sampled_s=0.
    def observe(self,time_start,dt,positions,masses):
        weight=max(0.,min(time_start+dt,self.end)-max(time_start,self.start))
        if not weight:return
        c=self.boxes.concentrations(positions,masses)
        value=self.thermo.evaluate(c).volume_fraction if self.thermo is not None else c
        self.integral+=weight*value;self.sampled_s+=weight
    def mean(self):
        if not self.sampled_s:raise RuntimeError("No samples in requested averaging window")
        return self.integral/self.sampled_s
