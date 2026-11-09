"""Domain boundary policies, independent of particle integration and physics."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AxisBoundary:
    _allow_equal_bounds = False
    lower: float
    upper: float
    mode: str='exiting'
    def __post_init__(self):
        equal_bounds = self._allow_equal_bounds and self.lower == self.upper and np.isfinite(self.lower)
        if (not (self.lower < self.upper or equal_bounds)
                or self.mode not in ('exiting', 'periodic', 'reflection')):
            raise ValueError('Invalid boundary')
    def apply(self,position,fluctuation):
        p=np.asarray(position).copy();v=np.asarray(fluctuation).copy();width=self.upper-self.lower
        alive=np.ones(p.shape,dtype=bool)
        if self.mode=='exiting':alive=(p>self.lower)&(p<self.upper)
        elif self.mode=='periodic':
            outside=(p<self.lower)|(p>self.upper);p[outside]=self.lower+np.mod(p[outside]-self.lower,width)
        else:
            outside=(p<self.lower)|(p>self.upper)
            phase=np.mod(p[outside]-self.lower,2*width)
            reflected=phase>width
            original=p[outside].copy()
            p[outside]=self.lower+np.where(reflected,2*width-phase,phase)
            crossing=np.floor((original-self.lower)/width).astype(np.int64)
            crossing-=((original>self.upper)&(np.mod(original-self.lower,width)==0)).astype(np.int64)
            v[outside]*=np.where(crossing%2,-1.,1.)
        return p,v,alive

class DomainBoundaries:
    def __init__(self,axes):
        if len(axes)!=3:raise ValueError("Three axis policies required")
        self.axes=tuple(axes)
    def apply(self,positions,fluctuations):
        p=np.array(positions,copy=True);v=np.array(fluctuations,copy=True);alive=np.ones(len(p),dtype=bool)
        for axis,policy in enumerate(self.axes):
            p[:,axis],v[:,axis],ok=policy.apply(p[:,axis],v[:,axis]);alive&=ok
        return p,v,alive


class NativeAxisBoundary(AxisBoundary):
    """CPU QES float32 wrapping and reflection rules."""

    _allow_equal_bounds = True

    def apply(self, position, fluctuation):
        p = np.array(position, dtype=np.float32, copy=True)
        v = np.array(fluctuation, dtype=np.float32, copy=True)
        lower, upper = np.float32(self.lower), np.float32(self.upper)
        alive = np.isfinite(p)
        if self.mode == 'exiting':
            return p, v, alive & (p > lower) & (p < upper)
        if self.mode == 'periodic':
            width = upper-lower
            if width == 0:
                return p, v, alive
            # The two loops preserve the closed interval and native float32
            # rounding. A no-progress step would loop forever in the source.
            for threshold, direction in ((lower, 1), (upper, -1)):
                outside = alive & (p < threshold if direction == 1 else p > threshold)
                increment = np.float32(direction)*width
                while outside.any():
                    previous = p[outside]
                    updated = previous+increment
                    alive[outside] &= updated != previous
                    p[outside] = updated
                    outside = alive & (p < threshold if direction == 1 else p > threshold)
        else:
            for _ in range(100):
                below = alive & (p < lower)
                above = alive & (p > upper)
                if not (below | above).any():
                    break
                p[below] = lower-(p[below]-lower)
                p[above] = upper-(p[above]-upper)
                v[below | above] *= np.float32(-1)
            # The source unconditionally returns ACTIVE after the loop,
            # including after its warning at the 100-bounce limit.
        return p, v, alive


class NativeDomainBoundaries(DomainBoundaries):
    """Apply CPU boundaries in axis order while the particle remains active."""

    def apply(self, positions, fluctuations, *, active=True):
        p = np.array(positions, dtype=np.float32, copy=True)
        v = np.array(fluctuations, dtype=np.float32, copy=True)
        alive = np.broadcast_to(active, (len(p),)).copy()
        for axis, policy in enumerate(self.axes):
            if not alive.any():
                break
            p[alive, axis], v[alive, axis], alive[alive] = policy.apply(p[alive, axis], v[alive, axis])
        return p, v, alive
