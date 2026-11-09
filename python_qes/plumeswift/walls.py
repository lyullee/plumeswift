"""Voxel wall reflection with exact segment traversal, separated from advection.

This retains the stair-step impermeable-wall representation. Segment
traversal avoids missing a wall when a numerical step crosses several cells.
It is not a replacement for the separately tracked triangle/cut-cell model.
"""
import numpy as np

class NoWalls:
    def reflect(self,positions,displacement,fluctuations):
        return positions+displacement,fluctuations.copy()


def endpoint_cells(sampler, positions):
    """Return bounded candidate indices and an in-grid mask."""
    p = np.asarray(positions)
    i = np.searchsorted(sampler.xf, p[:, 0], side='right')-1
    j = np.searchsorted(sampler.yf, p[:, 1], side='right')-1
    k = np.searchsorted(sampler.zf, p[:, 2], side='right')-1
    indices = (k, j, i)
    limits = (len(sampler.z), len(sampler.y), len(sampler.x))
    inside = np.logical_and.reduce([(idx >= 0) & (idx < limit) for idx, limit in zip(indices, limits)])
    return tuple(idx.clip(0, limit-1) for idx, limit in zip(indices, limits)), inside


def cell_grid_mask(indices, shape):
    return np.logical_and.reduce([(axis >= 0) & (axis < size)
                                  for axis, size in zip(indices, shape, strict=True)])


def fluid_cell_mask(indices, solid):
    """Look up solidity only for indices within the allocated cell array."""
    active = cell_grid_mask(indices, solid.shape)
    active[active] = ~solid[tuple(axis[active] for axis in indices)]
    return active


class SolidExitWalls(NoWalls):
    """Deactivate endpoints in solid cells or outside the allocated grid."""
    rejects_invalid_endpoints = True

    def __init__(self, sampler, solid):
        self.sampler, self.solid = sampler, solid

    def active_mask(self, positions):
        if self.sampler.native_math:
            finite = np.isfinite(positions).all(axis=1)
            finite[finite] = fluid_cell_mask(self.sampler.candidate_cell_indices(positions[finite]), self.solid)
            return finite
        indices, inside = endpoint_cells(self.sampler, positions)
        return inside & ~self.solid[indices]


class VoxelWalls:
    def __init__(self,sampler,solid,max_crossings=256, *, endpoint_only=False, compiled=False):
        self.sampler=sampler;self.solid=np.asarray(solid,dtype=bool);self.max_crossings=max_crossings
        self.endpoint_only = endpoint_only
        self.compiled = compiled and not endpoint_only and self.solid.shape == (
            len(sampler.zf)-1, len(sampler.yf)-1, len(sampler.xf)-1)
        if self.compiled:
            from .numerics.compiled_transport import uniform_spacing

            self._spacing = uniform_spacing((sampler.xf, sampler.yf, sampler.zf))
    def reflect(self,positions,displacement,fluctuations):
        if self.compiled and self.max_crossings > 0:
            inputs = tuple(np.asarray(a) for a in (positions, displacement, fluctuations))
            if all(a.dtype == np.float64 and a.ndim == 2 and a.shape == inputs[0].shape
                   and a.shape[1] == 3 and np.isfinite(a).all() for a in inputs):
                from .numerics.compiled_transport import reflect_voxels

                axes = self.sampler.xf, self.sampler.yf, self.sampler.zf
                epsilon = min(self.sampler.dx, self.sampler.dy, float(np.min(np.diff(axes[2]))))*1e-10
                return reflect_voxels(*inputs, axes, self._spacing, self.solid, epsilon, self.max_crossings)
        p=np.array(positions,copy=True);remaining=np.array(displacement,copy=True);v=np.array(fluctuations,copy=True)
        pending=np.arange(len(p));axes=[self.sampler.xf,self.sampler.yf,self.sampler.zf]
        if self.endpoint_only:
            end = p+remaining
            indices, inside = endpoint_cells(self.sampler, end)
            check = (inside & self.solid[indices]) | (end[:, 2] < self.sampler.zf[1])
            p[~check] = end[~check]
            pending = np.flatnonzero(check)
        epsilon=min(self.sampler.dx,self.sampler.dy,float(np.min(np.diff(self.sampler.zf))))*1e-10
        for _ in range(self.max_crossings):
            if not len(pending):return p,v
            q=p[pending];d=remaining[pending]
            # Face traversal uses geometric cells. The upstream nearest
            # sampler's index bias can exceed the epsilon nudge at a face.
            (k,j,i),inside=endpoint_cells(self.sampler,q)
            if not inside.all():
                raise ValueError('Wall reflection starts outside the geometric grid')
            indices=np.column_stack([i,j,k])
            direction=np.sign(d).astype(int);times=np.full_like(d,np.inf)
            for axis,faces in enumerate(axes):
                boundary=faces[indices[:,axis]+(direction[:,axis]>0)]
                np.divide(boundary-q[:,axis],d[:,axis],out=times[:,axis],where=d[:,axis]!=0)
            axis=np.argmin(times,axis=1);fraction=times[np.arange(len(q)),axis]
            done=fraction>=1-1e-13
            ids=pending[done];p[ids]+=remaining[ids];remaining[ids]=0
            crossing=~done
            if not crossing.any():return p,v
            ids=pending[crossing];a=axis[crossing];f=np.maximum(fraction[crossing],0)
            step=direction[crossing]
            neighbour=indices[crossing].copy();neighbour[np.arange(len(ids)),a]+=step[np.arange(len(ids)),a]
            valid=np.all(neighbour>=0,axis=1)&np.all(neighbour<np.array(self.solid.shape[::-1]),axis=1)
            # Exiting grid is handled by the independent domain policy.
            outside_ids=ids[~valid];p[outside_ids]+=remaining[outside_ids];remaining[outside_ids]=0
            if not valid.any():return p,v
            ids=ids[valid];a=a[valid];f=f[valid];neighbour=neighbour[valid];step=step[valid]
            hit=self.solid[neighbour[:,2],neighbour[:,1],neighbour[:,0]]
            p[ids]+=f[:,None]*remaining[ids];remaining[ids]*=(1-f)[:,None]
            bounced=ids[hit];normal=a[hit]
            remaining[bounced,normal]*=-1;v[bounced,normal]*=-1
            # Place the segment on the correct side of the face, without
            # discarding a finite fraction of the travelled distance.
            nudges=step[np.arange(len(ids)),a].astype(float);nudges[hit]*=-1
            p[ids,a]+=epsilon*nudges
            pending=ids
        raise RuntimeError("Voxel-wall traversal exceeded numerical crossing limit")
