"""Coordinate-based trilinear/nearest interpolation on QES staggered fields."""
from dataclasses import dataclass
import numpy as np
from .contracts import ModelDomainError

@dataclass
class Trilinear:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    def __post_init__(self):
        for a in [self.x,self.y,self.z]:
            if len(a)<2 or not np.isfinite(a).all()or np.any(np.diff(a)<=0):raise ValueError("Interpolation axes must increase")
    @staticmethod
    def _positions(positions):
        pos=np.asarray(positions,dtype=float)
        if pos.ndim!=2 or pos.shape[1]!=3 or not np.isfinite(pos).all():raise ValueError("Expected finite Nx3 coordinates")
        return pos

    @staticmethod
    def _axis_weights(axis, positions):
        index = np.clip(np.searchsorted(axis, positions, side='left')-1, 0, len(axis)-2)
        weight = (positions-axis[index])/(axis[index+1]-axis[index])
        # At most half-cell extrapolation at a physical staggered boundary.
        if np.any(weight < -.500001) or np.any(weight > 1.500001):
            raise ModelDomainError("Interpolation outside grid support")
        return index, weight

    def weights(self,positions):
        pos = self._positions(positions)
        return [self._axis_weights(axis, column) for axis, column in zip((self.x,self.y,self.z), pos.T, strict=True)]

    def __call__(self,values,positions):
        return self._interpolate(values, self.weights(positions))

    def _interpolate(self, values, weights):
        values=np.asarray(values)
        if values.shape[:3]!=(len(self.z),len(self.y),len(self.x)):raise ValueError("Field does not match interpolation axes")
        (i,wx),(j,wy),(k,wz)=weights
        xs, ys, zs = (i, i+1), (j, j+1), (k, k+1)
        xw, yw, zw = (1-wx, wx), (1-wy, wy), (1-wz, wz)
        xy = tuple(tuple(x_weight*y_weight for x_weight in xw) for y_weight in yw)
        out=np.zeros((len(i),)+values.shape[3:]);tail=(None,)*(values.ndim-3)
        for dk in [0,1]:
            for dj in [0,1]:
                for di in [0,1]:
                    weight=xy[dj][di]*zw[dk]
                    out+=values[zs[dk],ys[dj],xs[di]]*weight[(slice(None),)+tail]
        return out


class CompiledTrilinear(Trilinear):
    def __post_init__(self):
        super().__post_init__()
        from .numerics.compiled_transport import uniform_spacing

        self.spacing = uniform_spacing((self.x, self.y, self.z))

    def __call__(self, values, positions):
        from .numerics.compiled_transport import trilinear

        values = np.asarray(values)
        if values.ndim not in (3, 4) or values.dtype not in (np.float32, np.float64):
            return super().__call__(values, positions)
        if values.shape[:3] != (len(self.z), len(self.y), len(self.x)):
            raise ValueError("Field does not match interpolation axes")
        pos = self._positions(positions)
        if values.ndim == 3:
            return trilinear(values[..., None], pos, self.x, self.y, self.z, self.spacing)[:, 0]
        return trilinear(values, pos, self.x, self.y, self.z, self.spacing)


class NativeTrilinear:
    """CPU InterpTriLinear float32 weights with float32 or float64 field values."""

    def __init__(self, spacing, offsets, z, *, value_dtype=np.float32):
        self.spacing = np.float32(spacing)
        self.offsets = np.asarray(offsets, dtype=np.float32)*self.spacing
        self.z = np.asarray(z, dtype=np.float32)
        self.value_dtype = np.dtype(value_dtype)
        if self.value_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError('Native interpolation supports float32 or float64 fields')

    def __call__(self, values, positions):
        pos = np.asarray(positions, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3 or not np.isfinite(pos).all():
            raise ValueError('Expected finite Nx3 coordinates')
        values = np.asarray(values, dtype=self.value_dtype)
        xy = pos[:, :2]-self.offsets
        # C++ adds a double literal to the divisor but divides in float for the weights.
        indices = np.floor(xy.astype(np.float64)/(float(self.spacing)+1e-7)).astype(np.intp)
        i, j = indices.T
        k = np.searchsorted(self.z, pos[:, 2], side='left')-1
        nz, ny, nx = values.shape[:3]
        plane_size = ny*nx
        base = k*plane_size+j*nx+i
        # The source uses flattened addresses. A y index outside its nominal
        # row can still address a valid value when native dy aliases dx.
        if (np.any(k < 0) or np.any(k+1 >= len(self.z))
                or np.any(base < 0) or np.any(base+plane_size+nx+1 >= nz*plane_size)):
            raise ModelDomainError('Particle outside native interpolation grid')
        a, b = ((xy/self.spacing).astype(np.float64)-indices).astype(np.float32).T
        c = (pos[:, 2]-self.z[k])/(self.z[k+1]-self.z[k])
        tail = (slice(None),)+(None,)*(values.ndim-3)
        weights = ((1-a)*(1-b), a*(1-b), a*b, (1-a)*b)
        weights = [weight[tail] for weight in weights]
        flat = values.reshape(nz*plane_size, *values.shape[3:])

        def plane(address):
            return (weights[0]*flat[address] + weights[1]*flat[address+1]
                    + weights[2]*flat[address+nx+1] + weights[3]*flat[address+nx])

        lower, upper = plane(base), plane(base+plane_size)
        return (upper-lower)*c[tail]+lower


class FieldSampler:
    """Sample grid quantities at particle positions in metres.

    Wind and turbulence fields use (z, y, x) storage. Mean wind is returned
    in m/s and stress as a symmetric (N, 3, 3) array in (m/s)². The mixing
    length returned with the turbulence controls each particle substep.
    """
    def __init__(self,coordinates,wind,turbulence,mode='trilinear',mixlength=None, *, native_spacing=None,
                 native_math=False, compiled=False):
        if mode not in ['trilinear','nearest']:raise ValueError(mode)
        self.mode=mode;self.c=coordinates;self.wind=wind;self.turb=turbulence
        self.native_math = native_math
        self.compiled = compiled and not native_math and mode == 'trilinear'
        self.native_spacing = native_spacing
        self.mixlength = (None if mixlength is None else
                          np.asarray(mixlength, dtype=np.float32 if native_math else None))
        self.x=np.asarray(coordinates['x']);self.y=np.asarray(coordinates['y']);self.z=np.asarray(coordinates['z'])
        if native_math and native_spacing is not None:
            self.dx, self.dy = map(float, native_spacing[:2])
        else:
            self.dx=float(np.median(np.diff(self.x)));self.dy=float(np.median(np.diff(self.y)))
        # Upstream x_face/y_face serialization is not reliable in older workspaces.
        self.xf=np.r_[self.x-self.dx/2,self.x[-1]+self.dx/2]
        self.yf=np.r_[self.y-self.dy/2,self.y[-1]+self.dy/2]
        self.zf=np.asarray(coordinates['z_face'])
        self.interp = {}
        if native_math:
            self.wind = {key: np.asarray(wind[key], dtype=np.float32) for key in 'uvw'}
            if mode == 'trilinear':
                dx = self.dx if native_spacing is None else native_spacing[0]
                self.interp = {
                    'u': NativeTrilinear(dx, (0, .5), self.z),
                    'v': NativeTrilinear(dx, (.5, 0), self.z),
                    'w': NativeTrilinear(dx, (.5, .5), self.zf),
                    'cell': NativeTrilinear(dx, (.5, .5), self.z),
                }
        else:
            xe=np.r_[self.x,self.x[-1]+self.dx];ye=np.r_[self.y,self.y[-1]+self.dy]
            ze=np.r_[self.z,self.z[-1]+(self.z[-1]-self.z[-2])]
            interp = CompiledTrilinear if self.compiled else Trilinear
            self.interp={'u':interp(self.xf,ye,ze),'v':interp(xe,self.yf,ze),
                         'w':interp(xe,ye,self.zf),'cell':interp(self.x,self.y,self.z)}
        names=['txx','txy','txz','tyy','tyz','tzz','div_tau_x','div_tau_y','div_tau_z','CoEps']
        self._packed_names=names+(['nuT']if 'nuT'in turbulence else [])+(['mixlength']if mixlength is not None else [])
        self._packed=np.stack([self.mixlength if key=='mixlength'else turbulence[key]for key in self._packed_names],axis=-1)
        if native_math:
            self._packed = self._packed.astype(np.float32, copy=False)
    def candidate_cell_indices(self, positions):
        p = np.asarray(positions)
        if self.native_spacing is not None and self.native_math:
            return self.native_candidate_cell_indices(p, self.native_spacing)
        i, j, k = (np.searchsorted(faces, axis, side='right')-1
                   for faces, axis in zip((self.xf, self.yf, self.zf), p.T, strict=True))
        return k,j,i

    def native_candidate_cell_indices(self, positions, spacing):
        """Packed native cell addressing independent of wind interpolation precision."""
        dx, _, dz = spacing
        p = np.asarray(positions, dtype=np.float32).astype(np.float64)
        dx, dz = map(float, (np.float32(dx), np.float32(dz)))
        i, j, k = np.floor((p + (0, 0, dz))/(np.array([dx, dx, dz])+1e-9)).astype(np.intp).T
        # Interp::getCellIndex uses dx for y and the minimum vertical width.
        nx, ny = len(self.x), len(self.y)
        k, row = np.divmod((k*ny+j)*nx+i, nx*ny)
        j, i = np.divmod(row, nx)
        return k, j, i
    def cell_indices(self,positions):
        k,j,i=self.candidate_cell_indices(positions)
        if np.any(i<0)or np.any(i>=len(self.x))or np.any(j<0)or np.any(j>=len(self.y))or np.any(k<0)or np.any(k>=len(self.z)):
            raise ModelDomainError("Particle outside cell grid")
        return k,j,i
    def sample_distance(self, positions):
        if self.mixlength is None:
            return None
        if self.native_spacing is not None:
            return self.mixlength[self.cell_indices(positions)]
        return self.sample_cells(self.mixlength, positions)
    def sample_wind(self,positions):
        if self.mode=='trilinear':
            if self.native_math or self.compiled:
                return np.column_stack([self.interp[key](self.wind[key],positions) for key in 'uvw'])
            px, py, pz = Trilinear._positions(positions).T
            u, v, w = (self.interp[key] for key in 'uvw')
            xf = Trilinear._axis_weights(u.x, px)
            yc = Trilinear._axis_weights(u.y, py)
            zc = Trilinear._axis_weights(u.z, pz)
            xc = Trilinear._axis_weights(v.x, px)
            yf = Trilinear._axis_weights(v.y, py)
            zf = Trilinear._axis_weights(w.z, pz)
            return np.column_stack((u._interpolate(self.wind['u'], (xf, yc, zc)),
                                    v._interpolate(self.wind['v'], (xc, yf, zc)),
                                    w._interpolate(self.wind['w'], (xc, yc, zf))))
        k,j,i=self.cell_indices(positions);u,v,w=[self.wind[key]for key in 'uvw']
        return np.column_stack([.5*(u[k,j,i]+u[k,j,i+1]),.5*(v[k,j,i]+v[k,j+1,i]),.5*(w[k,j,i]+w[k+1,j,i])])
    def sample_cells(self,values,positions):
        if self.mode=='trilinear':return self.interp['cell'](values,positions)
        return values[self.cell_indices(positions)]
    def sample_turbulence(self,positions):
        return self.sample_turbulence_and_distance(positions)[:4]
    def sample_initial_turbulence(self,positions):
        tau=self.sample_turbulence(positions)[0]
        variance = abs(np.diagonal(tau, axis1=1, axis2=2))
        if self.native_math:
            sigma = np.sqrt(variance.astype(np.float32))
            sigma[sigma == 0] = np.float32(1e-8)
        else:
            sigma = np.sqrt(np.maximum(variance, 1e-16))
        return tau,sigma
    def sample_turbulence_and_distance(self,positions):
        packed=self.sample_cells(self._packed,positions)
        components={key:packed[:,j]for j,key in enumerate(self._packed_names)}
        tau = np.empty((len(positions), 3, 3), dtype=np.float32 if self.native_math else float)
        tau[:,0,0]=components['txx'];tau[:,1,1]=components['tyy'];tau[:,2,2]=components['tzz']
        tau[:,0,1]=tau[:,1,0]=components['txy'];tau[:,0,2]=tau[:,2,0]=components['txz'];tau[:,1,2]=tau[:,2,1]=components['tyz']
        flux=packed[:,6:9]
        distance = (self.sample_distance(positions) if self.native_spacing is not None
                    else components.get('mixlength'))
        return tau,flux,np.maximum(packed[:,9],1e-6),components.get('nuT'),distance
