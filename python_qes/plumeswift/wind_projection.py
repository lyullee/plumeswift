"""QES Euler-Lagrange wind correction separated from geometry/Poisson solvers.

Equations adapted from QES-Public Solver_CPU.cpp (GPL-3.0), preserving the
coefficients, vertical metric and update ranges. Both Python Poisson methods
solve the same multiplier equation. Building empirical closures are separate
ports and are not replaced by a different wake model here.
"""
import numpy as np
from .contracts import PhysicalCoefficients

def rhs_from_workspace(workspace,physics=None,*,array_module=np):
    physics=physics or PhysicalCoefficients();a=workspace.arrays;c=workspace.coordinates
    e,f,g,h,m,n=[a[k]for k in ['e','f','g','h','m','n']]
    u,v,w=[a[k]for k in ['u0','v0','w0']]
    # Some upstream workspace files contain an inconsistent first face
    # coordinate. Derive uniform horizontal metrics from cell centres.
    nz,ny,nx=e.shape;dx=float(np.median(np.diff(c['x'])));dy=float(np.median(np.diff(c['y'])))
    dz=a['dz_array'].astype(float,copy=False);result=array_module.zeros(e.shape)
    inner=(slice(1,-1),)*3
    hu=(e[inner]*u[1:nz-1,1:ny-1,2:nx]-f[inner]*u[1:nz-1,1:ny-1,1:nx-1])*dx
    hv=(g[inner]*v[1:nz-1,2:ny,1:nx-1]-h[inner]*v[1:nz-1,1:ny-1,1:nx-1])*dy
    up_metric=(dz[1:-1]*.5*(dz[1:-1]+dz[2:]))[:,None,None]
    down_metric=(dz[1:-1]*.5*(dz[1:-1]+dz[:-2]))[:,None,None]
    hw=m[inner]*up_metric*w[2:nz,1:ny-1,1:nx-1]-n[inner]*down_metric*w[1:nz-1,1:ny-1,1:nx-1]
    result[inner]=-2*physics.alpha_horizontal**2*(hu+hv+hw)
    return result

def corrected_velocity(workspace,multiplier,physics=None,*,array_module=np,dtype=float):
    physics=physics or PhysicalCoefficients();a=workspace.arrays;c=workspace.coordinates;p=array_module.asarray(multiplier)
    u,v,w=[a[k].astype(dtype,copy=True)for k in ['u0','v0','w0']]
    nz,ny,nx=p.shape;dx=float(np.median(np.diff(c['x'])));dy=float(np.median(np.diff(c['y'])))
    u[1:nz-1,1:ny,1:nx]+=a['f'][1:nz-1,1:ny,1:nx]*dx*(p[1:nz-1,1:ny,1:nx]-p[1:nz-1,1:ny,:nx-1])/(2*physics.alpha_horizontal**2)
    v[1:nz-1,1:ny,1:nx]+=a['h'][1:nz-1,1:ny,1:nx]*dy*(p[1:nz-1,1:ny,1:nx]-p[1:nz-1,:ny-1,1:nx])/(2*physics.alpha_horizontal**2)
    w[1:nz-1,1:ny,1:nx]+=a['n'][1:nz-1,1:ny,1:nx]*a['dz_array'][1:nz-1,None,None]*(p[1:nz-1,1:ny,1:nx]-p[:nz-2,1:ny,1:nx])/(2*physics.alpha_vertical**2)
    zero_solid_faces(u,v,w,a['icellflag'],array_module=array_module)
    return {'u':u,'v':v,'w':w}

def zero_solid_faces(u,v,w,flags,*,array_module=np):
    solid=(flags==0)|(flags==2);solid[0]=False
    k,j,i=array_module.where(solid)
    u[k,j,i]=0;u[k,j,i+1]=0;v[k,j,i]=0;v[k,j+1,i]=0;w[k,j,i]=0;w[k+1,j,i]=0

def project(workspace,solver,physics=None, *, initial=None):
    if hasattr(solver, 'project'):
        if initial is not None:
            return solver.project(workspace, physics, initial=initial)
        return solver.project(workspace, physics)
    rhs=rhs_from_workspace(workspace,physics)
    multiplier,report=solver.solve(workspace.stencil(),rhs,initial) if initial is not None else solver.solve(workspace.stencil(),rhs)
    return corrected_velocity(workspace,multiplier,physics),multiplier,report
