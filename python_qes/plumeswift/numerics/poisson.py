"""Replaceable solvers for the same seven-point QES multiplier equation.

The stencil is supplied by the physical/geometric model, not invented by a
solver. Outer sides/top are fixed multipliers, the bottom is mirrored.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,bicgstab
from ..contracts import NumericPolicy,SolverReport


NATIVE_WIND_SOLVERS = frozenset(('cpu-native-rb', 'cpu-native-serial', 'cuda-dynamic-legacy',
                                 'cuda-device-parallel', 'cuda-global-memory', 'cuda-shared-memory',
                                 'cpu-historical-sor'))


@dataclass(frozen=True)
class NativeWindPolicy(NumericPolicy):
    """XML defaults and no-sweep limit from QES SimulationParameters."""

    relative_tolerance: float = 1e-9
    max_iterations: int = 500
    minimum_iterations = 0

    def __post_init__(self):
        if (not np.isfinite(self.relative_tolerance) or self.relative_tolerance < 0
                or self.absolute_tolerance <= 0 or self.max_iterations < self.minimum_iterations):
            raise ValueError('Invalid numerical convergence policy')


def make_poisson(backend, policy=None, *, native_cuda_boundary=False):
    if backend == 'cpu':
        return KrylovPoisson(policy)
    if backend == 'cpu-historical-sor':
        from .native_cpu_rb import HistoricalCPUSOR
        return HistoricalCPUSOR(policy or NativeWindPolicy())
    if backend in ('cpu-native-rb', 'cpu-native-serial'):
        from .native_cpu_rb import NativeRedBlackSOR, NativeSerialSOR
        solver = NativeRedBlackSOR if backend == 'cpu-native-rb' else NativeSerialSOR
        return solver(policy)
    if backend == 'cuda-sor':
        from .cuda_poisson import CudaRedBlackSOR
        return CudaRedBlackSOR(policy)
    if backend == 'cuda-dynamic-legacy':
        from .cuda_dynamic_legacy import CudaDynamicLegacy
        return CudaDynamicLegacy(policy)
    if backend == 'cuda-device-parallel':
        from .cuda_device_parallel import CudaDeviceParallel
        return CudaDeviceParallel(policy)
    if backend in ('cuda-global-memory', 'cuda-shared-memory'):
        from .cuda_memory_legacy import CudaGlobalMemory, CudaSharedMemory
        solver = CudaGlobalMemory if backend == 'cuda-global-memory' else CudaSharedMemory
        return solver(policy, native_boundary=native_cuda_boundary)
    raise ValueError(f'Unknown pressure solver backend {backend!r}')


@dataclass
class Stencil:
    east: np.ndarray
    west: np.ndarray
    north: np.ndarray
    south: np.ndarray
    up: np.ndarray
    down: np.ndarray
    fluid: np.ndarray
    def __post_init__(self):
        self.shape=self.east.shape
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError('Stencil needs at least two cells on every axis, including boundary cells')
        for a in [self.east,self.west,self.north,self.south,self.up,self.down]:
            if a.shape!=self.shape or not np.isfinite(a).all()or np.any(a<0):raise ValueError("Invalid Poisson coefficients")
        if self.fluid.shape!=self.shape:raise ValueError("Incompatible fluid mask")
        self.diagonal=sum([self.east,self.west,self.north,self.south,self.up,self.down])
        self.active=self.fluid.copy().astype(bool)
        self.active[[0,-1],:,:]=False;self.active[:,[0,-1],:]=False;self.active[:,:,[0,-1]]=False
        self.active&=self.diagonal>0
        self.indices=np.flatnonzero(self.active.ravel())
    def boundaries(self,p):
        p[0]=p[1]
        p[-1]=0;p[:,0]=0;p[:,-1]=0;p[:,:,0]=0;p[:,:,-1]=0
        return p
    def neighbours(self,p):
        return (self.east[1:-1,1:-1,1:-1]*p[1:-1,1:-1,2:]+
                self.west[1:-1,1:-1,1:-1]*p[1:-1,1:-1,:-2]+
                self.north[1:-1,1:-1,1:-1]*p[1:-1,2:,1:-1]+
                self.south[1:-1,1:-1,1:-1]*p[1:-1,:-2,1:-1]+
                self.up[1:-1,1:-1,1:-1]*p[2:,1:-1,1:-1]+
                self.down[1:-1,1:-1,1:-1]*p[:-2,1:-1,1:-1])
    def apply(self,p):
        q=np.zeros(self.shape,dtype=float)
        q[1:-1,1:-1,1:-1]=self.diagonal[1:-1,1:-1,1:-1]*p[1:-1,1:-1,1:-1]-self.neighbours(p)
        return q
    def residual(self,p,rhs):
        r=self.apply(p)+rhs
        return float(np.max(abs(r[self.active]),initial=0))

class RedBlackSOR:
    def __init__(self,policy: NumericPolicy|None=None,omega=1.78):
        self.policy=policy or NumericPolicy();self.omega=omega
        if not 0<omega<2:raise ValueError("SOR requires 0 < omega < 2")
    def solve(self,stencil,rhs,initial=None):
        ts=time.perf_counter();s=stencil;policy=self.policy
        rhs=np.asarray(rhs,dtype=float)
        if rhs.shape!=s.shape or not np.isfinite(rhs).all():raise ValueError("Invalid Poisson RHS")
        p=np.zeros(s.shape)if initial is None else np.array(initial,dtype=float,copy=True)
        s.boundaries(p)
        limit=policy.absolute_tolerance+policy.relative_tolerance*np.max(abs(rhs[s.active]),initial=0)
        residual=s.residual(p,rhs);change=0.;iteration=0
        parity=np.indices(tuple(n-2 for n in s.shape)).sum(axis=0)%2
        masks=[(parity==i)&s.active[1:-1,1:-1,1:-1]for i in [0,1]]
        diag=s.diagonal[1:-1,1:-1,1:-1];safe=np.where(diag>0,diag,1)
        while residual>limit and iteration<policy.max_iterations:
            change=0.
            for mask in masks:
                target=(s.neighbours(p)-rhs[1:-1,1:-1,1:-1])/safe
                inner=p[1:-1,1:-1,1:-1];update=self.omega*(target[mask]-inner[mask])
                change=max(change,float(np.max(abs(update),initial=0)));inner[mask]+=update
                s.boundaries(p)
            iteration+=1
            residual=s.residual(p,rhs)
        if not np.isfinite(p).all():raise FloatingPointError("Non-finite Poisson iterate")
        report=SolverReport(bool(residual<=limit),iteration,residual,change,time.perf_counter()-ts,"numpy.red-black-SOR")
        if not report.converged:raise RuntimeError(f"Poisson convergence failed: {report}")
        return p,report

class KrylovPoisson:
    """BiCGSTAB + Jacobi, same stencil/boundaries; no momentum model change."""
    def __init__(self,policy: NumericPolicy|None=None):self.policy=policy or NumericPolicy()
    def solve(self,stencil,rhs,initial=None):
        ts=time.perf_counter();s=stencil;idx=s.indices
        if not np.isfinite(rhs).all():raise ValueError("Non-finite Poisson RHS")
        def matvec(x):
            p=np.zeros(s.shape);p.ravel()[idx]=x;s.boundaries(p)
            return s.apply(p).ravel()[idx]
        diagonal=s.diagonal.copy();diagonal[1]-=s.down[1]
        d=diagonal.ravel()[idx]
        if np.any(d<=0):raise ValueError("Singular or disconnected pressure cells")
        op=LinearOperator((len(idx),len(idx)),matvec=matvec,dtype=float)
        preconditioner=LinearOperator(op.shape,matvec=lambda x:x/d,dtype=float)
        iterations=[0]
        def count(_):iterations[0]+=1
        b=-np.asarray(rhs).ravel()[idx]
        limit=self.policy.absolute_tolerance+self.policy.relative_tolerance*np.max(abs(b),initial=0)
        x0=None if initial is None else np.asarray(initial).ravel()[idx]
        # Enforce a common maximum-equation-residual target for both solvers.
        # L2 <= limit implies Linf <= limit; using rtol*||b||2 alone can hide
        # a large local residual on a grid with many cells.
        solution,info=bicgstab(op,b,x0=x0,M=preconditioner,rtol=0.,
                   atol=limit,maxiter=self.policy.max_iterations,callback=count)
        p=np.zeros(s.shape);p.ravel()[idx]=solution;s.boundaries(p)
        residual=s.residual(p,rhs)
        converged=bool(info==0 and residual<=limit*1.01)
        report=SolverReport(converged,iterations[0],residual,None,time.perf_counter()-ts,"scipy.BiCGSTAB-Jacobi")
        if not converged or not np.isfinite(p).all():raise RuntimeError(f"Poisson convergence failed: {report}")
        return p,report
