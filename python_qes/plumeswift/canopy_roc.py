"""Row-organized canopy equations from QES CanopyROC.cpp (GPL-3)."""
from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from .geometry import PolygonBuilding
from .upwind import _round
from .vegetation import CanopyPrism


def stability(height, reciprocal):
    if reciprocal >= 0:
        return -5*height*reciprocal, -5*reciprocal
    argument = 1-15*height*reciprocal
    x = argument**.25
    derivative = -3.75*reciprocal*argument**-.75
    psi = np.log(.5*(1+x*x)*(.5*(1+x))**2)-2*np.arctan(x)+.5*np.pi
    return psi, derivative*(2*x/(1+x*x)+2/(1+x)-2/(1+x*x))


@dataclass(frozen=True)
class RowCanopy(CanopyPrism):
    vertices: tuple
    height: float
    understory: float
    row_spacing: float
    row_width: float
    row_angle: float
    lad_effective: float
    lad_average: float
    porosity: float
    tke_max: float
    sigma_w: float
    upstream_ustar: float
    displacement: float
    reciprocal: float
    roughness: float
    thin_fence: bool = False
    base_height: float = 0.

    def __post_init__(self):
        PolygonBuilding(self.vertices, self.height, self.base_height)
        if len(self.vertices) != 5:
            raise ValueError('ROC requires four footprint vertices')
        if not np.isfinite((self.understory, self.row_spacing, self.row_width, self.row_angle,
                            self.lad_effective, self.lad_average, self.porosity, self.tke_max,
                            self.sigma_w, self.upstream_ustar, self.displacement, self.reciprocal,
                            self.roughness)).all():
            raise ValueError('ROC parameters must be finite')
        if not (0 <= self.understory < self.height and 0 < self.row_width < self.row_spacing
                and self.lad_effective > 0 and self.lad_average >= 0 and 0 < self.porosity < 1
                and self.tke_max >= 0 and self.sigma_w >= 0 and self.upstream_ustar > 0
                and 0 <= self.displacement < min(self.height, 10.) and self.roughness > 0):
            raise ValueError('Invalid ROC row geometry, drag, or upstream profile parameters')

    def flag(self, workspace, state, number):
        columns = super().flag(workspace, state, number, cell_flag=None, base_face_offset=-1)
        top = max((state.top_index[j, i] for j, i in columns), default=0)
        if ceil(1.5*top) > len(workspace.coordinates['z']):
            raise ValueError('Domain is too short for ROC canopy')
        return columns

    def apply_rows(self, workspace, state):
        a, c = workspace.arrays, workspace.coordinates
        dz_faces, z = c['z_face'], c['z']
        dx = c['x'][1]-c['x'][0]
        polygon = np.asarray(self.vertices[:-1], dtype=np.float32)
        centroid = polygon.mean(axis=0)
        ic, jc = (_round(p/dx)-1 for p in centroid)
        start = np.trunc(polygon.min(axis=0)/dx).astype(int)
        end = np.trunc(polygon.max(axis=0)/dx+1).astype(int)
        if min(start) < 1:
            raise ValueError('ROC requires one cell upwind of the footprint index bounds')
        u, v = a['u0'], a['v0']
        nz = u.shape[0]
        far_wind = np.array((u[nz-5, jc, ic], v[nz-5, jc, ic]))
        speed = np.linalg.norm(far_wind)
        flow = RowFlow(self, a['icellflag'].shape)
        if speed == 0 or not np.any(state.top_index[start[1]:end[1], start[0]:end[0]]):
            return flow
        wind = far_wind/speed
        normal = flow.normal
        signed = polygon@normal
        order = signed*np.dot(wind, normal)
        flow.upwind, downwind = signed[np.argmin(order)], signed[np.argmax(order)]
        angle = np.arccos(np.clip(abs(np.dot(wind, flow.parallel)), 0., 1.))
        flow.sine = np.sin(max(np.deg2rad(.5), angle))
        flow.stream_spacing = (self.row_spacing-self.row_width)/flow.sine
        terrain = a['terrain'][jc, ic]
        k_top, k_bot = np.searchsorted(dz_faces, (terrain+self.height, terrain+self.understory))
        cross_top = abs(flow.cross(u[k_top, jc, ic], v[k_top, jc, ic]))
        cross_bot = abs(flow.cross(u[k_bot, jc, ic], v[k_bot, jc, ic]))
        if cross_top == 0:
            raise ValueError('ROC entrance length is undefined at zero cross-row reference wind')
        drag_length = 2/flow.lad
        entrance_length = drag_length*np.log(cross_top*self.height/(self.upstream_ustar*drag_length))
        # The native vector has N_e + 1 entries. Bound the degenerate
        # zero-row entrance case so the zeroth (unit) term remains defined.
        flow.entrance = max(0, ceil(entrance_length/self.row_spacing)+2)
        entry_porosity = flow.bleed**(flow.entrance+1)
        spread = .28*(1-entry_porosity)/(1+entry_porosity)
        flow.spread_top = np.hypot(spread, 2*self.sigma_w/cross_top)
        flow.spread_bot = np.hypot(spread, 2*self.sigma_w/(cross_top if cross_bot < 1e-6 else cross_bot))
        modified = []
        cross_tke = parallel_ratio = 0.
        for j in range(start[1], end[1]+1):
            for i in range(start[0], end[0]+1):
                bottom, top = state.bot_index[j, i], state.top_index[j, i]
                if bottom >= top:
                    continue
                iy, ix = j-1, i-1
                base = state.base[j, i]
                k = np.arange(bottom, len(z))
                relative = dz_faces[k-1]-base
                distance = abs(flow.signed(ix*dx, iy*dx)-flow.upwind)
                distance_end = abs(flow.signed(ix*dx, iy*dx)-downwind)
                rows, local = flow.row_offset(distance)
                mid = min(top-1, np.searchsorted(z, base+.5*(self.height+self.understory), side='right')-1)
                stream = (local-self.row_width)/flow.sine
                attenuation = flow.shelter(k <= mid, relative, rows, stream, local-self.row_width)
                cross, parallel = flow.rotate(u[k, iy, ix], v[k, iy, ix])
                ref = np.searchsorted(z, 10.+base, side='right')-1
                if ref < bottom:
                    ref = 0
                flow.set_parallel_reference(flow.along(u[ref, iy, ix], v[ref, iy, ix]))
                if distance > self.row_spacing and local < self.row_width:
                    inside = (relative >= self.understory) & (relative <= self.height)
                    cross_factor = np.where(inside, flow.bleed*attenuation[1], 1.)
                    actual_cross, actual_parallel = cross*cross_factor, parallel
                    a['icellflag'][k[inside], iy, ix] = 28
                    cross_tke = cross_factor[-1]
                    turbulence_parallel = parallel_ratio*parallel
                else:
                    if distance > self.row_spacing:
                        cross_factor = attenuation[0]*attenuation[1]
                        actual_cross = cross*cross_factor
                        cross_tke = cross_factor[-1]
                    else:
                        # Native retains the prior cross-row TKE multiplier
                        # in the first aisle, where cross wind is unaltered.
                        cross_factor, actual_cross = cross_tke, cross.copy()
                    flags = a['icellflag'][k, iy, ix]
                    flags.fill(30)
                    gap = self.row_spacing-local
                    if distance_end > self.row_spacing and gap < .5*self.height:
                        flow.displacement_zone(relative, gap, actual_cross, flags)
                    a['icellflag'][k, iy, ix] = flags
                    actual_parallel = flow.parallel_profile(relative)
                    turbulence_parallel = actual_parallel
                    parallel_ratio = abs(actual_parallel[-1]/parallel[-1]) if parallel[-1] != 0 else 0.
                flow.set_tke((k, iy, ix), cross_factor*cross, turbulence_parallel, cross, parallel)
                modified.append((k, iy, ix, *flow.unrotate(actual_cross, actual_parallel)))
        for k, j, i, u_new, v_new in modified:
            u[k, j, i], v[k, j, i] = u_new, v_new
        return flow

    def apply_wake(self, workspace, state, number, _defects):
        flow = state.row_flows[number]
        if flow.entrance is None:
            return
        a, c = workspace.arrays, workspace.coordinates
        polygon = np.asarray(self.vertices, dtype=np.float32)
        dx, dy = c['x'][1]-c['x'][0], c['y'][1]-c['y'][0]
        step = .5*min(dx, dy)
        ic, jc = (_round(p/dx)-1 for p in polygon[:-1].mean(axis=0))
        u, v = a['u0'], a['v0']
        wind = np.array((u[-5, jc, ic], v[-5, jc, ic]))
        speed = np.linalg.norm(wind)
        if speed == 0:
            return
        wind /= speed
        ny, nx = a['terrain'].shape
        terrain_id = a.get('terrain_id', np.ones((ny, nx), dtype=int))
        paths = {}
        for first, second in zip(polygon[:-1], polygon[1:]):
            if _inside_polygon(.5*(first+second)+wind, polygon):
                continue
            edge_length = np.linalg.norm(second.astype(float)-first)
            steps = floor(edge_length/step)
            edge_step = (second.astype(float)-first)/max(1, steps)
            for edge_index in range(steps+1):
                origin = first+edge_index*edge_step
                distance = abs(flow.signed(*origin)-flow.upwind)
                rows, offset = flow.row_offset(distance)
                for wake_index in range(floor(7.5*self.height/step)+1):
                    position = origin+wake_index*step*wind
                    i, j = np.ceil(position/dx).astype(int)-1
                    if not (0 <= i < nx and 0 <= j < ny):
                        break
                    row_distance = abs(flow.signed(*position)-flow.upwind)
                    paths[j, i] = rows, offset+row_distance-distance-self.row_width
        # QES commits all modified face values after traversal; the last
        # ray to visit each column supplies its result.
        for (j, i), (rows, local) in paths.items():
            k = np.arange(terrain_id[j, i], len(c['z']))
            height = a['terrain'][j, i]
            relative = c['z_face'][k-1]-height
            below = c['z'][k] <= height+.5*(self.height+self.understory)
            local_factor, upstream = flow.shelter(below, relative, rows, local/flow.sine, local+self.row_width,
                                                  all_entry_rows=True)
            cross, parallel = flow.rotate(u[k, j, i], v[k, j, i])
            actual_cross = cross*local_factor*upstream
            actual_parallel = flow.parallel_profile(relative)
            flow.set_tke((k, j, i), actual_cross, actual_parallel, cross, parallel)
            if state.native_geometry:
                # The native wake multiplies |v_new|/|v_old| by v_old.
                # At zero parallel wind this is NaN and the ROC closures skip it.
                flow.tke_factor[k[parallel == 0], j, i] = 0.
            u[k, j, i], v[k, j, i] = flow.unrotate(actual_cross, actual_parallel)
            a['icellflag'][k, j, i] = 33


def _inside_polygon(point, vertices):
    x, y = point
    crossings = 0
    for (xa, ya), (xb, yb) in zip(vertices[:-1], vertices[1:]):
        if (ya <= y < yb) or (yb <= y < ya):
            crossings += x < xa+(y-ya)*(xb-xa)/(yb-ya)
    return crossings % 2 != 0


class RowFlow:
    """Transient ROC wind references shared by row and wake closures."""
    def __init__(self, canopy, shape):
        self.canopy = canopy
        angle = np.deg2rad(canopy.row_angle)
        self.cs, self.sn = np.cos(angle), np.sin(angle)
        self.normal = np.array((self.cs, -self.sn))
        self.parallel = np.array((self.sn, self.cs))
        self.bleed = canopy.porosity if canopy.thin_fence else canopy.porosity**.4
        self.lad = (1-self.bleed)/canopy.row_spacing if canopy.thin_fence else canopy.lad_effective
        ratio = canopy.row_spacing/canopy.height
        self.shear_slope = 1.3945*ratio/(ratio+.4209)-1.4524
        self.tke_factor = np.zeros(shape)
        self.ustar, self.v_top, self.parallel_attenuation = 0., 0., 0.
        self.entrance = None

    def signed(self, x, y):
        return x*self.cs-y*self.sn

    def cross(self, u, v):
        return self.cs*u-self.sn*v

    def along(self, u, v):
        return self.sn*u+self.cs*v

    def rotate(self, u, v):
        return self.cross(u, v), self.along(u, v)

    def unrotate(self, u, v):
        return self.cs*u+self.sn*v, -self.sn*u+self.cs*v

    def row_offset(self, distance):
        spacing = self.canopy.row_spacing
        rows = floor(distance/spacing)
        return rows, distance-rows*spacing

    def shelter(self, below_mid, relative, rows, stream, local, *, all_entry_rows=False):
        model = self.canopy
        lower = below_mid & (model.understory > 0)
        spread = np.where(lower, self.spread_bot, self.spread_top)
        origin = np.where(lower, model.understory, max(0., self.shear_slope*local+model.height))
        upstream_origin = np.where(lower, model.understory, max(0., self.shear_slope*(model.row_spacing-model.row_width)+model.height))
        difference = np.where(lower, -1., 1.)
        blend = np.exp(np.log(.01)/(7.5*model.height)*stream)

        def factor(signed_height, thickness):
            return 1+blend*.5*(1-self.bleed)*(np.tanh(1.5*signed_height/thickness)-1)

        count = min(rows, self.entrance)
        local_factor = factor(difference*(relative-origin), spread*(self.stream_spacing*count+stream)+1e-5)
        upstream = np.ones(relative.shape)
        for n in range(1, (self.entrance if all_entry_rows else count)+1):
            upstream *= factor(difference*(relative-upstream_origin), spread*self.stream_spacing*n+1e-5)
        return local_factor, upstream

    def set_parallel_reference(self, reference):
        model = self.canopy
        height = model.height-model.displacement
        ref = 10.-model.displacement
        psi_h, gradient = stability(height, model.reciprocal)
        psi_ref, _ = stability(ref, model.reciprocal)
        self.ustar = .4*reference/(np.log(ref/model.roughness)-psi_ref)
        self.v_top = self.ustar/.4*(np.log(height/model.roughness)-psi_h)
        self.parallel_attenuation = (model.height/max(abs(self.v_top+1e-5), np.finfo(float).tiny)
                                     *abs(self.ustar)/.4*(1/height-gradient))

    def parallel_profile(self, relative):
        model = self.canopy
        result = self.v_top*np.exp(self.parallel_attenuation*(np.minimum(relative, model.height)/model.height-1))
        above = relative > model.height
        displaced = relative[above]-model.displacement
        psi, _ = stability(displaced, model.reciprocal)
        result[above] = self.ustar/.4*(np.log(displaced/model.roughness)-psi)
        return result

    def displacement_zone(self, z, gap, cross, flags):
        model = self.canopy
        distance = .5*model.height
        drag = (1.98*(1-model.porosity)/model.row_width if model.thin_fence else .5*model.lad_average)
        deficit = -drag*self.bleed*distance
        log_cutoff = np.log(.01)
        if model.understory == 0:
            top = .6*model.height*np.sqrt(1-gap/distance)
            active = z <= top
            blend = -np.expm1(log_cutoff*(top-z[active])/top)
        else:
            mid = .5*(model.height+model.understory)
            thickness = .5*(model.height-model.understory)*np.sqrt(1-gap/distance)
            delta = abs(z-mid)
            active = delta <= thickness
            blend = -np.expm1(log_cutoff*(1-delta[active]/thickness))
        flags[active] = 31
        cross[active] *= 1+deficit*blend

    def set_tke(self, index, model_cross, model_parallel, cross, parallel):
        speed = np.hypot(cross, parallel)
        self.tke_factor[index] = np.divide(np.hypot(model_cross, model_parallel), speed,
                                           out=np.zeros_like(speed), where=speed != 0)
