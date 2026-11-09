"""Row canopy turbulent viscosity and canopy-top stress."""
import numpy as np


def apply_viscosity(workspace, flow, mixing, energy, viscosity):
    model = flow.canopy
    a, c = workspace.arrays, workspace.coordinates
    k, j, i = np.nonzero(flow.tke_factor[:-1])
    height = a['terrain'][j, i]
    faces = c['z_face']
    canopy_length = 2*.4**3/(.5*flow.lad)
    low = np.searchsorted(faces, canopy_length/.4+height)
    high = np.searchsorted(faces, canopy_length/.4+height+model.displacement)
    relative = faces[k+1]-height
    index = k, j, i
    mixing[index] = np.where(k <= low, .4*relative,
                             np.where(k <= high, canopy_length, .4*(relative-model.displacement)))
    energy[index] = flow.tke_factor[index]**2*model.tke_max
    viscosity[index] = .55*np.sqrt(energy[index])*mixing[index]


def apply_stress(workspace, flow, velocity, energy, viscosity, stress):
    a, c = workspace.arrays, workspace.coordinates
    u, v, w = (velocity[component] for component in 'uvw')
    k, j, i = np.nonzero(flow.tke_factor)
    top = np.searchsorted(c['z_face'], a['terrain'][j, i]+flow.canopy.height)
    inside = (k < top) & (top > 0) & (top < len(c['z_face'])-2)
    k, j, i, h = k[inside], j[inside], i[inside], top[inside]
    dx = c['x'][1]-c['x'][0]
    dz = c['z_face'][h+1]-c['z_face'][h-1]

    def samples(field, offsets):
        # QES offsets are applied to linear staggered addresses, including
        # at domain edges. Clamp only outside the allocated volume.
        offsets = np.asarray(offsets)
        nz, ny, nx = field.shape
        center = (h*ny+j)*nx+i
        delta = (offsets[:, 0]*ny+offsets[:, 1])*nx+offsets[:, 2]
        return field.ravel()[np.clip(center[:, None]+delta, 0, nz*ny*nx-1)]

    def derivative(field, spacing, offsets):
        p, q, r, s = samples(field, offsets).T
        return (p+q-r-s)/(4*spacing)

    du_dx = derivative(u, dx, ((0, 0, 1), (0, 0, 2), (0, 0, 0), (0, 0, -1)))
    dv_dy = derivative(v, dx, ((0, 1, 0), (0, 2, 0), (0, 0, 0), (0, -1, 0)))
    dw_dz = derivative(w, dz, ((1, 0, 0), (2, 0, 0), (0, 0, 0), (-1, 0, 0)))
    du_dy = derivative(u, dx, ((0, 1, 0), (0, 1, 1), (0, -1, 0), (0, -1, 1)))
    dv_dx = derivative(v, dx, ((0, 0, 1), (0, 1, 1), (0, 0, -1), (0, 1, -1)))
    du_dz = derivative(u, dz, ((1, 0, 0), (1, 0, 1), (-1, 0, 0), (-1, 0, 1)))
    dw_dx = derivative(w, dx, ((0, 0, 1), (1, 0, 1), (0, 0, -1), (1, 0, -1)))
    dv_dz = derivative(v, dz, ((1, 0, 0), (1, 1, 0), (-1, 0, 0), (-1, 1, 0)))
    dw_dy = derivative(w, dx, ((0, 1, 0), (1, 1, 0), (0, -1, 0), (1, -1, 0)))
    index = k, j, i
    nu = viscosity[index]
    for component, gradient in enumerate((du_dx, dv_dy, dw_dz)):
        stress[index+(component, component)] = (2/3)*energy[index]-2*nu*gradient
    for first, second, gradient in ((0, 1, du_dy+dv_dx), (0, 2, du_dz+dw_dx), (1, 2, dv_dz+dw_dy)):
        stress[index+(first, second)] = stress[index+(second, first)] = -nu*gradient
