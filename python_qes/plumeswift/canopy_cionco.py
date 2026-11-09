"""Legacy domain-wide Cionco canopy routine from Canopy.cpp (GPL-3)."""
import numpy as np


def apply_cionco(workspace, state):
    """Apply the auxiliary closure to an already initialized canopy state."""
    arrays = workspace.arrays
    z = workspace.coordinates['z']
    ny, nx = state.top.shape
    for j, i in zip(*np.nonzero(state.top > 0.)):
        height, roughness = state.height[j, i], state.roughness[j, i]
        if height <= 0. or roughness <= 0.:
            raise ValueError('Cionco requires positive canopy height and roughness')
        layer = state.top_index[j, i]-1
        if not 0 <= layer < len(z):
            raise ValueError('Cionco canopy top index is outside the grid')
        displacement = state.column_displacement(j, i, sentinel_only=True)
        column = state.attenuation[:, j, i]
        for k in range(1, len(z)):
            if z[k] < state.base[j, i]:
                continue
            relative = z[k]-arrays['terrain'][j, i]
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                if z[k] < state.top[j, i]:
                    if column[layer] <= 0.:
                        continue
                    layer = k
                    neighbours = [value for value in column[max(0, k-1):k+2:2] if value > 0.]
                    attenuation = (column[k]+sum(neighbours))/(1+len(neighbours))
                    factor = (np.log((height-displacement)/roughness)*np.exp(attenuation*(relative/height-1))
                              /np.log(relative/roughness))
                else:
                    factor = np.log((relative-displacement)/roughness)/np.log(relative/roughness)
            if not 0. <= factor <= 1.:
                factor = 1.
            arrays['u0'][k, j, i] *= factor
            arrays['v0'][k, j, i] *= factor
            if i < nx-1 and state.top[j, i+1] == 0.:
                arrays['u0'][k, j, i+1] *= factor
            if j < ny-1 and state.top[j+1, i] == 0.:
                arrays['v0'][k, j+1, i] *= factor
