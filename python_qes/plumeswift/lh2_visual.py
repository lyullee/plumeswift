"""Shared concentration observer and offline 3D export for LH₂ examples."""

import numpy as np


def visual_dependencies():
    """Import the optional visualization stack or give the install command."""
    try:
        import plotly.graph_objects as go
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise ValueError('Install the visualization extra with: python -m pip install "plumeswift[visual]"') from error
    return go, marching_cubes


class GridAverage:
    """Time-average hydrogen volume fraction in cells after each particle step."""

    def __init__(self, sampler, shape, thermo, volume, start, end):
        self.sampler = sampler
        self.integral = np.zeros(shape)
        self.thermo = thermo
        self.volume = volume
        self.start = start
        self.end = end
        self.sampled_s = 0.

    def observe(self, time, dt, positions, masses):
        weight = max(0., min(time + dt, self.end) - max(time, self.start))
        if not weight:
            return
        ids = np.ravel_multi_index(self.sampler.cell_indices(positions), self.integral.shape)
        mass = np.bincount(ids, weights=masses)
        occupied = np.flatnonzero(mass)
        fraction = self.thermo.evaluate(mass[occupied] / self.volume).volume_fraction
        self.integral.ravel()[occupied] += weight * fraction
        self.sampled_s += weight


def write_figure(path, concentration, coordinates, levels, position, title, *, building=None, sensors=None):
    go, marching_cubes = visual_dependencies()
    axes = [coordinates[key] for key in 'zyx']
    spacing = tuple(axis[1] - axis[0] for axis in axes)
    origin = [axis[0] for axis in axes]
    fluid = np.isfinite(concentration)
    values = np.where(fluid, concentration, 0.)
    peak = float(np.nanmax(concentration))
    traces = []
    shown = []
    for level, color, opacity in zip(levels, ('#169ebb', '#5db878', '#f08a37'),
                                     (.26, .40, .68), strict=True):
        if not 0 < level < peak:
            continue
        try:
            vertices, faces, _, _ = marching_cubes(values, level, spacing=spacing, mask=fluid)
        except RuntimeError as error:
            if 'No surface found' not in str(error):
                raise
            continue
        z, y, x = (vertices + origin).T
        i, j, k = faces.T
        traces.append(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity,
            name=f'{level:g} vol%', showlegend=True,
            hovertemplate=f'{level:g} vol% H₂<extra></extra>',
        ))
        shown.append(float(level))
    if building is not None:
        bx = [p[0] for p in building.vertices[:-1]] * 2
        by = [p[1] for p in building.vertices[:-1]] * 2
        traces.append(go.Mesh3d(
            x=bx, y=by, z=[0.] * 4 + [building.height] * 4,
            i=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            j=[1, 5, 2, 6, 3, 7, 0, 4, 5, 6],
            k=[5, 4, 6, 5, 7, 6, 4, 7, 6, 7],
            color='#8d969e', name='Obstacle', showlegend=True, hoverinfo='name',
        ))
    traces.append(go.Scatter3d(
        x=[position[0]], y=[position[1]], z=[position[2]], mode='markers',
        marker=dict(size=5, color='#a61f37', symbol='diamond'), name='Release',
    ))
    if sensors:
        traces.append(go.Scatter3d(
            x=[r['x_m'] for r in sensors], y=[r['y_m'] for r in sensors],
            z=[r['z_m'] for r in sensors], mode='markers', name='FFI sensors',
            marker=dict(size=4, color='#263449', symbol='circle-open'),
            customdata=[[r['sensor'], r['observed_volpct'], r['predicted_volpct']] for r in sensors],
            hovertemplate='%{customdata[0]}<br>Measured: %{customdata[1]:.3f} vol%'
                          '<br>PlumeSwift sensor box: %{customdata[2]:.3f} vol%<extra></extra>',
        ))
    if not shown:
        title += ' · No requested isosurface intersects the grid'
    figure = go.Figure(traces)
    figure.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(xaxis_title='East (m)', yaxis_title='North (m)', zaxis_title='Height (m)',
                   aspectmode='data', camera=dict(eye=dict(x=-.9, y=-2.12, z=.85))),
    )
    figure.write_html(path, include_plotlyjs=True, config={'responsive': True})
    return shown
