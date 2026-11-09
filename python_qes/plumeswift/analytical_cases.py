"""Prescribed continuous-release test cases from QES testCases (GPL-3.0)."""
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .io import _staged_netcdf_path


@dataclass(frozen=True)
class Case:
    stem: str
    lengths: tuple[int, int, int]
    dz: float
    source: tuple[int, int, int]
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    boxes: tuple[int, int, int]
    duration: int
    average_start: int
    courant: float
    wall: str


CASES = {
    'uniform-x': Case('UniformFlow_xDir', (100, 100, 140), 1., (20, 50, 70),
                      (0, 1, 1), (100, 99, 139), (20, 49, 69), 2100, 300, 0., 'doNothing'),
    'uniform-y': Case('UniformFlow_yDir', (100, 100, 140), 1., (50, 20, 70),
                      (1, 0, 1), (99, 100, 139), (49, 20, 69), 1400, 200, 0., 'stairstepReflection'),
    'powerlaw-x': Case('PowerLawBLFlow_xDir', (200, 100, 20), .5, (20, 50, 4),
                       (0, 0, 0), (200, 100, 20), (36, 51, 20), 2100, 300, .4, 'stairstepReflection'),
    'powerlaw-y': Case('PowerLawBLFlow_yDir', (100, 100, 20), .5, (50, 20, 4),
                       (0, 0, 0), (100, 100, 20), (51, 18, 20), 2100, 300, .3, 'stairstepReflection'),
}


def _profiles(z, uniform, x_direction):
    zero = np.zeros_like(z)
    if uniform:
        ustar = .174
        speed = np.full_like(z, 2.)
        tke = np.full_like(z, (ustar/.55)**2)
        eps = 5.7*ustar**3/(.4*z)
        if x_direction:
            eps *= (1-.85*z/1000)**1.5
            stress_scale = (1-z/1000)**1.5
            txx, tyy, tzz = ((factor*ustar)**2*stress_scale for factor in (2.5, 1.78, 1.27))
            txz, tyz = -ustar**2*stress_scale, zero
        else:
            tke[0] = -tke[1]
            txx, tyy, tzz = (np.full_like(z, (factor*ustar)**2) for factor in (1.78, 2.5, 1.27))
            txz, tyz = zero, np.full_like(z, -ustar**2)
    else:
        speed = np.empty_like(z)
        speed[1:] = 4.8*z[1:]**.15
        speed[0] = -speed[1]
        speed[-1] = speed[-2]
        ustar = np.empty_like(z)
        ustar[1:] = .4*z[1:]*(.15*4.8*z[1:]**(.15-1))
        ustar[0] = ustar[1]
        ustar[-1] = ustar[-2]
        eps = 5.7*ustar**3/(.4*z)
        eps[-1] = eps[-2]
        tke = (ustar/.55)**2
        shear = -ustar**2
        if x_direction:
            txx, tyy, tzz = ((factor*ustar)**2 for factor in (2.5, 2.3, 1.2))
            txz, tyz = shear, zero
        else:
            txx, tyy, tzz = ((factor*ustar)**2 for factor in (2.3, 2.5, 1.3))
            for field in (txx, tyy, tzz, tke, shear):
                field[0] = -field[1]
            txz, tyz = zero, shear
    return speed, dict(txx=txx, txy=zero, txz=txz, tyy=tyy, tyz=tyz, tzz=tzz, CoEps=eps, tke=tke)


def _write_fields(path, coords, fields, *, faces=(), legacy_vertical_faces=False):
    from netCDF4 import Dataset

    cells = ('z', 'y', 'x')
    with _staged_netcdf_path(path) as staged, Dataset(str(staged), 'w') as nc:
        nc.createDimension('t', None)
        for axis in faces:
            nc.createDimension(axis, len(coords[axis[0]])+1)
        for axis in 'xyz':
            nc.createDimension(axis, len(coords[axis]))
        for axis, values in {'t': 0., **coords}.items():
            time = axis == 't'
            dimensions = () if time else ('z',) if axis == 'z_face' and legacy_vertical_faces else (axis,)
            var = nc.createVariable(axis, 'f4', dimensions)
            var.long_name = 'time' if time else f'{axis[0]}-distance'
            var.units = 's' if time else 'm'
            if len(axis) == 1:
                var.cartesian_axis = axis.upper()
            var[...] = values
        for key, values in fields.items():
            if key in ('u', 'v', 'w'):
                axes, name, units = faces, f'{"xyz"["uvw".index(key)]}-component velocity', 'm s-1'
            elif key == 'icellflag':
                axes, name, units = cells, 'icell flag value', '--'
            elif key == 'CoEps':
                axes, name, units = cells, 'dissipation rate of tke', 'm2 s-3'
            elif key == 'tke':
                axes, name, units = cells, 'tke = turbulence kinetic energy', 'm2 s-2'
            else:
                pair = key[1:].translate(str.maketrans('xyz', 'uvw'))
                axes, name, units = cells, f'{key} = {pair} stress', 'm2 s-2'
            var = nc.createVariable(key, 'i4' if key == 'icellflag' else 'f4', ('t', *axes))
            var.long_name = name
            var.units = units
            var[0] = values


def _write_config(path, case, uniform, *, steps, particles_per_step):
    root = ET.Element('QESPlumeParameters')

    def section(parent, tag, values):
        node = ET.SubElement(parent, tag)
        for key, value in values.items():
            ET.SubElement(node, key).text = str(value)
        return node

    duration = case.duration if steps is None else steps
    start = case.average_start if steps is None else 0
    frequency = 100 if case.stem == 'UniformFlow_yDir' else 60
    section(root, 'plumeParameters', dict(simDur=duration, timeStep=1, CourantNumber=case.courant,
                                          invarianceTol=1e-10, C_0=1, updateFrequency_particleLoop=10000,
                                          updateFrequency_timeLoop=min(frequency, duration)))
    section(root, 'particleOutputParameters', dict(outputFrequency=100, outputFields='all'))
    bounds = {f'boxBounds{axis}{edge}': value for edge, coords in enumerate((case.lower, case.upper), 1)
              for axis, value in zip('XYZ', coords, strict=True)}
    section(root, 'collectionParameters', dict(timeAvgStart=start, timeAvgFreq=duration-start,
                                               **bounds, **dict(zip(('nBoxesX', 'nBoxesY', 'nBoxesZ'), case.boxes))))
    source = ET.SubElement(ET.SubElement(root, 'sourceParameters'), 'source')
    section(source, 'releaseType_continuous', dict(particlePerTimestep=particles_per_step))
    section(source, 'sourceGeometry_Point', dict(zip(('posX', 'posY', 'posZ'), case.source)))
    section(root, 'boundaryConditions', dict(xBCtype='exiting', yBCtype='exiting',
                                             zBCtype='reflection' if uniform else 'exiting',
                                             wallReflection=case.wall))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8')


def prepare_analytical_case(name, output, *, steps=None, particles_per_step=200):
    """Generate the historical NetCDF schema and matching plume XML."""
    case = CASES[name]
    if (steps is not None and steps < 1) or particles_per_step < 1:
        raise ValueError('Steps and particles per step must be positive')
    uniform = name.startswith('uniform')
    x_direction = name.endswith('-x')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coords = {axis: (np.arange(int(length/spacing)+2)-.5)*spacing
              for axis, length, spacing in zip('xyz', case.lengths, (1., 1., case.dz), strict=True)}
    speed, profiles = _profiles(coords['z'], uniform, x_direction)
    cell_shape = tuple(len(coords[axis]) for axis in 'zyx')
    face_shape = tuple(size+1 for size in cell_shape)
    velocity = np.zeros(face_shape, dtype=np.float32)
    # The uniform Y script writes on the same slab as the uniform X script.
    if uniform or x_direction:
        velocity[:-1, :-1, :] = speed[:, None, None]
    else:
        velocity[:-1, :, :-1] = speed[:, None, None]
    zero = np.broadcast_to(np.float32(0), face_shape)
    wind = dict(u=velocity if x_direction else zero, v=zero if x_direction else velocity, w=zero)
    flags = np.ones(cell_shape, dtype=np.int32)
    if name != 'uniform-y':
        flags[0] = 2
    wind['icellflag'] = flags
    wind_path = output/f'{case.stem}_windsWk.nc'
    turb_path = output/f'{case.stem}_turbOut.nc'
    xml_path = output/f'{case.stem}_ContRelease.xml'
    z_faces = np.arange(face_shape[0])-1 if uniform else np.arange(cell_shape[0])
    _write_fields(wind_path, dict(coords, z_face=z_faces*case.dz), wind,
                  faces=('z_face', 'y_face', 'x_face'), legacy_vertical_faces=not uniform)
    _write_fields(turb_path, coords, {key: np.broadcast_to(profile[:, None, None], cell_shape)
                                    for key, profile in profiles.items()})
    _write_config(xml_path, case, uniform, steps=steps, particles_per_step=particles_per_step)
    return wind_path, turb_path, xml_path


def run_analytical_case(name, output, *, prepare_only=False, steps=None, particles_per_step=200, seed=0):
    wind, turbulence, xml = prepare_analytical_case(name, output, steps=steps,
                                                    particles_per_step=particles_per_step)
    report = dict(wind=str(wind), turbulence=str(turbulence), config=str(xml))
    if not prepare_only:
        from .plume_run import run_plume

        report['plume'] = run_plume(wind, xml, output, turbulence_path=turbulence, seed=seed,
                                    native_serial=True, native_layout=True, native_timestamps=True,
                                    native_collection_format=True, native_collection_math=True,
                                    output_basename=xml.stem)
    return report
