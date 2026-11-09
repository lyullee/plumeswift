"""Shared XML and transport for the original file-based instantaneous releases."""
from xml.etree import ElementTree as ET


def write_instantaneous_config(path, step, particles, source_bounds, *,
                               duration, output_period, collection_upper, boxes, vertical_bc):
    root = ET.Element('QESPlumeParameters')

    def section(parent, tag, **values):
        node = ET.SubElement(parent, tag)
        for key, value in values.items():
            ET.SubElement(node, key).text = str(value)
        return node

    section(root, 'plumeParameters', simDur=duration, timeStep=step, CourantNumber=0, invarianceTol=1e-10,
            C_0=1, updateFrequency_particleLoop=10000, updateFrequency_timeLoop=output_period)
    section(root, 'particleOutputParameters', outputFrequency=output_period, outputFields='all')
    collection = {f'boxBounds{axis}{edge}': value
                  for axis, upper in zip('XYZ', collection_upper, strict=True)
                  for edge, value in ((1, 0), (2, upper))}
    section(root, 'collectionParameters', timeAvgStart=0, timeAvgFreq=output_period,
            **collection, **{'nBoxes'+axis: count for axis, count in zip('XYZ', boxes, strict=True)})
    source = section(section(root, 'sourceParameters'), 'source')
    section(source, 'releaseType_instantaneous', numPar=particles)
    # QES 2.3.1 misorders the six output arguments in FullDomain.
    box = {edge+axis: value for axis, (lo, hi) in zip('XYZ', source_bounds, strict=True)
           for edge, value in (('min', lo), ('max', hi))}
    section(source, 'sourceGeometry_Cube', **box)
    section(root, 'boundaryConditions', xBCtype='periodic', yBCtype='periodic', zBCtype=vertical_bc,
            wallReflection='doNothing')
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8')


def run_instantaneous_plume(wind, turbulence, xml, output, *, seed, native_serial):
    from .plume_run import run_plume

    return run_plume(wind, xml, output, turbulence_path=turbulence, seed=seed, particle_csv=True,
                     native_serial=native_serial, native_arithmetic=True, native_layout=True,
                     native_timestamps=True, native_collection_format=True, native_collection_math=True,
                     output_basename=xml.stem)
