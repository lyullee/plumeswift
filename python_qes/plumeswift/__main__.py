"""Small inspection/property CLI, not a fake replacement for all native flags."""
import argparse,json
import sys
from pathlib import Path
from dataclasses import asdict, replace
from . import __version__
from .contracts import PhysicalCoefficients


def _cpu_seed_argument(value):
    return None if value == 'clock' else int(value)


def _add_legacy_particle_output(parser, long_option='--particleOutput'):
    parser.add_argument('-l', long_option, action='store_true', dest='particle_csv',
                        help='Write particle snapshots in the original CSV format')
    csv_format = parser.add_mutually_exclusive_group()
    csv_format.add_argument('--native-particle-csv', action='store_true',
                            help='Write particle snapshots in the original CSV format, like -l')
    csv_format.add_argument('--precise-particle-csv', action='store_true',
                            help='Write particle snapshots with full numerical precision')


def main(argv=None, *, launcher=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ['argparser']:
        from .argument_demo import argument_demo

        try:
            print(argument_demo(argv[1:]), end='')
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from None
        return
    parser=argparse.ArgumentParser(description='Modular Python QES port (full-feature migration incomplete)')
    command=parser.add_subparsers(dest='command',required=True)
    command.add_parser('argparser', add_help=False, help='Run the original scratch option parser example')
    xml_example = command.add_parser('xml-example', help='Run an original scratch XML parser example')
    xml_example.add_argument('xml', type=Path)
    xml_example.add_argument('--simulation', action='store_true', help='Use the historical simulation parameters example')
    managed = command.add_parser('plume-managed-container', help='Run the original particle slot reuse example')
    managed.add_argument('--steps', type=int, default=10_000)
    managed.add_argument('--release-per-step', type=int, default=1000)
    managed.add_argument('--seed', type=int, default=0)
    command.add_parser('info')
    lh2_demo = command.add_parser('lh2-demo', help='Run the bundled LH₂ example and export an interactive 3D plume')
    lh2_input = lh2_demo.add_mutually_exclusive_group()
    lh2_input.add_argument('--input', type=Path, help='Read a modified case JSON')
    lh2_input.add_argument('--write-input', type=Path, help='Copy the bundled case JSON without running')
    lh2_demo.add_argument('--output', type=Path, default=Path('lh2-demo'), help='Result directory (default: lh2-demo)')
    lh2_demo.add_argument('--open', action='store_true', help='Open the completed 3D result in your browser')
    ffi_lh2 = command.add_parser('lh2-ffi', help='Compare FFI LH₂ experiments from the official report')
    ffi_lh2.add_argument('--report', type=Path, required=True, help='Local copy of the official FFI 20/03101 PDF')
    ffi_lh2.add_argument('--tests', type=int, choices=range(1, 8), nargs='+', default=list(range(1, 8)),
                         help='Select test numbers (default: all seven)')
    ffi_lh2.add_argument('--workers', type=int, default=1, help='Number of independent case processes')
    ffi_lh2.add_argument('--output', type=Path, default=Path('ffi-results'), help='Result directory')
    ffi_lh2.add_argument('--no-3d', action='store_true', help='Write sensor comparisons without 3D fields')
    ffi_lh2.add_argument('--open', action='store_true', help='Open the completed report in your browser')
    runtime_info = command.add_parser('runtime-info', help='Show the original QES startup banner and runtime diagnostics')
    runtime_info.add_argument('--cuda', action='store_true', help='Query CUDA devices using the optional CuPy backend')
    runtime_info.add_argument('--optix', action='store_true', help='Report the installed NVIDIA PyOptiX binding version')
    sinusoidal = command.add_parser('plume-sinusoidal', help='Run the original sinusoidal-stress plume regression')
    sinusoidal.add_argument('--particles', type=int, default=200_000)
    sinusoidal.add_argument('--duration', type=float, default=12.)
    sinusoidal.add_argument('--seed', type=int, default=0)
    sinusoidal.add_argument('--native-rng', action='store_true', help='Use the Linux QES transport and geometry streams')
    uniform = command.add_parser('plume-uniform', help='Run the original uniform-flow Gaussian plume regression')
    uniform.add_argument('--particles-per-step', type=int, default=400)
    uniform.add_argument('--duration', type=float, default=2100.)
    uniform.add_argument('--averaging-start', type=float, default=300.)
    uniform.add_argument('--seed', type=int, default=0)
    from .analytical_cases import CASES

    analytical = command.add_parser('plume-analytical-case', help='Run a prescribed QES continuous-release test case')
    analytical.add_argument('case', choices=CASES)
    analytical.add_argument('--output', type=Path, required=True)
    analytical.add_argument('--prepare-only', action='store_true')
    analytical.add_argument('--steps', type=int, help='Override the run length and average over the entire run')
    analytical.add_argument('--particles-per-step', type=int, default=200)
    analytical.add_argument('--seed', type=int, default=0)
    diagnostics = command.add_parser('plume-analytical-results', help='Compare a continuous-release output with its analytical solution')
    diagnostics.add_argument('case', choices=CASES)
    diagnostics.add_argument('collection', type=Path)
    diagnostics.add_argument('--output', type=Path, required=True)
    diagnostics.add_argument('--xml', type=Path, help='Use the run configuration and recorded averaging time')
    diagnostics.add_argument('--record', type=int, default=-1)
    diagnostics.add_argument('--plot', action='store_true', help='Write profile and one-to-one PDF figures')
    from .sinusoidal_case import TIME_STEPS

    sinusoidal_case = command.add_parser('plume-sinusoidal-case', help='Run the file-based Sinusoidal3D timestep series')
    sinusoidal_case.add_argument('--output', type=Path, required=True)
    sinusoidal_case.add_argument('--step', type=float, choices=TIME_STEPS, action='append')
    sinusoidal_case.add_argument('--particles', type=int, default=100_000)
    sinusoidal_case.add_argument('--seed', type=int, default=0)
    sinusoidal_case.add_argument('--prepare-only', action='store_true')
    sinusoidal_case.add_argument('--native-serial', action='store_true')
    sinusoidal_case.add_argument('--plot', action='store_true')
    from .tabulated_plume_cases import TABULATED_CASES

    tabulated = command.add_parser('plume-tabulated-case', help='Run Channel3D or BaileyLES from the supplied profile files')
    tabulated.add_argument('case', choices=TABULATED_CASES)
    tabulated.add_argument('--profiles', type=Path, required=True, help='Directory containing the original .txt fields')
    tabulated.add_argument('--output', type=Path, required=True)
    tabulated.add_argument('--step', type=float, action='append')
    tabulated.add_argument('--duration', type=float)
    tabulated.add_argument('--particles', type=int, default=100_000)
    tabulated.add_argument('--seed', type=int, default=0)
    tabulated.add_argument('--prepare-only', action='store_true')
    tabulated.add_argument('--native-serial', action='store_true')
    tabulated.add_argument('--plot', action='store_true')
    particle_results = command.add_parser('plume-particle-results', help='Analyze archived particle NetCDF or a CSV snapshot')
    particle_results.add_argument('snapshot', type=Path)
    particle_results.add_argument('turbulence', type=Path)
    particle_results.add_argument('--output', type=Path, required=True)
    particle_results.add_argument('--step', type=float, required=True)
    particle_results.add_argument('--height', type=float, required=True)
    particle_results.add_argument('--record', type=int, default=-1)
    particle_results.add_argument('--les', action='store_true')
    particle_results.add_argument('--plot', action='store_true')
    epa = command.add_parser('plume-epa-results', help='Extract the original EPA array concentration sections')
    epa.add_argument('collection', type=Path)
    epa.add_argument('--output', type=Path, required=True)
    epa.add_argument('--xml', type=Path)
    epa.add_argument('--record', type=int, default=-1)
    epa.add_argument('--plot', action='store_true')
    example=command.add_parser('output-example', help='Write the QES concentration and spectra NetCDF example')
    example.add_argument('--output', type=Path, default=Path('test.nc'))
    example.add_argument('--start', help='Simulation start as an ISO datetime without a timezone offset')
    wrf_dem = command.add_parser('wrf2dem', help='Convert WRF or WRF-SFIRE terrain and winds into GeoTIFF and QES XML')
    wrf_dem.add_argument('wrf_input', type=Path)
    wrf_dem.add_argument('dem_output', type=Path)
    wrf_dem.add_argument('--sample', type=int, default=12, help='Atmospheric station sampling stride')
    wrf_dem.add_argument('--nz', type=int, default=300, help='Vertical QES cells at one metre spacing')
    wrf_diagnostic = command.add_parser('wrf-coupling-test', help='Exchange diagnostic unit winds with a WRF test peer')
    wrf_diagnostic.add_argument('wrf_input', type=Path)
    wrf_diagnostic.add_argument('--rounds', type=int, default=10)
    wrf_diagnostic.add_argument('--timeout', type=float, default=60., help='Seconds to wait per frame')
    wrf_input_demo = command.add_parser('wrf-input-test', help='Dump the original NetCDF pressure subset')
    wrf_input_demo.add_argument('pressure_input', type=Path)
    wrf_input_demo.add_argument('--wrf-input', type=Path, help='Also inspect a WRF fire-mesh wind frame')
    wrf_input_demo.add_argument('--wrf-live', action='store_true', help='Wait for the original WRF startup marker')
    wrf_input_demo.add_argument('--timeout', type=float, default=60., help='Seconds to wait for a live startup frame')
    wind_input = command.add_parser('wind-input-test', help='Construct the wind workspace before applying a wind profile')
    wind_input.add_argument('-q', '--qesWindsParamFile', dest='xml', type=Path, required=True)
    wind_input.add_argument('-o', '--output', type=Path, help='Save the constructor workspace as NetCDF')
    wind=command.add_parser('wind',help='Run supported QES XML wind components entirely in Python')
    wind.add_argument('xml',type=Path)
    wind.add_argument('--output',type=Path,required=True)
    wind.add_argument('--no-turbulence',action='store_true')
    wind.add_argument('--initialize-only', action='store_true', help='Save initial wind and stencil before pressure and turbulence')
    wind.add_argument('--native-workspace-format', action='store_true', help='Write the original windsWk.nc layout')
    wind.add_argument('--native-turbulence-format', action='store_true', help='Write the original turbOut.nc layout')
    visualization = wind.add_mutually_exclusive_group()
    visualization.add_argument('--no-visualization', action='store_true', help='Skip cell-centred windsOut.nc')
    visualization.add_argument('--native-visualization-format', action='store_true',
                               help='Write the original cell-centred windsOut.nc layout')
    wind.add_argument('--met-variant',choices=['native-cpu','native-gpu'],default='native-cpu')
    wind.add_argument('--meteorology-backend', choices=('cpu', 'cuda'), default='cpu',
                      help='Use CUDA for multi-station Barnes interpolation')
    wind.add_argument('--wrf-timeout', type=float, help='Seconds to wait for each live WRF frame')
    wind.add_argument('--terrain-obj', type=Path, help='Export the terrain triangle mesh as an indexed OBJ')
    legacy_wind = command.add_parser('qesWinds', prog=launcher, help='Run standalone wind with original QES flags and file names')
    legacy_wind.add_argument('-q', '--qesWindsParamFile', type=Path, required=True, dest='xml')
    legacy_wind.add_argument('-o', '--outbasename', dest='basename')
    legacy_wind.add_argument('-s', '--solvetype', type=int, choices=(1, 2, 3, 4), default=1)
    legacy_wind.add_argument('-w', '--wkout', action='store_true')
    legacy_wind.add_argument('-t', '--turbcomp', action='store_true')
    legacy_wind.add_argument('-m', '--terrainout', action='store_true')
    legacy_wind.add_argument('-x', '--windsolveroff', action='store_true',
                             help='Record the request as in the pinned standalone driver, which still solves')
    legacy_wind.add_argument('-f', '--firemode', action='store_true',
                             help='Record the request. WRF exchange is enabled in the XML.')
    legacy_wind.add_argument('-v', '--verbose', action='store_true')
    legacy_wind.add_argument('--wrf-timeout', type=float, help='Seconds to wait for each live WRF frame')
    legacy_qes = command.add_parser('qes', prog=launcher, help='Run the original coupled wind and plume command')
    legacy_qes.add_argument('-q', '--qesWindsParamFile', type=Path, required=True, dest='wind_xml')
    legacy_qes.add_argument('-p', '--qesPlumeParamFile', type=Path, dest='plume_xml')
    legacy_qes.add_argument('-o', '--outbasename', dest='basename')
    legacy_qes.add_argument('-s', '--solvetype', type=int, choices=(1, 2, 3, 4), default=1)
    legacy_qes.add_argument('-w', '--wkout', action='store_true')
    legacy_qes.add_argument('-t', '--turbcomp', action='store_true')
    legacy_qes.add_argument('-z', '--visuout', action='store_true', help='Visualization is already enabled with -o')
    legacy_qes.add_argument('-m', '--terrainout', action='store_true')
    _add_legacy_particle_output(legacy_qes)
    legacy_qes.add_argument('-x', '--windsolveroff', action='store_true',
                            help='Record the request. The pinned coupled driver still solves.')
    legacy_qes.add_argument('-v', '--verbose', action='store_true')
    legacy_qes.add_argument('--native-threads', type=int, default=1, help='Linux QES worker streams to replay')
    legacy_qes.add_argument('--source-seed', type=int, help='Seed native-style source geometry independently')
    replay=command.add_parser('wind-replay',help='Reproject initial winds with coefficients in a saved workspace')
    replay.add_argument('workspace',type=Path)
    replay.add_argument('--output',type=Path,required=True)
    replay.add_argument('--native-input', action='store_true',
                        help='Read stored NetCDF values and timestamps using legacy QES conventions')
    replay.add_argument('--wall-roughness', type=float, help='Apply post-projection wall log correction with this roughness in metres')
    replay.add_argument('--tolerance', type=float, help='Override the pressure solver tolerance for this replay')
    replay.add_argument('--max-iterations', type=int, help='Override the pressure solver iteration limit')
    compare_winds=command.add_parser('compare-winds', help='Compare QES wind workspace NetCDF records')
    compare_winds.add_argument('-a', '--windsfile1', type=Path, required=True)
    compare_winds.add_argument('-b', '--windsfile2', type=Path, required=True)
    compare_winds.add_argument('--by-index', action='store_true', help='Compare stored entries even when axes or timestamps differ')
    compare_winds.add_argument('--format', choices=('json', 'legacy-text'), default='json')
    compare_solvers=command.add_parser('compare-solvers', help='Compare pressure backends from one initialized QES wind case')
    compare_solvers.add_argument('-q', '--xml', type=Path, required=True)
    from .wind_compare import SOLVER_BACKENDS
    solver_selection = compare_solvers.add_mutually_exclusive_group()
    solver_selection.add_argument('--against', choices=SOLVER_BACKENDS, action='append',
                                  help='Backend to compare against CPU BiCGSTAB (default: cpu-sor); repeat to compare more')
    solver_selection.add_argument('-s', '--solve-type', type=int, choices=(1, 2, 3, 4),
                                  help='Original compareSolvers selection: all, dynamic, global or shared')
    compare_solvers.add_argument('--format', choices=('json', 'legacy-table'), default='json')
    for entry in (compare_winds, compare_solvers):
        entry.add_argument('--precision', choices=('native', 'float64'), default='native',
                           help='Use original float32 comparison arithmetic or compare full float64 values')
    plume=command.add_parser('plume',help='Run QES wind and base particle models')
    plume.add_argument('wind_input',type=Path,help='Wind XML, or a staggered workspace with --turbulence-nc')
    plume.add_argument('plume_xml',type=Path)
    plume.add_argument('--output',type=Path,required=True)
    native_order = plume.add_mutually_exclusive_group()
    native_order.add_argument('--native-serial', action='store_true',
                              help='Replay the Linux QES single-worker Langevin stream and slot order')
    native_order.add_argument('--native-threads', type=int,
                              help='Replay the Linux QES OpenMP stream assignment for this worker count')
    native_order.add_argument('--native-arithmetic', action='store_true',
                              help='Use native float32 particle arithmetic with vectorized transport')
    plume.add_argument('--native-parallel', action='store_true',
                       help='Run the replay workers concurrently (requires --native-threads >= 2)')
    plume.add_argument('--source-seed', type=int,
                       help='Seed native-style source geometry streams independently of transport')
    plume.add_argument('--turbulence-nc',type=Path,help='Read wind and turbulence fields from NetCDF')
    plume.add_argument('--terrain-obj', type=Path, help='Load terrain triangles for meshReflection')
    plume.add_argument('--particle-csv',action='store_true',help='Write per-model particle CSV snapshots')
    plume.add_argument('--deposition',action='store_true',help='Activate the deposition kernel for flagged models')
    plume.add_argument('--native-workspace-format', action='store_true', help='Also write windsWk.nc from the coupled wind XML')
    plume.add_argument('--native-turbulence-format', action='store_true', help='Also write turbOut.nc from the coupled wind XML')
    plume.add_argument('--native-visualization-format', action='store_true', help='Also write windsOut.nc from the coupled wind XML')
    legacy_plume = command.add_parser('qesPlume', prog=launcher, help='Run standalone plume with original QES command flags and file names')
    legacy_plume.add_argument('-p', '--qesPlumeParamFile', dest='plume_xml')
    legacy_plume.add_argument('-f', '--projectQESFiles', dest='project')
    legacy_plume.add_argument('-w', '--inputWINDSFile', type=Path, dest='wind_input')
    legacy_plume.add_argument('-t', '--inputTURBFile', type=Path, dest='turbulence_nc')
    legacy_plume.add_argument('-o', '--outputbasename', dest='basename')
    _add_legacy_particle_output(legacy_plume)
    legacy_plume.add_argument('-v', '--verbose', action='store_true')
    legacy_plume.add_argument('--native-threads', type=int, default=1, help='Linux QES worker streams to replay')
    legacy_plume.add_argument('--native-parallel', action='store_true',
                              help='Run the replay workers concurrently (requires --native-threads >= 2)')
    legacy_plume.add_argument('--source-seed', type=int, help='Seed native-style source geometry independently')
    gpu_plume=command.add_parser('plume-gpu', help='Run the experimental CUDA plume on a saved uniform grid')
    gpu_plume.add_argument('wind_input', type=Path)
    gpu_plume.add_argument('turbulence_nc', type=Path)
    gpu_plume.add_argument('--output', type=Path, required=True, help='Output NPZ archive')
    gpu_plume.add_argument('--steps', type=int, required=True)
    gpu_source = gpu_plume.add_mutually_exclusive_group(required=True)
    gpu_source.add_argument('--release-per-step', type=int)
    gpu_source.add_argument('--source-xml', type=Path, help='Read release windows and source geometry from QES plume XML')
    gpu_plume.add_argument('--source-model', help='Choose a particle model tag from --source-xml')
    gpu_plume.add_argument('--source-seed', type=int, help='Seed native-style source geometry independently')
    gpu_plume.add_argument('--capacity', type=int, required=True)
    gpu_plume.add_argument('--source', type=float, nargs=3, help='Fixed point for --release-per-step (default 20 50 70)')
    gpu_plume.add_argument('--boxes', type=int, nargs=3, default=(20, 49, 69))
    gpu_plume.add_argument('--seed', type=int, default=1234)
    gpu_plume.add_argument('--record', type=int, default=0)
    gpu_plume.add_argument('--wind-series', action='store_true', help='Apply subsequent records at their timestamps')
    gpu_plume.add_argument('--collection-start', type=int, default=0)
    gpu_plume.add_argument('--collection-bounds', type=float, nargs=6,
                           metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                           help='Sampling box bounds (default: transport domain)')
    gpu_plume.add_argument('--transport-bounds', type=float, nargs=6,
                           metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                           help='Transport boundary locations inside the interpolation domain')
    gpu_plume.add_argument('--boundaries', nargs=3, choices=('exiting', 'periodic', 'reflection'),
                           default=('exiting',)*3, metavar='MODE', help='Domain boundary modes in XYZ order')
    fire=command.add_parser('fire',help='Run CPU coupled fire and wind')
    fire.add_argument('xml',type=Path)
    fire.add_argument('--output',type=Path,required=True)
    fire.add_argument('--turbulence',action='store_true')
    fire.add_argument('--native-turbulence-format', action='store_true', help='Compute and save the original turbOut.nc layout')
    fire.add_argument('--native-fire-format', action='store_true', help='Save the original fireOutput.nc layout')
    fire.add_argument('--native-flame-sampling', action='store_true',
                      help='Use the Linux QES narrow-band startup wind-sampling level')
    fire.add_argument('--full-level-set', action='store_true', help='Use the original full-grid fire level set')
    fire.add_argument('--plume',type=Path,help='QES plume XML for fire smoke')
    fire.add_argument('--particle-csv',action='store_true',help='Write smoke particle CSV snapshots')
    fire_order = fire.add_mutually_exclusive_group()
    fire_order.add_argument('--native-serial', action='store_true', help='Replay the Linux QES single-worker plume stream')
    fire_order.add_argument('--native-threads', type=int, help='Replay the Linux QES OpenMP plume worker streams')
    fire.add_argument('--source-seed', type=int, help='Seed native-style source geometry streams independently')
    potential=fire.add_mutually_exclusive_group(required=True)
    potential.add_argument('--heat-potential',type=Path)
    potential.add_argument('--no-fire-winds',action='store_true')
    fire.add_argument('--heat-potential-backend', choices=('cpu', 'cuda'), default='cpu')
    fire.add_argument('--front-backend', choices=('cpu', 'cuda'), default='cpu')
    legacy_fire = command.add_parser('qesFire', prog=launcher, add_help=False, help='Run fire with original QES options and file names')
    legacy_fire.add_argument('-?', '--help', action='help')
    legacy_fire.add_argument('-q', '--qesWindsParamFile', type=Path, required=True, dest='xml')
    legacy_fire.add_argument('-p', '--qesPlumeParamFile', type=Path, dest='plume_xml')
    legacy_fire.add_argument('-o', '--outbasename', dest='basename')
    legacy_fire.add_argument('-s', '--solvetype', type=int, choices=(1, 2, 3, 4), default=1)
    legacy_fire.add_argument('-b', '--fireout', action='store_true')
    legacy_fire.add_argument('-t', '--turbcomp', action='store_true')
    _add_legacy_particle_output(legacy_fire, '--doParticleDataOutput')
    legacy_fire.add_argument('-i', '--fireWindsOff', action='store_true', dest='no_fire_winds')
    legacy_fire.add_argument('-x', '--windsolveroff', action='store_true', help='The original driver still solves')
    legacy_fire.add_argument('-w', '--wkout', action='store_true', help='Accepted by the original fire driver without a writer')
    legacy_fire.add_argument('-z', '--visuout', action='store_true', help='Accepted by the original fire driver without a writer')
    legacy_fire.add_argument('-h', '--terrainout', action='store_true', help='Accepted by the original fire driver without a writer')
    legacy_fire.add_argument('-v', '--verbose', action='store_true')
    legacy_fire.add_argument('--heat-potential', type=Path)
    legacy_fire.add_argument('--native-threads', type=int, default=1)
    for driver in (plume, legacy_plume, legacy_qes, fire, legacy_fire):
        driver.add_argument('--seed', type=_cpu_seed_argument, default=0, metavar='INTEGER|clock',
                            help='Transport seed, or clock to seed from the current Unix second')
        if driver is legacy_fire:
            continue
        has_corrected_option = driver in (legacy_plume, legacy_qes)
        release_indexing = driver.add_mutually_exclusive_group() if has_corrected_option else driver
        native_indexing_default = None if driver is legacy_qes else driver is legacy_plume
        release_indexing.add_argument('--native-release-indexing', action='store_true', default=native_indexing_default,
                                      help='Replay original CPU slot indexing for release solidity and CSV metadata')
        if has_corrected_option:
            release_indexing.add_argument('--corrected-release-indexing', action='store_false',
                                          dest='native_release_indexing',
                                          help='Check release solidity and metadata at each newly allocated slot')
    fire_release_indexing = legacy_fire.add_mutually_exclusive_group()
    fire_release_indexing.add_argument('--native-release-indexing', action='store_true', default=None,
                                       help='Use the original CPU slot indexing for fire plumes (default)')
    fire_release_indexing.add_argument('--corrected-release-indexing', action='store_false',
                                       dest='native_release_indexing',
                                       help='Check release solidity and metadata at each newly allocated slot')
    for driver in (legacy_plume, legacy_qes, legacy_fire):
        driver.add_argument('--native-full-domain-bounds', action='store_true',
                            help='Reproduce the original full-domain source bound ordering')
    for driver in (legacy_wind, legacy_qes, legacy_fire):
        driver.add_argument('--cpu-serial', action='store_true',
                            help='Use the original serial CPU solver selected by builds without OpenMP')
    legacy_fire.add_argument('--source-seed', type=int)
    for entry in (wind, replay, plume, fire):
        entry.add_argument('--poisson', choices=('cpu', *SOLVER_BACKENDS[1:]), default='cpu',
                           help='Wind pressure solver backend')
    for entry in (legacy_qes, plume, legacy_fire, fire):
        entry.add_argument('--wrf-timeout', type=float, help='Seconds to wait for the WRF startup frame')
    for entry in (wind, plume, fire):
        entry.add_argument('--turbulence-gradient', choices=('cpu', 'cuda'), default='cpu',
                           help='Backend for the turbulence velocity gradient')
        entry.add_argument('--native-sensor-math', action='store_true',
                           help='Use original CPU float32 wind initialization arithmetic')
        entry.add_argument('--native-wake-math', action='store_true',
                           help='Use original CPU mixed-precision polygon-wake arithmetic')
        entry.add_argument('--native-building-math', action='store_true',
                           help='Use original mixed-precision upwind, wake, sidewall and rooftop arithmetic')
        entry.add_argument('--native-turbulence-math', action='store_true',
                           help='Use original CPU mixed-precision local turbulence arithmetic')
    for entry in (plume, fire):
        entry.add_argument('--native-collection-format', action='store_true',
                           help='Write concentration NetCDF with the native QES schema and precision')
        entry.add_argument('--native-collection-math', action='store_true',
                           help='Use native QES float32 collection binning and accumulation')
    thermo=command.add_parser('thermo')
    thermo.add_argument('--provider',choices=['legacy','coolprop'],default='legacy')
    thermo.add_argument('--ambient-K',type=float,default=276.45)
    thermo.add_argument('--liquid-fraction',type=float,default=.559)
    thermo.add_argument('--RH',type=float,default=75)
    thermo.add_argument('--concentration',type=float,nargs='+',required=True)
    args=parser.parse_args(argv)
    if args.command=='xml-example':
        from .xml_demo import run_xml_demo

        stdout, stderr, status = run_xml_demo(args.xml, simulation=args.simulation)
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        if status:
            raise SystemExit(status)
    elif args.command=='info':
        from .assembly import default_registry
        path=Path(__file__).resolve().parents[1]/'component_status.json'
        coverage=json.loads(path.read_text())if path.exists()else {'all_features_ported':False}
        print(json.dumps({'version':__version__,'all_features_ported':False,'components':default_registry().names(),
             'physical_defaults':asdict(PhysicalCoefficients()),'coverage':coverage},indent=2))
    elif args.command=='lh2-demo':
        from .lh2_demo import example_input, run_demo

        try:
            if args.write_input is not None:
                with args.write_input.open('x', encoding='utf-8') as file:
                    file.write(example_input())
                print(args.write_input)
                return
            raw = args.input.read_text(encoding='utf-8-sig') if args.input is not None else example_input()
            result = run_demo(json.loads(raw), args.output)
        except (KeyError, TypeError, ValueError, OSError) as error:
            lh2_demo.error(str(error))
        print(json.dumps(result, indent=2))
        if args.open:
            import webbrowser

            webbrowser.open((args.output / 'plume.html').resolve().as_uri())
    elif args.command=='lh2-ffi':
        from .ffi_lh2 import run_ffi

        try:
            result = run_ffi(args.tests, args.output, args.workers, report=args.report, render=not args.no_3d)
        except (ValueError, OSError, RuntimeError) as error:
            ffi_lh2.error(str(error))
        print(json.dumps(dict(result['overall'], wall_s=result['wall_s'],
                              report=str(args.output / 'index.html')), indent=2))
        if args.open:
            import webbrowser

            webbrowser.open((args.output / 'index.html').resolve().as_uri())
    elif args.command=='runtime-info':
        from .runtime_info import report_startup_error, write_runtime_banner

        try:
            write_runtime_banner(cuda=args.cuda, optix=args.optix)
        except RuntimeError as exc:
            report_startup_error(exc)
            raise SystemExit(1) from None
    elif args.command=='output-example':
        from datetime import datetime
        from .output_example import write_output_example
        start = datetime.fromisoformat(args.start) if args.start is not None else None
        write_output_example(args.output, start=start)
    elif args.command=='wrf2dem':
        from .wrf_dem import convert_wrf_dem
        print(json.dumps(convert_wrf_dem(args.wrf_input, args.dem_output, sample=args.sample, nz=args.nz), indent=2))
    elif args.command=='wrf-coupling-test':
        from .wrf_diagnostic import run_coupling_diagnostic
        print(json.dumps(run_coupling_diagnostic(args.wrf_input, rounds=args.rounds, timeout=args.timeout), indent=2))
    elif args.command=='wrf-input-test':
        from .wrf_input_demo import pressure_dump

        if args.wrf_live and args.wrf_input is None:
            wrf_input_demo.error('--wrf-live requires --wrf-input')
        print(pressure_dump(args.pressure_input), end='', flush=args.wrf_live)
        if args.wrf_input is not None:
            from .wrf import FireMeshInput
            from .wrf_exchange import read_wrf_startup

            snapshot = (read_wrf_startup(args.wrf_input, timeout=args.timeout) if args.wrf_live
                        else FireMeshInput.read(args.wrf_input))
            print(json.dumps(snapshot.describe(), indent=2), file=sys.stderr)
    elif args.command=='wind-input-test':
        from .wind_run import inspect_wind_input

        print(json.dumps(inspect_wind_input(args.xml, output=args.output), indent=2))
    elif args.command=='wind':
        from .wind_case import read_wind_case
        from .wind_run import write_wind_run
        import time
        start=time.perf_counter()
        case=read_wind_case(args.xml,compute_turbulence=not (args.no_turbulence or args.initialize_only),wrf_timeout=args.wrf_timeout,
                           poisson_backend=args.poisson, turbulence_gradient=args.turbulence_gradient,
                           native_sensor_math=args.native_sensor_math, native_wake_math=args.native_wake_math,
                           native_turbulence_math=args.native_turbulence_math, native_building_math=args.native_building_math,
                           meteorology_backend=args.meteorology_backend)
        if args.terrain_obj is not None and case.terrain is None:
            parser.error('--terrain-obj requires a terrain input')
        if args.initialize_only and case.wrf_exchange is not None:
            parser.error('--initialize-only requires offline wind input')
        if args.native_turbulence_format and case.turbulence is None:
            parser.error('--native-turbulence-format requires turbulence computation and turbParams')
        if args.meteorology_backend == 'cpu':
            case.meteorology_variant=args.met_variant
        args.output.mkdir(parents=True,exist_ok=True)
        workspace_file = 'windsWk.nc' if args.native_workspace_format else 'winds.nc'
        turbulence_file = 'turbOut.nc' if args.native_turbulence_format else 'turbulence.nc'
        summaries = write_wind_run(
            case, workspace_path=args.output/workspace_file,
            turbulence_path=args.output/turbulence_file if case.turbulence is not None else None,
            visualization_path=None if args.no_visualization else args.output/'windsOut.nc',
            terrain_obj=args.terrain_obj, initialize_only=args.initialize_only,
            native_workspace_format=args.native_workspace_format, native_turbulence_format=args.native_turbulence_format,
            native_visualization_format=args.native_visualization_format)
        import hashlib
        manifest={'all_features_ported':False,'solver':summaries[0]['solver'],
                  'wall_s':time.perf_counter()-start,'turbulence':summaries[0]['turbulence'], 'steps':summaries,
                  'input_xml':str(args.xml.resolve()),'native_runtime_used':False,
                  'workspace_file':workspace_file,'initialize_only':args.initialize_only,
                  'turbulence_file':None if case.turbulence is None else turbulence_file,
                  'visualization_file':None if args.no_visualization else 'windsOut.nc',
                  'input_xml_sha256':hashlib.sha256(args.xml.read_bytes()).hexdigest(),
                  'configuration':case.manifest()}
        (args.output/'execution.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
        print(json.dumps(manifest,indent=2,allow_nan=False))
    elif args.command=='qesWinds':
        from .wind_case import read_wind_case
        from .wind_run import legacy_wind_backend, write_wind_run

        cpu = args.solvetype == 1
        if args.cpu_serial and not cpu:
            legacy_wind.error('--cpu-serial requires --solvetype 1')
        backend = legacy_wind_backend(args.solvetype, cpu_serial=args.cpu_serial)
        case = read_wind_case(args.xml, compute_turbulence=args.turbcomp, wrf_timeout=args.wrf_timeout,
                              poisson_backend=backend, native_sensor_math=cpu, native_building_math=cpu,
                              native_turbulence_math=args.turbcomp, meteorology_backend='cpu' if cpu else 'cuda',
                              legacy_turbulence_xml=True, legacy_wrf_protocol=True)
        case = replace(case, native_staircase_metrics=True)
        if args.turbcomp and case.turbulence is None:
            legacy_wind.error('--turbcomp requires turbParams')
        if args.basename and args.terrainout and case.terrain is None:
            legacy_wind.error('--terrainout requires a terrain input')
        products = {}
        if args.basename:
            for key, suffix, enabled in (
                ('visualization_path', '_windsOut.nc', True),
                ('workspace_path', '_windsWk.nc', args.wkout),
                ('turbulence_path', '_turbOut.nc', args.turbcomp),
                ('terrain_obj', '_terrainOut.obj', args.terrainout),
            ):
                if enabled:
                    products[key] = Path(args.basename + suffix)
        run_products = products.copy()
        terrain_path = run_products.pop('terrain_obj', None)
        if terrain_path is not None:
            # The original standalone driver exports the DEM before loading
            # solver coefficients or opening any wind output.
            case.terrain.mesh(case.domain).write_obj(terrain_path, indexed=True)
        steps = write_wind_run(case, **run_products, native_workspace_format=True,
                               native_turbulence_format=True, native_visualization_format=True)
        report = dict(backend=backend, solvetype=args.solvetype, steps=steps,
                      products={key: str(path) for key, path in products.items()},
                      windsolveroff_requested=args.windsolveroff, firemode_requested=args.firemode)
        if args.verbose:
            report['configuration'] = case.manifest()
        print(json.dumps(report, indent=2, allow_nan=False))
    elif args.command=='qes':
        from .plume_run import run_legacy_qes

        if args.cpu_serial and args.solvetype != 1:
            legacy_qes.error('--cpu-serial requires --solvetype 1')
        native_release_indexing = args.native_release_indexing
        if native_release_indexing is None:
            native_release_indexing = args.plume_xml is not None
        particle_csv = args.particle_csv or args.native_particle_csv or args.precise_particle_csv
        result = run_legacy_qes(
            args.wind_xml, args.plume_xml, basename=args.basename, solver_type=args.solvetype,
            workspace=args.wkout, turbulence=args.turbcomp, terrain=args.terrainout,
            particle_csv=particle_csv, seed=args.seed, native_threads=args.native_threads,
            source_seed=args.source_seed, verbose=args.verbose, wrf_timeout=args.wrf_timeout,
            native_release_indexing=native_release_indexing,
            native_full_domain_bounds=args.native_full_domain_bounds,
            native_particle_csv=particle_csv and not args.precise_particle_csv,
            progress_stream=sys.stderr, cpu_serial=args.cpu_serial)
        result['windsolveroff_requested'] = args.windsolveroff
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command=='wind-replay':
        from .contracts import NumericPolicy
        from .wind_replay import replay_wind
        from .numerics.poisson import make_poisson
        overrides = {name: value for name, value in (
            ('relative_tolerance', args.tolerance), ('max_iterations', args.max_iterations)
        ) if value is not None}
        solver = make_poisson(args.poisson, NumericPolicy(**overrides))
        print(json.dumps(replay_wind(args.workspace, args.output, solver=solver,
                                     wall_roughness=args.wall_roughness, native_input=args.native_input), indent=2))
    elif args.command=='compare-winds':
        from .wind_compare import compare_wind_files
        result = compare_wind_files(args.windsfile1, args.windsfile2, precision=args.precision,
                                    by_index=args.by_index, native_layout=args.format == 'legacy-text',
                                    include_speed=args.format != 'legacy-text')
        if args.format == 'legacy-text':
            from .wind_compare_text import format_workspace_stats
            print(format_workspace_stats(result), end='')
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command=='compare-solvers':
        from .wind_compare import compare_legacy_solvers, compare_solvers
        legacy = ('cuda-dynamic-legacy', 'cuda-global-memory', 'cuda-shared-memory')
        backends = {1: legacy, **{index: (name,) for index, name in enumerate(legacy, 2)}}
        selected = args.against or backends.get(args.solve_type, ('cpu-sor',))
        run_comparison = compare_legacy_solvers if args.solve_type is not None else compare_solvers
        result = run_comparison(args.xml, backends=selected, precision=args.precision)
        if args.format == 'legacy-table':
            from .wind_compare_text import format_solver_series
            print(format_solver_series(result), end='')
            for record in result['records']:
                for name, candidate in record['candidates'].items():
                    if not candidate['solver']['converged']:
                        print(f"{record['timestamp']} {name} did not converge", file=sys.stderr)
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command=='plume-managed-container':
        from .particle_slot_regression import run_managed_container

        report = run_managed_container(steps=args.steps, release_count=args.release_per_step, seed=args.seed)
        print(json.dumps(report, indent=2))
        if not report['invariants_passed']:
            raise SystemExit(1)
    elif args.command=='plume-sinusoidal':
        from .plume_regression import run_sinusoidal_regression

        report = run_sinusoidal_regression(particles=args.particles, duration=args.duration, seed=args.seed,
                                           native_rng=args.native_rng)
        print(json.dumps(report, indent=2, allow_nan=False))
        if not report['passed']:
            raise SystemExit(1)
    elif args.command=='plume-uniform':
        from .plume_regression import run_uniform_regression

        report = run_uniform_regression(particles_per_step=args.particles_per_step, duration=args.duration,
                                        averaging_start=args.averaging_start, seed=args.seed)
        print(json.dumps(report, indent=2, allow_nan=False))
        if not report['passed']:
            raise SystemExit(1)
    elif args.command=='plume-analytical-case':
        from .analytical_cases import run_analytical_case

        print(json.dumps(run_analytical_case(args.case, args.output, prepare_only=args.prepare_only,
                                             steps=args.steps, particles_per_step=args.particles_per_step,
                                             seed=args.seed), indent=2))
    elif args.command=='plume-analytical-results':
        from .analytical_results import write_analytical_results

        print(json.dumps(write_analytical_results(args.case, args.collection, args.output, record=args.record,
                                                  xml=args.xml, plot=args.plot), indent=2, allow_nan=False))
    elif args.command=='plume-sinusoidal-case':
        from .sinusoidal_case import run_sinusoidal_case

        print(json.dumps(run_sinusoidal_case(args.output, steps=args.step or TIME_STEPS, particles=args.particles,
                                             seed=args.seed, prepare_only=args.prepare_only,
                                             native_serial=args.native_serial, plot=args.plot), indent=2))
    elif args.command=='plume-tabulated-case':
        from .tabulated_plume_cases import run_tabulated_case

        print(json.dumps(run_tabulated_case(args.case, args.profiles, args.output, steps=args.step,
                                            duration=args.duration, particles=args.particles, seed=args.seed,
                                            prepare_only=args.prepare_only, native_serial=args.native_serial,
                                            plot=args.plot), indent=2))
    elif args.command=='plume-particle-results':
        from .prescribed_diagnostics import write_particle_diagnostics

        report = write_particle_diagnostics(args.snapshot, args.turbulence, args.output, args.step,
                                            height=args.height, les=args.les, record=args.record)
        if args.plot:
            from .prescribed_figures import write_profile_figures

            cases = {f'{args.step:g}': {'diagnostics': report}}
            report['plots'] = write_profile_figures(args.output.parent, args.output.stem, cases,
                                                    ghosts=True, les=args.les)
        print(json.dumps(report, indent=2, allow_nan=False))
    elif args.command=='plume-epa-results':
        from .epa_results import write_epa_results

        print(json.dumps(write_epa_results(args.collection, args.output, record=args.record,
                                           xml=args.xml, plot=args.plot), indent=2, allow_nan=False))
    elif args.command=='plume':
        from .plume_run import run_plume
        print(json.dumps(run_plume(args.wind_input, args.plume_xml, args.output, seed=args.seed,
                                   turbulence_path=args.turbulence_nc, particle_csv=args.particle_csv,
                                   deposition=args.deposition, terrain_obj=args.terrain_obj,
                                   poisson_backend=args.poisson, turbulence_gradient=args.turbulence_gradient,
                                   native_serial=args.native_serial, source_seed=args.source_seed,
                                   native_arithmetic=args.native_arithmetic,
                                   native_threads=args.native_threads, native_parallel=args.native_parallel,
                                   native_release_indexing=args.native_release_indexing,
                                   native_collection_format=args.native_collection_format,
                                   native_collection_math=args.native_collection_math,
                                   native_workspace_format=args.native_workspace_format,
                                   native_turbulence_format=args.native_turbulence_format,
                                   native_visualization_format=args.native_visualization_format,
                                   native_sensor_math=args.native_sensor_math,
                                   native_wake_math=args.native_wake_math,
                                   native_turbulence_math=args.native_turbulence_math,
                                   native_building_math=args.native_building_math, wrf_timeout=args.wrf_timeout), indent=2))
    elif args.command=='qesPlume':
        from .plume_progress import write_legacy_options
        from .plume_run import run_plume
        from .runtime_info import report_startup_error

        error = None
        if not args.plume_xml:
            error = 'qesPlumeParamFile not specified'
        elif args.project is None:
            if args.wind_input is None:
                error = 'inputWINDSFile not specified!'
            elif args.turbulence_nc is None:
                error = 'inputTURBFile not specified!'
            elif not args.basename:
                error = 'No output basename set -> output turned off '
        if error is not None:
            report_startup_error(error)
            raise SystemExit(1)
        if args.native_parallel and args.native_threads < 2:
            legacy_plume.error('--native-parallel requires --native-threads >= 2')
        if args.project is not None:
            basename = args.project
            wind, turbulence = Path(basename+'_windsWk.nc'), Path(basename+'_turbOut.nc')
        else:
            basename, wind, turbulence = args.basename, args.wind_input, args.turbulence_nc
        prefix = Path(f'{basename}_')
        basename_component = prefix.name.removesuffix('_')
        particle_csv = args.particle_csv or args.native_particle_csv or args.precise_particle_csv
        write_legacy_options(sys.stderr, plume=args.plume_xml, wind=wind, turbulence=turbulence, basename=basename,
                             verbose=args.verbose, particle_output=particle_csv)
        result = run_plume(wind, args.plume_xml, prefix.parent, turbulence_path=turbulence,
                           output_basename=basename_component, allow_empty_models=True, particle_csv=particle_csv,
                           seed=args.seed, native_threads=args.native_threads, source_seed=args.source_seed,
                           native_parallel=args.native_parallel,
                           native_collection_format=True, native_collection_math=True, progress_stream=sys.stderr,
                           native_release_indexing=args.native_release_indexing,
                           native_timestamps=True, native_layout=True, legacy_plume_xml=True,
                           native_particle_csv=particle_csv and not args.precise_particle_csv,
                           native_full_domain_bounds=args.native_full_domain_bounds)
        if args.verbose:
            result['files'] = dict(plume_xml=str(args.plume_xml), wind=str(wind), turbulence=str(turbulence),
                                   output_basename=str(basename))
        print(json.dumps(result, indent=2))
    elif args.command=='plume-gpu':
        from .cuda_plume_run import run_cuda_plume
        bounds = {}
        for name in ('collection_bounds', 'transport_bounds'):
            values = getattr(args, name)
            if values is not None:
                bounds[name] = list(zip(values[::2], values[1::2]))
        print(json.dumps(run_cuda_plume(args.wind_input, args.turbulence_nc, args.output, steps=args.steps,
                                   release_count=args.release_per_step, capacity=args.capacity, source=args.source,
                                   boxes=args.boxes, seed=args.seed, record=args.record,
                                   collection_start=args.collection_start, boundaries=args.boundaries,
                                   wind_series=args.wind_series, source_xml=args.source_xml,
                                   source_model=args.source_model, source_seed=args.source_seed, **bounds),
                         indent=2))
    elif args.command=='qesFire':
        from .fire_run import run_legacy_fire

        if args.cpu_serial and args.solvetype != 1:
            legacy_fire.error('--cpu-serial requires --solvetype 1')
        particle_csv = args.particle_csv or args.native_particle_csv or args.precise_particle_csv
        result = run_legacy_fire(
            args.xml, args.plume_xml, basename=args.basename, solver_type=args.solvetype,
            fire_output=args.fireout, turbulence=args.turbcomp, particle_csv=particle_csv,
            no_fire_winds=args.no_fire_winds, heat_potential=args.heat_potential,
            seed=args.seed, native_threads=args.native_threads, source_seed=args.source_seed,
            native_release_indexing=args.native_release_indexing,
            native_full_domain_bounds=args.native_full_domain_bounds,
            native_particle_csv=not args.precise_particle_csv,
            verbose=args.verbose, wrf_timeout=args.wrf_timeout, progress_stream=sys.stderr,
            cpu_serial=args.cpu_serial)
        result['windsolveroff_requested'] = args.windsolveroff
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command=='fire':
        from .fire_run import read_coupled_fire, write_fire_run
        case = read_coupled_fire(args.xml, heat_potential=args.heat_potential,
                                 compute_turbulence=args.turbulence or args.native_turbulence_format,
                                 plume=args.plume, seed=args.seed, particle_csv=args.particle_csv,
                                 poisson_backend=args.poisson, turbulence_gradient=args.turbulence_gradient,
                                 heat_potential_backend=args.heat_potential_backend, front_backend=args.front_backend,
                                 native_serial=args.native_serial, native_threads=args.native_threads,
                                 native_release_indexing=args.native_release_indexing,
                                 source_seed=args.source_seed,
                                 native_sensor_math=args.native_sensor_math, native_wake_math=args.native_wake_math,
                                 native_turbulence_math=args.native_turbulence_math, native_building_math=args.native_building_math,
                                 wrf_timeout=args.wrf_timeout, native_flame_sampling=args.native_flame_sampling,
                                 narrow_band=not args.full_level_set)
        print(json.dumps(write_fire_run(case, args.output, native_collection_format=args.native_collection_format,
                                        native_collection_math=args.native_collection_math,
                                        native_turbulence_format=args.native_turbulence_format,
                                        native_fire_format=args.native_fire_format), indent=2))
    else:
        from .thermo import LegacyEquilibrium,CoolPropEquilibrium
        provider=LegacyEquilibrium if args.provider=='legacy'else CoolPropEquilibrium
        model=provider(ambient_temperature=args.ambient_K,liquid_fraction=args.liquid_fraction,relative_humidity=args.RH)
        state=model.evaluate(args.concentration)
        result={key:value.tolist()for key,value in asdict(state).items()}
        print(json.dumps({'states':result,'model':model.describe()},indent=2,allow_nan=False))

def qes():
    return main(['qes', *sys.argv[1:]], launcher='qes')


def qes_winds():
    return main(['qesWinds', *sys.argv[1:]], launcher='qesWinds')


def qes_plume():
    return main(['qesPlume', *sys.argv[1:]], launcher='qesPlume')


def qes_fire():
    return main(['qesFire', *sys.argv[1:]], launcher='qesFire')


if __name__ == '__main__':
    main()
