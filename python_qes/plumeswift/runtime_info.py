"""QES startup and CUDA device diagnostics for the Python compatibility drivers."""
import sys


_BORDER = '#' * 67
_SEPARATOR = '=' * 67
_HEADER = (
    _BORDER,
    '#                                                                 #',
    '#                        Welcome to QES                           #',
    '#                                                                 #',
    _BORDER,
    'Version: v2.3.1',
)


def _version(number):
    return f'{number // 1000}.{number % 100 // 10}'


def _write_devices(runtime, output, diagnostics):
    count = runtime.getDeviceCount()
    if not count:
        report_warning('There are no available device(s) that support CUDA', diagnostics)
        return
    print(f'[CUDA]\t Detected {count} CUDA Capable device(s)', file=output)
    driver = _version(runtime.driverGetVersion())
    version = _version(runtime.runtimeGetVersion())
    for device in range(count):
        properties = runtime.getDeviceProperties(device)
        name = properties['name'].decode(errors='replace').rstrip('\0')
        memory = properties['totalGlobalMem']
        clock = properties['clockRate']
        print(f'\t Device {device}: {name}',
              f'\t | CUDA Driver Version / Runtime Version: {driver} / {version}',
              f'\t | CUDA Capability Major/Minor version number: {properties["major"]}.{properties["minor"]}',
              f'\t | Total amount of global memory: {memory / 1048576:.0f} MBytes ({memory} bytes)',
              f'\t | GPU Max Clock rate:  {clock / 1000:g} MHz ({clock / 1_000_000:g} GHz)',
              f'\t | PCI: BusID={properties["pciBusID"]}, DeviceID={properties["pciDeviceID"]}, '
              f'DomainID={properties["pciDomainID"]}', sep='\n', file=output)


def write_runtime_banner(*, cuda=False, optix=False, output=None, diagnostics=None):
    """Write the pinned QES banner and inspect optional backends on request."""
    output = sys.stdout if output is None else output
    diagnostics = sys.stderr if diagnostics is None else diagnostics
    print(*_HEADER, sep='\n', file=output)
    print('* CUDA support available!' if cuda else '* No CUDA support - CPU Only Computations!', file=output)
    if optix:
        from .optix_ray_mixing import load_optix

        binding = load_optix()
        version = '.'.join(map(str, binding.version()))
        print(f'* OptiX Python binding: {version}', file=output)
    if cuda:
        try:
            from cupy.cuda import runtime
        except ImportError as exc:
            raise RuntimeError('The CUDA backend requires CuPy built for the installed CUDA toolkit') from exc
        print('-' * 66, file=output)
        try:
            _write_devices(runtime, output, diagnostics)
        except runtime.CUDARuntimeError as exc:
            raise RuntimeError(f'CUDA device query returned {exc.status}\n\t-> {exc}') from exc
    print(_BORDER, file=output)


def report_startup_error(error, *, stream=None):
    """Print the native-style fatal error separator for a failed probe."""
    stream = sys.stderr if stream is None else stream
    print('', _SEPARATOR, f'[ERROR]\t {error}', _SEPARATOR, sep='\n', file=stream)


def report_warning(message, stream):
    print(f'[!! WARNING !!]\t {message}', file=stream)
