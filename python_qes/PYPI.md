# PlumeSwift

An independently maintained Python port of QES v2.3.1, previously released
as QESx. It implements public QES algorithms for 3D wind
fields, Lagrangian particle dispersion and fire-related workflows. The Python
API also includes experimental liquid hydrogen thermodynamics, buoyancy and
drag coupling. This alpha has partial upstream coverage and has been compared
against selected original QES cases and FFI concentration measurements.

## Install

```shell
python -m pip install --pre plumeswift
plumeswift --help
plumeswift runtime-info
```

Run the bundled synthetic liquid hydrogen example and open its interactive 3D
concentration plot in a browser:

```shell
python -m pip install "plumeswift[visual]==0.1.0a4"
plumeswift lh2-demo --output lh2-result --open
```

Use `plumeswift lh2-demo --write-input lh2-case.json` to export the editable input.
See the repository README for the input fields and output file layout.

Download FFI-RAPPORT 20/03101 from the [FFI publication page](https://www.ffi.no/en/publications-archive/large-scale-leakage-of-liquid-hydrogen-lh2-tests-related-to-bunkering-and-maritime-use-of-liquid-hydrogen)
and save it as `ffi-report.pdf`. PlumeSwift reads the sensor means from the local
PDF and links the comparisons and interactive 3D plumes in the result index.

```shell
python -m pip install "plumeswift[fast,visual,ffi]==0.1.0a4"
plumeswift lh2-ffi --report ffi-report.pdf --workers 5 --output ffi-results --open
```

Use `--tests 3` for a single experiment or `--no-3d` for sensor comparisons alone.

The optional `fast` extra adds Numba for the physical-coordinate interpolation,
voxel-wall traversal and 3×3 GLE solve. The `thermo` extra adds CoolProp.
The package exposes the `plumeswift` command and Python module. The original
command interfaces are available as subcommands, for example `plumeswift qesWinds`.

The source repository documents supported options, research limitations,
provenance and the experimental status of the LH₂ model. The current evidence
does not establish the model's suitability for determining safety distances.

The QES-derived code is GPL-3.0-only. The glibc-derived numerical components
retain LGPL-2.1-or-later. See the included license files and NOTICE.
