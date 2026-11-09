# Attribution and implementation scope

PlumeSwift is an independently maintained modified Python port of QES.
The initial port was publicly released on 2026-11-09 as QESx. On 2026-11-09
PlumeSwift 0.1.0a4 renamed the project and moved extraction of FFI
observations to the user's machine. The port is based on QES 2.3.1, commit
`dced7ec571487be7538eca17bfc25bb3262fb6d0`.

Copyright © 2025 University of Utah and University of Minnesota Duluth.
Original QES copyright notices also identify Behnam Bozorgmehr, Jeremy A.
Gibbs, Fabien Margairaz, Eric R. Pardyjak, Zachary Patterson, Rob Stoll,
Lucas Ulmer and Pete Willemsen (2025). QES-Fire notices also identify
Matthew Moody and Brian Bailey (2025). QES-derived code is distributed
under GNU GPL version 3; see LICENSE. The software comes without warranty
as described in LICENSE. The port inventory records the upstream paths.

The WRF input code credits Pete Willemsen (2019). The upstream argument
parser credits the Department of Computer Science, University of
Minnesota-Duluth (2009). Their upstream GPL version notices remain
applicable to those components.

The Python package contains no copied C++/CUDA source, native QES extension,
native executable launcher, or embedded native QES library. Translating
algorithms does not remove license/attribution obligations. NumPy, SciPy,
NetCDF4 and optional CoolProp are third-party numerical libraries with their
own licenses and compiled internals.

`plumeswift/fire_data/HeatPot.nc` is the unmodified QES-Fire lookup table from
`data/FireFiles/HeatPot.nc` at the QES revision identified above.

The inverse WRF converter in `plumeswift/utm.py` follows `WRFInput.cpp`'s
`UTMConv`, which credits SPECFEM3D 2.1 by Dimitri Komatitsch and Jeroen Tromp
(2012, GPL-2.0-or-later).

`plumeswift/numerics/linux_float_angles.py` adapts the finite-angle reductions and
polynomials from [glibc 2.35 libm](https://sourceware.org/git/?p=glibc.git;a=tree;f=sysdeps/ieee754/flt-32;hb=glibc-2.35).
It preserves the Sun Microsystems fdlibm notice and is distributed under
LGPL-2.1-or-later. See `LICENSE.LGPL-2.1`.
`plumeswift/numerics/linux_float_power.py` adapts the log2 and exp2 tables and
polynomials from the same glibc release for the wake exponent and is also
distributed under LGPL-2.1-or-later.
`plumeswift/numerics/linux_float_transcendentals.py` adapts the single-precision
natural logarithm and exponential from the same release under LGPL-2.1-or-later.

The FFI benchmark uses model inputs based on FFI-RAPPORT 20/03101, Jorunn Aaneby,
Thor Gjesdal and Øyvind Albert Voie (2021), principally Appendix A for the
sensor records. The [source report](https://www.ffi.no/en/publications-archive/large-scale-leakage-of-liquid-hydrogen-lh2-tests-related-to-bunkering-and-maritime-use-of-liquid-hydrogen)
is © Norwegian Defence Research Establishment (FFI). Sensor positions,
averaging windows and measured concentrations are read locally from the
user's copy of the report. The report and measured sensor records are
outside the software distribution and its license.

Consult `port_inventory.json` and README.md for the incomplete migration
status. Component tests are not proof of full-program equivalence or LH2
experimental qualification.
