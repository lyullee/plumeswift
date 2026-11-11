# FLADIS ammonia dispersion validation closeout

The FLADIS campaign is closed for PlumeSwift. **Keep the current model defaults.** The best downstream results came from an experimental relaxation of vertical slip. It improved Trials 16 and 24 but increased the error for Trial 9. The final test diluted vertical momentum by inferred entrained air mass. It kept Trial 9 closer to its reference performance, but recovered little of the downstream improvement in Trials 16 and 24. These options remain useful for diagnosis.

## Upstream QES and ammonia evidence

QES is the upstream Quick Environmental Simulation framework, and QES-Plume is its Lagrangian dispersion component. Its published validation covers analytical passive tracer dispersion and a wind tunnel experiment that used ethane as an effectively neutrally buoyant tracer in an array of cubical buildings. The public documentation describes generic tracer and particle sources. A search of the upstream publication, public documentation, archived source and examples at local upstream commit `dced7ec`, and the JRIII ammonia submission roster found no documented upstream QES ammonia release application or ammonia field validation. This conclusion is limited to the public sources reviewed. The FLADIS results below belong to PlumeSwift and its ammonia benchmark extensions. [QES-Plume paper](https://doi.org/10.5194/gmd-16-5729-2023) · [Upstream repository](https://github.com/UtahEFD/QES-Public) · [QES-Plume input documentation](https://qes-documentation.readthedocs.io/en/latest/QES-Plume.html) · [JRIII submission roster](https://admlc.com/jriii/)

## Field data and comparison method

The benchmark uses the SMEDIS field observations and JRIII equivalent source specifications for pressure-liquefied ammonia releases in FLADIS Trials 9, 16 and 24. Each simulation starts at the prescribed fully vaporized equivalent source, a disk 1.5 m above the ground with 12 mol% NH₃. The source represents the jet after vaporization. The campaign uses flat terrain, with no resolved upstream building wakes. [SMEDIS field data](https://admlc.com/smedis-dataset/) · [JRIII modeling protocol](https://admlc.com/wp-content/uploads/2025/07/jriii-initial-modeling-exercise-description-v2.4-1.pdf) · [Benchmark method](../results/fladis_benchmark/METHOD.txt)

| Trial | NH₃ release rate (kg/s) | Equivalent source temperature (K) | Averaging window (s) | Far arc radius (m) |
|---|---:|---:|---:|---:|
| 9 | 0.40 | 203.7 | 600 | 238 |
| 16 | 0.27 | 203.9 | 600 | 240 |
| 24 | 0.46 | 204.0 | 400 | 238 |

The distance tables pair predictions with the same 153 surveyed receptors, including zero observations and the additional measurement heights. The 20, 70 and nominal 240 m groups contain 58, 31 and 64 receptors across the three trials. Model averaging starts at simulated release onset. The spreadsheets do not identify the exact start of the corresponding window in the original field time series.

**Distance error (%)** is `100 × Σ|prediction − observation| / Σ observation`, with both sums taken over the indicated receptors. The 70 m and farther summary pools all 95 receptors at 70 and nominal 240 m before calculating the ratio. Values may exceed 100%. Each cell in the distance tables lists the errors in order of **20 m / 70 m / nominal 240 m**. At an individual receptor, **signed relative error (%)** is `100 × (prediction − observation) / observation`. Positive values indicate overprediction.

## Wind, turbulence and thermodynamic trials

The early series tested ground heat exchange, a synthetic coherent wind-direction perturbation and a vertical turbulence profile scaled by the reported vertical velocity variance. Subsequent runs incorporated measured 10 m mast directions along the flow and across the far arc. The lateral interpolation also altered the mean along-flow speed through its divergence compensation. The wind perturbations represent summary variability. Time-resolved field measurements and resolved building wakes would be needed to test a site wind reconstruction. [Original benchmark](../results/fladis_benchmark/METHOD.txt) · [Mast wind method](../results/fladis_benchmark/mast_wind/METHOD.txt) · [Lateral wind method](../results/fladis_benchmark/lateral_wind/METHOD.txt)

| Configuration | Trial 9 (%) | Trial 16 (%) | Trial 24 (%) | Pooled 70 m and farther (%) |
|---|---:|---:|---:|---:|
| Original dry equivalent-vapor benchmark, 120 particles/s | 70.7 / 301.0 / 529.2 | 56.9 / 341.4 / 410.7 | 42.1 / 197.4 / 489.9 | 278.8 |
| Early sensitivity control, 40 particles/s | 91.1 / 460.9 / 555.6 | 65.4 / 447.9 / 422.9 | 49.1 / 274.2 / 541.6 | 371.1 |
| Ground heat exchange only | 93.0 / 359.6 / 410.0 | 68.4 / 311.1 / 218.1 | 49.1 / 206.6 / 316.2 | 267.8 |
| Synthetic wind direction variation only | 88.3 / 451.6 / 540.4 | 59.5 / 431.7 / 409.4 | 47.5 / 268.7 / 546.8 | 362.6 |
| Measured vertical variance scaling only | 46.3 / 265.9 / 532.6 | 72.3 / 488.2 / 402.8 | 77.7 / 344.9 / 643.0 | 379.0 |
| Ground heat, wind variation and vertical variance, 40 particles/s | 49.2 / 206.5 / 333.2 | 69.3 / 341.5 / 205.5 | 80.0 / 284.9 / 275.7 | 278.1 |
| Ground heat, wind variation and vertical variance, 120 particles/s | 41.9 / 99.6 / 211.0 | 62.1 / 282.0 / 236.3 | 61.5 / 251.6 / 352.8 | 230.6 |
| Along-flow interpolation of mast directions | 42.2 / 103.5 / 220.9 | 68.7 / 278.9 / 162.3 | 61.6 / 251.7 / 347.9 | 227.2 |
| Along-flow and lateral interpolation of mast directions | 44.5 / 100.2 / 215.4 | 69.0 / 287.6 / 166.6 | 62.7 / 248.4 / 385.0 | 228.2 |

The thermodynamic series tested stochastic midpoint integration, humid air with water condensation, freezing and re-evaporation, and a leave-one-out estimate for force density. The latter removes one average marker's contribution when estimating buoyancy and pressure forces. Receptor concentrations and the mass and energy ledgers still include all markers. At 120 markers/s, a single marker in a 1 m³ force cell represents roughly 3,100–5,300 ppmv of instantaneous concentration, so its contribution to the force can matter even at low time-averaged receptor concentrations. [Thermodynamic method and source conventions](../results/fladis_benchmark/mixing_thermo/METHOD.txt)

The fixed-enthalpy humidity tests carry over the dry source energy and yield equilibrium source temperatures of roughly 216–220 K. The fixed-temperature cases recalculate source enthalpy to preserve the prescribed temperature and 12 mol% NH₃. The **diagnostic reference** for the later studies is the lateral-wind case with the moist, fixed-temperature source, midpoint integration and leave-one-out force density. It is a benchmark configuration with experimental options. The original equivalent-vapor benchmark and the diagnostic reference are both listed so readers can distinguish the starting point of the campaign from the control for the later comparisons.

| Configuration | Trial 9 (%) | Trial 16 (%) | Trial 24 (%) | Pooled 70 m and farther (%) |
|---|---:|---:|---:|---:|
| Dry lateral-wind control | 44.5 / 100.2 / 215.4 | 69.0 / 287.6 / 166.6 | 62.7 / 248.4 / 385.0 | 228.2 |
| Stochastic midpoint only | 42.0 / 95.8 / 223.3 | 61.3 / 281.2 / 190.4 | 55.6 / 255.1 / 430.9 | 232.8 |
| Humid air, fixed dry source enthalpy | 34.7 / 79.2 / 232.7 | 69.8 / 307.1 / 175.6 | 64.1 / 266.2 / 408.6 | 238.7 |
| Humid air and midpoint, fixed dry source enthalpy | 35.6 / 72.1 / 234.0 | 61.3 / 298.6 / 197.0 | 55.2 / 269.7 / 442.2 | 240.0 |
| Humid air and midpoint, fixed source temperature | 39.9 / 99.7 / 279.2 | 66.2 / 328.9 / 211.7 | 57.3 / 280.6 / 461.0 | 259.4 |
| Diagnostic reference with leave-one-out force density | 37.1 / 31.3 / 24.6 | 56.2 / 185.7 / 86.3 | 49.5 / 177.9 / 272.9 | 145.7 |

## Buoyancy and momentum diagnostics

Each later test used the diagnostic reference's wind, source, receptor configuration and averaging windows. The reference uses 120 particles/s, a 0.25 s main time step with adaptive substeps, 2 × 2 × 0.25 m cells and a ground heat-transfer coefficient of 20 W/(m² K). All three trials received the same treatment in each comparison. No coefficients were fitted to measured receptor concentrations.

| Configuration | Trial 9 (%) | Trial 16 (%) | Trial 24 (%) | Pooled 70 m and farther (%) |
|---|---:|---:|---:|---:|
| Diagnostic reference with leave-one-out force density | 37.1 / 31.3 / 24.6 | 56.2 / 185.7 / 86.3 | 49.5 / 177.9 / 272.9 | 145.7 |
| Vertical buoyancy removed | 57.4 / 46.1 / 40.3 | 74.1 / 41.0 / 25.6 | 80.3 / 54.7 / 83.8 | 49.7 |
| 240 particles/s | 37.0 / 30.4 / 26.3 | 58.1 / 191.0 / 93.1 | 50.0 / 192.3 / 295.6 | 154.9 |
| 0.50 m buoyancy density layers | 37.0 / 29.5 / 26.5 | 56.3 / 193.3 / 86.3 | 48.3 / 187.0 / 281.4 | 151.7 |
| Linearized force density | 40.8 / 34.1 / 27.6 | 63.7 / 180.2 / 84.2 | 59.2 / 172.4 / 275.2 | 142.6 |
| Vertical slip relaxation | 46.1 / 41.9 / 36.0 | 55.3 / 44.1 / 24.8 | 35.9 / 58.4 / 112.8 | 52.3 |
| Linearized force density and slip relaxation | 41.8 / 40.4 / 33.3 | 55.6 / 42.9 / 27.3 | 29.3 / 53.6 / 113.0 | 49.5 |
| Momentum dilution by inferred air mass | 43.3 / 31.8 / 26.8 | 49.1 / 169.2 / 85.0 | 36.3 / 173.1 / 266.9 | 140.0 |

- **Vertical buoyancy removal.** The vertical buoyancy acceleration was set to zero. Horizontal pressure, drag and ground heating were still applied. The downstream error fell from 145.7% to 49.7%, while the 20 m error increased from 48.1% to 72.5% across the three trials. Trial 9 also worsened downstream. The test shows strong sensitivity to vertical buoyancy in Trials 16 and 24. [Force study metrics](../results/fladis_benchmark/force_study/metrics.csv)
- **Particle count and buoyancy layers.** Increasing the rate to 240 particles/s or pooling pairs of vertical layers for the buoyancy density estimate gave pooled downstream errors of 154.9% and 151.7%, respectively. These two changes provide a sensitivity check at one alternative resolution each. [Force study run checks](../results/fladis_benchmark/force_study/run_checks.csv)
- **Linearized force density.** Linearizing the moist equation of state about ambient air gave 142.6% downstream error. The calculated density anomaly differed by about 0.03% at 1,000 ppmv in the specified dilute-state diagnostic. That test isolates equation-of-state nonlinearity within the existing thermodynamic closure. [Density audit](../results/fladis_benchmark/drift_study/density_audit.csv)
- **Vertical slip relaxation.** The original quadratic drag can preserve downward slip after the density anomaly weakens. With a 1 m drag length, a drag coefficient of 0.5 and only 0.1 m/s of downward slip, the zero-force half-life is about 40 s. The experimental update uses the local integral correlation time from the existing turbulence model. It reduced pooled downstream error to 52.3%, or to 49.5% when combined with the density linearization. Trial 9 worsened under both changes. The relaxation of mean slip is an experimental closure based on the turbulence time scale. [Drift study metrics](../results/fladis_benchmark/drift_study/metrics.csv) · [QES-Plume turbulence formulation](https://doi.org/10.5194/gmd-16-5729-2023)
- **Dilution of vertical momentum by inferred air mass.** Each marker starts with the mixture mass inferred from its first occupied cell. Subsequent increases in inferred mixture mass per unit tracer represent entrainment of air at zero relative vertical velocity. Existing vertical momentum is distributed over the increased mass. The local estimate provides an effective inertia and may retain the effect of sparse-cell sampling. Pooled downstream error reached 140.0%. Trial 9 remained close to its reference downstream error, but Trials 16 and 24 retained only a small part of the improvement seen with time-scale relaxation. [Momentum dilution metrics](../results/fladis_benchmark/entrainment_study/metrics.csv)

## Vertical profiles and external comparison

The 70 m center tower separates a surface concentration error from a vertical distribution error. In Trial 24, slip relaxation changed the signed error from +208.6% to +30.4% at 0.1 m, and from −88.6% to −61.4% at 9 m. In Trial 16, it changed the 4 m error from −8.6% to +132.0%. The final momentum-dilution test left the Trial 24 errors at +212.3% and −86.9% at 0.1 and 9 m. All four heights are recorded in the archived tables. `results/fladis_benchmark/force_study/vertical_profiles.csv` (local archive) · `results/fladis_benchmark/drift_study/vertical_profiles.csv` (local archive) · `results/fladis_benchmark/entrainment_study/vertical_profiles.csv` (local archive)

The public FLACS comparator is Gexcon's provided-source CERC submission under Pasquill D. The table below uses signed relative errors against the published SMEDIS arc maximum for both the diagnostic reference scan and FLACS. These are peak statistics, distinct from the paired-receptor metric above. At 70 m the reported observation can come from another height than the 0.5 m model scan. The public FLACS submission provides arc maxima, so the comparison is limited to the reported peak statistic. [JRIII submissions](https://admlc.com/jriii/) · `results/fladis_benchmark/mixing_thermo/arc_comparison.csv` (local archive)

| Trial | Distance | Diagnostic reference | FLACS D |
|---|---:|---:|---:|
| 9 | 20 m | -16.7% | +37.4% |
| 9 | 70 m | -8.7% | +56.5% |
| 9 | 240 m | -13.5% | +125.2% |
| 16 | 20 m | +87.1% | -14.9% |
| 16 | 70 m | +413.2% | +22.1% |
| 16 | 240 m | +195.6% | -8.7% |
| 24 | 20 m | +67.5% | -24.2% |
| 24 | 70 m | +283.3% | +3.2% |
| 24 | 240 m | +735.1% | +248.9% |

## Digital twin applicability

This evidence supports retrospective scenario screening and comparative plume visualization for releases similar to these three equivalent-vapor cases. It establishes performance for time-averaged receptor concentrations in the tested geometry and weather conditions. The campaign did not evaluate live sensor assimilation, time-to-detection, transient peak tracking, false alarm rates or end-to-end update latency. Runtime logs reflect concurrently executed jobs and do not establish a real-time service requirement.

Quantitative alarm thresholds, exposure estimates and emergency decisions require separate validation for the deployment site and its operating conditions. Trial 24 still had a 240 m error of 112.8% even with the experimental slip relaxation. The model also uses an equivalent vapor source and a water-only phase change option. Flashing-jet development, ammonia dissolution and heat of solution, aerosol transport and deposition, and site-specific building wakes lie outside this validation. Grid and particle sensitivities used a single alternative setting and a fixed seed, so the observed errors do not define a statistical uncertainty interval or a convergence bound.

The closeout decision retains the current production defaults, including vertical buoyancy and the existing drag update. Leave-one-out density, alternate force density, time-scale relaxation and inferred-air momentum dilution were evaluated as optional benchmark configurations. Their results and operating assumptions remain available for future validation.

## Reproducibility and archived evidence

The benchmark inputs and environment setup are listed in the [base method](../results/fladis_benchmark/METHOD.txt) and [thermodynamic method](../results/fladis_benchmark/mixing_thermo/METHOD.txt). From the repository root, the final controlled studies use [the force runner](../results/fladis_benchmark/fladis_force_study.py), [the drift runner](../results/fladis_benchmark/fladis_drift_study.py) and [the momentum dilution runner](../results/fladis_benchmark/fladis_entrainment_study.py). The [force comparator](../results/fladis_benchmark/compare_fladis_force.py) and [drift comparator](../results/fladis_benchmark/compare_fladis_drift.py) regenerate the paired metrics from completed runs. The result paths in this document refer to the local benchmark archive. The tables retain the principal numerical evidence for copies of the document distributed without that archive.
