# PlumeSwift 0.1.0a5 research release

This alpha adds optional liquid hydrogen physics, measured weather inputs and research modules for stability, fence wakes and moist ammonia. The FLADIS campaign closes with the existing model defaults. The new comparisons characterize where the model is sensitive to these options and where its errors limit use in a digital twin.

## Added modules and interfaces

| Area | Modules | Role and interface |
|---|---|---|
| Cryogenic LH₂ source and transport | `lh2_case`, `lh2_jet`, `lh2_pool`, `lh2_energy`, `numerics.compiled_lh2` | Use nozzle, ambient and optional substrate inputs to estimate flash, jet handoff or pool evaporation. Carry specific enthalpy with the particles to estimate vapor fraction, temperature and density. The FFI CLI exposes `--physics legacy\|jet\|pool\|dense`. |
| Dense cloud and heat exchange | `lh2_dense`, `lh2_ground_heat`, `lh2_entrainment`, `lh2_plume_forces`, `lh2` | Optional shallow-layer pressure, ground heating, ambient exchange and plume-section forces. Handoff and carrier-air options track mixture inertia separately from released hydrogen mass. |
| Measured weather | `ffi_weather`, `ffi_lh2` | Read local wind records for time-windowed forcing and independent case comparisons. The FFI command still takes a locally supplied copy of the official report for receptor observations. |
| Atmospheric stability and barriers | `surface_layer`, `fence_wake` | Supply surface-layer Reynolds stresses and dissipation from meteorological inputs, or a solid-fence wake and shear contribution from geometry and wind inputs. |
| Ammonia research | `thermodynamics.ammonia_moist`, `thermodynamics.particle_density`, `lh2`, `particles` | Evaluate moist NH₃/air temperature, density and gas volume fraction from concentration and enthalpy. The benchmark can opt into leave-one-out force density, alternative vertical drift updates or mixture inertia tracking. These are Python research interfaces and benchmark configurations. |

See the [user guide](USER_GUIDE.md) for command examples and input fields. The default FFI setting remains `legacy`.

## Validation since 0.1.0a4

| Dataset | Archived comparison | Interpretation |
|---|---|---|
| FFI liquid hydrogen, Trials 1–7 | Across 210 matched concentrations, the archived thermophysics study reports an MAE of 0.709 vol% for legacy, 0.547 vol% for its ground-heat-only variant and 0.706 vol% with all three tested terms enabled. At 4 vol%, ground heat alone detected 6 of 9 observed positives with one false positive. [Aggregate results](evidence/a5-ffi-thermophysics.csv) | The result depends on the physics combination and source assumptions. It does not establish an alarm threshold for a new site. |
| FFI measured wind and plume-section forces | The archived joint wind perturbation gave 7 detections out of 9 versus 8 for the axis-only rule. In the separate measured-wind plume-force comparison, MAE changed from 0.452 to 0.446 vol% and detections from 6 to 7. [Wind metrics](evidence/a5-ffi-wind.csv) · [Plume force metrics](evidence/a5-ffi-plume-forces.csv) | These are separate experiments with different controls. Small MAE differences do not establish general improvement. |
| Prairie Grass PG8 and PG17 | The stability variant changed MAE from 6.68 to 3.39 ppm for PG8 and from 11.82 to 13.92 ppm for PG17. [Matched aggregate metrics](evidence/a5-smedis-cases.csv) | A change that improves one stability case may worsen another. |
| SMEDIS propane EEC360, EEC361 and EEC362 | Weather, grid, fence shear, dense-layer transport, approach and wake wind, and variability sensitivities tested the zero, one and two barrier cases. [Fence metrics](evidence/a5-fence-wake.csv) · [Dense-layer metrics](evidence/a5-dense-wake.csv) · [Wind and variability metrics](evidence/a5-variability.csv) | Comparisons distinguish errors in plume arrival and wind direction from errors in wake mixing. They do not establish a universal barrier correction. |
| SMEDIS propane EEC550 and EEC551 | In the 30-receptor holdout, MAEs were 0.356 and 0.568 vol%. The observed upwind increase and lee-side decreases had the same signs in the model. [Holdout sensor metrics](evidence/a5-eec55-sensors.csv) · [Regional contrasts](evidence/a5-eec55-regions.csv) | Cases use their reported averaging windows and wind inputs. Concentration magnitudes still differ. |
| PRESLHY E3.5 Trial 10 | Separate holdout investigated blocked sample-line intervals, concentration time histories and 4 vol% arrival. It also tested the nozzle cooling transient using the pressure and temperature record. | The holdout was conducted independently of FFI. This release summary does not quote scores without the corresponding archived metric table. |
| FLADIS ammonia Trials 9, 16 and 24 | The [closeout report](FLADIS_VALIDATION.md) includes distance-wise errors, vertical profiles, the FLACS comparison and the decision to retain defaults. | The strongest downstream sensitivity was vertical slip decay. Neither the time-scale closure nor inferred-air momentum dilution met the cross-trial adoption criteria. |

Public evidence contains aggregate metrics and methods. Supply original field records locally from the cited sources when reproducing a benchmark. The package includes synthetic inputs and source configurations. The release does not bundle the field report PDFs, instrument workbooks or receptor-level archives.

The validation supports research scenario comparisons for conditions represented by the benchmarks. Live sensor assimilation, real-time update latency, threshold false alarm rates and site-specific emergency decisions require separate validation. See the [FLADIS digital twin applicability section](FLADIS_VALIDATION.md#digital-twin-applicability) for the ammonia limits.
