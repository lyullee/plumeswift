# PlumeSwift user guide

This guide uses PlumeSwift `0.1.0a4` with Python 3.11 or newer. It covers three workflows: a customizable liquid hydrogen (LH₂) demo, a QES-style wind and passive dispersion case, and the FFI field benchmark. For the project overview and available install extras, see the [README](../README.md).

- [Set up and check the installation](#1-set-up-and-check-the-installation)
- [Create and edit an LH₂ case](#2-create-and-edit-an-lh2-case)
- [Write wind and plume XML](#3-write-wind-and-plume-xml)
- [Run and interpret the FFI benchmark](#4-run-and-interpret-the-ffi-benchmark)
- [Read results with Python](#5-read-results-with-python)
- [Troubleshoot and reproduce a run](#6-troubleshoot-and-reproduce-a-run)

## 1. Set up and check the installation

Run `python -m venv .venv` in your working directory. Activate the environment with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS or Linux. Install the release and the extras used here.

```shell
python -m pip install "plumeswift[fast,visual,ffi]==0.1.0a4"
plumeswift info
plumeswift plume-managed-container --steps 20 --release-per-step 16 --seed 0
```

Check that `info` reports version `0.1.0a4` and that the smoke test prints `"invariants_passed": true`. The CPU examples require no GPU. The `fast` extra enables optional Numba acceleration in supported workflows, `visual` enables LH₂ 3D output, and `ffi` enables local PDF parsing. If the shell cannot find `plumeswift`, use `python -m plumeswift` with the same interpreter that installed the package.

Each example writes into a named directory. Use a fresh output directory for each parameter change so its results remain traceable to its inputs.

## 2. Create and edit an LH2 case

First run the bundled example, then export its settings for editing.

```shell
plumeswift lh2-demo --output lh2-reference --open
plumeswift lh2-demo --write-input lh2-case.json
```

The first command writes a reference run. The second only writes the JSON input. To try a lower release rate, change `"release_rate_kg_s": 0.35` to `"release_rate_kg_s": 0.25` in `lh2-case.json`, then rerun.

```shell
plumeswift lh2-demo --input lh2-case.json --output lh2-low-rate --open
```

The [bundled JSON](../python_qes/plumeswift/example_data/lh2.json) contains every field required by this command. Keep the full structure when editing. JSON keys use double quotes and JSON has no comments or trailing commas.

| Field | Default | How to set it |
| --- | --- | --- |
| `grid_cells`, `cell_size_m` | `[48, 28, 20]`, `[2, 2, 1]` | Counts and spacings in x, y, z order. The demo requires at least 3 × 3 × 2 cells. Use a domain that covers the region you plan to inspect. |
| `wind_speed_m_s`, `wind_from_deg` | `4`, `270` | Speed at the reference height and a meteorological bearing. A bearing of 270° is wind from the west, moving east. |
| `wind_reference_height_m`, `roughness_length_m` | `10`, `0.02` | Reference wind measurement height and aerodynamic roughness length, both in metres. |
| `building` | `x_m=34`, `y_m=25`, `length_m=8`, `width_m=6`, `height_m=5` | Define the rectangular obstacle by its lower x/y corner and dimensions in metres. |
| `source_position_m` | `[16, 28, 0.5]` | Release coordinates in x, y, z order. Place the source in a fluid cell strictly inside the simulation boundaries. |
| `release_rate_kg_s`, `particles_per_s` | `0.35`, `100` | Physical hydrogen mass flow and numerical particle rate. Change the numerical rate independently when checking sampling sensitivity. |
| `ambient_temperature_k`, `liquid_mass_fraction`, `relative_humidity_pct` | `293.15`, `0.6`, `50` | Temperature in kelvin, liquid mass fraction on a 0–1 scale, and relative humidity on a 0–100 percent scale. |
| `duration_s`, `average_from_s`, `step_s` | `24`, `8`, `0.2` | Set a positive step and an averaging start from zero up to, but before, the end of the run. |
| `seed`, `isosurfaces_volpct` | `20261107`, `[0.5, 2, 4]` | Use a fixed integer seed for repeat comparisons. Specify three distinct positive contour levels at or below 100 vol%. |

The input grid is specified in x, y, z order, and the exported concentration array is indexed `[z, y, x]`. Coordinates are local. x points east, y points north, and z is height above the flat ground in this example.

| Output | What to check |
| --- | --- |
| `case.json` | Compare the settings actually used in the reference and modified runs. |
| `concentration.npz` | `h2_mean_volpct` stores gas-phase hydrogen volume concentration averaged over the requested window. `solid` identifies solid cells, whose concentrations are masked with `NaN`. The `coord_x`, `coord_y`, and `coord_z` arrays give cell centres in metres. |
| `plume.html` | Drag to rotate, scroll to zoom, and toggle levels in the legend. This standalone file works offline. For headless runs, omit `--open` and open the file later. |
| JSON printed to the terminal | Check the averaged duration, peak of the mean field, rendered isosurface levels and mass accounting. |

For the default case, the averaging duration is 16 seconds. A constant 0.35 kg/s release over 24 seconds supplies 8.4 kg. The lower-rate case supplies 6.0 kg. Compare `released_mass_kg` with this input calculation and inspect `mass_balance_error_kg` for the numerical residual. `escaped_mass_kg` is mass transported out of the modeled domain. A contour appears where the gridded field crosses its level. The plotted concentration is a time average, so an isosurface represents that average over the selected interval.

## 3. Write wind and plume XML

For a custom wind and passive dispersion case, create the two files below in the same working directory. This example uses one stationary wind profile and a rectangular obstacle. It uses the QES-style `wind` and `plume` commands. Use the previous section for the demo's LH₂ equilibrium and momentum coupling.

Save as `winds.xml`.

```xml
<QESWindsParameters>
  <simulationParameters>
    <domain>40 24 16</domain>
    <cellSize>2 2 1</cellSize>
    <totalTimeIncrements>1</totalTimeIncrements>
    <maxIterations>3000</maxIterations>
    <tolerance>1e-6</tolerance>
    <meshTypeFlag>0</meshTypeFlag>
  </simulationParameters>
  <metParams>
    <sensor>
      <site_coord_flag>1</site_coord_flag>
      <site_xcoord>8</site_xcoord>
      <site_ycoord>8</site_ycoord>
      <timeSeries>
        <timeStamp>2020-01-01T01:00:00</timeStamp>
        <boundaryLayerFlag>1</boundaryLayerFlag>
        <siteZ0>0.02</siteZ0>
        <height>10</height>
        <speed>4</speed>
        <direction>270</direction>
      </timeSeries>
    </sensor>
  </metParams>
  <turbParams>
    <method>1</method>
  </turbParams>
  <buildingsParams>
    <numBuildings>1</numBuildings>
    <wakeFlag>2</wakeFlag>
    <rectangularBuilding>
      <xStart>34</xStart>
      <yStart>20</yStart>
      <length>8</length>
      <width>6</width>
      <height>5</height>
      <baseHeight>0</baseHeight>
      <buildingRotation>0</buildingRotation>
    </rectangularBuilding>
  </buildingsParams>
</QESWindsParameters>
```

`domain` is the number of cells per axis, and `cellSize` is their spacing in metres. The sensor uses local coordinates and supplies a 4 m/s wind at 10 m height. Set the wind direction as the bearing **from** which it blows. Use the same local origin for obstacles and sources. Here the obstacle occupies x = 34–42 m and y = 20–26 m before any grid representation.

Save as `plume.xml`.

```xml
<QESPlumeParameters>
  <plumeParameters>
    <simDur>20</simDur>
    <timeStep>0.2</timeStep>
    <CourantNumber>0.25</CourantNumber>
    <invarianceTol>1e-8</invarianceTol>
  </plumeParameters>
  <collectionParameters>
    <timeAvgStart>0</timeAvgStart>
    <timeAvgFreq>10</timeAvgFreq>
    <boxBoundsX1>2</boxBoundsX1><boxBoundsX2>78</boxBoundsX2><nBoxesX>38</nBoxesX>
    <boxBoundsY1>2</boxBoundsY1><boxBoundsY2>46</boxBoundsY2><nBoxesY>22</nBoxesY>
    <boxBoundsZ1>0</boxBoundsZ1><boxBoundsZ2>14</boxBoundsZ2><nBoxesZ>14</nBoxesZ>
  </collectionParameters>
  <sourceParameters>
    <source>
      <releaseType_continuous>
        <particlePerTimestep>20</particlePerTimestep>
        <massPerSec>350</massPerSec>
      </releaseType_continuous>
      <sourceGeometry_Point>
        <posX>16</posX><posY>23</posY><posZ>1.5</posZ>
      </sourceGeometry_Point>
    </source>
  </sourceParameters>
  <boundaryConditions>
    <xBCtype>exiting</xBCtype>
    <yBCtype>exiting</yBCtype>
    <zBCtype>reflection</zBCtype>
    <wallReflection>stairstepReflection</wallReflection>
  </boundaryConditions>
</QESPlumeParameters>
```

The continuous source emits 20 numerical particles per time step. Legacy plume XML uses **g/s** for `massPerSec`, so `350` means 0.35 kg/s. The LH₂ JSON field `release_rate_kg_s` instead uses **kg/s**. Set only one mass input for the continuous source. A `timeAvgStart` of 0 and `timeAvgFreq` of 10 seconds request two collection windows across this 20-second run. The collection box bounds and counts define the output sampling grid separately from the wind grid.

```shell
plumeswift wind winds.xml --output wind-results
plumeswift plume winds.xml plume.xml --output plume-results
```

Inspect the wind command's JSON for solver convergence and filenames. The default wind output includes `winds.nc` for the staggered workspace, `turbulence.nc` for turbulence fields, and `windsOut.nc` for visualization. To reuse the saved workspace with its matching turbulence file, run the following.

```shell
plumeswift plume wind-results/winds.nc plume.xml --turbulence-nc wind-results/turbulence.nc --output plume-replay
```

The plume command reports particle counts and mass accounting by model. This input releases 7 kg in 20 seconds and writes two records to `plume-results/base_plumeOut.nc`, ending at 10 and 20 seconds. `t_avg` gives the averaging duration for each record. `c` is mass concentration in **g/m³**, indexed by time and then z, y, x. Divide by 1000 for kg/m³. Converting mass concentration to a gas volume concentration requires the relevant physical model and conditions.

Read the last averaging window from the working directory that contains `plume-results`.

```python
from netCDF4 import Dataset

with Dataset("plume-results/base_plumeOut.nc") as dataset:
    end_s = float(dataset["t"][-1])
    duration_s = float(dataset["t_avg"][-1])
    concentration = dataset["c"][-1]
    print(f"Window: {end_s - duration_s:.1f}–{end_s:.1f} s")
    print(f"Peak mean concentration: {float(concentration.max()):.4f} g/m³")
```

## 4. Run and interpret the FFI benchmark

Download the full FFI-RAPPORT 20/03101 publication, including Appendix A, and save it as `ffi-report.pdf`. PlumeSwift extracts the measured averages from your copy of the PDF. [Official FFI report page](https://www.ffi.no/en/publications-archive/large-scale-leakage-of-liquid-hydrogen-lh2-tests-related-to-bunkering-and-maritime-use-of-liquid-hydrogen)

Start with one test, then run all seven. Use a quoted absolute path after `--report` if the PDF is stored elsewhere.

```shell
plumeswift lh2-ffi --report ffi-report.pdf --tests 3 --output ffi-test3 --open
plumeswift lh2-ffi --report ffi-report.pdf --workers 5 --output ffi-all --open
plumeswift lh2-ffi --report ffi-report.pdf --tests 1 3 --no-3d --output ffi-sensors
```

Without `--workers`, cases run sequentially. `--workers 5` allows up to five independent case processes. `--no-3d` skips concentration grid and HTML plume generation while retaining the sensor reports and the index page. First-run Numba compilation and rendering can affect wall-clock timings.

Open `index.html` for the summary and links to each test. The top-level `sensor_comparison.csv` combines all selected cases. A directory such as `test3` holds its own CSV, `summary.json`, `case.json`, and generated `winds.xml`, as well as `plume.html` and `concentration.npz` when 3D output is enabled.

- The CSV records observed and predicted averages in vol%, and `error_pp` is predicted minus observed, in percentage points. For example, 0.75 vol% predicted against 1.00 vol% observed gives −0.25 percentage points.
- Use `observed_start_s` and `observed_end_s` for the reported measurement window, and `model_start_s` and `model_end_s` for the model window. The benchmark matches their durations after a 100-second model spin-up. Their absolute start times can differ.
- Compare measurements against the **sensor predictions**. They are calculated over 2 × 2 × 0.2 m boxes centred on the instruments. The 3D isosurfaces use a separate gridded average. The FFI grid and CSV use x and y relative to the release point, with z as height above ground.
- In `summary.json`, RMSE gives greater weight to large errors than MAE does. Bias is the signed mean error. `measured_at_least_4` counts observed averages of at least 4 vol%, and `detected_at_least_4` counts those same sensors where the prediction also reaches 4 vol%.

The documented run reports 210 sensor averages, RMSE of approximately **1.9843 percentage points**, and a bias of about **−0.7048 percentage points**. The model reached 4 vol% at two of the nine sensors whose observed averages reached that level. The benchmark uses prescribed wind on an open site, and FFI derived the reported concentrations from oxygen depletion. Keep the full PDF and locally generated files that contain observations separate from distributable software artifacts. [Benchmark protocol](LH2_direct_benchmark.md) · [Validation evidence](evidence/ffi5_comparison.json)

## 5. Read results with Python

This example reads the modified demo from section 2, selects the grid layer nearest 1.5 m and prints the mean concentration at a specified point. Save it as `inspect_lh2.py` and run `python inspect_lh2.py` from the same working directory.

```python
import numpy as np

with np.load("lh2-low-rate/concentration.npz") as result:
    x, y, z = (result[f"coord_{axis}"] for axis in "xyz")
    k = int(np.abs(z - 1.5).argmin())
    j = int(np.abs(y - 28.0).argmin())
    i = int(np.abs(x - 24.0).argmin())
    plane = result["h2_mean_volpct"][k]
    if result["solid"][k, j, i]:
        raise ValueError("The requested cell is solid")
    print(f"Cell centre: ({x[i]:.2f}, {y[j]:.2f}, {z[k]:.2f}) m")
    print(f"Cell mean: {plane[j, i]:.4f} vol%")
    print(f"Peak on this layer: {np.nanmax(plane):.4f} vol%")
```

This is a nearest-cell query of the mean grid. For a receptor time average in a custom Python workflow, `SensorBoxes(centres, widths)` defines the centres and x/y/z widths in metres. `TimeAveragedSensors(boxes, start_s, end_s, thermo)` weights samples by their overlap with the requested time window. With a thermodynamic model, its `mean()` returns gas volume fraction on a 0–1 scale. Multiply by 100 for vol%.

You can also inspect the LH₂ coupling for a single particle. This example passes positions in metres, masses in kilograms, drift in m/s, cell IDs, a cell volume in m³ and a time step in seconds.

```python
import numpy as np
from plumeswift.lh2 import LH2Momentum
from plumeswift.thermo import LegacyEquilibrium

equilibrium = LegacyEquilibrium(
    ambient_temperature=276.85, liquid_fraction=0.559, relative_humidity=75.0
)
force = LH2Momentum(equilibrium)
next_drift, diagnostics = force.update(
    np.array([[0.0, 0.0, 0.5]]),
    np.array([0.005]),
    np.zeros((1, 3)),
    np.array([0]),
    2.0,
    0.2,
)
print(next_drift, diagnostics)
```

The coupling sums particle mass per cell and divides by cell volume before evaluating mixture properties. The returned drift is the component added to the mean wind in the particle coupling. See the [module index](MODULE_REFERENCE.md) for API signatures and the [README's API example](../README.md#use-the-python-api) to run a full case from Python.

## 6. Troubleshoot and reproduce a run

| Symptom | What to check |
| --- | --- |
| PDF parsing fails | Use the complete official FFI PDF, check the path, and install the `ffi` extra into the environment that runs the command. |
| The browser does not open | Open `plume.html` or `index.html` directly. On a remote or headless machine, omit `--open` and copy the relevant output directory to a machine with a browser. |
| A requested isosurface is missing | Check the reported rendered levels and the mean concentration range in the NPZ file. A level must intersect the modeled field for a surface to appear. |
| Wind does not converge or the source is rejected | Check the grid and geometry, locate the source in the fluid interior, and inspect the solver residual before interpreting plume output. |
| Concentration seems inconsistent across outputs | Check `vol%` versus `g/m³` or `kg/m³`, array axis order, origin and averaging window. Compare the same quantity at the same sampling location. |

Record the package version, the exact input files, the seed, and any changed solver, acceleration or output options. When testing sensitivity, change one setting at a time and compare the same averaging windows and coordinates.

For older research workflows that compare external models or run case-level and particle-level timing studies, see the [development notes](DEVELOPMENT.md) and [FFI protocol](LH2_direct_benchmark.md). These workflows use additional reference files and have different output and timing scopes from the packaged FFI command.

The [component status](../python_qes/component_status.json) identifies available and partially verified QES paths. Separate validation is needed for applications that depend on flashing, independent liquid droplets or pools, ground heat transfer, wind feedback from gas density, ignition, or safety distances. Cite the version, the relevant QES methods, and the experimental data used. [Citation metadata](../CITATION.cff) · [License and third-party notices](../python_qes/NOTICE.md)
