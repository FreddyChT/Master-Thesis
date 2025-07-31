# Master-Thesis DataBlade

This repository contains Python utilities and validation data for running
two-dimensional turbine blade simulations with [SU2](https://su2code.github.io/).

```
Blades/
  <bladeName>/           # input files for each blade
    ises.databladeVALIDATION
    blade.databladeVALIDATION
    ...
  <bladeName>/results/   # simulation outputs, organized per run
```

## How the tool works

`analysis_datablade.py` drives a complete CFD run:

1. `mesh_datablade.py` builds a gmsh mesh from the blade geometry.
2. `configSU2_datablade.py` prepares an SU2 configuration file.
3. SU2 is executed and its logs are captured.
4. `post_processing_datablade.py` extracts performance metrics and plots.

Each execution creates a timestamped folder under
`Blades/<bladeName>/results/` containing the mesh, configuration, SU2 outputs
and a `rerun.py` script. Invoke `rerun.py --mode rerun` to repeat the full
simulation or `rerun.py --mode replot` to regenerate plots from existing data.

## Available routines

- `analysis_datablade.py` – run the full pipeline for a single blade.
- `analysis_incidence.py` – sweep angle of attack for a blade.
- `analysis_mesh.py` – compute mesh quality metrics (gmsh optional).
- `analysis_turbulence.py` – compare turbulence model behaviour.
- `analysis_spleen.py` – prototype research analysis.
- `mesh_datablade.py` – generate gmsh meshes for blades.
- `configSU2_datablade.py` – create SU2 configuration files.
- `post_processing_datablade.py` – parse SU2 results and produce plots.
- `liveParaview_datablade.py` – stream solution fields into ParaView.
- `report_datablade.py` – gather summaries from previous runs.
- `utils.py` – geometry and data-processing helpers.

## Running analysis

To analyze a single blade, pass its name with the `--blade` option:

```
python analysis_datablade.py --blade Blade_1
```

Results are stored in `Blades/Blade_1/results/Test_<num>_<date>/` along with a
`rerun.py` script for reproducing the run. Multiple blades can be processed at
once:

```
python analysis_datablade.py --blades Blade_1 Blade_2 Blade_3
```

Each blade receives its own results directory and `rerun.py` file.

## Generating run reports

Run `report_datablade.py` to collate information from previous SU2 runs. The
script prompts for the date string (e.g. `03-07-2025`) and test number, scans
each `Blades/*/results/Test_<num>_<date>` directory and extracts the mesh
quality lines and the tail of the Performance Summary from `su2.log` or
`run_summary.txt`.

Outputs are written to `reports/<date>_Test_<num>/` next to the script. The
folder contains a text summary named `<date>_Test_<num>_report.txt` plus bar
plots of convergence time, iteration count, mesh size and mesh quality metrics
(minimum orthogonality angle, maximum CV face area aspect ratio and maximum CV
sub-volume ratio). Bars are colored red when a simulation failed to converge.

## Environment setup

Create a new conda environment and install the required packages:

```
conda create -n datablade python=3.10
conda activate datablade
```

| Package          | Installation command                                           |
|------------------|----------------------------------------------------------------|
| numpy            | `conda install conda-forge::numpy`                             |
| pandas           | `conda install conda-forge::pandas`                            |
| matplotlib       | `conda install conda-forge::matplotlib`                        |
| scipy            | `conda install conda-forge::scipy`                             |
| gmsh             | `conda install conda-forge::gmsh=4.12.2`<br>`pip install gmsh` |
| paraview         | `conda install paraview=5.12.0`                                |
| pythonocc-core   | `conda install conda-forge::pythonocc-core`                    |
| latex (texlive)  | `conda install conda-forge::texlive-core`                      |
| openpyxl         | `pip install openpyxl`                                         |

Ensure that the [SU2](https://su2code.github.io/) executable is available in
your `PATH` to run the simulations.