# Master-Thesis DataBlade

The repository stores validation files for different turbine blades.

```
Blades/
  <bladeName>/           # input files for each blade
    ises.databladeVALIDATION
    blade.databladeVALIDATION
    ...
  <bladeName>/results/   # simulation outputs, organized per run
```

`DataBladeAnalysis v8.py` now automatically creates a timestamped folder under the
corresponding `Blades/<bladeName>/results/` directory every time it is run. All
meshes, configuration files and SU2 results are saved in this run directory.

Each new result folder also contains a `rerun.py` script with all the
parameters used for that run. Execute it with `--mode rerun` to repeat the full
simulation or with `--mode replot` to regenerate the plots from the existing
output files.

## Running analysis

To analyze a single blade, pass its name with the `--blade` option:

```
python analysis_datablade.py --blade Blade_1
```

The results will be stored in a timestamped subfolder of `Blades/Blade_1/results/` along with a `rerun.py` script for reproducing the run.

Once the bulk option is implemented, multiple blades can be processed in one command:

```
python analysis_datablade.py --blades Blade_1 Blade_2 Blade_3
```

A separate results directory and `rerun.py` file will be created for each specified blade.