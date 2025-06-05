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

Run the script with:

```bash
python DataBladeAnalysis\ v8.py --blade <name> --cores <n>
```

where `<name>` is the blade directory and `<n>` the number of MPI cores for SU2.
Post-processing helpers live in `datablade_post.py`.
