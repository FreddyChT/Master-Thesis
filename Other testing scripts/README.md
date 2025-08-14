# Other Testing Scripts

In the spirit of thoroughness, this folder contains all other routines used for 
the experimental and sensitivity validation carried out in the development of 
this tool. For the curious user, in order for these routines to run, they 
must be moved out of this folder into the main folder that contains all other scripts.

The scripts that contain `_spleen` are simply alternative routines that perform the
same operations as their `_datablade` counterparts but leverage the SPLEEN database.

Furthermore, the following scripts perform specific studies aside to the main blade
simulation scripts:

1. `analysis_turbulence` performs a turbulence sensitivity analysis by prompting 
the user with a TI range and step increase
2. `analysis_incidence` performs an incidence sensitivity analysis by prompting
the user with an inlet angle range and step increase
3. `analysis_mesh` performs a mesh sensitivity analysis on the aerodynamic coefficients
comparing user-determined target mesh sizes and outputs the GCI index of the 
three finest meshes. The meshes for the GCI computation can also be user-determined.
4. `model_comparison` performs a comparison between two turbulence and transition 
models, plotting the results into a single plot.
5. `analysis_parametric_spleen` performs a parametric-type study of the identified
meshing "knobs" such that their influence can be determined on a specific model.  
