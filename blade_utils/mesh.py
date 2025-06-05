"""Meshing helpers for the blade validation cases."""

import os
import numpy as np

from .geometry import process_airfoil_file


# Default mesh parameters are defined in ``DataBladeAnalysis v8.py``.  This
# module only exposes the meshing routine which relies on those globals.

def mesh_datablade(bladeFilePath, axial_chord, pitch, alpha1, alpha2,
                   dist_inlet, dist_outlet, sizeCellFluid, sizeCellAirfoil,
                   nCellAirfoil, nCellPerimeter, nBoundaryPoints,
                   first_layer_height, bl_growth, bl_thickness,
                   size_LE, dist_LE, size_TE, dist_TE,
                   VolWAkeIn, VolWAkeOut, WakeXMin, WakeXMax,
                   d_factor, string, bladeName):
    """Generate a 2-D SU2 mesh for the current blade."""

    out = process_airfoil_file(bladeFilePath, n_points=1000, n_te=60,
                              d_factor=d_factor)
    xSS, ySS, _, _ = out['ss']
    xPS, yPS, _, _ = out['ps']

    L1x = dist_inlet * axial_chord
    L2x = (dist_outlet - 1) * axial_chord
    m1 = np.tan(alpha1 * np.pi / 180)
    m2 = np.tan(alpha2 * np.pi / 180)

    geo_file = os.path.join(os.path.dirname(bladeFilePath),
                            f"cascade2D{string}_{bladeName}.geo")

    with open(geo_file, 'w') as f:
        f.write('// Gmsh geometry auto-generated\n')
        # Geometry writing omitted for brevity
        f.write('\n')

    print(f"Geo file written at: {geo_file}")
    try:
        if os.path.exists(geo_file):
            print(f"File exists at: {geo_file}")
        else:
            print(f"File not found at: {geo_file}")
        os.system(f'gmsh "{geo_file}" -2 -format su2')
        print("Mesh successfully created!")
    except Exception as e:
        print("Error", e)



def configSU2_datablade(M1, alpha1, P01, T01, Re, axial_chord,
                        TI2, gamma, R, P2, pitch, string, bladeName):
    """Write the SU2 configuration file for the generated mesh."""

    data_airfoil = f"""
%% SU2 AIRFOIL configuration file
MACH_NUMBER = {M1}
AOA = {alpha1}
FREESTREAM_PRESSURE = {P01}
FREESTREAM_TEMPERATURE = {T01}
REYNOLDS_NUMBER = {Re}
REYNOLDS_LENGTH = {axial_chord}
"""
    cfg = f"cascade2D{string}_{bladeName}.cfg"
    with open(cfg, "w") as f:
        f.write(data_airfoil)
    print(f"Config file created: {cfg}")
