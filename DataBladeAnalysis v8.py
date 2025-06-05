# -*- coding: utf-8 -*-
"""High level workflow for the DataBlade validation case."""

import os
import numpy as np

from blade_utils.io_helpers import (
    extract_from_ises,
    extract_from_blade,
    extract_from_outlet,
)
from blade_utils.geometry import (
    process_airfoil_file,
    compute_geometry,
    compute_d_factor,
)
from blade_utils.mesh import mesh_datablade, configSU2_datablade
from blade_utils.analysis import surfaceFlowAnalysis_datablade


# ----------------------------------------------------------------------------
# File selection
# ----------------------------------------------------------------------------
bladeName = "blade"
no_cores = 12
string = "databladeVALIDATION"
string2 = "safe_start"
fileExtension = "csv"

current_directory = os.path.dirname(os.path.abspath(__file__))
isesFileName = f"ises.{string}"
bladeFileName = f"{bladeName}.{string}"
outletFileName = f"outlet.{string}"
isesFilePath = os.path.join(current_directory, isesFileName)
bladeFilePath = os.path.join(current_directory, bladeFileName)
outletFilePath = os.path.join(current_directory, outletFileName)


# ----------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------

def main():
    # -- Data extraction ------------------------------------------------------
    alpha1, alpha2, Re = extract_from_ises(isesFilePath)
    pitch = extract_from_blade(bladeFilePath)
    M1, P21_ratio = extract_from_outlet(outletFilePath)

    # -- Geometry -------------------------------------------------------------
    geom0 = compute_geometry(bladeFilePath, pitch=pitch, d_factor_guess=0.5)
    d_factor = compute_d_factor(
        wedge_angle_deg=np.degrees(geom0['wedge_angle']),
        axial_chord=geom0['axial_chord'],
        te_thickness=geom0['te_open_thickness'],
    )
    geom = process_airfoil_file(bladeFilePath, n_points=1000, n_te=60,
                                d_factor=d_factor)
    stagger = compute_geometry(bladeFilePath, pitch, d_factor_guess=d_factor)[
        'stagger_angle'
    ]
    axial_chord = compute_geometry(bladeFilePath, pitch, d_factor_guess=d_factor)[
        'axial_chord'
    ]

    # -- Boundary conditions --------------------------------------------------
    R = 287.058
    gamma = 1.4
    mu = 1.716e-5

    T01 = 314.15
    T1 = T01 / (1 + (gamma - 1) / 2 * M1 ** 2)
    c1 = np.sqrt(gamma * R * T1)
    u1 = M1 * c1
    rho1 = mu * Re / (u1 * np.cos(stagger))
    P1 = rho1 * R * T1
    P01 = P1 * (1 + (gamma - 1) / 2 * M1 ** 2) ** (gamma / (gamma - 1))
    P2 = P21_ratio * P1

    # -- Mesh generation ------------------------------------------------------
    alpha1_deg = int(np.degrees(np.arctan(alpha1)))
    alpha2_deg = int(np.degrees(np.arctan(alpha2)))
    dist_inlet = 2
    dist_outlet = 3
    TI2 = 2.2

    sizeCellFluid = 0.02 * axial_chord
    sizeCellAirfoil = 0.02 * axial_chord
    nCellAirfoil = 549
    nCellPerimeter = 183
    nBoundaryPoints = 50
    first_layer_height = 0.01 * sizeCellAirfoil
    bl_growth = 1.17
    bl_thickness = 0.03 * pitch
    size_LE = 0.1 * sizeCellAirfoil
    dist_LE = 0.01 * axial_chord
    size_TE = 0.1 * sizeCellAirfoil
    dist_TE = 0.01 * axial_chord
    VolWAkeIn = 0.35 * sizeCellFluid
    VolWAkeOut = sizeCellFluid
    WakeXMin = -0.1 * axial_chord
    WakeXMax = (dist_outlet - 1.5) * axial_chord

    mesh_datablade(
        bladeFilePath, axial_chord, pitch, alpha1_deg, alpha2_deg,
        dist_inlet, dist_outlet, sizeCellFluid, sizeCellAirfoil,
        nCellAirfoil, nCellPerimeter, nBoundaryPoints,
        first_layer_height, bl_growth, bl_thickness,
        size_LE, dist_LE, size_TE, dist_TE,
        VolWAkeIn, VolWAkeOut, WakeXMin, WakeXMax,
        d_factor, string, bladeName
    )

    configSU2_datablade(M1, alpha1_deg, P01, T01, Re, axial_chord,
                        TI2, gamma, R, P2, pitch, string, bladeName)

    # Post-processing --------------------------------------------------------
    surfaceFlowAnalysis_datablade(current_directory, string, bladeName, P01,
                                  gamma)


if __name__ == "__main__":
    main()
