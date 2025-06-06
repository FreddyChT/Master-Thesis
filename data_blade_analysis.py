import argparse
import numpy as np
import os
from pathlib import Path
from datetime import datetime

import utils
import mesh_datablade
import configSU2_datablade
import surfaceFlowAnalysis_datablade

# ---------------------- USER PARAMETERS ----------------------
# Defaults can be overridden from the command line
BLADEROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run blade analysis")
    parser.add_argument('--blade', default='blade', help='Blade name')
    parser.add_argument('--no_cores', type=int, default=12, help='MPI cores for SU2')
    parser.add_argument('--suffix', default='databladeVALIDATION', help='File name suffix')
    parser.add_argument('--opt_tag', default='safe_start', help='Optimization tag')
    parser.add_argument('--file_ext', default='csv', help='Output file extension')
    args = parser.parse_args()

    bladeName = args.blade
    no_cores = args.no_cores
    string = args.suffix
    string2 = args.opt_tag
    fileExtension = args.file_ext

    base_dir = BLADEROOT
    blade_dir = base_dir / 'Blades' / bladeName

    run_root = blade_dir / 'results'
    run_root.mkdir(exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    n = 1
    while (run_root / f'Test_{n}_{date_str}').exists():
        n += 1
    run_dir = run_root / f'Test_{n}_{date_str}'
    run_dir.mkdir()

    isesFilePath = blade_dir / f'ises.{string}'
    bladeFilePath = blade_dir / f'{bladeName}.{string}'
    outletFilePath = blade_dir / f'outlet.{string}'

    alpha1, alpha2, Re = utils.extract_from_ises(isesFilePath)
    pitch = utils.extract_from_blade(bladeFilePath)
    M1, P21_ratio = utils.extract_from_outlet(outletFilePath)

    geom0 = utils.compute_geometry(bladeFilePath, pitch=pitch, d_factor_guess=0.5)
    d_factor = utils.compute_d_factor(np.degrees(geom0['wedge_angle']),
                                      axial_chord=geom0['axial_chord'],
                                      te_thickness=geom0['te_open_thickness'])
    print(f"Updated d_factor = {d_factor:.3f}")
    geom = utils.compute_geometry(bladeFilePath, pitch=pitch, d_factor_guess=d_factor)

    stagger = geom['stagger_angle']
    axial_chord = geom['axial_chord']

    R = 287.058
    gamma = 1.4
    mu = 1.716e-5

    T01 = 314.15
    T1 = T01 / (1 + (gamma - 1)/2 * M1**2)
    c1 = np.sqrt(gamma * R * T1)
    u1 = M1 * c1
    rho1 = mu * Re / (u1 * np.cos(stagger))
    P1 = rho1 * R * T1
    P01 = P1 * (1 + (gamma - 1)/2 * M1**2)**(gamma/(gamma - 1))
    P2 = P21_ratio * P1

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
    n_points = 1000
    n_te = 60
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

    # expose variables to modules
    for mod in (mesh_datablade, configSU2_datablade, surfaceFlowAnalysis_datablade):
        mod.bladeName = bladeName
        mod.no_cores = no_cores
        mod.string = string
        mod.string2 = string2
        mod.fileExtension = fileExtension
        mod.base_dir = base_dir
        mod.blade_dir = blade_dir
        mod.run_dir = run_dir
        mod.isesFilePath = isesFilePath
        mod.bladeFilePath = bladeFilePath
        mod.outletFilePath = outletFilePath
        mod.alpha1 = alpha1_deg
        mod.alpha2 = alpha2_deg
        mod.Re = Re
        mod.M1 = M1
        mod.P21_ratio = P21_ratio
        mod.pitch = pitch
        mod.d_factor = d_factor
        mod.stagger = stagger
        mod.axial_chord = axial_chord
        mod.R = R
        mod.gamma = gamma
        mod.mu = mu
        mod.T01 = T01
        mod.P01 = P01
        mod.P2 = P2
        mod.dist_inlet = dist_inlet
        mod.dist_outlet = dist_outlet
        mod.TI2 = TI2
        mod.sizeCellFluid = sizeCellFluid
        mod.sizeCellAirfoil = sizeCellAirfoil
        mod.nCellAirfoil = nCellAirfoil
        mod.nCellPerimeter = nCellPerimeter
        mod.nBoundaryPoints = nBoundaryPoints
        mod.first_layer_height = first_layer_height
        mod.bl_growth = bl_growth
        mod.bl_thickness = bl_thickness
        mod.size_LE = size_LE
        mod.dist_LE = dist_LE
        mod.size_TE = size_TE
        mod.dist_TE = dist_TE
        mod.VolWAkeIn = VolWAkeIn
        mod.VolWAkeOut = VolWAkeOut
        mod.WakeXMin = WakeXMin
        mod.WakeXMax = WakeXMax

    mesh_datablade.mesh_datablade()
    configSU2_datablade.configSU2_datablade()
    surfaceFlowAnalysis_datablade.surfaceFlowAnalysis_datablade()

if __name__ == '__main__':
    main()
