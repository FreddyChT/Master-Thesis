# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:38:00 2025

@author: fredd


Mesh sensitivity study utilities.

This script runs a sequence of SU2 simulations with increasing
mesh resolution for a chosen blade. The resulting lift and drag
coefficients are collected and the Grid Convergence Index (GCI)
is evaluated for the finest three meshes.

The mesh is generated and the solver is executed using the same
helper modules used by :mod:`analysis_datablade`. Six runs are
performed with approximate element counts of 10k, 20k, 30k, 40k,
80k and 120k. Only the last three are used for the GCI estimate.

Results are plotted as a function of the number of mesh elements
and printed to standard output.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

import utils
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

# Attempt to import the optional pyGCS/pyGCI package.
try:
    from pyGCS import gci
except Exception:  # pragma: no cover - library may not be installed
    try:
        from pyGCI import gci
    except Exception:
        gci = None


BLADEROOT = Path(__file__).resolve().parent


def _compute_gci(values, elems):
    """Return a basic GCI estimation.

    Parameters
    ----------
    values : sequence of float
        Solution values on the coarse, medium and fine meshes.
    elems : sequence of int
        Number of elements for the same meshes.
    """
    if gci is not None:
        # pyGCS/pyGCI style API: gci(phi, h) where h is 1/sqrt(N)
        h = [1.0 / math.sqrt(n) for n in elems]
        return gci(values, h)

    # Fallback: simple GCI computation following ASME guidelines.
    f1, f2, f3 = values  # coarse → fine
    n1, n2, n3 = elems
    h1, h2, h3 = 1.0 / math.sqrt(n1), 1.0 / math.sqrt(n2), 1.0 / math.sqrt(n3)
    r21 = h2 / h1
    r32 = h3 / h2
    p = math.log(abs((f3 - f2)/(f2 - f1))) / math.log(r32)
    f_ext = f3 + (f3 - f2)/(r32**p - 1)
    gci_fine = 1.25 * abs((f_ext - f3)/(f3)) * 100.0
    gci_med = 1.25 * abs((f_ext - f2)/(f2)) * 100.0
    return dict(p=p, gci_fine=gci_fine, gci_med=gci_med)


def _update_mesh_params(scale: float, axial_chord: float) -> None:
    """Scale global mesh parameters for modules by ``scale``."""
    mesh_datablade.sizeCellFluid = 0.04 * axial_chord / scale
    mesh_datablade.sizeCellAirfoil = 0.02 * axial_chord / scale
    mesh_datablade.nCellAirfoil = max(1, int(549 * scale))
    mesh_datablade.nCellPerimeter = max(1, int(183 * scale))
    mesh_datablade.nBoundaryPoints = 50 # Only controls smoothness of curved airfoil-like boundary line (no effect on mesh)

    configSU2_datablade.sizeCellFluid = mesh_datablade.sizeCellFluid
    configSU2_datablade.sizeCellAirfoil = mesh_datablade.sizeCellAirfoil
    configSU2_datablade.nCellAirfoil = mesh_datablade.nCellAirfoil
    configSU2_datablade.nCellPerimeter = mesh_datablade.nCellPerimeter
    configSU2_datablade.nBoundaryPoints = mesh_datablade.nBoundaryPoints


def run_one(blade: str, run_dir: Path, scale: float) -> tuple[int, float, float]:
    """Run a single simulation for ``blade`` with mesh scaled by ``scale``.

    Returns
    -------
    nelem : int
        Number of mesh elements produced.
    cl : float
        Final lift coefficient.
    cd : float
        Final drag coefficient.
    """
    blade_dir = BLADEROOT / 'Blades' / blade
    ises = blade_dir / 'ises.databladeVALIDATION'
    blade_file = blade_dir / 'blade.databladeVALIDATION'
    outlet_file = blade_dir / 'outlet.databladeVALIDATION'

    alpha1, alpha2, Re, M2, P2_P0a = utils.extract_from_ises(ises)
    pitch = utils.extract_from_blade(blade_file)
    geom0 = utils.compute_geometry(blade_file, pitch=pitch, d_factor_guess=0.5)
    d_factor = utils.compute_d_factor(
        math.degrees(geom0['wedge_angle']),
        axial_chord=geom0['axial_chord'],
        te_thickness=geom0['te_open_thickness'])
    geom = utils.compute_geometry(blade_file, pitch=pitch, d_factor_guess=d_factor)

    stagger = math.degrees(geom['stagger_angle'])
    axial_chord = geom['axial_chord']
    chord = geom['chord_length']
    pitch2chord = pitch / chord
    alpha1_deg = int(math.degrees(math.atan(alpha1)))
    alpha2_deg = int(math.degrees(math.atan(alpha2)))

    R = 287.058
    gamma = 1.4
    mu = 1.846e-5
    T01 = 300.0
    P1, P01 = utils.freestream_total_pressure(Re, M2, axial_chord, T01)
    M1 = utils.compute_Mx(P01, P1, gamma)
    P2 = P2_P0a * P01
    T02 = T01
    T2 = T02 / (1 + (gamma - 1)/2 * M2**2)
    c2 = math.sqrt(gamma * R * T2)
    u2 = M2 * c2
    rho2 = mu * Re / (u2 * math.cos(math.radians(stagger)))
    TI = 3.5

    dist_inlet = 1
    dist_outlet = 1.5

    bl = utils.compute_bl_parameters(u2, rho2, mu, axial_chord,
                                     n_layers=25, y_plus_target=1.0)
    first_layer_height = bl['first_layer_height']
    bl_growth = bl['bl_growth']
    bl_thickness = bl['bl_thickness']

    # Prepare modules
    for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
        mod.bladeName = blade
        mod.no_cores = 12
        mod.string = 'databladeVALIDATION'
        mod.fileExtension = 'csv'
        mod.base_dir = BLADEROOT
        mod.blade_dir = blade_dir
        mod.run_dir = run_dir
        mod.isesFilePath = ises
        mod.bladeFilePath = blade_file
        mod.outletFilePath = outlet_file
        mod.alpha1 = alpha1_deg
        mod.alpha2 = alpha2_deg
        mod.d_factor = d_factor
        mod.stagger = stagger
        mod.axial_chord = axial_chord
        mod.chord = chord
        mod.pitch = pitch
        mod.pitch2chord = pitch2chord
        mod.R = R
        mod.gamma = gamma
        mod.mu = mu
        mod.T01 = T01
        mod.P1 = P1
        mod.P01 = P01
        mod.M1 = M1
        mod.P2 = P2
        mod.P2_P0a = P2_P0a
        mod.M2 = M2
        mod.T02 = T02
        mod.T2 = T2
        mod.c2 = c2
        mod.u2 = u2
        mod.rho2 = rho2
        mod.Re = Re
        mod.TI = TI
        mod.dist_inlet = dist_inlet
        mod.dist_outlet = dist_outlet
        mod.first_layer_height = first_layer_height
        mod.bl_growth = bl_growth
        mod.bl_thickness = bl_thickness
        mod.size_LE = 0.1 * 0.02 * axial_chord
        mod.dist_LE = 0.01 * axial_chord
        mod.size_TE = 0.1 * 0.02 * axial_chord
        mod.dist_TE = 0.01 * axial_chord
        mod.VolWAkeIn = 0.35 * 0.04 * axial_chord
        mod.VolWAkeOut = 0.04 * axial_chord
        mod.WakeXMin = 0.1 * axial_chord
        mod.WakeXMax = (dist_outlet + 1) * axial_chord
        mod.WakeYMin = -5 * pitch
        mod.WakeYMax = 5 * pitch

    _update_mesh_params(scale, axial_chord)

    mesh_datablade.mesh_datablade()
    configSU2_datablade.configSU2_datablade()
    configSU2_datablade.runSU2_datablade()
    post_processing_datablade.post_processing_datablade()

    mesh_file = run_dir / f"cascade2D_databladeVALIDATION_{blade}.su2"
    nelem = 0
    with open(mesh_file, 'r') as f:
        for line in f:
            if line.startswith('NELEM'):
                nelem = int(line.split('=')[1])
                break

    hist_file = run_dir / f"history_databladeVALIDATION_{blade}.csv"
    hist = pd.read_csv(hist_file)
    cd = hist['   "CD(blade1)"   '].iat[-1]
    cl = hist['   "CL(blade1)"   '].iat[-1]
    return nelem, cl, cd


def main():
    parser = argparse.ArgumentParser(description='Run mesh convergence study')
    parser.add_argument('--blade', default='Blade_1', help='Blade name')
    args = parser.parse_args()

    blade = args.blade
    run_root = BLADEROOT / 'Blades' / blade / 'results'
    run_root.mkdir(exist_ok=True)
    study_dir = run_root / f"MeshStudy_{datetime.now().strftime('%d-%m-%Y_%H%M')}"
    study_dir.mkdir()

    targets = [1e4, 2e4, 3e4, 4e4, 8e4, 1.2e5] #[1e4, 2e4, 3e4, 4e4, 8e4, 1.2e5]
    baseline = 8.6e4  #approximate reference for default parameters

    results = []
    for target in targets:
        scale = math.sqrt(target / baseline)
        run_dir = study_dir / f"run_{int(target/1000)}k"
        run_dir.mkdir()
        nelem, cl, cd = run_one(blade, run_dir, scale)
        results.append(dict(n=nelem, cl=cl, cd=cd))
        print(f"Mesh {nelem} elements -> CL={cl:.5f}, CD={cd:.5f}")

    elems = [r['n'] for r in results]
    Cls = [r['cl'] for r in results]
    Cds = [r['cd'] for r in results]

    # GCI based on last three meshes (40k, 80k, 120k approx)
    gci_cl = _compute_gci(Cls[3:6], elems[3:6]) #_compute_gci(cls[3:6], elems[3:6])
    gci_cd = _compute_gci(Cds[3:6], elems[3:6]) #_compute_gci(cds[3:6], elems[3:6])

    print('\nGCI results (CL):', gci_cl)
    print('GCI results (CD):', gci_cd)

    plt.figure(figsize=(6, 4))
    plt.plot(elems, Cls, 'o-', label='$C_l$')
    plt.plot(elems, Cds, 's-', label='$C_d$')
    plt.xlabel('Number of Elements')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(study_dir / 'mesh_convergence.svg', format='svg')


if __name__ == '__main__':
    main()
