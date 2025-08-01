# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:38:00 2025

@author: fredd


Mesh sensitivity study utilities for the SPLEEN blade.

This variant of :mod:`analysis_mesh` is tailored for the "Blade_0"
geometry (a.k.a. the SPLEEN blade).  The boundary conditions are
set to the nominal experimental case ``Re = 90k`` and ``Ma = 0.70``
following the procedure implemented in :mod:`analysis_spleen`.
The script runs a sequence of SU2 simulations with increasing mesh
resolution and collects the aerodynamic coefficients as well as basic
mesh-quality statistics.  The Grid Convergence Index (GCI) is evaluated
for the finest three meshes.

Results are plotted as a function of the number of mesh elements and
printed to standard output.
"""

from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime
import time
import re
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


def prepare_spleen_params():
    """Return geometry and flow parameters for the nominal SPLEEN case."""

    blade = "Blade_0"
    blade_dir = BLADEROOT / "Blades" / blade
    blade_file = blade_dir / "blade.databladeVALIDATION"

    # Geometry values from the SPLEEN data set
    pitch = 32.950e-3
    chord = 52.285e-3
    axial_chord = 47.614e-3
    stagger = 24.40
    pitch2chord = pitch / chord
    alpha1_deg = 37.3
    alpha2_deg = -53.80
    d_factor = 0.0

    # Boundary conditions for nominal case Re=90k, Ma=0.70
    R = 287.058
    gamma = 1.4
    mu = 1.716e-5
    T01 = 300.0
    M2 = 0.90
    Re = 70000

    P1, P01 = utils.freestream_total_pressure(Re, M2, axial_chord, T01)
    M1 = utils.compute_Mx(P01, P1, gamma)
    T02 = T01
    T2 = T02 / (1 + (gamma - 1) / 2 * M2 ** 2)
    c2 = (gamma * R * T2) ** 0.5
    u2 = M2 * c2
    rho2 = mu * Re / (u2 * math.cos(math.radians(stagger)))
    P2 = P01 / (1 + (gamma - 1) / 2 * M2 ** 2) ** (gamma / (gamma - 1))
    TI = 2.0

    # Mesh parameters
    dist_inlet = 2.0
    dist_outlet = 3.0
    x_plane = 1.5

    sizeCellFluid = 0.04 * axial_chord
    sizeCellAirfoil = 0.02 * axial_chord
    nCellAirfoil = 300
    nCellPerimeter = 183
    nBoundaryPoints = 50
    bl = utils.compute_bl_parameters(
        u2,
        rho2,
        mu,
        axial_chord,
        n_layers=25,
        y_plus_target=1.0,
        x_ref_yplus=1/1000,
    )
    first_layer_height = bl["first_layer_height"]
    bl_growth = bl["bl_growth"]
    bl_thickness = bl["bl_thickness"]
    size_LE = 0.1 * sizeCellAirfoil
    dist_LE = 0.01 * axial_chord
    size_TE = 0.1 * sizeCellAirfoil
    dist_TE = 0.01 * axial_chord
    VolWAkeIn = 0.35 * sizeCellFluid
    VolWAkeOut = sizeCellFluid
    WakeXMin = 0.1 * axial_chord
    WakeXMax = (dist_outlet + 0.5) * axial_chord

    return dict(
        blade=blade,
        blade_dir=blade_dir,
        blade_file=blade_file,
        pitch=pitch,
        chord=chord,
        axial_chord=axial_chord,
        stagger=stagger,
        pitch2chord=pitch2chord,
        alpha1_deg=alpha1_deg,
        alpha2_deg=alpha2_deg,
        d_factor=d_factor,
        R=R,
        gamma=gamma,
        mu=mu,
        T01=T01,
        P1=P1,
        P01=P01,
        M1=M1,
        P2=P2,
        M2=M2,
        T02=T02,
        T2=T2,
        c2=c2,
        u2=u2,
        rho2=rho2,
        Re=Re,
        TI=TI,
        dist_inlet=dist_inlet,
        dist_outlet=dist_outlet,
        x_plane=x_plane,
        sizeCellFluid=sizeCellFluid,
        sizeCellAirfoil=sizeCellAirfoil,
        nCellAirfoil=nCellAirfoil,
        nCellPerimeter=nCellPerimeter,
        nBoundaryPoints=nBoundaryPoints,
        first_layer_height=first_layer_height,
        bl_growth=bl_growth,
        bl_thickness=bl_thickness,
        size_LE=size_LE,
        dist_LE=dist_LE,
        size_TE=size_TE,
        dist_TE=dist_TE,
        VolWAkeIn=VolWAkeIn,
        VolWAkeOut=VolWAkeOut,
        WakeXMin=WakeXMin,
        WakeXMax=WakeXMax,
    )


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


_TABLE_LINE = re.compile(r"\|\s*(.*?)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|")


def _parse_mesh_quality(log_file: Path) -> dict[str, float]:
    """Extract mesh quality statistics from a SU2 log file."""
    metrics: dict[str, float] = {}
    lines = log_file.read_text().splitlines()
    in_table = False
    for line in lines:
        if "Mesh Quality Metric" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("+"):
                continue
            if not line.strip() or not line.startswith("|"):
                break
            m = _TABLE_LINE.search(line)
            if not m:
                continue
            name, vmin, vmax = m.groups()
            try:
                lo = float(vmin)
                hi = float(vmax)
            except ValueError:
                continue
            name = name.lower()
            if "orthogonality" in name:
                metrics["orthogonality_min"] = lo
                metrics["orthogonality_max"] = hi
            elif "aspect ratio" in name:
                metrics["aspect_ratio_min"] = lo
                metrics["aspect_ratio_max"] = hi
            elif "sub-volume" in name:
                metrics["subvol_ratio_min"] = lo
                metrics["subvol_ratio_max"] = hi
    return metrics


def _gmsh_quality(mesh_file: Path) -> dict[str, float]:
    """Return additional quality metrics using gmsh if available."""
    quality: dict[str, float] = {}
    try:
        import gmsh  # type: ignore
    except Exception:
        return quality
    try:
        gmsh.initialize()
        gmsh.open(str(mesh_file))
        try:
            gmsh.option.setNumber("Mesh.QualityType", 5)  # aspect ratio
            _, vals = gmsh.model.mesh.getQuality()
            if vals:
                quality["aspect_ratio_min"] = min(vals)
                quality["aspect_ratio_max"] = max(vals)
        except Exception:
            pass
        try:
            gmsh.option.setNumber("Mesh.QualityType", 7)  # skewness
            _, vals = gmsh.model.mesh.getQuality()
            if vals:
                quality["skewness_min"] = min(vals)
                quality["skewness_max"] = max(vals)
        except Exception:
            pass
        try:
            gmsh.option.setNumber("Mesh.QualityType", 2)  # jacobian ratio
            _, vals = gmsh.model.mesh.getQuality()
            if vals:
                quality["jacobian_ratio_min"] = min(vals)
                quality["jacobian_ratio_max"] = max(vals)
        except Exception:
            pass
        gmsh.finalize()
    except Exception:
        try:
            gmsh.finalize()
        except Exception:
            pass
    return quality


def _run_diverged(summary_file: Path) -> bool:
    """Return ``True`` if the SU2 run reported divergence."""
    try:
        text = summary_file.read_text().lower()
    except OSError:
        return False
    return "diverged" in text


def _update_mesh_params(scale: float, params: dict) -> None:
    """Scale mesh-related parameters for all helper modules."""

    # Base values from the nominal configuration
    base_fluid = params["sizeCellFluid"]
    base_airfoil = params["sizeCellAirfoil"]
    base_n_airfoil = params["nCellAirfoil"]
    base_n_perimeter = params["nCellPerimeter"]

    mesh_datablade.sizeCellFluid = base_fluid / scale
    mesh_datablade.sizeCellAirfoil = base_airfoil / scale
    mesh_datablade.nCellAirfoil = max(1, int(base_n_airfoil * scale))
    mesh_datablade.nCellPerimeter = max(1, int(base_n_perimeter * scale))
    mesh_datablade.nBoundaryPoints = params["nBoundaryPoints"]

    configSU2_datablade.sizeCellFluid = mesh_datablade.sizeCellFluid
    configSU2_datablade.sizeCellAirfoil = mesh_datablade.sizeCellAirfoil
    configSU2_datablade.nCellAirfoil = mesh_datablade.nCellAirfoil
    configSU2_datablade.nCellPerimeter = mesh_datablade.nCellPerimeter
    configSU2_datablade.nBoundaryPoints = mesh_datablade.nBoundaryPoints

    # ``post_processing_datablade`` relies on ``sizeCellFluid`` for wake sampling
    post_processing_datablade.sizeCellFluid = mesh_datablade.sizeCellFluid



def run_one(run_dir: Path, scale: float) -> tuple[int, float, float, dict[str, float]]:
    """Run a single simulation for the SPLEEN blade with mesh scaled by ``scale``."""

    params = prepare_spleen_params()
    blade = params["blade"]

    # Configure modules with fixed SPLEEN parameters
    for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
        mod.bladeName = blade
        mod.no_cores = 12
        mod.string = "databladeVALIDATION"
        mod.fileExtension = "csv"
        mod.base_dir = BLADEROOT
        mod.blade_dir = params["blade_dir"]
        mod.run_dir = run_dir
        mod.bladeFilePath = params["blade_file"]

        mod.alpha1 = params["alpha1_deg"]
        mod.alpha2 = params["alpha2_deg"]
        mod.d_factor = params["d_factor"]
        mod.stagger = params["stagger"]
        mod.axial_chord = params["axial_chord"]
        mod.chord = params["chord"]
        mod.pitch = params["pitch"]
        mod.pitch2chord = params["pitch2chord"]
        mod.R = params["R"]
        mod.gamma = params["gamma"]
        mod.mu = params["mu"]
        mod.T01 = params["T01"]
        mod.P1 = params["P1"]
        mod.P01 = params["P01"]
        mod.M1 = params["M1"]
        mod.P2 = params["P2"]
        mod.M2 = params["M2"]
        mod.T02 = params["T02"]
        mod.T2 = params["T2"]
        mod.c2 = params["c2"]
        mod.u2 = params["u2"]
        mod.rho2 = params["rho2"]
        mod.Re = params["Re"]
        mod.TI = params["TI"]
        mod.dist_inlet = params["dist_inlet"]
        mod.dist_outlet = params["dist_outlet"]
        mod.x_plane = params["x_plane"]
        mod.first_layer_height = params["first_layer_height"]
        mod.bl_growth = params["bl_growth"]
        mod.bl_thickness = params["bl_thickness"]
        mod.size_LE = params["size_LE"]
        mod.dist_LE = params["dist_LE"]
        mod.size_TE = params["size_TE"]
        mod.dist_TE = params["dist_TE"]
        mod.VolWAkeIn = params["VolWAkeIn"]
        mod.VolWAkeOut = params["VolWAkeOut"]
        mod.WakeXMin = params["WakeXMin"]
        mod.WakeXMax = params["WakeXMax"]

    _update_mesh_params(scale, params)

    start_t = time.perf_counter()
    mesh_datablade.mesh_datablade()
    mesh_time = time.perf_counter() - start_t
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

    log_file = run_dir / "su2.log"
    metrics = _parse_mesh_quality(log_file)
    metrics.setdefault("mesh_time", mesh_time)
    mesh_msh = run_dir / f"cascade2D_databladeVALIDATION_{blade}.msh"
    metrics.update({k: v for k, v in _gmsh_quality(mesh_msh).items() if v is not None})
    summary_file = run_dir / "run_summary.txt"
    metrics["diverged"] = _run_diverged(summary_file)

    return nelem, cl, cd, metrics


def main():
    params = prepare_spleen_params()
    blade = params["blade"]
    run_root = BLADEROOT / "Blades" / blade / "results"
    run_root.mkdir(exist_ok=True)
    study_dir = run_root / f"MeshStudy_{datetime.now().strftime('%d-%m-%Y_%H%M')}"
    study_dir.mkdir()

    targets = [1e4, 2e4, 3e4, 4e4, 8e4, 1.2e5]
    baseline = 8.6e4

    results = []
    for target in targets:
        scale = math.sqrt(target / baseline)
        run_dir = study_dir / f"run_{int(target/1000)}k"
        run_dir.mkdir()
        nelem, cl, cd, mstat = run_one(run_dir, scale)
        entry = dict(n=nelem, cl=cl, cd=cd)
        entry.update(mstat)
        results.append(entry)
        print(
            f"Mesh {nelem} elements -> CL={cl:.5f}, CD={cd:.5f}, "
            f"mesh time={mstat.get('mesh_time', 0):.2f}s"
        )

    elems = [r['n'] for r in results]
    Cls = [r['cl'] for r in results]
    Cds = [r['cd'] for r in results]
    mesh_times = [r.get('mesh_time') for r in results]
    ar_min = [r.get('aspect_ratio_min') for r in results]
    ar_max = [r.get('aspect_ratio_max') for r in results]
    orth_min = [r.get('orthogonality_min') for r in results]
    orth_max = [r.get('orthogonality_max') for r in results]
    skew_min = [r.get('skewness_min') for r in results]
    skew_max = [r.get('skewness_max') for r in results]
    jac_min = [r.get('jacobian_ratio_min') for r in results]
    jac_max = [r.get('jacobian_ratio_max') for r in results]
    subvol_max = [r.get('subvol_ratio_max') for r in results]
    diverged = [r.get('diverged') for r in results]

    # GCI based on last three meshes (40k, 80k, 120k approx)
    gci_cl = _compute_gci(Cls[3:6], elems[3:6]) #_compute_gci(cls[3:6], elems[3:6])
    gci_cd = _compute_gci(Cds[3:6], elems[3:6]) #_compute_gci(cds[3:6], elems[3:6])

    print('\nGCI results (CL):', gci_cl)
    print('GCI results (CD):', gci_cd)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(elems, Cls, 'o-', label='$C_l$', color='tab:blue')
    if any(diverged):
        ax1.plot([e for e, d in zip(elems, diverged) if d],
                 [c for c, d in zip(Cls, diverged) if d],
                 'x', color='red', label='diverged')
    ax1.set_xlabel('Number of Elements')
    ax1.set_ylabel('$C_l$', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(elems, Cds, 's-', label='$C_d$', color='tab:orange')
    if any(diverged):
        ax2.plot([e for e, d in zip(elems, diverged) if d],
                 [c for c, d in zip(Cds, diverged) if d],
                 'x', color='red')
    ax2.set_ylabel('$C_d$', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    # Scientific formatter with 10^-2 scale
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, -2))
    formatter.set_scientific(True)
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(study_dir / 'mesh_convergence.svg', format='svg')


    def _plot(values, ylabel, filename):
        if all(v is None for v in values):
            return
        plt.figure(figsize=(6, 4))
        plt.plot(elems, values, 'o-')
        if any(diverged):
            plt.plot([e for e, d in zip(elems, diverged) if d],
                     [v for v, d in zip(values, diverged) if d],
                     'x', color='red')
        plt.xlabel('Number of Elements')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(study_dir / filename, format='svg')

    _plot(mesh_times, 'Meshing time [s]', 'mesh_time.svg')
    _plot(ar_max, 'Aspect ratio (max)', 'aspect_ratio_max.svg')
    #_plot(ar_min, 'Aspect ratio (min)', 'aspect_ratio_min.svg')
    #_plot(orth_max, 'Orthogonality (max)', 'orthogonality_max.svg')
    _plot(orth_min, 'Orthogonality (min)', 'orthogonality_min.svg')
    _plot(skew_max, 'Skewness (max)', 'skewness_max.svg')
    #_plot(skew_min, 'Skewness (min)', 'skewness_min.svg')
    _plot(jac_max, 'Jacobian ratio (max)', 'jacobian_ratio_max.svg')
    _plot(jac_min, 'Jacobian ratio (min)', 'jacobian_ratio_min.svg')
    _plot(subvol_max, 'Sub-volume ratio (max)', 'subvol_ratio_max.svg')


if __name__ == '__main__':
    main()