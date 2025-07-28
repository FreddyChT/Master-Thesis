# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:45:23 2025

@author: fredd

Run turbulence intensity sensitivity study for a single blade.

A small Tkinter dialog asks for the blade name and the TI sweep
(start, end and step). For each TI value the usual SU2 workflow
(mesh generation, solver run and post-processing) is executed.
All results are saved in a dedicated folder under ``Blades/<blade>/results``.
Finally two plots are produced in that directory:

* ``mach_ti_sweep.svg``  – Mach number distribution coloured by TI.
* ``ti_vs_rms.svg``      – RMS error versus turbulence intensity.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

BLADEROOT = Path(__file__).resolve().parent


class TIRangeDialog(simpledialog.Dialog):
    """Dialog asking for blade name and TI range."""

    def body(self, master):
        tk.Label(master, text="Blade name").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar(value="Blade_0")
        tk.Entry(master, textvariable=self.name_var).grid(row=0, column=1)

        tk.Label(master, text="TI start (%)").grid(row=1, column=0, sticky="w")
        self.start_var = tk.DoubleVar(value=1.0)
        tk.Entry(master, textvariable=self.start_var).grid(row=1, column=1)

        tk.Label(master, text="TI end (%)").grid(row=2, column=0, sticky="w")
        self.end_var = tk.DoubleVar(value=5.0)
        tk.Entry(master, textvariable=self.end_var).grid(row=2, column=1)

        tk.Label(master, text="Step (%)").grid(row=3, column=0, sticky="w")
        self.step_var = tk.DoubleVar(value=0.5)
        tk.Entry(master, textvariable=self.step_var).grid(row=3, column=1)
        return master

    def apply(self):
        self.result = (
            self.name_var.get().strip(),
            float(self.start_var.get()),
            float(self.end_var.get()),
            float(self.step_var.get()),
        )


def ask_inputs() -> tuple[str, float, float, float]:
    root = tk.Tk()
    root.withdraw()
    dlg = TIRangeDialog(root, title="TI sweep")
    root.destroy()
    if not dlg.result:
        raise SystemExit("Inputs required")
    return dlg.result


def _prepare_modules(blade: str, run_dir: Path, TI: float) -> dict:
    """Set up global parameters for helper modules and return basics."""
    blade_dir = BLADEROOT / "Blades" / blade
    ises = blade_dir / "ises.databladeVALIDATION"
    blade_file = blade_dir / "blade.databladeVALIDATION"
    outlet_file = blade_dir / "outlet.databladeVALIDATION"

    alpha1, alpha2, Re, M2, P2_P0a = utils.extract_from_ises(ises)
    pitch = utils.extract_from_blade(blade_file)

    geom0 = utils.compute_geometry(blade_file, pitch=pitch, d_factor_guess=0.5)
    d_factor = utils.compute_d_factor(
        np.degrees(geom0["wedge_angle"]),
        axial_chord=geom0["axial_chord"],
        te_thickness=geom0["te_open_thickness"],
    )
    geom = utils.compute_geometry(blade_file, pitch=pitch, d_factor_guess=d_factor)

    stagger = np.degrees(geom["stagger_angle"])
    axial_chord = geom["axial_chord"]
    chord = geom["chord_length"]
    pitch2chord = pitch / chord
    alpha1_deg = int(np.degrees(np.arctan(alpha1)))
    alpha2_deg = int(np.degrees(np.arctan(alpha2)))

    R = 287.058
    gamma = 1.4
    mu = 1.846e-5
    T01 = 300.0
    P1, P01 = utils.freestream_total_pressure(Re, M2, axial_chord, T01)
    M1 = utils.compute_Mx(P01, P1, gamma)
    P2 = P2_P0a * P01
    T02 = T01
    T2 = T02 / (1 + (gamma - 1) / 2 * M2 ** 2)
    c2 = np.sqrt(gamma * R * T2)
    u2 = M2 * c2
    rho2 = mu * Re / (u2 * np.cos(np.radians(stagger)))

    dist_inlet = 1.0
    dist_outlet = 1.5
    x_plane = 1.0

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
        x_ref_yplus=1 / 1000,
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

    for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
        mod.bladeName = blade
        mod.no_cores = 12
        mod.string = "databladeVALIDATION"
        mod.fileExtension = "csv"
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
        mod.x_plane = x_plane
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

    return {
        "P01": P01,
        "gamma": gamma,
        "blade_dir": blade_dir,
    }


def _compute_distributions(run_dir: Path, blade: str, P01: float, gamma: float):
    surf = run_dir / f"surface_flowdatabladeVALIDATION_{blade}.csv"
    df = pd.read_csv(surf)
    _, _, ss, ps = utils.SU2_organize(df)
    sss = utils.surface_fraction(ss["x"].values, ss["y"].values)
    sps = utils.surface_fraction(ps["x"].values, ps["y"].values)
    mach_ss = utils.compute_Mx(P01, ss["Pressure"].values, gamma)
    mach_ps = utils.compute_Mx(P01, ps["Pressure"].values, gamma)
    return sss, sps, mach_ss, mach_ps


def _compute_rms(blade_dir: Path, blade: str, sss, sps, mach_ss, mach_ps):
    mfile = blade_dir / "machDistribution.databladeVALIDATION"
    ps_frac, ss_frac, ps_mach, ss_mach = utils.MISES_machDataGather(mfile)
    if len(ps_frac) == 0 and len(ss_frac) == 0:
        return np.nan
    su2_ss = np.interp(ss_frac, sss, mach_ss)
    su2_ps = np.interp(ps_frac, sps, mach_ps)
    diff = np.concatenate([su2_ss - ss_mach, su2_ps - ps_mach])
    return float(np.sqrt(np.nanmean(diff ** 2)) * 100)


def run_one(blade: str, run_dir: Path, TI: float) -> tuple[np.ndarray, np.ndarray, float]:
    params = _prepare_modules(blade, run_dir, TI)
    mesh_datablade.mesh_datablade()
    configSU2_datablade.configSU2_datablade()
    proc, logf = configSU2_datablade.runSU2_datablade(background=True)
    proc.wait()
    logf.close()
    configSU2_datablade._summarize_su2_log(run_dir / "su2.log")
    post_processing_datablade.post_processing_datablade()

    sss, sps, mach_ss, mach_ps = _compute_distributions(
        run_dir, blade, params["P01"], params["gamma"]
    )
    rms = _compute_rms(params["blade_dir"], blade, sss, sps, mach_ss, mach_ps)
    frac = np.concatenate([-sps, sss])
    mach = np.concatenate([mach_ps, mach_ss])
    return frac, mach, rms


def main():
    blade, start, end, step = ask_inputs()
    tis = np.arange(start, end + 1e-9, step)

    results_dir = BLADEROOT / "Blades" / blade / "results"
    results_dir.mkdir(exist_ok=True)
    sweep_dir = results_dir / f"TISweep_{datetime.now().strftime('%d-%m-%Y_%H%M')}"
    sweep_dir.mkdir()

    all_frac: list[np.ndarray] = []
    all_mach: list[np.ndarray] = []
    rms_vals: list[float] = []

    for TI in tis:
        run_dir = sweep_dir / f"TI_{TI:.1f}".replace(".", "p")
        run_dir.mkdir()
        frac, mach, rms = run_one(blade, run_dir, TI)
        all_frac.append(frac)
        all_mach.append(mach)
        rms_vals.append(rms)

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(tis)))
    plt.figure(figsize=(6, 4))
    for ti, frac, mach, col in zip(tis, all_frac, all_mach, colors):
        plt.plot(abs(frac), mach, color=col, label=f"{ti:.1f} %")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(tis.min(), tis.max()))
    sm.set_array([])
    plt.colorbar(sm, label="TI [%]")
    plt.xlabel(r"$s/s_{total}$")
    plt.ylabel("Mach")
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(sweep_dir / "mach_ti_sweep.svg", format="svg")

    plt.figure(figsize=(6, 4))
    plt.plot(tis, rms_vals, "o-")
    plt.xlabel("Turbulence intensity [%]")
    plt.ylabel("Mach RMS error [%]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(sweep_dir / "ti_vs_rms.svg", format="svg")


if __name__ == "__main__":
    main()