# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:46:18 2025

@author: fredd

Incidence sweep for the SPLEEN blade.

This is a specialised version of :mod:`analysis_incidence` where the
geometry and flow parameters correspond to the SPLEEN reference blade
(``Blade_0``).  The boundary conditions are fixed to the nominal case
``Re = 90k`` and ``Ma = 0.70``.  Only the inlet incidence angle is
varied across the sweep.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

BLADEROOT = Path(__file__).resolve().parent
DEFAULT_SUFFIX = "databladeVALIDATION"
DEFAULT_EXT = "csv"
DEFAULT_CORES = 12


class IncidenceDialog(simpledialog.Dialog):
    """Dialog asking for incidence sweep parameters."""

    def body(self, master):
        tk.Label(master, text="Incidence start [deg]").grid(row=0, column=0, sticky="w")
        self.start_var = tk.StringVar(value="-2.0")
        tk.Entry(master, textvariable=self.start_var).grid(row=0, column=1)

        tk.Label(master, text="Incidence end [deg]").grid(row=1, column=0, sticky="w")
        self.end_var = tk.StringVar(value="2.0")
        tk.Entry(master, textvariable=self.end_var).grid(row=1, column=1)

        tk.Label(master, text="Step [deg]").grid(row=2, column=0, sticky="w")
        self.step_var = tk.StringVar(value="0.2")
        tk.Entry(master, textvariable=self.step_var).grid(row=2, column=1)

        return tk.Entry(master)

    def apply(self):
        self.result = (
            self.start_var.get().strip(),
            self.end_var.get().strip(),
            self.step_var.get().strip(),
        )


def ask_user_inputs():
    root = tk.Tk()
    root.withdraw()
    dlg = IncidenceDialog(root, title="Incidence sweep")
    root.destroy()
    if not dlg.result:
        raise SystemExit("Inputs required")
    start, end, step = dlg.result
    return float(start), float(end), float(step)


def load_mach_distribution(surf_csv: Path, P01: float, gamma: float):
    """Return fractional surface coordinate and Mach array from a SU2 CSV."""
    df = pd.read_csv(surf_csv)
    _, _, ss, ps = utils.SU2_organize(df)
    s_ss = utils.surface_fraction(ss["x"].values, ss["y"].values)
    s_ps = utils.surface_fraction(ps["x"].values, ps["y"].values)
    mach_ss = utils.compute_Mx(P01, ss["Pressure"].values, gamma)
    mach_ps = utils.compute_Mx(P01, ps["Pressure"].values, gamma)
    x = np.concatenate([-s_ps[::-1], s_ss])
    mach = np.concatenate([mach_ps[::-1], mach_ss])
    return x, mach


def load_mises_distribution(mises_file: Path):
    """Return fractional surface coordinate and Mach array from a MISES file."""
    ps_f, ss_f, ps_m, ss_m = utils.MISES_machDataGather(mises_file)
    if ps_f.size and ss_f.size:
        frac = np.concatenate([-ps_f[::-1], ss_f])
        mach = np.concatenate([ps_m[::-1], ss_m])
        return frac, mach
    return np.array([]), np.array([])


def read_rms(summary_file: Path) -> float | None:
    """Extract RMS error from ``summary_file`` if present."""
    try:
        lines = summary_file.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if "Mach RMS error" in line:
            try:
                return float(line.split(":", 1)[1])
            except Exception:
                return None
    return None


def prepare_spleen_params():
    """Return geometry and flow parameters for the SPLEEN nominal case."""

    blade = "Blade_0"
    blade_dir = BLADEROOT / "Blades" / blade
    blade_file = blade_dir / f"blade.{DEFAULT_SUFFIX}"

    pitch = 32.950e-3
    chord = 52.285e-3
    axial_chord = 47.614e-3
    stagger = 24.40
    pitch2chord = pitch / chord
    alpha1_deg = 37.3
    alpha2_deg = -53.80
    d_factor = 0.0

    R = 287.058
    gamma = 1.4
    mu = 1.716e-5
    T01 = 300.0
    M2 = 0.70
    Re = 90_000

    P1, P01 = utils.freestream_total_pressure(Re, M2, axial_chord, T01)
    M1 = utils.compute_Mx(P01, P1, gamma)
    P2 = P01 / (1 + (gamma - 1) / 2 * M2 ** 2) ** (gamma / (gamma - 1))
    T02 = T01
    T2 = T02 / (1 + (gamma - 1) / 2 * M2 ** 2)
    c2 = np.sqrt(gamma * R * T2)
    u2 = M2 * c2
    rho2 = mu * Re / (u2 * np.cos(np.radians(stagger)))
    TI = 2.0

    dist_inlet = 1.0
    dist_outlet = 2.0
    x_plane = 1.5

    bl = utils.compute_bl_parameters(u2, rho2, mu, axial_chord, n_layers=25, y_plus_target=1.0)
    first_layer_height = bl["first_layer_height"]
    bl_growth = bl["bl_growth"]
    bl_thickness = bl["bl_thickness"]

    params = dict(
        blade_dir=blade_dir,
        blade_file=blade_file,
        alpha1_deg=alpha1_deg,
        alpha2_deg=alpha2_deg,
        d_factor=d_factor,
        stagger=stagger,
        axial_chord=axial_chord,
        chord=chord,
        pitch=pitch,
        pitch2chord=pitch2chord,
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
        first_layer_height=first_layer_height,
        bl_growth=bl_growth,
        bl_thickness=bl_thickness,
        size_LE=0.1 * 0.02 * axial_chord,
        dist_LE=0.01 * axial_chord,
        size_TE=0.1 * 0.02 * axial_chord,
        dist_TE=0.01 * axial_chord,
        VolWAkeIn=0.35 * 0.04 * axial_chord,
        VolWAkeOut=0.04 * axial_chord,
        WakeXMin=0.1 * axial_chord,
        WakeXMax=(dist_outlet + 0.5) * axial_chord,
    )
    return params


def configure_modules(run_dir: Path, blade: str, alpha1_deg: float, params: dict) -> None:
    for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
        mod.bladeName = blade
        mod.no_cores = DEFAULT_CORES
        mod.string = DEFAULT_SUFFIX
        mod.fileExtension = DEFAULT_EXT
        mod.base_dir = BLADEROOT
        mod.blade_dir = params["blade_dir"]
        mod.run_dir = run_dir
        mod.bladeFilePath = params["blade_file"]
        mod.alpha1 = alpha1_deg
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
        mod.sizeCellFluid = 0.04 * params["axial_chord"]
        mod.sizeCellAirfoil = 0.02 * params["axial_chord"]
        mod.nCellAirfoil = 300
        mod.nCellPerimeter = 183
        mod.nBoundaryPoints = 50
        mod.size_LE = params["size_LE"]
        mod.dist_LE = params["dist_LE"]
        mod.size_TE = params["size_TE"]
        mod.dist_TE = params["dist_TE"]
        mod.VolWAkeIn = params["VolWAkeIn"]
        mod.VolWAkeOut = params["VolWAkeOut"]
        mod.WakeXMin = params["WakeXMin"]
        mod.WakeXMax = params["WakeXMax"]


def run_once(run_dir: Path, alpha1_deg: float, params: dict):
    blade = "Blade_0"
    configure_modules(run_dir, blade, alpha1_deg, params)
    mesh_datablade.mesh_datablade()
    configSU2_datablade.configSU2_datablade()
    configSU2_datablade.runSU2_datablade()
    post_processing_datablade.post_processing_datablade()

    surf_csv = run_dir / f"surface_flow{DEFAULT_SUFFIX}_{blade}.csv"
    frac, mach = load_mach_distribution(surf_csv, params["P01"], params["gamma"])
    rms = read_rms(run_dir / "run_summary.txt")
    return frac, mach, rms


def main():
    inc_start, inc_end, inc_step = ask_user_inputs()
    params = prepare_spleen_params()
    blade = "Blade_0"

    mises_file = params["blade_dir"] / f"machDistribution.{DEFAULT_SUFFIX}"
    mises_frac, mises_mach = load_mises_distribution(mises_file)

    run_root = params["blade_dir"] / "results"
    run_root.mkdir(exist_ok=True)
    study_dir = run_root / f"IncStudy_{datetime.now().strftime('%d-%m-%Y_%H%M')}"
    study_dir.mkdir()

    base_alpha = params["alpha1_deg"]
    angles = np.arange(inc_start, inc_end + 0.001, inc_step)

    all_frac = []
    all_mach = []
    rms_vals = []
    alpha_vals = []
    offset_vals = []

    for offset in angles:
        angle = base_alpha + offset
        run_dir = study_dir / f"inc_{offset:+.1f}"
        run_dir.mkdir()
        frac, mach, rms = run_once(run_dir, angle, params)
        all_frac.append(frac)
        all_mach.append(mach)
        rms_vals.append(rms)
        alpha_vals.append(angle)
        offset_vals.append(offset)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(inc_start, inc_end)

    fig, ax = plt.subplots(figsize=(6, 4))
    for f, m, offs in zip(all_frac, all_mach, offset_vals):
        ax.plot(np.abs(f), m, color=cmap(norm(offs)))
    if mises_frac.size:
        ax.scatter(np.abs(mises_frac), mises_mach, s=2, facecolors="none",
                   edgecolors="k", label="MISES", zorder=5)
    ax.set_xlabel("Surface fraction")
    ax.set_ylabel("Mach number")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Incidence offset [deg]")
    if mises_frac.size:
        ax.legend()
    fig.tight_layout()
    fig.savefig(study_dir / "mach_distributions.png", dpi=300)

    # Store Mach data for later use
    s_base = np.abs(all_frac[0])
    mach_df = pd.DataFrame({"s": s_base})
    for offs, frac, mach in zip(offset_vals, all_frac, all_mach):
        x_abs = np.abs(frac)
        if not np.allclose(x_abs, s_base):
            mach_interp = np.interp(s_base, x_abs, mach)
        else:
            mach_interp = mach
        mach_df[f"inc_{offs:+.1f}"] = mach_interp
    mach_df.to_csv(study_dir / "mach_distributions.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(offset_vals, rms_vals, "o-")
    plt.xlabel("Incidence offset [deg]")
    plt.ylabel("Mach RMS error [\%]")
    plt.tight_layout()
    plt.savefig(study_dir / "rms_vs_incidence.png", dpi=300)

    rms_df = pd.DataFrame({"alpha": alpha_vals, "offset": offset_vals, "rms": rms_vals})
    rms_df.to_csv(study_dir / "rms_results.csv", index=False)


if __name__ == "__main__":
    main()