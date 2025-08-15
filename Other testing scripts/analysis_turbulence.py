# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:45:23 2025

@author: fredd

Turbulence intensity sweep analysis for SU2 blades.

Runs a series of SU2 simulations for a chosen blade while varying the
incoming turbulence intensity. The mesh, solver and post-processing steps
are executed for each intensity value and the resulting Mach distribution
and RMS error are recorded. A colour-mapped plot of all Mach distributions
and a plot of RMS error versus TI are saved in the study directory.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import utils
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

BLADEROOT = Path(__file__).resolve().parent
DEFAULT_SUFFIX = "databladeVALIDATION"
DEFAULT_EXT = "csv"
DEFAULT_CORES = 12


class TIDialog(simpledialog.Dialog):
    """Dialog asking for blade name and TI parameters."""

    def body(self, master):
        tk.Label(master, text="Blade name").grid(row=0, column=0, sticky="w")
        self.blade_var = tk.StringVar(value="Blade_1")
        tk.Entry(master, textvariable=self.blade_var).grid(row=0, column=1)

        tk.Label(master, text="TI start [%]").grid(row=1, column=0, sticky="w")
        self.start_var = tk.StringVar(value="1.0")
        tk.Entry(master, textvariable=self.start_var).grid(row=1, column=1)

        tk.Label(master, text="TI end [%]").grid(row=2, column=0, sticky="w")
        self.end_var = tk.StringVar(value="5.0")
        tk.Entry(master, textvariable=self.end_var).grid(row=2, column=1)

        tk.Label(master, text="Step [%]").grid(row=3, column=0, sticky="w")
        self.step_var = tk.StringVar(value="0.5")
        tk.Entry(master, textvariable=self.step_var).grid(row=3, column=1)

        return tk.Entry(master)

    def apply(self):
        self.result = (
            self.blade_var.get().strip(),
            self.start_var.get().strip(),
            self.end_var.get().strip(),
            self.step_var.get().strip(),
        )


def ask_user_inputs():
    root = tk.Tk()
    root.withdraw()
    dlg = TIDialog(root, title="TI sweep")
    root.destroy()
    if not dlg.result:
        raise SystemExit("Inputs required")
    blade, start, end, step = dlg.result
    return blade, float(start), float(end), float(step)


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


def prepare_common(blade: str):
    blade_dir = BLADEROOT / "Blades" / blade
    ises = blade_dir / f"ises.{DEFAULT_SUFFIX}"
    blade_file = blade_dir / f"blade.{DEFAULT_SUFFIX}"
    outlet_file = blade_dir / f"outlet.{DEFAULT_SUFFIX}"

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

    bl = utils.compute_bl_parameters(u2, rho2, mu, axial_chord, n_layers=25, y_plus_target=1.0)
    first_layer_height = bl["first_layer_height"]
    bl_growth = bl["bl_growth"]
    bl_thickness = bl["bl_thickness"]

    params = dict(
        blade_dir=blade_dir,
        ises=ises,
        blade_file=blade_file,
        outlet_file=outlet_file,
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
        P2_P0a=P2_P0a,
        M2=M2,
        T02=T02,
        T2=T2,
        c2=c2,
        u2=u2,
        rho2=rho2,
        Re=Re,
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
        WakeXMax=(dist_outlet + 1) * axial_chord,
    )
    return params


def configure_modules(run_dir: Path, blade: str, TI: float, params: dict) -> None:
    for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
        mod.bladeName = blade
        mod.no_cores = DEFAULT_CORES
        mod.string = DEFAULT_SUFFIX
        mod.fileExtension = DEFAULT_EXT
        mod.base_dir = BLADEROOT
        mod.blade_dir = params["blade_dir"]
        mod.run_dir = run_dir
        mod.isesFilePath = params["ises"]
        mod.bladeFilePath = params["blade_file"]
        mod.outletFilePath = params["outlet_file"]
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
        mod.P2_P0a = params["P2_P0a"]
        mod.M2 = params["M2"]
        mod.T02 = params["T02"]
        mod.T2 = params["T2"]
        mod.c2 = params["c2"]
        mod.u2 = params["u2"]
        mod.rho2 = params["rho2"]
        mod.Re = params["Re"]
        mod.TI = TI
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


def run_once(run_dir: Path, blade: str, TI: float, params: dict):
    configure_modules(run_dir, blade, TI, params)
    mesh_datablade.mesh_datablade()
    configSU2_datablade.configSU2_datablade()
    configSU2_datablade.runSU2_datablade()
    post_processing_datablade.post_processing_datablade()

    surf_csv = run_dir / f"surface_flow{DEFAULT_SUFFIX}_{blade}.csv"
    frac, mach = load_mach_distribution(surf_csv, params["P01"], params["gamma"])
    rms = read_rms(run_dir / "run_summary.txt")
    return frac, mach, rms


def main():
    blade, ti_start, ti_end, ti_step = ask_user_inputs()
    params = prepare_common(blade)

    mises_file = params["blade_dir"] / f"machDistribution.{DEFAULT_SUFFIX}"
    mises_frac, mises_mach = load_mises_distribution(mises_file)

    run_root = params["blade_dir"] / "results"
    run_root.mkdir(exist_ok=True)
    study_dir = run_root / f"TISweep_{datetime.now().strftime('%d-%m-%Y_%H%M')}"
    study_dir.mkdir()

    tis = np.arange(ti_start, ti_end + 0.001, ti_step)

    all_frac = []
    all_mach = []
    rms_vals = []
    ti_vals = []

    for TI in tis:
        run_dir = study_dir / f"TI_{TI:.1f}".replace(".", "p")
        run_dir.mkdir()
        frac, mach, rms = run_once(run_dir, blade, TI, params)
        all_frac.append(frac)
        all_mach.append(mach)
        rms_vals.append(rms)
        ti_vals.append(TI)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(ti_start, ti_end)

    fig, ax = plt.subplots(figsize=(6, 4))
    for f, m, ti in zip(all_frac, all_mach, ti_vals):
        ax.plot(np.abs(f), m, color=cmap(norm(ti)))
    if mises_frac.size:
        ax.scatter(np.abs(mises_frac), mises_mach, s=2, facecolors="none",
                   edgecolors="k", label="MISES", zorder=5)
    ax.set_xlabel("Surface fraction")
    ax.set_ylabel("Mach number")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Turbulence intensity [\%]")
    if mises_frac.size:
        ax.legend()
    fig.tight_layout()
    fig.savefig(study_dir / "mach_distributions.png", dpi=300)

    # Store Mach data for later use
    s_base = np.abs(all_frac[0])
    mach_df = pd.DataFrame({"s": s_base})
    for val, frac, mach in zip(ti_vals, all_frac, all_mach):
        x_abs = np.abs(frac)
        if not np.allclose(x_abs, s_base):
            mach_interp = np.interp(s_base, x_abs, mach)
        else:
            mach_interp = mach
        mach_df[f"TI_{val:.1f}"] = mach_interp
    mach_df.to_csv(study_dir / "mach_distributions.csv", index=False)

    # ── RMS‑vs‑TI plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ti_vals, rms_vals, "o-")
    ax.set_xlabel(r"Turbulence intensity [\%]")  # escaped ‘%’ as before
    ax.set_ylabel("Mach RMS error")
    
    # format y‑axis numbers to two decimals
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig.tight_layout()
    fig.savefig(study_dir / "rms_vs_ti.png", dpi=300)

    rms_df = pd.DataFrame({"TI": ti_vals, "rms": np.round(rms_vals, 3)})
    rms_df.to_csv(study_dir / "rms_results.csv", index=False)


if __name__ == "__main__":
    main()