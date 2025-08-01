# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:52:24 2025

@author: fredd

Compare two SU2 runs for a given blade.

This routine plots, in the same figure, the Mach number distribution,
skin–friction coefficient, total pressure loss and RMS residuals for two
separate runs.  MISES results are also overlaid when available.
"""

import argparse
import re
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{helvet}",
})


def _parse_config(cfg_path: Path):
    """Return inlet total pressure and pitch from a SU2 ``cfg`` file."""
    p01 = None
    pitch = None
    with open(cfg_path) as f:
        for line in f:
            if line.strip().startswith("MARKER_INLET"):
                numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+e?[+-]?[0-9]*", line)
                if len(numbers) >= 2:
                    p01 = float(numbers[1])
            if line.strip().startswith("MARKER_PERIODIC"):
                numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+e?[+-]?[0-9]*", line)
                if len(numbers) >= 9:
                    pitch = float(numbers[8])
    if p01 is None or pitch is None:
        raise ValueError("Could not extract MARKER_INLET or MARKER_PERIODIC from config")
    return p01, pitch


def load_run(run_dir: Path):
    """Gather all relevant SU2 data from *run_dir*."""
    run_dir = Path(run_dir)
    surf_file = next(run_dir.glob("surface_flow*.csv"))
    m = re.match(r"surface_flow(.+)_([^_]+)\.csv", surf_file.name)
    if not m:
        raise ValueError(f"Unexpected surface file name: {surf_file.name}")
    string, blade = m.group(1), m.group(2)

    df = pd.read_csv(surf_file)
    _, _, ss, ps = utils.SU2_organize(df)
    s_ss = utils.surface_fraction(ss['x'].values, ss['y'].values)
    s_ps = utils.surface_fraction(ps['x'].values, ps['y'].values)
    mach_ss = ss['Mach'].values
    mach_ps = ps['Mach'].values
    cf_ss = np.sqrt(ss['Skin_Friction_Coefficient_x'].values**2 +
                    ss['Skin_Friction_Coefficient_y'].values**2)
    cf_ps = np.sqrt(ps['Skin_Friction_Coefficient_x'].values**2 +
                    ps['Skin_Friction_Coefficient_y'].values**2)

    # Pressure loss related data
    vol_file = next(run_dir.glob("restart_flow*.csv"))
    vol_df = pd.read_csv(vol_file)
    cfg_file = next(run_dir.glob("cascade2D*.cfg"))
    p01, pitch = _parse_config(cfg_file)
    x_plane = vol_df['x'].max()
    plane_df = utils.SU2_extract_plane_data(vol_df, x_plane, pitch, 0.0)
    if plane_df is None:
        alpha2 = 0.0
    else:
        alpha2 = np.arctan2(plane_df['Velocity_y'].mean(),
                             plane_df['Velocity_x'].mean())
    loss_df = utils.SU2_total_pressure_loss(
        vol_df, x_plane, pitch, p01, alpha_m=alpha2,
        smooth=True, window_length=15, polyorder=4)
    if loss_df is None:
        y_loss = loss = np.array([])
    else:
        y_loss = loss_df['y_norm'].values
        loss = loss_df['loss'].values

    # Residual history
    hist_file = next(run_dir.glob("history*.csv"))
    hist = pd.read_csv(hist_file)
    iter_hist = hist['Inner_Iter'].values
    rho = hist['    "rms[Rho]"    '].values
    rhou = hist['    "rms[RhoU]"   '].values
    rhoe = hist['    "rms[RhoE]"   '].values

    return {
        'run_dir': run_dir,
        'blade': blade,
        'string': string,
        's_ss': s_ss,
        's_ps': s_ps,
        'mach_ss': mach_ss,
        'mach_ps': mach_ps,
        'cf_ss': cf_ss,
        'cf_ps': cf_ps,
        'y_loss': y_loss,
        'loss': loss,
        'iter': iter_hist,
        'rho': rho,
        'rhou': rhou,
        'rhoe': rhoe,
        'pitch': pitch,
        'x_plane': x_plane,
        'p01': p01,
    }


def load_mises(blade_dir: Path, string: str, x_plane: float, pitch: float, p01: float):
    """Load MISES distributions for the selected blade."""
    blade_dir = Path(blade_dir)
    mach_file = blade_dir / f"machDistribution.{string}"
    if utils.file_nonempty(mach_file):
        ps_frac, ss_frac, ps_mach, ss_mach = utils.MISES_machDataGather(mach_file)
        exp_mach_s = np.concatenate([ps_frac, ss_frac])
        exp_mach = np.concatenate([ps_mach, ss_mach])
    else:
        exp_mach_s = exp_mach = None

    bl_file = blade_dir / f"bl.{string}"
    if utils.file_nonempty(bl_file):
        ps_bl, ss_bl = utils.MISES_blDataGather(bl_file)
        exp_cf_s = np.concatenate([-ps_bl['s'].values, ss_bl['s'].values])
        exp_cf = np.concatenate([ps_bl['Cf'].values, ss_bl['Cf'].values])
    else:
        exp_cf_s = exp_cf = None

    field_file = blade_dir / f"field.{string}"
    mises_res = utils.MISES_total_pressure_loss(
        field_file, x_plane, pitch, p01,
        atol=0.025, smooth=True, window_length=15, polyorder=4)
    if mises_res is not None:
        mises_pitch = mises_res['y_norm'].values
        mises_loss = mises_res['loss'].values
    else:
        mises_pitch = mises_loss = np.array([])

    return exp_mach_s, exp_mach, exp_cf_s, exp_cf, mises_pitch, mises_loss


def plot_surface(run1, run2, quantity, data1, data2,
                 exp_s=None, exp_data=None,
                 mirror_ps=False, blade="Blade", out_dir=Path('.')):
    """Helper to plot surface quantities for two runs."""
    sps1 = -run1['s_ps'] if mirror_ps else run1['s_ps']
    sps2 = -run2['s_ps'] if mirror_ps else run2['s_ps']
    plt.plot(run1['s_ss'], data1[0], color='C0', linestyle='-', label='Test 1')
    plt.plot(sps1, data1[1], color='C0', linestyle='-')
    plt.plot(run2['s_ss'], data2[0], color='C1', linestyle='--', label='Test 2')
    plt.plot(sps2, data2[1], color='C1', linestyle='--')
    if exp_s is not None and exp_data is not None:
        plt.scatter(exp_s, exp_data, s=0.5, color='red', label='MISES')
    plt.xlabel(r'S/S_{total}' if not mirror_ps else r'S/S_{total} (PS<0)')
    plt.ylabel(f'{quantity} - {blade}')
    if mirror_ps:
        plt.xlim(-1, 1)
    else:
        plt.xlim(0, 1)
    plt.legend(loc='upper left', edgecolor='k', fancybox=False)
    fname = f'comparison_{quantity.replace(" ", "_")}_{blade}.svg'
    plt.savefig(out_dir / fname, format='svg', bbox_inches='tight')
    plt.show()


def plot_loss(run1, run2, mises_pitch, mises_loss, blade="Blade", out_dir=Path('.')):
    plt.plot(run1['y_loss'], run1['loss'], color='C0', linestyle='-', label='Test 1')
    plt.plot(run2['y_loss'], run2['loss'], color='C1', linestyle='--', label='Test 2')
    if len(mises_pitch):
        plt.plot(mises_pitch, mises_loss, color='red', linestyle=':', label='MISES')
    plt.xlabel('y/pitch')
    plt.ylabel(f'Total pressure loss - {blade}')
    plt.xlim(-0.6, 0.6)
    plt.legend(loc='upper left', edgecolor='k', fancybox=False)
    fname = f'comparison_pressure_loss_{blade}.svg'
    plt.savefig(out_dir / fname, format='svg', bbox_inches='tight')
    plt.show()


def plot_residuals(run1, run2, blade="Blade", out_dir=Path('.')):
    colors = ['C0', 'C1', 'C2']
    labels = [r'$\rho$', r'$\rho u$', r'$\rho E$']
    cols = ['rho', 'rhou', 'rhoe']
    for col, label, color in zip(cols, labels, colors):
        plt.plot(run1['iter'], run1[col], color=color, linestyle='-', label=f'{label} - Test 1')
        plt.plot(run2['iter'], run2[col], color=color, linestyle='--', label=f'{label} - Test 2')
    plt.xlabel('Iteration')
    plt.ylabel(f'RMS residual - {blade}')
    plt.legend(loc='upper right', ncol=2, edgecolor='k', fancybox=False)
    fname = f'comparison_rms_residual_{blade}.svg'
    plt.savefig(out_dir / fname, format='svg', bbox_inches='tight')
    plt.show()


def main():
    sys.argv = ['python model_comparison.py', 
                'Blade_2',
                'Test_1_25-07-2025',
                'Test_1_01-08-2025',
                #'--label1', 'SA-BCM',
                #'--label2', 'k-w-SST-LM',
                '--output', 'Blades/Blade_2/results/plots']
    
    parser = argparse.ArgumentParser(description='Compare two SU2 runs for the same blade.')
    parser.add_argument('blade', help='Blade name, e.g. Blade_1')
    parser.add_argument('test1', help='Name of first test folder inside results directory')
    parser.add_argument('test2', help='Name of second test folder inside results directory')
    parser.add_argument('--output', type=Path, default=None, help='Directory for saving plots (default: test1 folder)')
    args = parser.parse_args()

    blade_dir = Path('Blades') / args.blade
    run1_dir = blade_dir / 'results' / args.test1
    run2_dir = blade_dir / 'results' / args.test2

    run1 = load_run(run1_dir)
    run2 = load_run(run2_dir)
    if run1['blade'] != run2['blade']:
        raise ValueError('Runs correspond to different blades')

    out_dir = args.output or run1_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_mach_s, exp_mach, exp_cf_s, exp_cf, mises_pitch, mises_loss = (
        load_mises(blade_dir, run1['string'], run1['x_plane'], run1['pitch'], run1['p01']))

    blade = run1['blade']
    plot_surface(run1, run2, 'Mach Number',
                 (run1['mach_ss'], run1['mach_ps']),
                 (run2['mach_ss'], run2['mach_ps']),
                 exp_s=exp_mach_s, exp_data=exp_mach,
                 mirror_ps=False, blade=blade, out_dir=out_dir)

    plot_surface(run1, run2, 'Skin Friction Coefficient',
                 (run1['cf_ss'], run1['cf_ps']),
                 (run2['cf_ss'], run2['cf_ps']),
                 exp_s=exp_cf_s, exp_data=exp_cf,
                 mirror_ps=True, blade=blade, out_dir=out_dir)

    plot_loss(run1, run2, mises_pitch, mises_loss, blade=blade, out_dir=out_dir)
    plot_residuals(run1, run2, blade=blade, out_dir=out_dir)


if __name__ == '__main__':
    main()