# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 14:52:24 2025

@author: fredd
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{helvet}"
})


def load_su2_surface(run_dir: Path):
    """Extract surface data from a single SU2 run directory."""
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

    yplus_ss = ss['Y_Plus'].values
    yplus_ps = ps['Y_Plus'].values

    cf_ss = np.sqrt(ss['Skin_Friction_Coefficient_x'].values**2 +
                    ss['Skin_Friction_Coefficient_y'].values**2)
    cf_ps = np.sqrt(ps['Skin_Friction_Coefficient_x'].values**2 +
                    ps['Skin_Friction_Coefficient_y'].values**2)

    return {
        'run_dir': run_dir,
        'string': string,
        'blade': blade,
        's_ss': s_ss,
        's_ps': s_ps,
        'mach_ss': mach_ss,
        'mach_ps': mach_ps,
        'yplus_ss': yplus_ss,
        'yplus_ps': yplus_ps,
        'cf_ss': cf_ss,
        'cf_ps': cf_ps,
    }


def load_experimental(blade_dir: Path, string: str):
    """Load experimental Mach and skin friction distributions."""
    mach_file = blade_dir / f"machDistribution.{string}"
    ps_frac, ss_frac, ps_mach, ss_mach = utils.MISES_machDataGather(mach_file)
    if len(ps_frac) and len(ss_frac):
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

    return exp_mach_s, exp_mach, exp_cf_s, exp_cf


def plot_comparison(run1, run2, quantity, data1, data2,
                    exp_s=None, exp_data=None,
                    label1='SA-BCM', label2='k-w-SST-LM',
                    blade='Blade', out_dir=Path('.')):
    """Generic helper for comparative plotting."""
    plt.plot(run1['s_ss'], data1[0], color='tab:blue', label=label1)
    plt.plot(run1['s_ps'], data1[1], color='tab:blue')
    plt.plot(run2['s_ss'], data2[0], color='tab:green', label=label2)
    plt.plot(run2['s_ps'], data2[1], color='tab:green')
    if exp_s is not None and exp_data is not None:
        plt.scatter(exp_s, exp_data, s=0.5, color='red', label='EXP')
    plt.xlabel(r'S/S_{total}')
    plt.ylabel(f'{quantity} - {blade}')
    plt.xlim(0, 1)
    plt.legend(loc='upper left', edgecolor='k', fancybox=False)
    fname = f'comparison_{quantity.replace(" ", "_")}_{blade}.svg'
    plt.savefig(out_dir / fname, format='svg', bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare two SU2 turbulence model runs.')
    parser.add_argument('test1', type=Path, help='Run directory for first model (e.g. SA-BCM).')
    parser.add_argument('test2', type=Path, help='Run directory for second model (e.g. SST-LM).')
    parser.add_argument('--label1', default='SA-BCM', help='Legend label for first run.')
    parser.add_argument('--label2', default='k-w-SST-LM', help='Legend label for second run.')
    parser.add_argument('--output', type=Path, default=None, help='Directory for saving plots (default: test1).')
    args = parser.parse_args()

    run1 = load_su2_surface(args.test1)
    run2 = load_su2_surface(args.test2)

    if run1['blade'] != run2['blade']:
        raise ValueError('Runs correspond to different blades')

    out_dir = args.output or run1['run_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    blade_dir = run1['run_dir'].parent.parent
    exp_mach_s, exp_mach, exp_cf_s, exp_cf = load_experimental(blade_dir, run1['string'])

    plot_comparison(run1, run2, 'Mach Number',
                    (run1['mach_ss'], run1['mach_ps']),
                    (run2['mach_ss'], run2['mach_ps']),
                    exp_s=exp_mach_s, exp_data=exp_mach,
                    label1=args.label1, label2=args.label2,
                    blade=run1['blade'], out_dir=out_dir)

    plot_comparison(run1, run2, 'Y Plus',
                    (run1['yplus_ss'], run1['yplus_ps']),
                    (run2['yplus_ss'], run2['yplus_ps']),
                    label1=args.label1, label2=args.label2,
                    blade=run1['blade'], out_dir=out_dir)

    plot_comparison(run1, run2, 'Skin Friction Coefficient',
                    (run1['cf_ss'], run1['cf_ps']),
                    (run2['cf_ss'], run2['cf_ps']),
                    exp_s=exp_cf_s, exp_data=exp_cf,
                    label1=args.label1, label2=args.label2,
                    blade=run1['blade'], out_dir=out_dir)


if __name__ == '__main__':
    main()
