# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 16:43:50 2025

@author: fredd

Compare two SU2 SPLEEN runs for a given blade.

This routine plots, in the same figure, the Mach number distribution,
skin–friction coefficient and total pressure loss  for two separate runs. 
Experimental results are also overlaid when available.
"""


import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import utils
from analysis_spleen import load_exp_blade_pt, load_exp_pl06


def _parse_params(run_dir: Path):
    keys = {
        'bladeName': str,
        'string': str,
        'P01': float,
        'gamma': float,
        'alpha2_deg': float,
        'x_plane': float,
        'pitch': float,
        'axial_chord': float,
        'sizeCellFluid': float,
        'Re': float,
        'M2': float,
    }
    params = {}
    rerun = run_dir / 'rerun.py'
    if not rerun.exists():
        raise FileNotFoundError(f'Could not find rerun.py in {run_dir}')
    with open(rerun, 'r') as f:
        for line in f:
            line = line.strip()
            for k, cast in keys.items():
                if line.startswith(f"{k} ="):
                    val = line.split('=', 1)[1].split('#')[0].strip().strip('"\'')
                    params[k] = cast(val)
    missing = [k for k in keys if k not in params]
    if missing:
        raise ValueError(f'Missing parameters in rerun.py: {missing}')
    return params


def _load_run(run_dir: Path):
    params = _parse_params(run_dir)
    blade = params['bladeName']
    string = params['string']

    surf_file = run_dir / f'surface_flow{string}_{blade}.csv'
    restart_file = run_dir / f'restart_flow_{string}_{blade}.csv'

    df = pd.read_csv(surf_file)
    _, _, dataSS, dataPS = utils.SU2_organize(df)

    s_normSS = utils.surface_fraction(dataSS['x'].values, dataSS['y'].values)
    s_normPS = utils.surface_fraction(dataPS['x'].values, dataPS['y'].values)
    machSS = utils.compute_Mx(params['P01'], dataSS['Pressure'].values, params['gamma'])
    machPS = utils.compute_Mx(params['P01'], dataPS['Pressure'].values, params['gamma'])
    cfSS = dataSS['Skin_Friction_Coefficient_x'].values
    cfPS = dataPS['Skin_Friction_Coefficient_x'].values

    vol_df = pd.read_csv(restart_file)
    p_loc = (params['x_plane'] + 1) * params['axial_chord']
    res = utils.SU2_total_pressure_loss(
        vol_df,
        p_loc,
        params['pitch'],
        params['P01'],
        params['alpha2_deg'],
        atol=params['sizeCellFluid']/2,
        smooth=True,
        window_length=15,
        polyorder=4,
    )
    y_norm = res['y_norm']
    loss = res['loss']
    order = np.argsort(y_norm)
    y_norm = y_norm[order]
    loss = loss[order]

    return {
        'params': params,
        's_normSS': s_normSS,
        's_normPS': s_normPS,
        'machSS': machSS,
        'machPS': machPS,
        'cfSS': cfSS,
        'cfPS': cfPS,
        'pitch': y_norm,
        'loss': loss,
    }


def compare_runs(run1: Path, run2: Path):
    run1 = run1.resolve()
    run2 = run2.resolve()
    data1 = _load_run(run1)
    data2 = _load_run(run2)

    base_dir = Path(__file__).resolve().parent
    p = data1['params']
    Re_tag = str(int(round(p['Re'] / 1000)))
    M_tag = f"{int(round(p['M2'] * 100)):03d}"
    ss_frac, ps_frac, ss_mach, ps_mach = load_exp_blade_pt(
        base_dir, p['P01'], p['gamma'], Re_tag, M_tag)
    exp_x = np.concatenate([ps_frac, ss_frac])
    exp_m = np.concatenate([ps_mach, ss_mach])
    exp_pitch, exp_loss = load_exp_pl06(base_dir, p['P01'], Re_tag, M_tag)

    out_dir = run1.parent / f"Comparison_{run1.name}_vs_{run2.name}"
    out_dir.mkdir(exist_ok=True)
    blade = p['bladeName']

    # Mach number distribution
    plt.figure()
    plt.plot(data1['s_normSS'], data1['machSS'], color='C0', linestyle='-', label='SA-BCM')
    plt.plot(data1['s_normPS'], data1['machPS'], color='C0', linestyle='-')
    plt.plot(data2['s_normSS'], data2['machSS'], color='C1', linestyle='--', label='k-w-SST-LM')
    plt.plot(data2['s_normPS'], data2['machPS'], color='C1', linestyle='--')
    plt.scatter(exp_x, exp_m, s=2, color='red', label='Exp')
    plt.xlabel(r'S/S_{total}')
    plt.ylabel(f'Mach Number - {blade}')
    plt.legend()
    plt.savefig(out_dir / f'mach_comparison_{blade}.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Skin friction coefficient
    plt.figure()
    plt.plot(data1['s_normSS'], data1['cfSS'], color='C0', linestyle='-', label='SA-BCM')
    plt.plot(-data1['s_normPS'], data1['cfPS'], color='C0', linestyle='-')
    plt.plot(data2['s_normSS'], data2['cfSS'], color='C1', linestyle='--', label='k-w-SST-LM')
    plt.plot(-data2['s_normPS'], data2['cfPS'], color='C1', linestyle='--')
    plt.xlabel(r'S/S_{total}')
    plt.ylabel(f'Skin Friction Coefficient - {blade}')
    plt.xlim(-1, 1)
    plt.legend()
    plt.savefig(out_dir / f'cf_comparison_{blade}.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Pressure loss
    plt.figure()
    plt.plot(data1['pitch'], data1['loss'], color='C0', linestyle='-', label='SA-BCM')
    plt.plot(data2['pitch'], data2['loss'], color='C1', linestyle='--', label='k-w-SST-LM')
    plt.scatter(exp_pitch, exp_loss, s=2, color='red', label='Exp')
    plt.xlabel('y/pitch')
    plt.ylabel(f'Total pressure loss - {blade}')
    plt.xlim(-0.6, 0.6)
    plt.legend()
    plt.savefig(out_dir / f'loss_comparison_{blade}.svg', format='svg', bbox_inches='tight')
    plt.show()


def main():
    sys.argv = ['python model_comparison_spleen.py', 
                'Blades/Blade_0/results/Test_1_31-07-2025',
                'Blades/Blade_0/results/Test_2_31-07-2025',
                #'--label1', 'SA-BCM',
                #'--label2', 'k-w-SST-LM',
                #'--output', 'Blades/Blade_0/results/plots'
                ]
    parser = argparse.ArgumentParser(description='Compare two SPLEEN runs')
    parser.add_argument('run1', type=Path, help='Path to first test directory')
    parser.add_argument('run2', type=Path, help='Path to second test directory')
    args = parser.parse_args()
    compare_runs(args.run1, args.run2)


if __name__ == '__main__':
    main()