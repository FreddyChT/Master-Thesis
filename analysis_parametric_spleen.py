# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 21:46:41 2025

@author: fredd
"""

import sys
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

# Allow importing helper utilities from the Experimental Data folder
sys.path.append(str(Path(__file__).resolve().parent / 'Experimental Data'))
import analysis_spleen as spleen


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _default_setup():
    """Return baseline parameters for the Re=70k, M=0.90 SPLEEN case."""
    base_dir = Path(__file__).resolve().parent
    bladeName = 'Blade_0'
    blade_dir = base_dir / 'Blades' / bladeName
    blade_dir.mkdir(exist_ok=True)
    results_root = blade_dir / 'results'
    results_root.mkdir(exist_ok=True)

    csv_geom = base_dir / 'Experimental Data' / 'SPLEENC1_Geometry_Airfoil_2D_v1.csv'
    blade_dat = blade_dir / 'blade.databladeVALIDATION'
    if not blade_dat.exists():
        spleen.csv_to_dat(csv_geom, blade_dat)

    pitch = 32.950e-3
    chord = 52.285e-3
    axial_chord = 47.614e-3
    stagger = 24.40
    pitch2chord = pitch / chord
    alpha1_deg = 37.3
    alpha2_deg = -53.80
    d_factor = 0.0

    Re_exp = 70
    M_exp = 90
    Mach_levels = [70, 80, 90, 95]
    Re_levels = [65, 70, 100, 120]
    mach_index = Mach_levels.index(M_exp)
    re_index = Re_levels.index(Re_exp)
    flat_index = mach_index * len(Re_levels) + re_index
    P01_tests = [10009, 10779, 15399, 18478,
                 9295, 10010, 14301, 17161,
                 8821, 9500, 13571, 16285,
                 8652, 9318, 13311, 15974]
    P6_tests = [7216, 7771, 11101, 13321,
                6098, 6567, 9381, 11258,
                5216, 5617, 8024, 9629,
                4841, 5213, 7447, 8937]
    P01 = P01_tests[flat_index]
    P2 = P6_tests[flat_index]

    R = 287.058
    gamma = 1.4
    mu = 1.716e-5
    T01 = 300
    P1 = 9310.72429
    M2 = M_exp / 100
    Re = Re_exp * 1000

    def compute_Mx(P0x, Px):
        return math.sqrt(2/(gamma-1)*((P0x/Px)**((gamma-1)/gamma)-1))
    def compute_Tx(T0x, Mx):
        return T0x/(1+(gamma-1)/2*Mx**2)

    M1 = compute_Mx(P01, P1)
    T02 = T01
    T2 = compute_Tx(T02, M2)
    c2 = math.sqrt(gamma * R * T2)
    u2 = M2 * c2
    rho2 = mu * 70000 / (u2 * math.cos(math.radians(stagger)))
    TI = 2.0

    dist_inlet = 1.0
    dist_outlet = 2.0
    x_plane = 1.5
    sizeCellFluid = 0.04 * axial_chord
    sizeCellAirfoil = 0.02 * axial_chord
    nCellAirfoil = 300
    nCellPerimeter = 183
    nBoundaryPoints = 50
    n_layers = 25
    y_plus_target = 1
    x_ref_yplus = 1/1000
    bl = utils.compute_bl_parameters(u2, rho2, mu, axial_chord,
                                     n_layers, y_plus_target, x_ref_yplus)
    first_layer_height = bl['first_layer_height']
    bl_growth = bl['bl_growth']
    bl_thickness = bl['bl_thickness']
    size_LE = 0.1 * sizeCellAirfoil
    dist_LE = 0.01 * axial_chord
    size_TE = 0.1 * sizeCellAirfoil
    dist_TE = 0.01 * axial_chord
    VolWAkeIn = 0.35 * sizeCellFluid
    VolWAkeOut = sizeCellFluid
    WakeXMin = 0.1 * axial_chord
    WakeXMax = (dist_outlet + 0.5) * axial_chord

    params = {
        'pitch': pitch,
        'chord': chord,
        'axial_chord': axial_chord,
        'stagger': stagger,
        'pitch2chord': pitch2chord,
        'alpha1_deg': alpha1_deg,
        'alpha2_deg': alpha2_deg,
        'd_factor': d_factor,
        'R': R,
        'gamma': gamma,
        'mu': mu,
        'T01': T01,
        'P1': P1,
        'P01': P01,
        'P2': P2,
        'M1': M1,
        'M2': M2,
        'T02': T02,
        'T2': T2,
        'c2': c2,
        'u2': u2,
        'rho2': rho2,
        'Re': Re,
        'TI': TI,
        'dist_inlet': dist_inlet,
        'dist_outlet': dist_outlet,
        'x_plane': x_plane,
        'sizeCellFluid': sizeCellFluid,
        'sizeCellAirfoil': sizeCellAirfoil,
        'nCellAirfoil': nCellAirfoil,
        'nCellPerimeter': nCellPerimeter,
        'nBoundaryPoints': nBoundaryPoints,
        'first_layer_height': first_layer_height,
        'bl_growth': bl_growth,
        'bl_thickness': bl_thickness,
        'size_LE': size_LE,
        'dist_LE': dist_LE,
        'size_TE': size_TE,
        'dist_TE': dist_TE,
        'VolWAkeIn': VolWAkeIn,
        'VolWAkeOut': VolWAkeOut,
        'WakeXMin': WakeXMin,
        'WakeXMax': WakeXMax,
    }
    return params, blade_dat, results_root, bladeName, base_dir


def _assign_to_modules(params: Dict, run_dir: Path, blade_dat: Path,
                       bladeName: str, base_dir: Path) -> None:
    for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
        mod.bladeName = bladeName
        mod.no_cores = 12
        mod.string = 'databladeVALIDATION'
        mod.fileExtension = 'csv'
        mod.base_dir = base_dir
        mod.blade_dir = blade_dat.parent
        mod.run_dir = run_dir
        mod.bladeFilePath = blade_dat
        for k, v in params.items():
            setattr(mod, k, v)
        # Many modules expect ``alpha1``/``alpha2`` without the ``_deg`` suffix.
        # Provide those aliases to avoid ``NameError`` in routines like
        # ``mesh_datablade`` or ``configSU2_datablade``.
        mod.alpha1 = params['alpha1_deg']
        mod.alpha2 = params['alpha2_deg']


def _run_case(case_id: int, params: Dict, blade_dat: Path,
              results_root: Path, bladeName: str, base_dir: Path) -> Dict:
    run_dir = results_root / f'Case_{case_id}'
    # Allow rerunning the parametric analysis without raising an error when
    # the case directory already exists from a previous run.
    run_dir.mkdir(exist_ok=True)
    _assign_to_modules(params, run_dir, blade_dat, bladeName, base_dir)

    spleen.create_rerun_script(
        run_dir, bladeName, base_dir,
        12, 'databladeVALIDATION', 'csv',
        params['alpha1_deg'], params['alpha2_deg'], params['Re'],
        params['R'], params['gamma'], params['mu'],
        params['pitch'], params['d_factor'], params['stagger'],
        params['axial_chord'], params['chord'], params['pitch2chord'],
        params['T01'], params['T02'], params['T2'], params['P01'], params['P1'],
        params['M1'], params['M2'], params['P2'], params['c2'], params['u2'],
        params['rho2'], params['dist_inlet'], params['dist_outlet'],
        params['x_plane'], params['TI'], params['sizeCellFluid'],
        params['sizeCellAirfoil'], params['nCellAirfoil'],
        params['nCellPerimeter'], params['nBoundaryPoints'],
        params['first_layer_height'], params['bl_growth'],
        params['bl_thickness'], params['size_LE'], params['dist_LE'],
        params['size_TE'], params['dist_TE'], params['VolWAkeIn'],
        params['VolWAkeOut'], params['WakeXMin'], params['WakeXMax'])

    mesh_datablade.mesh_datablade()
    configSU2_datablade.configSU2_datablade()
    proc, logf = configSU2_datablade.runSU2_datablade(background=True)
    ret = proc.wait(); logf.close()
    configSU2_datablade._summarize_su2_log(run_dir / 'su2.log')

    hist_path = run_dir / f"history_databladeVALIDATION_{bladeName}.csv"
    surf_path = run_dir / f"surface_flowdatabladeVALIDATION_{bladeName}.csv"
    restart_path = run_dir / f"restart_flow_databladeVALIDATION_{bladeName}.csv"
    if ret != 0 or not (hist_path.exists() and surf_path.exists() and restart_path.exists()):
        return {'case_id': case_id, 'run_dir': run_dir, 'diverged': True}

    # Extract results
    hist = pd.read_csv(hist_path)
    cl = hist['   "CL(blade1)"   '].iloc[-1]
    cd = hist['   "CD(blade1)"   '].iloc[-1]

    surf = pd.read_csv(surf_path)
    _, _, dataSS, dataPS = utils.SU2_organize(surf)
    # Ensure both sides are oriented from leading edge to trailing edge
    if dataPS['x'].iloc[0] > dataPS['x'].iloc[-1]:
        dataPS = dataPS.iloc[::-1].reset_index(drop=True)
    if dataSS['x'].iloc[0] > dataSS['x'].iloc[-1]:
        dataSS = dataSS.iloc[::-1].reset_index(drop=True)

    # Global surface fraction from 0→1 along pressure then suction side
    def _arc_len(df):
        dx = np.diff(df['x'].values)
        dy = np.diff(df['y'].values)
        seg = np.sqrt(dx**2 + dy**2)
        arc = np.cumsum(seg)
        arc = np.insert(arc, 0, 0.0)
        return arc, arc[-1]

    arc_ps, len_ps = _arc_len(dataPS)
    arc_ss, len_ss = _arc_len(dataSS)
    # Normalize each side independently.  The experimental dataset already has
    # the pressure side oriented from trailing edge to leading edge, so mirror
    # the SU2 pressure-side data to match that convention.
    s_ps = arc_ps / len_ps
    s_ss = arc_ss / len_ss
    mach_ps = utils.compute_Mx(
        params['P01'], dataPS['Pressure'].values, params['gamma'])
    mach_ss = utils.compute_Mx(
        params['P01'], dataSS['Pressure'].values, params['gamma'])

    # Reorder the pressure-side arrays from trailing edge to leading edge so
    # that they share the same orientation as the experimental measurements.
    s_ps = s_ps[::-1]
    mach_ps = mach_ps[::-1]
    # Wake loss distribution
    vol_df = pd.read_csv(restart_path)
    p_loc = (params['x_plane'] + 1) * params['axial_chord']
    loss_df = utils.SU2_total_pressure_loss(
        vol_df, p_loc, params['pitch'], params['P01'], params['alpha2_deg'],
        atol=params['sizeCellFluid']/2, smooth=True, window_length=15, polyorder=4,
    )
    wake_y = loss_df['y_norm'].to_numpy()
    wake_loss = loss_df['loss'].to_numpy()
    order = np.argsort(wake_y)
    wake_y = wake_y[order]
    wake_loss = wake_loss[order]

    return {
        'case_id': case_id,
        'run_dir': run_dir,
        'cl': cl,
        'cd': cd,
        's_ss': s_ss,
        's_ps': s_ps,
        'mach_ss': mach_ss,
        'mach_ps': mach_ps,
        'wake_y': wake_y,
        'wake_loss': wake_loss,
    }


def main() -> None:
    params, blade_dat, results_root, bladeName, base_dir = _default_setup()

    print('Available variables to vary:')
    options = ['first_layer_height', 'bl_growth', 'bl_thickness',
               'sizeCellFluid', 'VolWAkeIn', 'VolWAkeOut',
               'dist_inlet', 'dist_outlet', 'x_plane']
    for v in options:
        print(' -', v, f'(default={params[v]})')
    chosen = input('Enter variables to modify (comma separated): ').split(',')
    chosen = [c.strip() for c in chosen if c.strip() in options]

    values: Dict[str, List[float]] = {}
    for var in chosen:
        raw = input(f'Provide three comma separated values for {var}: ')
        vals = [float(x) for x in raw.split(',')]
        if len(vals) != 3:
            raise ValueError(f'{var} requires exactly three values')
        values[var] = vals

    # Load experimental Mach distribution once
    ss_frac_e, ps_frac_e, ss_mach_e, ps_mach_e = spleen.load_exp_blade_pt(
        base_dir, params['P01'], params['gamma'], '70', '090')
    # Experimental wake loss distribution
    exp_pitch, exp_loss = spleen.load_exp_pl06(base_dir, params['P01'], '70', '090')
    order = np.argsort(exp_pitch)
    exp_pitch = exp_pitch[order]
    exp_loss = exp_loss[order]

    results = []
    for i in range(3):
        case_params = params.copy()
        for var in values:
            case_params[var] = values[var][i]
        res = _run_case(i+1, case_params, blade_dat, results_root, bladeName, base_dir)
        if res.get('diverged'):
            print(f'Case {i+1} diverged; excluding from plots.')
            continue
        results.append(res)

    case_ids = [r['case_id'] for r in results]
    # Experimental data already carries the correct orientation; keep as provided
    exp_x = np.concatenate([ps_frac_e, ss_frac_e])
    exp_m = np.concatenate([ps_mach_e, ss_mach_e])
    # Plot CL
    plt.figure()
    plt.plot(case_ids, [r['cl'] for r in results], marker='o')
    plt.xlabel('Case')
    plt.ylabel('C_l')
    plt.title('Lift coefficient')
    plt.xticks(case_ids, [f'Case {cid}' for cid in case_ids])
    plt.savefig(results_root / 'CL_comparison.svg', format='svg', bbox_inches='tight')

    # Plot CD
    plt.figure()
    plt.plot(case_ids, [r['cd'] for r in results], marker='o')
    plt.xlabel('Case')
    plt.ylabel('C_d')
    plt.title('Drag coefficient')
    plt.xticks(case_ids, [f'Case {cid}' for cid in case_ids])
    plt.savefig(results_root / 'CD_comparison.svg', format='svg', bbox_inches='tight')

    # Mach distribution plot
    plt.figure()
    plt.scatter(exp_x, exp_m, s=2, color='k', label='EXP')
    for r in results:
        s_comb = np.concatenate([r['s_ps'], r['s_ss']])
        m_comb = np.concatenate([r['mach_ps'], r['mach_ss']])
        plt.plot(s_comb, m_comb, label=f"Case {r['case_id']}")
    plt.xlabel(r'S/S_{side}')
    plt.ylabel('Mach number')
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig(results_root / 'Mach_comparison.svg', format='svg', bbox_inches='tight')

    # Wake total pressure loss plot
    plt.figure()
    plt.scatter(exp_pitch, exp_loss, s=2, color='k', label='EXP')
    min_loss = exp_loss.min()
    max_loss = exp_loss.max()
    for r in results:
        plt.plot(r['wake_y'], r['wake_loss'], label=f"Case {r['case_id']}")
        min_loss = min(min_loss, r['wake_loss'].min())
        max_loss = max(max_loss, r['wake_loss'].max())
    plt.xlabel('y/pitch')
    plt.ylabel('Total pressure loss')
    plt.xlim(-0.6, 0.6)
    plt.ylim(min_loss, max_loss)
    plt.legend()
    plt.savefig(results_root / 'Wake_loss_comparison.svg', format='svg', bbox_inches='tight')


if __name__ == '__main__':
    main()

