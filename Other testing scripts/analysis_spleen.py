
"""
Created on Thu Feb 13 17:54:39 2025

@author: Freddy Chica
@co-author: Francesco Porta

Notice: GPT-4o was heavily used for the elaboration of this script

Run SU2 analysis for the SPLEEN blade using the available
experimental data. The routine mirrors ``analysis_datablade`` but
operates only on the SPLEEN geometry delivered as CSV files.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt
import utils
import mesh_datablade
import configSU2_datablade
import post_processing_datablade


# ----------------------------------------------------------------------
# helper : convert provided CSV geometry to a Selig style .dat file
# ----------------------------------------------------------------------
def csv_to_dat(csv_path: Path, dat_path: Path) -> None:
    """Convert the SPLEEN CSV geometry into a simple Selig dat format."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=2)
    px, py, sx, sy = data[:,0], data[:,1], data[:,2], data[:,3]
    with open(dat_path, "w") as f:
        f.write("spleen\n")
        f.write("generated from csv\n")
        # start at TE on the pressure side and walk to the LE
        for x, y in zip(px[::-1], py[::-1]):
            f.write(f"{x:.8e} {y:.8e}\n")
        # then LE->TE on the suction side
        for x, y in zip(sx, sy):
            f.write(f"{x:.8e} {y:.8e}\n")


# ----------------------------------------------------------------------
# experimental data loaders
# ----------------------------------------------------------------------
def load_exp_blade_pt(base_dir: Path, P01: float, gamma: float,
                      Re_tag: str, M_tag: str):
    """Return experimental arc fraction and Mach number from Blade/PT file."""
    filesBlade = base_dir / "Experimental Data" / "Blade"
    fname = f"SPLEEN_C1_NC_St000_Re{Re_tag}_M{M_tag}_Blade_PT_s4970.xlsx"
    df = pd.read_excel(filesBlade / fname, usecols=[1, 2, 3])
    df.columns = ['x_over_c', 's_norm', 'Ps_P01']

    inside = (2.0/(gamma-1))*(df['Ps_P01']**((1-gamma)/gamma)-1.0)
    df['Mach'] = np.sqrt(np.clip(inside, 0.0, None))

    ss_mask = df['s_norm'] >= 0
    ps_mask = ~ss_mask
    ss_frac = df.loc[ss_mask, 's_norm'].to_numpy()
    ss_mach = df.loc[ss_mask, 'Mach'].to_numpy()
    ps_frac = -df.loc[ps_mask, 's_norm'].to_numpy()
    ps_mach = df.loc[ps_mask, 'Mach'].to_numpy()
    return ss_frac, ps_frac, ss_mach, ps_mach


def load_exp_pl06(base_dir: Path, P01: float, Re_tag: str, M_tag: str):
    """Return experimental PL06 pitch fraction and loss."""
    filesPL06 = base_dir / "Experimental Data" / "PL06"
    fname = f"SPLEEN_C1_NC_St000_Re{Re_tag}_M{M_tag}_PL06_L5HP_s5000.xlsx"
    df = pd.read_excel(filesPL06 / fname, usecols=[1,3,4,5,6,7,8,9])
    df.columns = ['y_g','d','pitch','rho','V_ax','P06_P01','Ps6_P01','ksi']
    df['P06'] = df['P06_P01'] * P01
    df['loss'] = (P01 - df['P06']) / P01
    return df['y_g'].to_numpy(), df['loss'].to_numpy()


def post_processing_spleen(run_dir: Path, base_dir: Path,
                           bladeName: str,
                           string: str,
                           P01: float, alpha2: float,
                           x_plane: float, pitch: float,
                           axial_chord: float,
                           sizeCellFluid: float,
                           gamma: float,
                           Re_tag: str = '70',
                           M_tag: str = '090'):
    """Plot SU2 results against SPLEEN experimental data."""
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   HISTORY FILE TRACKING - Residuals, Linear Solvers, CFL, CD, CL
    # ─────────────────────────────────────────────────────────────────────────────
    
    residuals_file = run_dir / f'history_{string}_{bladeName}.csv'
    hist = pd.read_csv(residuals_file)
    total_time = hist['    "Time(sec)"   '].sum()
    last_iter = hist['Inner_Iter'].iloc[-1]

    # RMS Tracking
    plt.plot(hist['Inner_Iter'], hist['    "rms[Rho]"    '], label=r'$\rho$')               # Density
    plt.plot(hist['Inner_Iter'], hist['    "rms[RhoU]"   '], label=r'$\rho u$')             # Momentum-x
    plt.plot(hist['Inner_Iter'], hist['    "rms[RhoE]"   '], label=r'$\rho E$')             # Energy
    #plt.plot(hist['Inner_Iter'], hist['    "rms[RhoV]"   '], label=r'$\rho v$')            # Momentum-y
    #plt.plot(hist['Inner_Iter'], hist['     "rms[nu]"    '], label='v')                    # Viscosity
    #plt.plot(hist['Inner_Iter'], hist['     "rms[k]"    '], label='k')                     # TKE
    #plt.plot(hist['Inner_Iter'], hist['     "rms[w]"    '], label='w')
    #plt.grid(alpha=0.3);
    ax = plt.gca()
    ax_time = ax.twinx()
    ax_time.plot([0, last_iter], [0, total_time], color='grey', linestyle='--', linewidth=1,
                 label='Time [s]')
    ax_time.set_ylabel('Time [s]')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_time.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'RMS residual - {bladeName}')
    plt.savefig(run_dir / f'rms_residual_{string}_{bladeName}.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Linear Solver Tracking
    plt.plot(hist['Inner_Iter'], hist['    "LinSolRes"   '], label='LSRes')                 # Linear Solver Residual
    plt.plot(hist['Inner_Iter'], hist['  "LinSolResTurb" '], label='LSResTurb')             # Linear Solver Residual
    #plt.grid(alpha=0.3);  
    plt.legend();  plt.xlabel('Iteration')
    plt.ylabel(f'Linear Solver residual - {bladeName}');
    plt.savefig(run_dir / f'linear_solver_residual_{string}_{bladeName}.svg', format='svg', bbox_inches='tight')
    plt.show()

    # CFL Tracking
    plt.plot(hist['Inner_Iter'], hist['     "Avg CFL"    '], label='CFL')                   # CFL used per iteration
    #plt.grid(alpha=0.3);  
    plt.legend();  plt.xlabel('Iteration')
    plt.ylabel(f'Average CFL - {bladeName}');
    plt.savefig(run_dir / f'cfl_{string}_{bladeName}.svg', format='svg', bbox_inches='tight')
    plt.show()

    # Aero Coefficients Tracking
    plt.plot(hist['Inner_Iter'], hist['   "CD(blade1)"   '], label='CD')                    # Drag Coefficient
    plt.plot(hist['Inner_Iter'], hist['   "CL(blade1)"   '], label='CL')                    # Lift Coefficient
    #plt.grid(alpha=0.3);  
    plt.legend();  plt.xlabel('Iteration')
    plt.ylabel(f'Aerodynamic Coefficients - {bladeName}');
    plt.savefig(run_dir / f'aero_coefficients_{string}_{bladeName}.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   SIMULATIONS DATA
    # ─────────────────────────────────────────────────────────────────────────────    
    
    su2_file = run_dir / f"surface_flowdatabladeVALIDATION_{bladeName}.csv"
    df = pd.read_csv(su2_file)
    _, _, dataSS, dataPS = utils.SU2_organize(df)

    s_normSS = utils.surface_fraction(dataSS['x'].values, dataSS['y'].values)
    s_normPS = utils.surface_fraction(dataPS['x'].values, dataPS['y'].values)
    machSS = utils.compute_Mx(P01, dataSS['Pressure'].values, gamma)
    machPS = utils.compute_Mx(P01, dataPS['Pressure'].values, gamma)

    yPlusSS = dataSS['Y_Plus'].values
    yPlusPS = dataPS['Y_Plus'].values
    cfSS = dataSS['Skin_Friction_Coefficient_x'].values
    cfPS = dataPS['Skin_Friction_Coefficient_x'].values

    ss_frac_e, ps_frac_e, ss_mach_e, ps_mach_e = load_exp_blade_pt(
        base_dir, P01, gamma, Re_tag, M_tag)
    exp_x = np.concatenate([ps_frac_e, ss_frac_e])
    exp_m = np.concatenate([ps_mach_e, ss_mach_e])
    case_label = f"Re={Re_tag}k, M={float(M_tag)/100:.2f}"

    utils.SU2_DataPlotting(s_normSS, s_normPS, machSS, machPS,
                           "Mach Number", 'databladeVALIDATION', run_dir, bladeName,
                           mirror_PS=False, exp_s=exp_x, exp_data=exp_m,
                           case_label=case_label)

    utils.SU2_DataPlotting(s_normSS, s_normPS, yPlusSS, yPlusPS,
                           "Y Plus", 'databladeVALIDATION', run_dir, bladeName,
                           mirror_PS=True, case_label=case_label)

    utils.SU2_DataPlotting(s_normSS, s_normPS, cfSS, cfPS,
                           "Skin Friction Coefficient", 'databladeVALIDATION', run_dir, bladeName,
                           mirror_PS=True, case_label=case_label)

    restart_file = run_dir / f"restart_flow_databladeVALIDATION_{bladeName}.csv"
    vol_df = pd.read_csv(restart_file)
    p_loc = (x_plane + 1) * axial_chord
    su2_res = utils.SU2_total_pressure_loss(vol_df, p_loc, pitch, P01, alpha2,
                                            atol=sizeCellFluid/2, smooth=True,
                                            window_length=15, polyorder=4)
    su2_pitch, su2_loss = su2_res['y_norm'], su2_res['loss']

    exp_pitch, exp_loss = load_exp_pl06(base_dir, P01, Re_tag, M_tag)

    plt.scatter(su2_pitch, su2_loss, s=0.5, label='SU2')
    plt.scatter(exp_pitch, exp_loss, s=0.5, color='red', label='EXP')
    plt.xlabel('y/pitch')
    plt.ylabel(f'Total pressure loss - {bladeName}')
    plt.xlim(-0.6, 0.6)
    plt.legend()
    plt.title(case_label)
    plt.savefig(run_dir / f"loss_pitch_databladeVALIDATION_{bladeName}.svg",
                format='svg', bbox_inches='tight')
    plt.show()


# ----------------------------------------------------------------------
# copy of the rerun script generator from analysis_datablade
# ----------------------------------------------------------------------
def create_rerun_script(run_dir, bladeName, base_dir,
                        no_cores, string, fileExtension,
                        alpha1_deg, alpha2_deg, Re, R, gamma, mu,
                        pitch, d_factor, stagger, axial_chord, chord, pitch2chord,
                        T01, T02, T2, P01, P1, M1, M2, P2, c2, u2, rho2,
                        dist_inlet, dist_outlet, x_plane, TI,
                        sizeCellFluid, sizeCellAirfoil,
                        nCellAirfoil, nCellPerimeter, nBoundaryPoints,
                        first_layer_height, bl_growth, bl_thickness,
                        size_LE, dist_LE, size_TE, dist_TE,
                        VolWAkeIn, VolWAkeOut,
                        WakeXMin, WakeXMax):
    """Write a runnable Python script inside *run_dir* to rerun or replot."""
    date_str = datetime.now().strftime('%d-%m-%Y, %H:%M:%S')
    script_path = Path(run_dir) / "rerun.py"
    content = f"""
#!/usr/bin/env python3
#Created on {date_str}
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

bladeName = {bladeName!r}
no_cores = {no_cores}
string = {string!r}
fileExtension = {fileExtension!r}

run_dir = Path(__file__).resolve().parent
base_dir = Path(__file__).resolve().parents[4]
blade_dir = base_dir / 'Blades' / bladeName
isesFilePath = blade_dir / f'ises.{string}'
bladeFilePath = blade_dir / f'blade.{string}'

alpha1_deg = {alpha1_deg}
alpha2_deg = {alpha2_deg}
d_factor = {d_factor}
stagger = {stagger}
axial_chord = {axial_chord}
chord = {chord}
pitch = {pitch}
pitch2chord = {pitch2chord}

R = {R}
gamma = {gamma}
mu = {mu}
T01 = {T01}
P1 = {P1}
P01 = {P01}
M1 = {M1}
P2 = {P2}
M2 = {M2}
T02 = {T02}
T2 = {T2}
c2 = {c2}
u2 = {u2}
rho2 = {rho2}
Re = {Re}
TI = {TI}

dist_inlet = {dist_inlet}
dist_outlet = {dist_outlet}
x_plane = {x_plane}
sizeCellFluid = {sizeCellFluid}
sizeCellAirfoil = {sizeCellAirfoil}
nCellAirfoil = {nCellAirfoil}
nCellPerimeter = {nCellPerimeter}
nBoundaryPoints = {nBoundaryPoints}
first_layer_height = {first_layer_height}
bl_growth = {bl_growth}
bl_thickness = {bl_thickness}
size_LE = {size_LE}
dist_LE = {dist_LE}
size_TE = {size_TE}
dist_TE = {dist_TE}
VolWAkeIn = {VolWAkeIn}
VolWAkeOut = {VolWAkeOut}
WakeXMin = {WakeXMin}
WakeXMax = {WakeXMax}

for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
    mod.bladeName = bladeName
    mod.no_cores = no_cores
    mod.string = string
    mod.fileExtension = fileExtension
    mod.base_dir = base_dir
    mod.blade_dir = blade_dir
    mod.run_dir = run_dir
    mod.isesFilePath = isesFilePath
    mod.bladeFilePath = bladeFilePath

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

def rerun():
    mesh_datablade.mesh_datablade()
    configSU2_datablade.configSU2_datablade()
    configSU2_datablade.runSU2_datablade()
    post_processing_datablade.post_processing_datablade()

def replot():
    post_processing_datablade.post_processing_datablade()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['rerun', 'replot'], default='replot')
    args = parser.parse_args()
    if args.mode == 'rerun':
        rerun()
    else:
        replot()
"""
    with open(script_path, "w") as f:
        f.write(content)


# ----------------------------------------------------------------------
# main routine
# ----------------------------------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parent
    bladeName = "Blade_0"

    # ------------------ directories ------------------
    blade_dir = base_dir / "Blades" / bladeName
    blade_dir.mkdir(exist_ok=True)
    results_root = blade_dir / "results"
    results_root.mkdir(exist_ok=True)
    date_str = datetime.now().strftime('%d-%m-%Y')
    n = 1
    while (results_root / f"Test_{n}_{date_str}").exists():
        n += 1
    run_dir = results_root / f"Test_{n}_{date_str}"
    run_dir.mkdir()

    # ------------------ geometry ------------------
    csv_geom = base_dir / "Experimental Data" / "SPLEENC1_Geometry_Airfoil_2D_v1.csv"
    blade_dat = blade_dir / "blade.databladeVALIDATION"
    if not blade_dat.exists():
        csv_to_dat(csv_geom, blade_dat)

    # Geometry values taken from the SPLEEN dataset
    pitch = 32.950e-3 #[m]
    chord = 52.285e-3 #[m]
    axial_chord = 47.614e-3 #[m]
    stagger = 24.40 #[deg]
    pitch2chord = pitch / chord
    alpha1_deg = 37.3
    alpha2_deg = -53.80
    d_factor = 0.0
    
    #---- TESTING SETTINGS ----
    Re_exp = 70
    M_exp = 90
    St_test     = '000'
    Re_test     = f'{Re_exp}'
    M_test      = f'0{M_exp}'

    # Define reference arrays for indexing
    Mach_levels = [70, 80, 90, 95]      # Corresponds to M = 0.70, 0.80, etc.
    Re_levels   = [65, 70, 100, 120]    # Corresponds to Re = 65k, 70k, etc.

    # Get indices
    mach_index = Mach_levels.index(M_exp)
    re_index   = Re_levels.index(Re_exp)

    # Compute flattened index (4 elements per Mach level, in your list layout)
    flat_index = mach_index * len(Re_levels) + re_index

    # Lookup Tables
    #Re         65k    70k    100k   120k                   From Table 5.1 - Measurement Techniques
    P01_tests = [                                   #Ma
                10009, 10779, 15399, 18478,   #0.70
                9295,  10010, 14301, 17161,   #0.80
                8821,  9500,  13571, 16285,   #0.90
                8652,  9318,  13311, 15974    #0.95
                ]

    #Re         65k   70k   100k   120k                     From Table 5.1 - Measurement Techniques
    P6_tests = [                                  #Ma
                7216, 7771, 11101, 13321,   #0.70
                6098, 6567, 9381,  11258,   #0.80
                5216, 5617, 8024,  9629,    #0.90
                4841, 5213, 7447,  8937     #0.95
                ]

    # Assign values
    P01_test  = P01_tests[flat_index]
    P6_test   = P6_tests[flat_index]
    
    # Optional display
    print(f"Selected P01: {P01_test}, P6: {P6_test}")
    
    # ------------------ boundary conditions ------------------
    R = 287.058
    gamma = 1.4
    mu = 1.716e-5
    T01 = 300
    P1 = 9310.72429
    P01 = P01_test
    P2 = P6_test
    M2 = M_exp / 100
    Re = Re_exp * 1000
    
    def compute_Mx(P0x, Px, gamma):
        return math.sqrt(2/(gamma-1)*((P0x/Px)**((gamma-1)/gamma)-1))
    def compute_Tx(T0x, Mx, gamma):
        return T0x/(1+(gamma-1)/2*Mx**2)
    def compute_Vx(Mx, gamma, R, Tx):
        return Mx*math.sqrt(gamma*R*Tx)
    M1 = compute_Mx(P01, P1, gamma)
    T02 = T01
    T2 = compute_Tx(T02, M2, gamma)
    c2 = math.sqrt(gamma * R * T2)
    u2 = M2 * c2
    rho2 = mu * 70000 / (u2 * math.cos(math.radians(stagger)))
    TI = 2.0
    
    # ------------------ mesh parameters ------------------
    dist_inlet = 1.0
    dist_outlet = 2.0
    x_plane = 1.5  # PL06 location measured from LE

    sizeCellFluid = 0.21 * axial_chord
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

    # expose variables to modules
    for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
        mod.bladeName = bladeName
        mod.no_cores = 12
        mod.string = 'databladeVALIDATION'
        mod.fileExtension = 'csv'
        mod.base_dir = base_dir
        mod.blade_dir = blade_dir
        mod.run_dir = run_dir
        mod.bladeFilePath = blade_dat

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

    create_rerun_script(run_dir, bladeName, base_dir,
                        12, 'databladeVALIDATION', 'csv',
                        alpha1_deg, alpha2_deg, Re, R, gamma, mu,
                        pitch, d_factor, stagger, axial_chord, chord, pitch2chord,
                        T01, T02, T2, P01, P1, M1, M2, P2, c2, u2, rho2,
                        dist_inlet, dist_outlet, x_plane, TI,
                        sizeCellFluid, sizeCellAirfoil,
                        nCellAirfoil, nCellPerimeter, nBoundaryPoints,
                        first_layer_height, bl_growth, bl_thickness,
                        size_LE, dist_LE, size_TE, dist_TE,
                        VolWAkeIn, VolWAkeOut,
                        WakeXMin, WakeXMax)

    mesh_datablade.mesh_datablade()
    configSU2_datablade.configSU2_datablade()
    proc, logf = configSU2_datablade.runSU2_datablade(background=True)
    proc.wait()
    logf.close()
    configSU2_datablade._summarize_su2_log(run_dir / "su2.log")
    post_processing_spleen(run_dir, base_dir, bladeName, 'databladeVALIDATION',
                           P01, alpha2_deg,
                           x_plane, pitch, axial_chord,
                           sizeCellFluid, gamma, Re_tag=Re_test, M_tag=M_test)

if __name__ == '__main__':
    main()