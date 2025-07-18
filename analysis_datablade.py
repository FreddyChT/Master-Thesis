# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:54:39 2025

@author: Freddy Chica
@co-author: Francesco Porta

Disclaimer: GPT-o3 & Codex were heavily used for the elaboration of this script
"""

import argparse
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import time
import sys
import utils
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

# You will need the following files to run this analysis:
# - Ises file
# - Blade file
# - Gridpar file
# - Mach Distribution file
# Others to be determined when checking other files

# ---------------------- USER PARAMETERS ----------------------
# Defaults can be overridden from the command line
BLADEROOT = Path(__file__).resolve().parent

def create_rerun_script(run_dir, bladeName, base_dir,
                        no_cores, string, fileExtension,
                        alpha1_deg, alpha2_deg, Re, R, gamma, mu, 
                        pitch, d_factor, stagger, axial_chord, chord, pitch2chord,
                        T01, T02, T2, P01, P1, M1, P2_P0a, M2, P2, c2, u2, rho2,
                        dist_inlet, dist_outlet, TI,
                        sizeCellFluid, sizeCellAirfoil,
                        nCellAirfoil, nCellPerimeter, nBoundaryPoints,
                        first_layer_height, bl_growth, bl_thickness,
                        size_LE, dist_LE, size_TE, dist_TE,
                        VolWAkeIn, VolWAkeOut,
                        WakeXMin, WakeXMax, WakeYMin, WakeYMax):
    """Write a runnable Python script inside *run_dir* to rerun or replot."""
    date_str = datetime.now().strftime('%d-%m-%Y, %H:%M:%S')
    script_path = Path(run_dir) / "rerun.py"
    content = f"""
#!/usr/bin/env python3
#Created on {date_str}
#@author: Freddy Chica
#Disclaimer: GPT-o3 & Codex were heavily used for the elaboration of this script

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

# ─────────────────────────────────────────────────────────────────────────────
#   INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
bladeName = {bladeName!r}
no_cores = {no_cores}
string = {string!r}
fileExtension = {fileExtension!r}

# ─────────────────────────────────────────────────────────────────────────────
#   FILE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
run_dir = Path(__file__).resolve().parent
base_dir = Path(__file__).resolve().parents[4]
blade_dir = base_dir / 'Blades' / bladeName
isesFilePath = blade_dir / f'ises.{string}'
bladeFilePath = blade_dir / f'{bladeName}.{string}'
outletFilePath = blade_dir / f'outlet.{string}'

# ─────────────────────────────────────────────────────────────────────────────
#   BLADE GEOMETRY 
# ─────────────────────────────────────────────────────────────────────────────
alpha1_deg = {alpha1_deg}
alpha2_deg = {alpha2_deg}
d_factor = {d_factor}
stagger = {stagger}
axial_chord = {axial_chord}
chord = {chord}
pitch = {pitch}
pitch2chord = {pitch2chord}

# ─────────────────────────────────────────────────────────────────────────────
#   BOUNDARY CONDITIONS 
# ───────────────────────────────────────────────────────────────────────────── 
R = {R}
gamma = {gamma}
mu = {mu}
T01 = {T01}
P1 = {P1}
P01 = {P01}
M1 = {M1}
P2 = {P2}
P2_P0a = {P2_P0a}
M2 = {M2}
T02 = {T02}
T2 = {T2}
c2 = {c2}
u2 = {u2}
rho2 = {rho2}
Re = {Re}
TI = {TI}

# ─────────────────────────────────────────────────────────────────────────────
#   MESHING
# ─────────────────────────────────────────────────────────────────────────────
dist_inlet = {dist_inlet}
dist_outlet = {dist_outlet}
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
WakeYMin = {WakeYMin}
WakeYMax = {WakeYMax}

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
    mod.outletFilePath = outletFilePath
    
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
    mod.WakeYMin = WakeYMin
    mod.WakeYMax = WakeYMax

def rerun():
    #mesh_datablade.mesh_datablade()
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
    with open(script_path, 'w') as f:
        f.write(content)


def main():
    
    # ─────────────────────────────────────────────────────────────────────────────
    #   INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────────
    #sys.argv = ['analysis_datablade.py', '--blades',
                #'Blade_1',  'Blade_2',  'Blade_3',  'Blade_4',  'Blade_5',  'Blade_6',  'Blade_7',  'Blade_8',  'Blade_9',  'Blade_10',
                #'Blade_11', 'Blade_12', 'Blade_13', 'Blade_14', 'Blade_15', 'Blade_16', 'Blade_17', 'Blade_18', 'Blade_19', 'Blade_20',
                #'Blade_21', 'Blade_22', 'Blade_23', 'Blade_24', 'Blade_25', 'Blade_26']
    
    # --- USER INPUTS 
    parser = argparse.ArgumentParser(description="Run blade analysis")
    parser.add_argument('--blade', default='Blade_1', help='Blade name')
    parser.add_argument('--blades', nargs='+', help='Process multiple blades')
    parser.add_argument('--no_cores', type=int, default=12, help='MPI cores for SU2')
    parser.add_argument('--suffix', default='databladeVALIDATION', help='File name suffix')
    parser.add_argument('--opt_tag', default='safe_start', help='Optimization tag')
    parser.add_argument('--file_ext', default='csv', help='Output file extension')
    args = parser.parse_args()
    
    # ---FILES DEFINITION 
    blades = args.blades if args.blades else [args.blade]
    no_cores = args.no_cores
    string = args.suffix
    fileExtension = args.file_ext

    base_dir = BLADEROOT
    for bladeName in blades:
        blade_dir = base_dir / 'Blades' / bladeName

        run_root = blade_dir / 'results'
        run_root.mkdir(exist_ok=True)
        date_str = datetime.now().strftime('%d-%m-%Y')
        n = 1
        while (run_root / f'Test_{n}_{date_str}').exists():
            n += 1
        run_dir = run_root / f'Test_{n}_{date_str}'
        run_dir.mkdir()
    
        isesFilePath = blade_dir / f'ises.{string}'
        bladeFilePath = blade_dir / f'blade.{string}'
        outletFilePath = blade_dir / f'outlet.{string}'
        
        # --- BLADE DATA EXTRACTION 
        alpha1, alpha2, Re, M2, P2_P0a = utils.extract_from_ises(isesFilePath)
        pitch = utils.extract_from_blade(bladeFilePath)
        #M1, P21_ratio = utils.extract_from_outlet(outletFilePath)
    
        # ─────────────────────────────────────────────────────────────────────────────
        #   BLADE GEOMETRY 
        # ─────────────────────────────────────────────────────────────────────────────
        geom0 = utils.compute_geometry(bladeFilePath, pitch=pitch, d_factor_guess=0.5)
        d_factor = utils.compute_d_factor(np.degrees(geom0['wedge_angle']),
                                          axial_chord=geom0['axial_chord'],
                                          te_thickness=geom0['te_open_thickness'])
        print(f"Updated d_factor = {d_factor:.3f}")
        geom = utils.compute_geometry(bladeFilePath, pitch=pitch, d_factor_guess=d_factor)
    
        stagger = np.degrees(geom['stagger_angle'])
        axial_chord = geom['axial_chord']
        chord = geom['chord_length']
        pitch2chord = pitch / chord
        alpha1_deg = int(np.degrees(np.arctan(alpha1)))
        alpha2_deg = int(np.degrees(np.arctan(alpha2)))
        
        # ─────────────────────────────────────────────────────────────────────────────
        #   BOUNDARY CONDITIONS 
        # ───────────────────────────────────────────────────────────────────────────── 
        R = 287.058 #[J/kg K]
        gamma = 1.4
        mu = 1.846e-5 #[kg/m s] # Obtained from table online @T=300
        T01 = 300 #[K]
        P1, P01 = utils.freestream_total_pressure(Re, M2, axial_chord, T01)
        M1 = utils.compute_Mx(P01, P1, gamma)
        P2 = P2_P0a * P01
        T02 = T01
        T2 = T02 / (1 + (gamma - 1)/2 * M2**2)
        c2 = np.sqrt(gamma * R * T2)
        u2 = M2 * c2
        rho2 = mu * Re / (u2 * np.cos(np.radians(stagger)))
        TI = 3.5 #%
        
        # ─────────────────────────────────────────────────────────────────────────────
        #   MESHING
        # ─────────────────────────────────────────────────────────────────────────────
        #--- INLET & OUTLET
        dist_inlet = 1
        dist_outlet = 1.5
        
        # -- GEOMETRY EXTRACTION 
        out = utils.process_airfoil_file(bladeFilePath, n_points=1000, n_te=60, d_factor=d_factor)
        xSS, ySS, _, _ = out['ss']
        xPS, yPS, _, _ = out['ps']
        
        # --- GENERAL MESH PARAMETERS 
        sizeCellFluid = 0.04 * axial_chord
        sizeCellAirfoil = 0.02 * axial_chord
        nCellAirfoil = 300
        nCellPerimeter = 183
        nBoundaryPoints = 50
        
        # --- AIRFOIL RESAMPLING AND TE CLOSING 
        n_points = 1000
        n_te = 60
        d_factor = d_factor
        
        # --- MESH BL PARAMETERS
        #first_layer_height = 0.01 * sizeCellAirfoil
        #bl_growth = 1.17
        #bl_thickness = 0.03 * pitch
        
        bl = utils.compute_bl_parameters(u2, rho2, mu, axial_chord,
                                         n_layers    = 25,           # keep in sync with gmsh Field[1].thickness
                                         y_plus_target = 1.0)
        
        first_layer_height = bl['first_layer_height']
        bl_growth          = bl['bl_growth']
        bl_thickness       = bl['bl_thickness']
        size_LE = 0.1 * sizeCellAirfoil
        dist_LE = 0.01 * axial_chord
        size_TE = 0.1 * sizeCellAirfoil
        dist_TE = 0.01 * axial_chord
        
        # --- REFINEMENT PARAMETERS 
        VolWAkeIn   = 0.35 * sizeCellFluid
        VolWAkeOut  = sizeCellFluid
        WakeXMin    = 0.1 * axial_chord 
        WakeXMax    = (dist_outlet + 1) * axial_chord
        WakeYMin    = -5 * pitch
        WakeYMax    =  5 * pitch

        # expose variables to modules
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
            mod.outletFilePath = outletFilePath
            
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
            mod.WakeYMin = WakeYMin
            mod.WakeYMax = WakeYMax
            
        create_rerun_script(run_dir, bladeName, base_dir,
                            no_cores, string, fileExtension,
                            alpha1_deg, alpha2_deg, Re, R, gamma, mu, 
                            pitch, d_factor, stagger, axial_chord, chord, pitch2chord,
                            T01, T02, T2, P01, P1, M1, P2_P0a, M2, P2, c2, u2, rho2,
                            dist_inlet, dist_outlet, TI,
                            sizeCellFluid, sizeCellAirfoil,
                            nCellAirfoil, nCellPerimeter, nBoundaryPoints,
                            first_layer_height, bl_growth, bl_thickness,
                            size_LE, dist_LE, size_TE, dist_TE,
                            VolWAkeIn, VolWAkeOut,
                            WakeXMin, WakeXMax, WakeYMin, WakeYMax)
        
        mesh_datablade.mesh_datablade()
        '''
        configSU2_datablade.configSU2_datablade()
        proc, logf = configSU2_datablade.runSU2_datablade(background=True)
        if utils.ask_view_live(bladeName):
            time.sleep(5)
            utils.launch_paraview_live(run_dir, bladeName, string)
        proc.wait()
        logf.close()
        configSU2_datablade._summarize_su2_log(run_dir / "su2.log")
        post_processing_datablade.post_processing_datablade()'''

if __name__ == '__main__':
    main()