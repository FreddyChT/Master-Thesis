
#!/usr/bin/env python3
#Created on 15-06-2025, 15:38:04
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
bladeName = 'Blade_12'
no_cores = 12
string = 'databladeVALIDATION'
fileExtension = 'csv'

# ─────────────────────────────────────────────────────────────────────────────
#   FILE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
run_dir = Path(__file__).resolve().parent
base_dir = Path(__file__).resolve().parents[4]
blade_dir = base_dir / 'Blades' / bladeName
isesFilePath = blade_dir / f'ises.databladeVALIDATION'
bladeFilePath = blade_dir / f'Blade_12.databladeVALIDATION'
outletFilePath = blade_dir / f'outlet.databladeVALIDATION'

# ─────────────────────────────────────────────────────────────────────────────
#   BLADE GEOMETRY 
# ─────────────────────────────────────────────────────────────────────────────
alpha1 = -19
alpha2 = 50
d_factor = 0.2
stagger = 0.39034999646006663
axial_chord = 1.0020728306546747
pitch = 0.908271

# ─────────────────────────────────────────────────────────────────────────────
#   BOUNDARY CONDITIONS 
# ───────────────────────────────────────────────────────────────────────────── 
R = 287.058
gamma = 1.4
mu = 1.846e-05
T01 = 300
P1 = 4569.91309367013
P01 = 5828.941408525037
M1 = 0.6000000000000003
P2 = 4569.913380049264
P2_P0a = 0.784004
M2 = 0.6
T02 = 300
T2 = 279.8507462686567
c2 = 335.3606323517167
u2 = 201.21637941103
rho2 = 0.059522764743853575
Re = 600000.0
TI = 3.5

# ─────────────────────────────────────────────────────────────────────────────
#   MESHING
# ─────────────────────────────────────────────────────────────────────────────
dist_inlet = 1
dist_outlet = 2
sizeCellFluid = 0.04008291322618699
sizeCellAirfoil = 0.020041456613093495
nCellAirfoil = 549
nCellPerimeter = 183
nBoundaryPoints = 50
first_layer_height = 2.238174076727937e-05
bl_growth = 1.2547905978872331
bl_thickness = 0.025497535671850178
size_LE = 0.0020041456613093494
dist_LE = 0.010020728306546748
size_TE = 0.0020041456613093494
dist_TE = 0.010020728306546748
VolWAkeIn = 0.014029019629165446
VolWAkeOut = 0.04008291322618699
WakeXMin = 0.10020728306546747
WakeXMax = 2.505182076636687
WakeYMin = -1.816542
WakeYMax = 1.816542

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
    mod.pitch = pitch
    
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
