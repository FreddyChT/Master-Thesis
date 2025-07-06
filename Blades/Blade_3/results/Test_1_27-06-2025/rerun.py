
#!/usr/bin/env python3
#Created on 27-06-2025, 16:00:46
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
bladeName = 'Blade_3'
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
bladeFilePath = blade_dir / f'Blade_3.databladeVALIDATION'
outletFilePath = blade_dir / f'outlet.databladeVALIDATION'

# ─────────────────────────────────────────────────────────────────────────────
#   BLADE GEOMETRY 
# ─────────────────────────────────────────────────────────────────────────────
alpha1_deg = -25.00001608804352
alpha2_deg = 65.00000081345597
d_factor = 0.2
stagger = 34.235049603945264
axial_chord = 1.001553825841012
chord = 1.2114545978163471
pitch = 1.04007
pitch2chord = 0.8585299043602058

# ─────────────────────────────────────────────────────────────────────────────
#   BOUNDARY CONDITIONS 
# ───────────────────────────────────────────────────────────────────────────── 
R = 287.058
gamma = 1.4
mu = 1.846e-05
T01 = 300
P1 = 5486.737465087765
P01 = 6508.437322718792
M1 = 0.5000000000000002
P2 = 5486.736323361073
P2_P0a = 0.843019
M2 = 0.5
T02 = 300
T2 = 285.7142857142857
c2 = 338.85572150990754
u2 = 169.42786075495377
rho2 = 0.07907350105264756
Re = 600000.0
TI = 3.5

# ─────────────────────────────────────────────────────────────────────────────
#   MESHING
# ─────────────────────────────────────────────────────────────────────────────
dist_inlet = 1
dist_outlet = 2
sizeCellFluid = 0.040062153033640475
sizeCellAirfoil = 0.020031076516820238
nCellAirfoil = 549
nCellPerimeter = 183
nBoundaryPoints = 50
first_layer_height = 2.0303435480739446e-05
bl_growth = 1.2594365896451976
bl_thickness = 0.024922083096760177
size_LE = 0.002003107651682024
dist_LE = 0.010015538258410119
size_TE = 0.002003107651682024
dist_TE = 0.010015538258410119
VolWAkeIn = 0.014021753561774165
VolWAkeOut = 0.040062153033640475
WakeXMin = 0.1001553825841012
WakeXMax = 2.50388456460253
WakeYMin = -5.20035
WakeYMax = 5.20035

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
