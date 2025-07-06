
#!/usr/bin/env python3
#Created on 12-06-2025, 15:01:00
#@author: Freddy Chica
#Disclaimer: GPT-o3 & Codex were heavily used for the elaboration of this script

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

bladeName = 'Blade_3'
no_cores = 12
string = 'databladeVALIDATION'
string2 = 'safe_start'
fileExtension = 'csv'
alpha1 = -25
alpha2 = 65
Re = 600000.0
M1 = 0.20769
P21_ratio = 0.86457
pitch = 1.04007
d_factor = 0.2
stagger = 0.5975143351724256
axial_chord = 1.001553825841012
R = 287.058
gamma = 1.4
mu = 1.716e-05
T01 = 314.15
P01 = 15615.897203574556
P2 = 13101.167477642835
dist_inlet = 1
dist_outlet = 2
TI2 = 2.2
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
WakeYMin = -2.08014
WakeYMax = 2.08014

run_dir = Path(__file__).resolve().parent
base_dir = Path(__file__).resolve().parents[4]
blade_dir = base_dir / 'Blades' / bladeName
isesFilePath = blade_dir / f'ises.databladeVALIDATION'
bladeFilePath = blade_dir / f'Blade_3.databladeVALIDATION'
outletFilePath = blade_dir / f'outlet.databladeVALIDATION'

for mod in (mesh_datablade, configSU2_datablade, post_processing_datablade):
    mod.bladeName = bladeName
    mod.no_cores = no_cores
    mod.string = string
    mod.string2 = string2
    mod.fileExtension = fileExtension
    mod.base_dir = base_dir
    mod.blade_dir = blade_dir
    mod.run_dir = run_dir
    mod.isesFilePath = isesFilePath
    mod.bladeFilePath = bladeFilePath
    mod.outletFilePath = outletFilePath
    mod.alpha1 = alpha1
    mod.alpha2 = alpha2
    mod.Re = Re
    mod.M1 = M1
    mod.P21_ratio = P21_ratio
    mod.pitch = pitch
    mod.d_factor = d_factor
    mod.stagger = stagger
    mod.axial_chord = axial_chord
    mod.R = R
    mod.gamma = gamma
    mod.mu = mu
    mod.T01 = T01
    mod.P01 = P01
    mod.P2 = P2
    mod.dist_inlet = dist_inlet
    mod.dist_outlet = dist_outlet
    mod.TI2 = TI2
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
