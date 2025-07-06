
#!/usr/bin/env python3
#Created on 12-06-2025, 14:44:23
#@author: Freddy Chica
#Disclaimer: GPT-o3 & Codex were heavily used for the elaboration of this script

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

bladeName = 'Blade_2'
no_cores = 12
string = 'databladeVALIDATION'
string2 = 'safe_start'
fileExtension = 'csv'
alpha1 = -40
alpha2 = 65
Re = 600000.0
M1 = 0.25147
P21_ratio = 0.87665
pitch = 1.08088
d_factor = 0.2
stagger = 0.5459503890616313
axial_chord = 1.0013968107613906
R = 287.058
gamma = 1.4
mu = 1.716e-05
T01 = 314.15
P01 = 12626.012647091482
P2 = 10592.257424290927
dist_inlet = 1
dist_outlet = 2
TI2 = 2.2
sizeCellFluid = 0.04005587243045563
sizeCellAirfoil = 0.020027936215227814
nCellAirfoil = 549
nCellPerimeter = 183
nBoundaryPoints = 50
first_layer_height = 2.0897502893263416e-05
bl_growth = 1.2580462887850652
bl_thickness = 0.025084906772970178
size_LE = 0.0020027936215227817
dist_LE = 0.010013968107613907
size_TE = 0.0020027936215227817
dist_TE = 0.010013968107613907
VolWAkeIn = 0.01401955535065947
VolWAkeOut = 0.04005587243045563
WakeXMin = 0.10013968107613908
WakeXMax = 2.5034920269034764
WakeYMin = -2.16176
WakeYMax = 2.16176

run_dir = Path(__file__).resolve().parent
base_dir = Path(__file__).resolve().parents[4]
blade_dir = base_dir / 'Blades' / bladeName
isesFilePath = blade_dir / f'ises.databladeVALIDATION'
bladeFilePath = blade_dir / f'Blade_2.databladeVALIDATION'
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
