
#!/usr/bin/env python3
#Created on 01-08-2025, 16:30:17
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

bladeName = 'Blade_0'
no_cores = 12
string = 'databladeVALIDATION'
fileExtension = 'csv'

run_dir = Path(__file__).resolve().parent
base_dir = Path(__file__).resolve().parents[4]
blade_dir = base_dir / 'Blades' / bladeName
isesFilePath = blade_dir / f'ises.databladeVALIDATION'
bladeFilePath = blade_dir / f'blade.databladeVALIDATION'

alpha1_deg = 37.3
alpha2_deg = -53.8
d_factor = 0.0
stagger = 24.4
axial_chord = 0.047614
chord = 0.052285
pitch = 0.03295
pitch2chord = 0.6301998661183896

R = 287.058
gamma = 1.4
mu = 1.716e-05
T01 = 300
P1 = 9310.72429
P01 = 16285
M1 = 0.9305955774058217
P2 = 9629
M2 = 0.9
T02 = 300
T2 = 258.1755593803787
c2 = 322.111632224696
u2 = 289.90046900222643
rho2 = 0.004549868667689753
Re = 120000
TI = 2.0

dist_inlet = 2.0
dist_outlet = 3.0
x_plane = 1.5
sizeCellFluid = 0.0019045599999999998
sizeCellAirfoil = 0.0009522799999999999
nCellAirfoil = 300
nCellPerimeter = 183
nBoundaryPoints = 50
first_layer_height = 7.852495232721763e-06
bl_growth = 1.2037336576193827
bl_thickness = 0.003935247851407715
size_LE = 9.522799999999999e-05
dist_LE = 0.00047613999999999995
size_TE = 9.522799999999999e-05
dist_TE = 0.00047613999999999995
VolWAkeIn = 0.0006665959999999999
VolWAkeOut = 0.0019045599999999998
WakeXMin = 0.0047614
WakeXMax = 0.166649

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
