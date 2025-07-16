# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 14:56:22 2025

@author: fredd
"""

# trace generated using paraview version 5.12.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# -----------------------------------------------------------------------------
# File locations
# -----------------------------------------------------------------------------
import sys
import glob

# pick the VTU file from the first command line argument or search for one
if len(sys.argv) > 1:
    volume_vtu = sys.argv[1]
else:
    matches = glob.glob("volume_flow_datablade*Blade_*.vtu")
    volume_vtu = matches[0] if matches else "volume_flow_databladeVALIDATION_Blade_25.vtu"
    
volume_vtu = volume_vtu.replace('\\', '/')
vtu_dir, vtu_file = volume_vtu.rsplit('/', 1) if '/' in volume_vtu else ('', volume_vtu)
stem = vtu_file.split('.', 1)[0]
blade_id = stem.rsplit('_', 1)[-1]
history_csv_path = f"{vtu_dir}/history_databladeVALIDATION_Blade_{blade_id}.csv" if vtu_dir else f"history_databladeVALIDATION_Blade_{blade_id}.csv"

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# get layout
layout1 = GetLayout()

# split cell
layout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')

# assign view to a particular cell in the layout
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=2)

# split cell
layout1.SplitVertical(2, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Line Chart View'
lineChartView2 = CreateView('XYChartView')

# assign view to a particular cell in the layout
AssignViewToLayout(view=lineChartView2, layout=layout1, hint=6)

# set active view
SetActiveView(renderView1)

# get active source.
volume_flow_databladeVALIDATION_Blade_25vtu = GetActiveSource()

# get display properties
volume_flow_databladeVALIDATION_Blade_25vtuDisplay = GetDisplayProperties(volume_flow_databladeVALIDATION_Blade_25vtu, view=renderView1)

# create a new 'Live Programmable Source'
liveProgrammableSource1 = LiveProgrammableSource(registrationName='LiveProgrammableSource1')

# set active source
SetActiveSource(volume_flow_databladeVALIDATION_Blade_25vtu)

# set active source
SetActiveSource(liveProgrammableSource1)

# Properties modified on liveProgrammableSource1
liveProgrammableSource1.OutputDataSetType = 'vtkUnstructuredGrid'
liveProgrammableSource1.Script = f"""# .vtu paraview
from paraview.vtk.vtkIOXML import vtkXMLUnstructuredGridReader as vtuReader
reader = vtuReader()
reader.SetFileName('{volume_vtu}')
reader.Update()
self.GetOutputDataObject(0).ShallowCopy(reader.GetOutput())"""
liveProgrammableSource1.ScriptRequestInformation = ''
liveProgrammableSource1.PythonPath = ''
liveProgrammableSource1.ScriptCheckNeedsUpdate = """import time 
# the update frequency is 3 seconds
UpdateFrequency = 3
if not hasattr(self, "_my_time"): 
  setattr(self, "_my_time", time.time()) 

t = time.time() 
lastTime = getattr(self, "_my_time") 

if t - lastTime > UpdateFrequency: 
  setattr(self, "_my_time", t) 
  self.SetNeedsUpdate(True)"""

# show data in view
liveProgrammableSource1Display = Show(liveProgrammableSource1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
liveProgrammableSource1Display.Representation = 'Surface'

# update the view to ensure updated data information
renderView1.Update()

# hide data in view
Hide(volume_flow_databladeVALIDATION_Blade_25vtu, renderView1)

# set scalar coloring
ColorBy(liveProgrammableSource1Display, ('POINTS', 'Mach'))

# rescale color and/or opacity maps used to include current data range
liveProgrammableSource1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
liveProgrammableSource1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Mach'
machLUT = GetColorTransferFunction('Mach')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
machLUT.ApplyPreset('Cool to Warm (Extended)', True)

# get opacity transfer function/opacity map for 'Mach'
machPWF = GetOpacityTransferFunction('Mach')

# get 2D transfer function for 'Mach'
machTF2D = GetTransferFunction2D('Mach')

# set active view
SetActiveView(lineChartView1)

# create a new 'Live Programmable Source'
liveProgrammableSource2 = LiveProgrammableSource(registrationName='LiveProgrammableSource2')

# Properties modified on liveProgrammableSource2
liveProgrammableSource2.OutputDataSetType = 'vtkTable'
liveProgrammableSource2.Script = f"""import numpy as np
import pandas as pd

data = pd.read_csv('{history_csv_path}',sep=',')
print(data.keys())
print(len(data.columns)) 
for name in data.keys(): 
  array = data[name].to_numpy() 
  output.RowData.append(array, name)"""
liveProgrammableSource2.ScriptRequestInformation = ''
liveProgrammableSource2.PythonPath = ''
liveProgrammableSource2.ScriptCheckNeedsUpdate = """import time 
# the update frequency is 1 seconds
UpdateFrequency = 1
if not hasattr(self, "_my_time"): 
  setattr(self, "_my_time", time.time()) 

t = time.time() 
lastTime = getattr(self, "_my_time") 

if t - lastTime > UpdateFrequency: 
  setattr(self, "_my_time", t) 
  self.SetNeedsUpdate(True)"""

# Create a new 'SpreadSheet View'
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024

# show data in view
liveProgrammableSource2Display = Show(liveProgrammableSource2, spreadSheetView1, 'SpreadSheetRepresentation')

# add view to a layout so it's visible in UI
AssignViewToLayout(view=spreadSheetView1, layout=layout1, hint=5)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
spreadSheetView1.Update()

# destroy spreadSheetView1
Delete(spreadSheetView1)
del spreadSheetView1

# close an empty frame
layout1.Collapse(12)

# set active view
SetActiveView(lineChartView1)

# set active source
SetActiveSource(liveProgrammableSource2)

# show data in view
liveProgrammableSource2Display = Show(liveProgrammableSource2, lineChartView1, 'XYChartRepresentation')

# Properties modified on liveProgrammableSource2Display
liveProgrammableSource2Display.SeriesOpacity = ['     "Avg CFL"    ', '1', '     "Max CFL"    ', '1', '     "Max DT"     ', '1', '     "Min CFL"    ', '1', '     "Min DT"     ', '1', '    "LinSolRes"   ', '1', '    "rms[Rho]"    ', '1', '    "rms[RhoE]"   ', '1', '    "rms[RhoU]"   ', '1', '    "Time(sec)"   ', '1', '  "LinSolResTurb" ', '1', ' "Avg_Mach(inlet)"', '1', ' "Avg_Temp(inlet)"', '1', 'Avg_Density(blade1)', '1', 'Avg_Density(inlet)', '1', 'Avg_Density(outlet)', '1', 'Avg_Enthalpy(blade1)', '1', 'Avg_Enthalpy(inlet)', '1', 'Avg_Enthalpy(outlet)', '1', 'Avg_Mach(blade1)', '1', 'Avg_Mach(outlet)', '1', 'Avg_Massflow(blade1)', '1', 'Avg_Massflow(inlet)', '1', 'Avg_Massflow(outlet)', '1', 'Avg_NormalVel(blade1)', '1', 'Avg_NormalVel(inlet)', '1', 'Avg_NormalVel(outlet)', '1', 'Avg_Press(blade1)', '1', 'Avg_Press(inlet)', '1', 'Avg_Press(outlet)', '1', 'Avg_Temp(blade1)', '1', 'Avg_Temp(outlet)', '1', 'Inner_Iter', '1', 'Linear_Solver_Iterations', '1', 'LinSolIterTurb', '1', 'Momentum_Distortion(inlet)', '1', 'Momentum_Distortion(outlet)', '1', 'Secondary_Strength(blade1)', '1', 'Secondary_Strength(inlet)', '1', 'Secondary_Strength(outlet)', '1', 'Uniformity(blade1)', '1', 'Uniformity(inlet)', '1', 'Uniformity(outlet)', '1']
liveProgrammableSource2Display.SeriesPlotCorner = ['     "Avg CFL"    ', '0', '     "Max CFL"    ', '0', '     "Max DT"     ', '0', '     "Min CFL"    ', '0', '     "Min DT"     ', '0', '    "LinSolRes"   ', '0', '    "Time(sec)"   ', '0', '    "rms[RhoE]"   ', '0', '    "rms[RhoU]"   ', '0', '    "rms[Rho]"    ', '0', '  "LinSolResTurb" ', '0', ' "Avg_Mach(inlet)"', '0', ' "Avg_Temp(inlet)"', '0', 'Avg_Density(blade1)', '0', 'Avg_Density(inlet)', '0', 'Avg_Density(outlet)', '0', 'Avg_Enthalpy(blade1)', '0', 'Avg_Enthalpy(inlet)', '0', 'Avg_Enthalpy(outlet)', '0', 'Avg_Mach(blade1)', '0', 'Avg_Mach(outlet)', '0', 'Avg_Massflow(blade1)', '0', 'Avg_Massflow(inlet)', '0', 'Avg_Massflow(outlet)', '0', 'Avg_NormalVel(blade1)', '0', 'Avg_NormalVel(inlet)', '0', 'Avg_NormalVel(outlet)', '0', 'Avg_Press(blade1)', '0', 'Avg_Press(inlet)', '0', 'Avg_Press(outlet)', '0', 'Avg_Temp(blade1)', '0', 'Avg_Temp(outlet)', '0', 'Inner_Iter', '0', 'LinSolIterTurb', '0', 'Linear_Solver_Iterations', '0', 'Momentum_Distortion(inlet)', '0', 'Momentum_Distortion(outlet)', '0', 'Secondary_Strength(blade1)', '0', 'Secondary_Strength(inlet)', '0', 'Secondary_Strength(outlet)', '0', 'Uniformity(blade1)', '0', 'Uniformity(inlet)', '0', 'Uniformity(outlet)', '0']
liveProgrammableSource2Display.SeriesLineStyle = ['     "Avg CFL"    ', '1', '     "Max CFL"    ', '1', '     "Max DT"     ', '1', '     "Min CFL"    ', '1', '     "Min DT"     ', '1', '    "LinSolRes"   ', '1', '    "Time(sec)"   ', '1', '    "rms[RhoE]"   ', '1', '    "rms[RhoU]"   ', '1', '    "rms[Rho]"    ', '1', '  "LinSolResTurb" ', '1', ' "Avg_Mach(inlet)"', '1', ' "Avg_Temp(inlet)"', '1', 'Avg_Density(blade1)', '1', 'Avg_Density(inlet)', '1', 'Avg_Density(outlet)', '1', 'Avg_Enthalpy(blade1)', '1', 'Avg_Enthalpy(inlet)', '1', 'Avg_Enthalpy(outlet)', '1', 'Avg_Mach(blade1)', '1', 'Avg_Mach(outlet)', '1', 'Avg_Massflow(blade1)', '1', 'Avg_Massflow(inlet)', '1', 'Avg_Massflow(outlet)', '1', 'Avg_NormalVel(blade1)', '1', 'Avg_NormalVel(inlet)', '1', 'Avg_NormalVel(outlet)', '1', 'Avg_Press(blade1)', '1', 'Avg_Press(inlet)', '1', 'Avg_Press(outlet)', '1', 'Avg_Temp(blade1)', '1', 'Avg_Temp(outlet)', '1', 'Inner_Iter', '1', 'LinSolIterTurb', '1', 'Linear_Solver_Iterations', '1', 'Momentum_Distortion(inlet)', '1', 'Momentum_Distortion(outlet)', '1', 'Secondary_Strength(blade1)', '1', 'Secondary_Strength(inlet)', '1', 'Secondary_Strength(outlet)', '1', 'Uniformity(blade1)', '1', 'Uniformity(inlet)', '1', 'Uniformity(outlet)', '1']
liveProgrammableSource2Display.SeriesLineThickness = ['     "Avg CFL"    ', '2', '     "Max CFL"    ', '2', '     "Max DT"     ', '2', '     "Min CFL"    ', '2', '     "Min DT"     ', '2', '    "LinSolRes"   ', '2', '    "Time(sec)"   ', '2', '    "rms[RhoE]"   ', '2', '    "rms[RhoU]"   ', '2', '    "rms[Rho]"    ', '2', '  "LinSolResTurb" ', '2', ' "Avg_Mach(inlet)"', '2', ' "Avg_Temp(inlet)"', '2', 'Avg_Density(blade1)', '2', 'Avg_Density(inlet)', '2', 'Avg_Density(outlet)', '2', 'Avg_Enthalpy(blade1)', '2', 'Avg_Enthalpy(inlet)', '2', 'Avg_Enthalpy(outlet)', '2', 'Avg_Mach(blade1)', '2', 'Avg_Mach(outlet)', '2', 'Avg_Massflow(blade1)', '2', 'Avg_Massflow(inlet)', '2', 'Avg_Massflow(outlet)', '2', 'Avg_NormalVel(blade1)', '2', 'Avg_NormalVel(inlet)', '2', 'Avg_NormalVel(outlet)', '2', 'Avg_Press(blade1)', '2', 'Avg_Press(inlet)', '2', 'Avg_Press(outlet)', '2', 'Avg_Temp(blade1)', '2', 'Avg_Temp(outlet)', '2', 'Inner_Iter', '2', 'LinSolIterTurb', '2', 'Linear_Solver_Iterations', '2', 'Momentum_Distortion(inlet)', '2', 'Momentum_Distortion(outlet)', '2', 'Secondary_Strength(blade1)', '2', 'Secondary_Strength(inlet)', '2', 'Secondary_Strength(outlet)', '2', 'Uniformity(blade1)', '2', 'Uniformity(inlet)', '2', 'Uniformity(outlet)', '2']
liveProgrammableSource2Display.SeriesMarkerStyle = ['     "Avg CFL"    ', '0', '     "Max CFL"    ', '0', '     "Max DT"     ', '0', '     "Min CFL"    ', '0', '     "Min DT"     ', '0', '    "LinSolRes"   ', '0', '    "Time(sec)"   ', '0', '    "rms[RhoE]"   ', '0', '    "rms[RhoU]"   ', '0', '    "rms[Rho]"    ', '0', '  "LinSolResTurb" ', '0', ' "Avg_Mach(inlet)"', '0', ' "Avg_Temp(inlet)"', '0', 'Avg_Density(blade1)', '0', 'Avg_Density(inlet)', '0', 'Avg_Density(outlet)', '0', 'Avg_Enthalpy(blade1)', '0', 'Avg_Enthalpy(inlet)', '0', 'Avg_Enthalpy(outlet)', '0', 'Avg_Mach(blade1)', '0', 'Avg_Mach(outlet)', '0', 'Avg_Massflow(blade1)', '0', 'Avg_Massflow(inlet)', '0', 'Avg_Massflow(outlet)', '0', 'Avg_NormalVel(blade1)', '0', 'Avg_NormalVel(inlet)', '0', 'Avg_NormalVel(outlet)', '0', 'Avg_Press(blade1)', '0', 'Avg_Press(inlet)', '0', 'Avg_Press(outlet)', '0', 'Avg_Temp(blade1)', '0', 'Avg_Temp(outlet)', '0', 'Inner_Iter', '0', 'LinSolIterTurb', '0', 'Linear_Solver_Iterations', '0', 'Momentum_Distortion(inlet)', '0', 'Momentum_Distortion(outlet)', '0', 'Secondary_Strength(blade1)', '0', 'Secondary_Strength(inlet)', '0', 'Secondary_Strength(outlet)', '0', 'Uniformity(blade1)', '0', 'Uniformity(inlet)', '0', 'Uniformity(outlet)', '0']
liveProgrammableSource2Display.SeriesMarkerSize = ['     "Avg CFL"    ', '4', '     "Max CFL"    ', '4', '     "Max DT"     ', '4', '     "Min CFL"    ', '4', '     "Min DT"     ', '4', '    "LinSolRes"   ', '4', '    "Time(sec)"   ', '4', '    "rms[RhoE]"   ', '4', '    "rms[RhoU]"   ', '4', '    "rms[Rho]"    ', '4', '  "LinSolResTurb" ', '4', ' "Avg_Mach(inlet)"', '4', ' "Avg_Temp(inlet)"', '4', 'Avg_Density(blade1)', '4', 'Avg_Density(inlet)', '4', 'Avg_Density(outlet)', '4', 'Avg_Enthalpy(blade1)', '4', 'Avg_Enthalpy(inlet)', '4', 'Avg_Enthalpy(outlet)', '4', 'Avg_Mach(blade1)', '4', 'Avg_Mach(outlet)', '4', 'Avg_Massflow(blade1)', '4', 'Avg_Massflow(inlet)', '4', 'Avg_Massflow(outlet)', '4', 'Avg_NormalVel(blade1)', '4', 'Avg_NormalVel(inlet)', '4', 'Avg_NormalVel(outlet)', '4', 'Avg_Press(blade1)', '4', 'Avg_Press(inlet)', '4', 'Avg_Press(outlet)', '4', 'Avg_Temp(blade1)', '4', 'Avg_Temp(outlet)', '4', 'Inner_Iter', '4', 'LinSolIterTurb', '4', 'Linear_Solver_Iterations', '4', 'Momentum_Distortion(inlet)', '4', 'Momentum_Distortion(outlet)', '4', 'Secondary_Strength(blade1)', '4', 'Secondary_Strength(inlet)', '4', 'Secondary_Strength(outlet)', '4', 'Uniformity(blade1)', '4', 'Uniformity(inlet)', '4', 'Uniformity(outlet)', '4']

# Properties modified on liveProgrammableSource2Display
liveProgrammableSource2Display.SeriesVisibility = []

# Properties modified on liveProgrammableSource2Display
liveProgrammableSource2Display.SeriesVisibility = ['    "rms[RhoE]"   ']

# Properties modified on liveProgrammableSource2Display
liveProgrammableSource2Display.SeriesVisibility = ['    "rms[RhoE]"   ', '    "rms[RhoU]"   ']

# Properties modified on liveProgrammableSource2Display
liveProgrammableSource2Display.SeriesVisibility = ['    "rms[Rho]"    ', '    "rms[RhoE]"   ', '    "rms[RhoU]"   ']

# set active view
SetActiveView(lineChartView2)

# show data in view
liveProgrammableSource2Display_1 = Show(liveProgrammableSource2, lineChartView2, 'XYChartRepresentation')

# Properties modified on liveProgrammableSource2Display_1
liveProgrammableSource2Display_1.SeriesOpacity = ['     "Avg CFL"    ', '1', '     "Max CFL"    ', '1', '     "Max DT"     ', '1', '     "Min CFL"    ', '1', '     "Min DT"     ', '1', '    "LinSolRes"   ', '1', '    "rms[Rho]"    ', '1', '    "rms[RhoE]"   ', '1', '    "rms[RhoU]"   ', '1', '    "Time(sec)"   ', '1', '  "LinSolResTurb" ', '1', ' "Avg_Mach(inlet)"', '1', ' "Avg_Temp(inlet)"', '1', 'Avg_Density(blade1)', '1', 'Avg_Density(inlet)', '1', 'Avg_Density(outlet)', '1', 'Avg_Enthalpy(blade1)', '1', 'Avg_Enthalpy(inlet)', '1', 'Avg_Enthalpy(outlet)', '1', 'Avg_Mach(blade1)', '1', 'Avg_Mach(outlet)', '1', 'Avg_Massflow(blade1)', '1', 'Avg_Massflow(inlet)', '1', 'Avg_Massflow(outlet)', '1', 'Avg_NormalVel(blade1)', '1', 'Avg_NormalVel(inlet)', '1', 'Avg_NormalVel(outlet)', '1', 'Avg_Press(blade1)', '1', 'Avg_Press(inlet)', '1', 'Avg_Press(outlet)', '1', 'Avg_Temp(blade1)', '1', 'Avg_Temp(outlet)', '1', 'Inner_Iter', '1', 'Linear_Solver_Iterations', '1', 'LinSolIterTurb', '1', 'Momentum_Distortion(inlet)', '1', 'Momentum_Distortion(outlet)', '1', 'Secondary_Strength(blade1)', '1', 'Secondary_Strength(inlet)', '1', 'Secondary_Strength(outlet)', '1', 'Uniformity(blade1)', '1', 'Uniformity(inlet)', '1', 'Uniformity(outlet)', '1']
liveProgrammableSource2Display_1.SeriesPlotCorner = ['     "Avg CFL"    ', '0', '     "Max CFL"    ', '0', '     "Max DT"     ', '0', '     "Min CFL"    ', '0', '     "Min DT"     ', '0', '    "LinSolRes"   ', '0', '    "Time(sec)"   ', '0', '    "rms[RhoE]"   ', '0', '    "rms[RhoU]"   ', '0', '    "rms[Rho]"    ', '0', '  "LinSolResTurb" ', '0', ' "Avg_Mach(inlet)"', '0', ' "Avg_Temp(inlet)"', '0', 'Avg_Density(blade1)', '0', 'Avg_Density(inlet)', '0', 'Avg_Density(outlet)', '0', 'Avg_Enthalpy(blade1)', '0', 'Avg_Enthalpy(inlet)', '0', 'Avg_Enthalpy(outlet)', '0', 'Avg_Mach(blade1)', '0', 'Avg_Mach(outlet)', '0', 'Avg_Massflow(blade1)', '0', 'Avg_Massflow(inlet)', '0', 'Avg_Massflow(outlet)', '0', 'Avg_NormalVel(blade1)', '0', 'Avg_NormalVel(inlet)', '0', 'Avg_NormalVel(outlet)', '0', 'Avg_Press(blade1)', '0', 'Avg_Press(inlet)', '0', 'Avg_Press(outlet)', '0', 'Avg_Temp(blade1)', '0', 'Avg_Temp(outlet)', '0', 'Inner_Iter', '0', 'LinSolIterTurb', '0', 'Linear_Solver_Iterations', '0', 'Momentum_Distortion(inlet)', '0', 'Momentum_Distortion(outlet)', '0', 'Secondary_Strength(blade1)', '0', 'Secondary_Strength(inlet)', '0', 'Secondary_Strength(outlet)', '0', 'Uniformity(blade1)', '0', 'Uniformity(inlet)', '0', 'Uniformity(outlet)', '0']
liveProgrammableSource2Display_1.SeriesLineStyle = ['     "Avg CFL"    ', '1', '     "Max CFL"    ', '1', '     "Max DT"     ', '1', '     "Min CFL"    ', '1', '     "Min DT"     ', '1', '    "LinSolRes"   ', '1', '    "Time(sec)"   ', '1', '    "rms[RhoE]"   ', '1', '    "rms[RhoU]"   ', '1', '    "rms[Rho]"    ', '1', '  "LinSolResTurb" ', '1', ' "Avg_Mach(inlet)"', '1', ' "Avg_Temp(inlet)"', '1', 'Avg_Density(blade1)', '1', 'Avg_Density(inlet)', '1', 'Avg_Density(outlet)', '1', 'Avg_Enthalpy(blade1)', '1', 'Avg_Enthalpy(inlet)', '1', 'Avg_Enthalpy(outlet)', '1', 'Avg_Mach(blade1)', '1', 'Avg_Mach(outlet)', '1', 'Avg_Massflow(blade1)', '1', 'Avg_Massflow(inlet)', '1', 'Avg_Massflow(outlet)', '1', 'Avg_NormalVel(blade1)', '1', 'Avg_NormalVel(inlet)', '1', 'Avg_NormalVel(outlet)', '1', 'Avg_Press(blade1)', '1', 'Avg_Press(inlet)', '1', 'Avg_Press(outlet)', '1', 'Avg_Temp(blade1)', '1', 'Avg_Temp(outlet)', '1', 'Inner_Iter', '1', 'LinSolIterTurb', '1', 'Linear_Solver_Iterations', '1', 'Momentum_Distortion(inlet)', '1', 'Momentum_Distortion(outlet)', '1', 'Secondary_Strength(blade1)', '1', 'Secondary_Strength(inlet)', '1', 'Secondary_Strength(outlet)', '1', 'Uniformity(blade1)', '1', 'Uniformity(inlet)', '1', 'Uniformity(outlet)', '1']
liveProgrammableSource2Display_1.SeriesLineThickness = ['     "Avg CFL"    ', '2', '     "Max CFL"    ', '2', '     "Max DT"     ', '2', '     "Min CFL"    ', '2', '     "Min DT"     ', '2', '    "LinSolRes"   ', '2', '    "Time(sec)"   ', '2', '    "rms[RhoE]"   ', '2', '    "rms[RhoU]"   ', '2', '    "rms[Rho]"    ', '2', '  "LinSolResTurb" ', '2', ' "Avg_Mach(inlet)"', '2', ' "Avg_Temp(inlet)"', '2', 'Avg_Density(blade1)', '2', 'Avg_Density(inlet)', '2', 'Avg_Density(outlet)', '2', 'Avg_Enthalpy(blade1)', '2', 'Avg_Enthalpy(inlet)', '2', 'Avg_Enthalpy(outlet)', '2', 'Avg_Mach(blade1)', '2', 'Avg_Mach(outlet)', '2', 'Avg_Massflow(blade1)', '2', 'Avg_Massflow(inlet)', '2', 'Avg_Massflow(outlet)', '2', 'Avg_NormalVel(blade1)', '2', 'Avg_NormalVel(inlet)', '2', 'Avg_NormalVel(outlet)', '2', 'Avg_Press(blade1)', '2', 'Avg_Press(inlet)', '2', 'Avg_Press(outlet)', '2', 'Avg_Temp(blade1)', '2', 'Avg_Temp(outlet)', '2', 'Inner_Iter', '2', 'LinSolIterTurb', '2', 'Linear_Solver_Iterations', '2', 'Momentum_Distortion(inlet)', '2', 'Momentum_Distortion(outlet)', '2', 'Secondary_Strength(blade1)', '2', 'Secondary_Strength(inlet)', '2', 'Secondary_Strength(outlet)', '2', 'Uniformity(blade1)', '2', 'Uniformity(inlet)', '2', 'Uniformity(outlet)', '2']
liveProgrammableSource2Display_1.SeriesMarkerStyle = ['     "Avg CFL"    ', '0', '     "Max CFL"    ', '0', '     "Max DT"     ', '0', '     "Min CFL"    ', '0', '     "Min DT"     ', '0', '    "LinSolRes"   ', '0', '    "Time(sec)"   ', '0', '    "rms[RhoE]"   ', '0', '    "rms[RhoU]"   ', '0', '    "rms[Rho]"    ', '0', '  "LinSolResTurb" ', '0', ' "Avg_Mach(inlet)"', '0', ' "Avg_Temp(inlet)"', '0', 'Avg_Density(blade1)', '0', 'Avg_Density(inlet)', '0', 'Avg_Density(outlet)', '0', 'Avg_Enthalpy(blade1)', '0', 'Avg_Enthalpy(inlet)', '0', 'Avg_Enthalpy(outlet)', '0', 'Avg_Mach(blade1)', '0', 'Avg_Mach(outlet)', '0', 'Avg_Massflow(blade1)', '0', 'Avg_Massflow(inlet)', '0', 'Avg_Massflow(outlet)', '0', 'Avg_NormalVel(blade1)', '0', 'Avg_NormalVel(inlet)', '0', 'Avg_NormalVel(outlet)', '0', 'Avg_Press(blade1)', '0', 'Avg_Press(inlet)', '0', 'Avg_Press(outlet)', '0', 'Avg_Temp(blade1)', '0', 'Avg_Temp(outlet)', '0', 'Inner_Iter', '0', 'LinSolIterTurb', '0', 'Linear_Solver_Iterations', '0', 'Momentum_Distortion(inlet)', '0', 'Momentum_Distortion(outlet)', '0', 'Secondary_Strength(blade1)', '0', 'Secondary_Strength(inlet)', '0', 'Secondary_Strength(outlet)', '0', 'Uniformity(blade1)', '0', 'Uniformity(inlet)', '0', 'Uniformity(outlet)', '0']
liveProgrammableSource2Display_1.SeriesMarkerSize = ['     "Avg CFL"    ', '4', '     "Max CFL"    ', '4', '     "Max DT"     ', '4', '     "Min CFL"    ', '4', '     "Min DT"     ', '4', '    "LinSolRes"   ', '4', '    "Time(sec)"   ', '4', '    "rms[RhoE]"   ', '4', '    "rms[RhoU]"   ', '4', '    "rms[Rho]"    ', '4', '  "LinSolResTurb" ', '4', ' "Avg_Mach(inlet)"', '4', ' "Avg_Temp(inlet)"', '4', 'Avg_Density(blade1)', '4', 'Avg_Density(inlet)', '4', 'Avg_Density(outlet)', '4', 'Avg_Enthalpy(blade1)', '4', 'Avg_Enthalpy(inlet)', '4', 'Avg_Enthalpy(outlet)', '4', 'Avg_Mach(blade1)', '4', 'Avg_Mach(outlet)', '4', 'Avg_Massflow(blade1)', '4', 'Avg_Massflow(inlet)', '4', 'Avg_Massflow(outlet)', '4', 'Avg_NormalVel(blade1)', '4', 'Avg_NormalVel(inlet)', '4', 'Avg_NormalVel(outlet)', '4', 'Avg_Press(blade1)', '4', 'Avg_Press(inlet)', '4', 'Avg_Press(outlet)', '4', 'Avg_Temp(blade1)', '4', 'Avg_Temp(outlet)', '4', 'Inner_Iter', '4', 'LinSolIterTurb', '4', 'Linear_Solver_Iterations', '4', 'Momentum_Distortion(inlet)', '4', 'Momentum_Distortion(outlet)', '4', 'Secondary_Strength(blade1)', '4', 'Secondary_Strength(inlet)', '4', 'Secondary_Strength(outlet)', '4', 'Uniformity(blade1)', '4', 'Uniformity(inlet)', '4', 'Uniformity(outlet)', '4']

# Properties modified on liveProgrammableSource2Display_1
liveProgrammableSource2Display_1.SeriesVisibility = []

# Properties modified on liveProgrammableSource2Display_1
liveProgrammableSource2Display_1.SeriesVisibility = ['     "Avg CFL"    ']

# set active view
SetActiveView(renderView1)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1757, 1099)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.7515408396720886, 1.5609371662139893, 13.89727520942688]
renderView1.CameraFocalPoint = [0.7515408396720886, 1.5609371662139893, 0.0]
renderView1.CameraParallelScale = 2.716152835436206


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------