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
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
bladeName = sys.argv[2]
suffix = sys.argv[3]

volume_vtu = run_dir / f"volume_flow_{suffix}_{bladeName}.vtu"
history_csv = run_dir / f"history_{suffix}_{bladeName}.csv"
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

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

# create reader for volume flow results
volume_flow_reader = XMLUnstructuredGridReader(
    registrationName=f'volume_flow_{suffix}_{bladeName}',
    FileName=[str(volume_vtu)]
)

# get display properties
volume_flow_readerDisplay = GetDisplayProperties(volume_flow_reader, view=renderView1)

# create a new 'Live Programmable Source'
liveProgrammableSource1 = LiveProgrammableSource(registrationName='LiveProgrammableSource1')

# set active source
SetActiveSource(volume_flow_reader)

# set active source
SetActiveSource(liveProgrammableSource1)

# Properties modified on liveProgrammableSource1
liveProgrammableSource1.OutputDataSetType = 'vtkUnstructuredGrid'
liveProgrammableSource1.Script = f"""# .vtu paraview
from paraview.vtk.vtkIOXML import vtkXMLUnstructuredGridReader as vtuReader
reader = vtuReader()
reader.SetFileName(r'{volume_vtu}')
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
Hide(volume_flow_reader, renderView1)

# set scalar coloring
ColorBy(liveProgrammableSource1Display, ('POINTS', 'Mach'))

# rescale color and/or opacity maps used to include current data range
liveProgrammableSource1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
liveProgrammableSource1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Mach'
machLUT = GetColorTransferFunction('Mach')

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

data = pd.read_csv(r'{history_csv}',sep=',')
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