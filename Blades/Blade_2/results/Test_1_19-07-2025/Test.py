# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Line Chart View'
lineChartView2 = CreateView('XYChartView')
lineChartView2.ViewSize = [868, 521]
lineChartView2.LegendPosition = [790, 489]
lineChartView2.LeftAxisUseCustomRange = 1
lineChartView2.LeftAxisRangeMinimum = -0.006670011896832721
lineChartView2.LeftAxisRangeMaximum = 0.02699756199840494
lineChartView2.BottomAxisUseCustomRange = 1
lineChartView2.BottomAxisRangeMinimum = -13.408318304318431
lineChartView2.BottomAxisRangeMaximum = 100.20897000838143
lineChartView2.RightAxisUseCustomRange = 1
lineChartView2.RightAxisRangeMinimum = -0.1179376773986337
lineChartView2.RightAxisRangeMaximum = 7.356263727344185
lineChartView2.TopAxisUseCustomRange = 1
lineChartView2.TopAxisRangeMinimum = 1.1705437844537152
lineChartView2.TopAxisRangeMaximum = 10.072792492248746

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1746, 522]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.CenterOfRotation = [0.751047670841217, 0.7598607104969026, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [0.751047670841217, 0.7598607104969026, 19.351978909705245]
renderView1.CameraFocalPoint = [0.751047670841217, 0.7598607104969026, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 5.2095092517178525
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'SpreadSheet View'
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.SplitVertical(0, 0.500000)
layout1.AssignView(1, renderView1)
layout1.SplitHorizontal(2, 0.500000)
layout1.AssignView(5, lineChartView2)
layout1.AssignView(6, spreadSheetView1)
layout1.SetSize(1746, 1044)

# ----------------------------------------------------------------
# restore active view
SetActiveView(lineChartView2)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
volume_flow_databladeVALIDATION_Blade_2vtu = XMLUnstructuredGridReader(registrationName='volume_flow_databladeVALIDATION_Blade_2.vtu', FileName=['C:\\Users\\fredd\\Documents\\GitHub\\Master-Thesis\\Blades\\Blade_2\\results\\Test_1_19-07-2025\\volume_flow_databladeVALIDATION_Blade_2.vtu'])
volume_flow_databladeVALIDATION_Blade_2vtu.PointArrayStatus = ['Mach', 'Pressure']
volume_flow_databladeVALIDATION_Blade_2vtu.TimeArray = 'None'

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=volume_flow_databladeVALIDATION_Blade_2vtu)
calculator1.Function = '( 6509.45782 - (Pressure * (1 + 0.5*0.4*Mach^2)^(1.4/0.4))) / 6509.45782'

# create a new 'Append Datasets'
appendDatasets2 = AppendDatasets(registrationName='AppendDatasets2', Input=calculator1)

# create a new 'Mesh Quality'
meshQuality1 = MeshQuality(registrationName='MeshQuality1', Input=calculator1)

# create a new 'Append Datasets'
appendDatasets3 = AppendDatasets(registrationName='AppendDatasets3', Input=calculator1)

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=volume_flow_databladeVALIDATION_Blade_2vtu)
calculator2.Function = 'coordsY/1.08088'

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=appendDatasets3)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]
slice1.PointMergeMethod = 'Uniform Binning'

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [1.9848788094059728, 1.8407407104969025, 0.0]
slice1.SliceType.Normal = [0.0, 1.0, 0.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.751047670841217, 1.8407407104969025, 0.0]

# create a new 'Append Datasets'
appendDatasets1 = AppendDatasets(registrationName='AppendDatasets1', Input=calculator1)

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=calculator1)
plotOverLine1.Point1 = [2.002792, -1.08088, 0.0]
plotOverLine1.Point2 = [2.002792, 1.08088, 0.0]

# create a new 'Append Datasets'
appendDatasets4 = AppendDatasets(registrationName='AppendDatasets4', Input=calculator1)

# create a new 'Plot Over Line'
plotOverLine2 = PlotOverLine(registrationName='PlotOverLine2', Input=appendDatasets4)
plotOverLine2.Point1 = [2.002792, -1.08088, 0.0]
plotOverLine2.Point2 = [2.002792, 1.08088, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'lineChartView2'
# ----------------------------------------------------------------

# show data from slice1
slice1Display = Show(slice1, lineChartView2, 'XYChartRepresentation')

# trace defaults for the display properties.
slice1Display.XArrayName = 'Points_Y'
slice1Display.SeriesVisibility = ['Result']
slice1Display.SeriesLabel = ['Mach', 'Mach', 'Pressure', 'Pressure', 'Result', 'Result', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
slice1Display.SeriesColor = ['Mach', '0', '0', '0', 'Pressure', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'Result', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'Points_X', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'Points_Y', '0.6', '0.3100022888532845', '0.6399938963912413', 'Points_Z', '1', '0.5000076295109483', '0', 'Points_Magnitude', '0.6500038147554742', '0.3400015259021897', '0.16000610360875867']
slice1Display.SeriesOpacity = ['Mach', '1', 'Pressure', '1', 'Result', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
slice1Display.SeriesPlotCorner = ['Mach', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Pressure', '0', 'Result', '0']
slice1Display.SeriesLabelPrefix = ''
slice1Display.SeriesLineStyle = ['Mach', '1', 'Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Pressure', '1', 'Result', '1']
slice1Display.SeriesLineThickness = ['Mach', '2', 'Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Pressure', '2', 'Result', '2']
slice1Display.SeriesMarkerStyle = ['Mach', '0', 'Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Pressure', '0', 'Result', '0']
slice1Display.SeriesMarkerSize = ['Mach', '4', 'Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'Pressure', '4', 'Result', '4']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from calculator1
calculator1Display = Show(calculator1, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'Result'
resultTF2D = GetTransferFunction2D('Result')

# get color transfer function/color map for 'Result'
resultLUT = GetColorTransferFunction('Result')
resultLUT.TransferFunction2D = resultTF2D
resultLUT.RGBPoints = [-0.01347142200431029, 0.231373, 0.298039, 0.752941, 0.10848758202770505, 0.865003, 0.865003, 0.865003, 0.23044658605972038, 0.705882, 0.0156863, 0.14902]
resultLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Result'
resultPWF = GetOpacityTransferFunction('Result')
resultPWF.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]
resultPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'
calculator1Display.ColorArrayName = ['POINTS', 'Result']
calculator1Display.LookupTable = resultLUT
calculator1Display.SelectTCoordArray = 'None'
calculator1Display.SelectNormalArray = 'None'
calculator1Display.SelectTangentArray = 'None'
calculator1Display.OSPRayScaleArray = 'Mach'
calculator1Display.OSPRayScaleFunction = 'Piecewise Function'
calculator1Display.Assembly = ''
calculator1Display.SelectOrientationVectors = 'None'
calculator1Display.ScaleFactor = 0.5060641229152679
calculator1Display.SelectScaleArray = 'Mach'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'Mach'
calculator1Display.GaussianRadius = 0.025303206145763396
calculator1Display.SetScaleArray = ['POINTS', 'Mach']
calculator1Display.ScaleTransferFunction = 'Piecewise Function'
calculator1Display.OpacityArray = ['POINTS', 'Mach']
calculator1Display.OpacityTransferFunction = 'Piecewise Function'
calculator1Display.DataAxesGrid = 'Grid Axes Representation'
calculator1Display.PolarAxes = 'Polar Axes Representation'
calculator1Display.ScalarOpacityFunction = resultPWF
calculator1Display.ScalarOpacityUnitDistance = 0.15714657027269077
calculator1Display.OpacityArrayName = ['POINTS', 'Mach']
calculator1Display.SelectInputVectors = [None, '']
calculator1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.6219130158424377, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.6219130158424377, 1.0, 0.5, 0.0]

# show data from appendDatasets1
appendDatasets1Display = Show(appendDatasets1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets1Display.Representation = 'Surface'
appendDatasets1Display.ColorArrayName = ['POINTS', 'Result']
appendDatasets1Display.LookupTable = resultLUT
appendDatasets1Display.SelectTCoordArray = 'None'
appendDatasets1Display.SelectNormalArray = 'None'
appendDatasets1Display.SelectTangentArray = 'None'
appendDatasets1Display.Position = [0.0, 1.08088, 0.0]
appendDatasets1Display.OSPRayScaleArray = 'Result'
appendDatasets1Display.OSPRayScaleFunction = 'Piecewise Function'
appendDatasets1Display.Assembly = ''
appendDatasets1Display.SelectOrientationVectors = 'None'
appendDatasets1Display.ScaleFactor = 0.5060641229152679
appendDatasets1Display.SelectScaleArray = 'Result'
appendDatasets1Display.GlyphType = 'Arrow'
appendDatasets1Display.GlyphTableIndexArray = 'Result'
appendDatasets1Display.GaussianRadius = 0.025303206145763396
appendDatasets1Display.SetScaleArray = ['POINTS', 'Result']
appendDatasets1Display.ScaleTransferFunction = 'Piecewise Function'
appendDatasets1Display.OpacityArray = ['POINTS', 'Result']
appendDatasets1Display.OpacityTransferFunction = 'Piecewise Function'
appendDatasets1Display.DataAxesGrid = 'Grid Axes Representation'
appendDatasets1Display.PolarAxes = 'Polar Axes Representation'
appendDatasets1Display.ScalarOpacityFunction = resultPWF
appendDatasets1Display.ScalarOpacityUnitDistance = 0.15714657027269077
appendDatasets1Display.OpacityArrayName = ['POINTS', 'Result']
appendDatasets1Display.SelectInputVectors = [None, '']
appendDatasets1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
appendDatasets1Display.ScaleTransferFunction.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
appendDatasets1Display.OpacityTransferFunction.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
appendDatasets1Display.PolarAxes.Translation = [0.0, 1.08088, 0.0]

# show data from appendDatasets2
appendDatasets2Display = Show(appendDatasets2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets2Display.Representation = 'Surface'
appendDatasets2Display.ColorArrayName = ['POINTS', 'Result']
appendDatasets2Display.LookupTable = resultLUT
appendDatasets2Display.SelectTCoordArray = 'None'
appendDatasets2Display.SelectNormalArray = 'None'
appendDatasets2Display.SelectTangentArray = 'None'
appendDatasets2Display.Position = [0.0, -1.08088, 0.0]
appendDatasets2Display.OSPRayScaleArray = 'Result'
appendDatasets2Display.OSPRayScaleFunction = 'Piecewise Function'
appendDatasets2Display.Assembly = ''
appendDatasets2Display.SelectOrientationVectors = 'None'
appendDatasets2Display.ScaleFactor = 0.5060641229152679
appendDatasets2Display.SelectScaleArray = 'Result'
appendDatasets2Display.GlyphType = 'Arrow'
appendDatasets2Display.GlyphTableIndexArray = 'Result'
appendDatasets2Display.GaussianRadius = 0.025303206145763396
appendDatasets2Display.SetScaleArray = ['POINTS', 'Result']
appendDatasets2Display.ScaleTransferFunction = 'Piecewise Function'
appendDatasets2Display.OpacityArray = ['POINTS', 'Result']
appendDatasets2Display.OpacityTransferFunction = 'Piecewise Function'
appendDatasets2Display.DataAxesGrid = 'Grid Axes Representation'
appendDatasets2Display.PolarAxes = 'Polar Axes Representation'
appendDatasets2Display.ScalarOpacityFunction = resultPWF
appendDatasets2Display.ScalarOpacityUnitDistance = 0.15714657027269077
appendDatasets2Display.OpacityArrayName = ['POINTS', 'Result']
appendDatasets2Display.SelectInputVectors = [None, '']
appendDatasets2Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
appendDatasets2Display.ScaleTransferFunction.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
appendDatasets2Display.OpacityTransferFunction.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
appendDatasets2Display.PolarAxes.Translation = [0.0, -1.08088, 0.0]

# show data from appendDatasets3
appendDatasets3Display = Show(appendDatasets3, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets3Display.Representation = 'Surface'
appendDatasets3Display.ColorArrayName = ['POINTS', 'Result']
appendDatasets3Display.LookupTable = resultLUT
appendDatasets3Display.SelectTCoordArray = 'None'
appendDatasets3Display.SelectNormalArray = 'None'
appendDatasets3Display.SelectTangentArray = 'None'
appendDatasets3Display.Position = [0.0, -2.1617, 0.0]
appendDatasets3Display.OSPRayScaleArray = 'Result'
appendDatasets3Display.OSPRayScaleFunction = 'Piecewise Function'
appendDatasets3Display.Assembly = ''
appendDatasets3Display.SelectOrientationVectors = 'None'
appendDatasets3Display.ScaleFactor = 0.5060641229152679
appendDatasets3Display.SelectScaleArray = 'Result'
appendDatasets3Display.GlyphType = 'Arrow'
appendDatasets3Display.GlyphTableIndexArray = 'Result'
appendDatasets3Display.GaussianRadius = 0.025303206145763396
appendDatasets3Display.SetScaleArray = ['POINTS', 'Result']
appendDatasets3Display.ScaleTransferFunction = 'Piecewise Function'
appendDatasets3Display.OpacityArray = ['POINTS', 'Result']
appendDatasets3Display.OpacityTransferFunction = 'Piecewise Function'
appendDatasets3Display.DataAxesGrid = 'Grid Axes Representation'
appendDatasets3Display.PolarAxes = 'Polar Axes Representation'
appendDatasets3Display.ScalarOpacityFunction = resultPWF
appendDatasets3Display.ScalarOpacityUnitDistance = 0.15714657027269077
appendDatasets3Display.OpacityArrayName = ['POINTS', 'Result']
appendDatasets3Display.SelectInputVectors = [None, '']
appendDatasets3Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
appendDatasets3Display.ScaleTransferFunction.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
appendDatasets3Display.OpacityTransferFunction.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
appendDatasets3Display.PolarAxes.Translation = [0.0, -2.1617, 0.0]

# show data from appendDatasets4
appendDatasets4Display = Show(appendDatasets4, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
appendDatasets4Display.Representation = 'Surface'
appendDatasets4Display.ColorArrayName = ['POINTS', 'Result']
appendDatasets4Display.LookupTable = resultLUT
appendDatasets4Display.SelectTCoordArray = 'None'
appendDatasets4Display.SelectNormalArray = 'None'
appendDatasets4Display.SelectTangentArray = 'None'
appendDatasets4Display.Position = [0.0, -3.24264, 0.0]
appendDatasets4Display.OSPRayScaleArray = 'Result'
appendDatasets4Display.OSPRayScaleFunction = 'Piecewise Function'
appendDatasets4Display.Assembly = ''
appendDatasets4Display.SelectOrientationVectors = 'None'
appendDatasets4Display.ScaleFactor = 0.5060641229152679
appendDatasets4Display.SelectScaleArray = 'Result'
appendDatasets4Display.GlyphType = 'Arrow'
appendDatasets4Display.GlyphTableIndexArray = 'Result'
appendDatasets4Display.GaussianRadius = 0.025303206145763396
appendDatasets4Display.SetScaleArray = ['POINTS', 'Result']
appendDatasets4Display.ScaleTransferFunction = 'Piecewise Function'
appendDatasets4Display.OpacityArray = ['POINTS', 'Result']
appendDatasets4Display.OpacityTransferFunction = 'Piecewise Function'
appendDatasets4Display.DataAxesGrid = 'Grid Axes Representation'
appendDatasets4Display.PolarAxes = 'Polar Axes Representation'
appendDatasets4Display.ScalarOpacityFunction = resultPWF
appendDatasets4Display.ScalarOpacityUnitDistance = 0.15714657027269077
appendDatasets4Display.OpacityArrayName = ['POINTS', 'Result']
appendDatasets4Display.SelectInputVectors = [None, '']
appendDatasets4Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
appendDatasets4Display.ScaleTransferFunction.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
appendDatasets4Display.OpacityTransferFunction.Points = [-0.01347142200431029, 0.0, 0.5, 0.0, 0.23044658605972038, 1.0, 0.5, 0.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
appendDatasets4Display.PolarAxes.Translation = [0.0, -3.24264, 0.0]

# show data from slice1
slice1Display_1 = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display_1.Representation = 'Surface'
slice1Display_1.ColorArrayName = ['POINTS', 'Result']
slice1Display_1.LookupTable = resultLUT
slice1Display_1.SelectTCoordArray = 'None'
slice1Display_1.SelectNormalArray = 'None'
slice1Display_1.SelectTangentArray = 'None'
slice1Display_1.OSPRayScaleArray = 'Result'
slice1Display_1.OSPRayScaleFunction = 'Piecewise Function'
slice1Display_1.Assembly = ''
slice1Display_1.SelectOrientationVectors = 'None'
slice1Display_1.ScaleFactor = 0.10808800458908081
slice1Display_1.SelectScaleArray = 'Result'
slice1Display_1.GlyphType = 'Arrow'
slice1Display_1.GlyphTableIndexArray = 'Result'
slice1Display_1.GaussianRadius = 0.005404400229454041
slice1Display_1.SetScaleArray = ['POINTS', 'Result']
slice1Display_1.ScaleTransferFunction = 'Piecewise Function'
slice1Display_1.OpacityArray = ['POINTS', 'Result']
slice1Display_1.OpacityTransferFunction = 'Piecewise Function'
slice1Display_1.DataAxesGrid = 'Grid Axes Representation'
slice1Display_1.PolarAxes = 'Polar Axes Representation'
slice1Display_1.SelectInputVectors = [None, '']
slice1Display_1.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
slice1Display_1.ScaleTransferFunction.Points = [-0.007047622742650492, 0.0, 0.5, 0.0, 0.2137826231425648, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
slice1Display_1.OpacityTransferFunction.Points = [-0.007047622742650492, 0.0, 0.5, 0.0, 0.2137826231425648, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for resultLUT in view renderView1
resultLUTColorBar = GetScalarBar(resultLUT, renderView1)
resultLUTColorBar.WindowLocation = 'Upper Right Corner'
resultLUTColorBar.Title = 'Result'
resultLUTColorBar.ComponentTitle = ''

# set color bar visibility
resultLUTColorBar.Visibility = 1

# show color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
appendDatasets1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
appendDatasets2Display.SetScalarBarVisibility(renderView1, True)

# show color legend
appendDatasets3Display.SetScalarBarVisibility(renderView1, True)

# show color legend
appendDatasets4Display.SetScalarBarVisibility(renderView1, True)

# show color legend
slice1Display_1.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup the visualization in view 'spreadSheetView1'
# ----------------------------------------------------------------

# show data from meshQuality1
meshQuality1Display = Show(meshQuality1, spreadSheetView1, 'SpreadSheetRepresentation')

# trace defaults for the display properties.
meshQuality1Display.Assembly = ''

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = [renderView1, lineChartView2, spreadSheetView1]
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.0

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(meshQuality1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.CatalystLiveTrigger = 'Time Step'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
