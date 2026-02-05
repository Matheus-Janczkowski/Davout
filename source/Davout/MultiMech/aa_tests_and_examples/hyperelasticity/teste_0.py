# script-version: 2.0
# Catalyst state generated using paraview version 5.13.1
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1049, 776]
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.CenterOfRotation = [0.15000000596046448, 0.5, 0.10000000149011612]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [1.353486759242743, 1.2298227470603695, 1.5953915342476168]
renderView1.CameraFocalPoint = [0.15000000596046448, 0.5, 0.10000000149011612]
renderView1.CameraViewUp = [-0.16175754636412246, 0.9319056569359231, -0.3246326273883827]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.5315072925992291
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.PolarGrid = 'Polar Grid Actor'

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1049, 776)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Xdmf3 Reader S'
displacementxdmf = Xdmf3ReaderS(registrationName='displacement.xdmf', FileName=['/home/matheus-janczkowski/Github/Davout/source/Davout/MultiMech/aa_tests_and_examples/hyperelasticity/displacement.xdmf'])
displacementxdmf.PointArrays = ['Displacement']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from displacementxdmf
displacementxdmfDisplay = Show(displacementxdmf, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'Displacement'
displacementTF2D = GetTransferFunction2D('Displacement')

# get color transfer function/color map for 'Displacement'
displacementLUT = GetColorTransferFunction('Displacement')
displacementLUT.TransferFunction2D = displacementTF2D
displacementLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 0.35417886449569913, 0.865003, 0.865003, 0.865003, 0.7083577289913983, 0.705882, 0.0156863, 0.14902]
displacementLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Displacement'
displacementPWF = GetOpacityTransferFunction('Displacement')
displacementPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.7083577289913983, 1.0, 0.5, 0.0]
displacementPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
displacementxdmfDisplay.Representation = 'Surface With Edges'
displacementxdmfDisplay.ColorArrayName = ['POINTS', 'Displacement']
displacementxdmfDisplay.LookupTable = displacementLUT
displacementxdmfDisplay.InterpolateScalarsBeforeMapping = 0
displacementxdmfDisplay.SelectNormalArray = 'None'
displacementxdmfDisplay.SelectTangentArray = 'None'
displacementxdmfDisplay.SelectTCoordArray = 'None'
displacementxdmfDisplay.TextureTransform = 'Transform2'
displacementxdmfDisplay.OSPRayScaleArray = 'Displacement'
displacementxdmfDisplay.OSPRayScaleFunction = 'Piecewise Function'
displacementxdmfDisplay.OSPRayMaterial = ''
displacementxdmfDisplay.Assembly = ''
displacementxdmfDisplay.SelectedBlockSelectors = ['']
displacementxdmfDisplay.SelectOrientationVectors = 'Displacement'
displacementxdmfDisplay.ScaleFactor = 0.1
displacementxdmfDisplay.SelectScaleArray = 'Displacement'
displacementxdmfDisplay.GlyphType = 'Arrow'
displacementxdmfDisplay.GlyphTableIndexArray = 'Displacement'
displacementxdmfDisplay.GaussianRadius = 0.005
displacementxdmfDisplay.SetScaleArray = ['POINTS', 'Displacement']
displacementxdmfDisplay.ScaleTransferFunction = 'Piecewise Function'
displacementxdmfDisplay.OpacityArray = ['POINTS', 'Displacement']
displacementxdmfDisplay.OpacityTransferFunction = 'Piecewise Function'
displacementxdmfDisplay.DataAxesGrid = 'Grid Axes Representation'
displacementxdmfDisplay.PolarAxes = 'Polar Axes Representation'
displacementxdmfDisplay.ScalarOpacityFunction = displacementPWF
displacementxdmfDisplay.ScalarOpacityUnitDistance = 0.10140399464321644
displacementxdmfDisplay.OpacityArrayName = ['POINTS', 'Displacement']
displacementxdmfDisplay.SelectInputVectors = ['POINTS', 'Displacement']
displacementxdmfDisplay.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
displacementxdmfDisplay.ScaleTransferFunction.Points = [-0.023747267201542854, 0.0, 0.5, 0.0, 0.02373075857758522, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
displacementxdmfDisplay.OpacityTransferFunction.Points = [-0.023747267201542854, 0.0, 0.5, 0.0, 0.02373075857758522, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for displacementLUT in view renderView1
displacementLUTColorBar = GetScalarBar(displacementLUT, renderView1)
displacementLUTColorBar.WindowLocation = 'Any Location'
displacementLUTColorBar.Title = 'Displacement'
displacementLUTColorBar.ComponentTitle = 'Magnitude'
displacementLUTColorBar.TitleFontFamily = 'Times'
displacementLUTColorBar.TitleFontSize = 22
displacementLUTColorBar.LabelFontFamily = 'Times'
displacementLUTColorBar.LabelFontSize = 22
displacementLUTColorBar.ScalarBarLength = 0.3300000000000003

# set color bar visibility
displacementLUTColorBar.Visibility = 1

# show color legend
displacementxdmfDisplay.SetScalarBarVisibility(renderView1, True)

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
animationScene1.ViewModules = renderView1
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 1.0
animationScene1.PlayMode = 'Snap To TimeSteps'

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(displacementxdmf)
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
