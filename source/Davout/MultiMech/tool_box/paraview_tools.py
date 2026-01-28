# Routine to externally control paraview and automate the extraction of
# simulation output

from paraview.simple import *

from ...PythonicUtilities import path_tools

from ...PythonicUtilities import programming_tools

########################################################################
#                           Frozen snapshots                           #
########################################################################

# Defines a function to call the frozen_snapshots function,as it must be
# ran externally using the pvpython interpreter of ParaView

def frozen_snapshots(input_fileName, field_name, input_path=None,
output_path=None, camera_position=None, color_map="Cool to Warm", 
output_imageFileName="plot.png", execution_rootPath=None):
    
    programming_tools.script_executioner("paraview_tools.py", 
    parent_path="source//tool_box", python_interpreter="pvpython",
    function_name="LOCAL_frozenSnapshots", arguments_list=[input_fileName,
    field_name], keyword_argumentsDict={"input_path": input_path, "out"+
    "put_path": output_path, "camera_position": camera_position, "colo"+
    "r_map": color_map, "output_imageFileName": output_imageFileName},
    execution_rootPath=execution_rootPath)

# Defines a function to control paraview to take a single or a set of
# frozen snapshots

def LOCAL_frozenSnapshots(input_fileName, field_name, input_path=None,
output_path=None, camera_position=None, color_map="Cool to Warm", 
output_imageFileName="plot.png"):
    
    print("Gibberish")

    float(a)
    
    # Verifies the input and output paths

    if input_path:

        input_fileName = path_tools.verify_path(input_path, 
        input_fileName)

    if output_path:

        output_imageFileName = path_tools.verify_path(output_path,
        output_imageFileName)

    # If the output path is None, but the input path is given, makes the
    # former equal to the latter

    elif input_path:

        output_path = input_path

        output_imageFileName = path_tools.verify_path(output_path,
        output_imageFileName)
    
    # Loads the simulation output data

    data = XDMFReader(FileNames=[input_fileName])

    data.PointArrayStatus = [field_name]

    # Shows data in view

    renderView = GetActiveViewOrCreate('RenderView')

    display = Show(data, renderView)

    # Sets color and representation

    ColorBy(display, ('POINTS', field_name))

    LookupTable = GetColorTransferFunction(field_name)

    display.SetScalarBarVisibility(renderView, True)

    # Applies color map

    LookupTable.ApplyPreset(color_map, True)

    # Sets camera angle

    if camera_position:

        renderView.CameraPosition = camera_position

    # Renders and saves

    Render()

    SaveScreenshot(output_imageFileName, renderView)

########################################################################
#                               Testing                                #
########################################################################

# Tests the frozen snapshots

"""frozen_snapshots("displacement.xdmf", "Displacement", input_path="/hom"+
"e/matheus-janczkowski/Github/MultiMech/tests/micropolar/bending_case/"+
"results/graphics/simulation_11", execution_rootPath="/home/matheus-janczkowski/Github/MultiMech")"""