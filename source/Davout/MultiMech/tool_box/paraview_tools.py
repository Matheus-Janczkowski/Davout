# Routine to externally control paraview and automate the extraction of
# simulation output

from paraview.simple import *

from ...PythonicUtilities import path_tools

from ...PythonicUtilities import programming_tools

from ...PythonicUtilities.string_tools import string_toList

########################################################################
#                           Frozen snapshots                           #
########################################################################

# Defines a function to call the frozen_snapshots function,as it must be
# ran externally using the pvpython interpreter of ParaView

def frozen_snapshots(input_fileName, field_name, input_path=None,
output_path=None, camera_position=None, color_map="Cool to Warm", 
output_imageFileName="plot.png", execution_rootPath=None, time=None,
time_step_index=None, camera_focal_point=None, camera_up_direction=None,
representation_type=None, legend_bar_position=None, legend_bar_length=
None, axes_color=None, image_resolution=None, get_attributes_render=None,
camera_parallel_scale=None, camera_rotation=None):
    
    programming_tools.script_executioner("source.Davout.MultiMech.tool"+
    "_box.paraview_tools", python_interpreter="pvpython", function_name=
    "LOCAL_frozenSnapshots", arguments_list=[input_fileName, field_name], 
    keyword_argumentsDict={"input_path": input_path, "output_path": 
    output_path, "camera_position": camera_position, "color_map": 
    color_map, "output_imageFileName": output_imageFileName, "time_ste"+
    "p_index": time_step_index, "time": time, "camera_focal_point":
    camera_focal_point, "camera_up_direction": camera_up_direction,
    "representation_type": representation_type, "legend_bar_position":
    legend_bar_position, "legend_bar_length": legend_bar_length, "axes"+
    "_color": axes_color, "image_resolution": image_resolution, "get_a"+
    "ttributes_render": get_attributes_render, "camera_parallel_scale":
    camera_parallel_scale, "camera_rotation": camera_rotation},
    execution_rootPath=execution_rootPath, run_as_module=True)

# Defines a function to control paraview to take a single or a set of
# frozen snapshots

def LOCAL_frozenSnapshots(input_fileName, field_name, input_path=None,
output_path=None, camera_position=None, color_map="Cool to Warm", 
output_imageFileName="plot.png", time_step_index=None, time=None,
camera_focal_point=None, camera_up_direction=None, representation_type=
None, legend_bar_position=None, legend_bar_length=None, axes_color=None,
image_resolution=None, get_attributes_render=None, camera_parallel_scale=
None, camera_rotation=None):
    
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

    # Selects the time step

    animation_scene = GetAnimationScene()

    animation_scene.UpdateAnimationUsingDataTimeSteps()

    # Verifies if the time value was given

    if time is not None:

        # Selects this time to show

        animation_scene.AnimationTime = float(time)

    # Otherwise, if the index of the time step was provided

    elif time_step_index is not None:

        # Selects this time to show

        times = animation_scene.TimeKeeper.TimestepValues

        animation_scene.AnimationTime = times[int(time_step_index)]

    # Shows data in view

    renderView = GetActiveViewOrCreate('RenderView')

    display = Show(data, renderView)

    Render()

    # Verifies if the color of the coordinate system triad is given

    if axes_color:

        # Verifies if it is a list

        if (axes_color[0]!="[" or axes_color[-1]!="]"):

            raise TypeError("'axes_color' in 'frozen_snapshots' must b"+
            "e a list of 3 components of RGB values---[0.0, 0.0, 0.0] "+
            "means black; [1.0, 1.0, 1.0] means white. Currently, it i"+
            "s: "+str(axes_color)+"; whose type is: "+str(type(
            axes_color)))
        
        # Converts the argument to a list and sets the parameter

        axes_color = string_toList(axes_color)

        renderView.OrientationAxesXColor = axes_color

        renderView.OrientationAxesYColor = axes_color

        renderView.OrientationAxesZColor = axes_color

    # Verifies if the representation is set

    if representation_type:

        # Sets a list of allowed representation options

        available_representations = ['Surface', 'Surface With Edges', 
        'Wireframe', 'Points', 'Volume', 'Outline', 'Feature Edges', 
        'Slice', 'Point Gaussian']

        # Verifies if it is an available representation type

        if not (representation_type in available_representations):

            available_names = ""

            for name in available_representations:

                available_names += "\n"+name

            raise ValueError("The provided 'representation_type' is '"+
            str(representation_type)+"'. But the available options are:"+
            available_names)
        
        # Sets the representation

        display.Representation = representation_type

    # Sets color

    ColorBy(display, ('POINTS', field_name))

    LookupTable = GetColorTransferFunction(field_name)

    display.SetScalarBarVisibility(renderView, True)

    # Sets the position of the legend

    if legend_bar_position:

        # Verifies if it is a list

        if (legend_bar_position[0]!="[" or legend_bar_position[-1]!="]"):

            raise TypeError("'legend_bar_position' in 'frozen_snapshot"+
            "s' must be a list of 2 components---[0.0,0.0] means left-"+
            "bottom, whereas [1.0, 1.0] means right-top. Currently, it"+
            " is: "+str(legend_bar_position)+"; whose type is: "+str(
            type(legend_bar_position)))
        
        # Converts the argument to a list and sets the parameter

        legend_bar_position = string_toList(legend_bar_position)

        scalarBar = GetScalarBar(LookupTable, renderView)

        scalarBar.Position = legend_bar_position

    # Sets the size of the legend

    if legend_bar_length:

        try:

            legend_bar_length = float(legend_bar_length)

        except:

            raise ValueError("Could not convert 'legend_bar_length' to"+
            " float in 'frozen_snapshots'. The current value is: "+str(
            legend_bar_length))
        
        # Sets the length

        scalarBar = GetScalarBar(LookupTable, renderView)

        scalarBar.ScalarBarLength = legend_bar_length

    # Applies color map

    if color_map:

        # Sets a list of allowed color maps

        available_color_maps = ['Cool to Warm', 'Warm to Cool', 'Rainb'+
        'ow Uniform', 'Viridis (matplotlib)', 'Plasma (matplotlib)', 
        'Inferno (matplotlib)', 'Magma (matplotlib)', 'Turbo', 'Jet',
        'Grayscale', 'Black-Body Radiation', 'Blue to Red Rainbow', 'C'+
        'old and Hot', 'X Ray', 'erdc_rainbow_dark', 'erdc_rainbow_bri'+
        'ght', 'Rainbow Desaturated']

        # Verifies if it is an available color map

        if not (color_map in available_color_maps):

            available_names = ""

            for name in available_color_maps:

                available_names += "\n"+name

            raise ValueError("The provided 'color_map' is '"+str(
            color_map)+"'. But the available options are:"+
            available_names)

        # Sets the color map

        LookupTable.ApplyPreset(color_map, True)

    # Sets camera position in space

    if camera_position:

        if (camera_position[0]!="[" or camera_position[-1]!="]"):

            raise TypeError("'camera_position' in 'frozen_snapshots' m"+
            "ust be a list of 3 components. Currently, it is: "+str(
            camera_position)+"; whose type is: "+str(type(
            camera_position)))
        
        # Converts the argument to a list

        camera_position = string_toList(camera_position)

        renderView.CameraPosition = camera_position

    # Sets camera focal point

    if camera_focal_point:

        if (camera_focal_point[0]!="[" or camera_focal_point[-1]!="]"):

            raise TypeError("'camera_focal_point' in 'frozen_snapshots"+
            "' must be a list of 3 components. Currently, it is: "+str(
            camera_focal_point)+"; whose type is: "+str(type(
            camera_focal_point)))
        
        # Converts the argument to a list

        camera_focal_point = string_toList(camera_focal_point)

        renderView.CameraFocalPoint = camera_focal_point

    # Sets camera upwards direction

    if camera_up_direction:

        if (camera_up_direction[0]!="[" or camera_up_direction[-1]!="]"):

            raise TypeError("'camera_up_direction' in 'frozen_snapshot"+
            "s' must be a list of 3 components. Currently, it is: "+str(
            camera_up_direction)+"; whose type is: "+str(type(
            camera_up_direction)))
        
        # Converts the argument to a list

        camera_up_direction = string_toList(camera_up_direction)

        renderView.CameraViewUp = camera_up_direction

    # Sets camera rotation

    if camera_rotation:

        if (camera_rotation[0]!="[" or camera_rotation[-1]!="]"):

            raise TypeError("'camera_rotation' in 'frozen_snapshot"+
            "s' must be a list of 3 components. Currently, it is: "+str(
            camera_rotation)+"; whose type is: "+str(type(
            camera_rotation)))
        
        # Converts the argument to a list

        camera_rotation = string_toList(camera_rotation)

        renderView.CenterOfRotation = camera_rotation

    # Sets the parallel scale

    if camera_parallel_scale:

        try:

            camera_parallel_scale = float(camera_parallel_scale)

        except:

            raise ValueError("Could not convert 'camera_parallel_scale"+
            "' to float in 'frozen_snapshots'. The current value is: "+
            str(camera_parallel_scale))
        
        # Sets the parallel scale

        renderView.CameraParallelScale = camera_parallel_scale

    # Verifies if image resolution has been provided

    if image_resolution:

        if (image_resolution[0]!="[" or image_resolution[-1]!="]"):

            raise TypeError("'image_resolution' in 'frozen_snapshots' "+
            "must be a list of 2 components. Currently, it is: "+str(
            image_resolution)+"; whose type is: "+str(type(
            image_resolution)))
        
        # Converts the argument to a list

        renderView.ViewSize = string_toList(image_resolution)

    else:

        # Otherwise, uses the default

        image_resolution = renderView.ViewSize

    # Renders and saves

    Render()

    SaveScreenshot(output_imageFileName, renderView)

    if get_attributes_render:

        print("\nThe attributes of the RenderView are:\n"+str(
        renderView.ListProperties()))

########################################################################
#                         Functions dispatching                        #
########################################################################

# Defines a section of code to dispatch all functions from this module

if __name__ == "__main__":

    import sys

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("function")

    parser.add_argument("args", nargs="*")

    # Parses keyword args like --input_path value

    known, unknown = parser.parse_known_args()

    # Creates a dictionary of keyword arguments

    kwargs = {}

    i = 0

    while i < len(unknown):

        key = unknown[i].lstrip("-")

        value = unknown[i+1]

        kwargs[key] = value

        i += 2

    func = globals()[known.function]

    func(*known.args, **kwargs)