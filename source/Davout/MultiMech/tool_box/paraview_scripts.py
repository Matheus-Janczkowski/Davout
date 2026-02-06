# Routine to externally control paraview and automate the extraction of
# simulation output

import paraview

paraview.options.offscreen = True

from paraview.simple import *

from PIL import Image

from pathlib import Path

from importlib import util

import sys

import numpy as np

########################################################################
#                           Imports preamble                           #
########################################################################

# Gets the parent paths of the current 

broken_path = Path(__file__).parents

# Imports path tools

specifications = util.spec_from_file_location("path_tools", 
broken_path[2]/"PythonicUtilities"/"path_tools.py")

path_tools = util.module_from_spec(specifications)

sys.modules["path_tools"] = path_tools

specifications.loader.exec_module(path_tools)

# Imports string tools

specifications = util.spec_from_file_location("string_tools", 
broken_path[2]/"PythonicUtilities"/"string_tools.py")

string_tools = util.module_from_spec(specifications)

sys.modules["string_tools"] = string_tools

specifications.loader.exec_module(string_tools)

########################################################################
#                           Frozen snapshots                           #
########################################################################

# Defines a function to control paraview to take a single or a set of
# frozen snapshots

def LOCAL_frozenSnapshots(input_fileName, field_name, input_path=None,
output_path=None, camera_position=None, color_map="Cool to Warm", 
output_imageFileName="plot.png", time_step_index=None, time=None,
camera_focal_point=None, camera_up_direction=None, representation_type=
None, legend_bar_position=None, legend_bar_length=None, axes_color=None,
size_in_pixels=None, get_attributes_render=None, camera_parallel_scale=
None, camera_rotation=None, legend_bar_font=None, legend_bar_font_file=
None, zoom_factor=None, plot_x_axis=None, plot_y_axis=None, plot_z_axis=
None, no_axes=None, component_to_plot=None, resolution_ratio=None,
transparent_background=None):
    
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

    # Makes sure the file ends with xdmf

    input_fileName = path_tools.take_outFileNameTermination(
    input_fileName)+".xdmf"
    
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

        axes_color = string_tools.string_toList(axes_color)

        renderView.OrientationAxesXColor = axes_color

        renderView.OrientationAxesYColor = axes_color

        renderView.OrientationAxesZColor = axes_color

    # Verifies if no axes are to be plotted

    if no_axes:

        plot_x_axis = "False" 

        plot_y_axis = "False" 

        plot_z_axis = "False" 

    # Verifies if the axes are to plotted

    if plot_x_axis:

        if plot_x_axis=="False":

            renderView.OrientationAxesXVisibility = False

    if plot_y_axis:

        if plot_y_axis=="False":

            renderView.OrientationAxesYVisibility = False

    if plot_z_axis:

        if plot_z_axis=="False":

            renderView.OrientationAxesZVisibility = False

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

    # Gets the components of the data

    data.UpdatePipeline()

    Render()

    info = data.GetPointDataInformation()

    array_info = info.GetArray(field_name)

    number_of_components = 1

    if hasattr(array_info, "GetNumberOfComponents"):

        number_of_components = array_info.GetNumberOfComponents()

    else:

        # Makes the component to plot automatically magnitude

        component_to_plot = "Magnitude"

    # Verifies the component to plot

    if component_to_plot is None:

        # Sets Magnitude as default

        component_to_plot = "Magnitude"
    
    # If the component is not Magnitude, verifies if it is indeed a valid
    # component

    if component_to_plot!="Magnitude":

        # Initializes a flag to check if the component has been found

        flag_component_found = False

        # Tries to convert the component to a number

        try:
            
            component_to_plot = int(component_to_plot)

            if component_to_plot>=number_of_components or (
            component_to_plot<0):

                flag_component_found = "out of bounds"
            
            else:

                flag_component_found = True

        except:

            # Iterates through the components

            for i in range(number_of_components):

                if component_to_plot==array_info.GetComponentName(i):

                    # Updates the flag to tell that the component has
                    # been found

                    flag_component_found = True 

                    break 

        # Verifies the flag

        if flag_component_found=="out of bounds":

            raise NameError("'component_to_plot' in 'frozen_snapshots'"+
            " is a number, "+str(component_to_plot)+", but it is not b"+
            "etween 0 and "+str(number_of_components-1))

        elif not flag_component_found:

            available_names = ""

            for i in range(number_of_components):

                available_names += "\n'"+str(array_info.GetComponentName(
                i))+"'   or   "+str(i)

            raise NameError("'component_to_plot' in 'frozen_snapshots'"+
            " is '"+str(component_to_plot)+"', but it is not an availa"
            "ble name. Check the list of available names:"+
            available_names)

    # Sets color

    # Takes care of scalar fields

    if number_of_components==1:

        ColorBy(display, ('POINTS', field_name))

    # Otherwise, allows for the required component

    else:

        ColorBy(display, ('POINTS', field_name, component_to_plot))

    Render()

    # Rescales the color

    display.RescaleTransferFunctionToDataRange(True, True)

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

        legend_bar_position = string_tools.string_toList(
        legend_bar_position)

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

    # Sets the font of the legend

    if legend_bar_font:

        scalarBar.TitleFontFamily = legend_bar_font

        scalarBar.LabelFontFamily = legend_bar_font

    # Otherwise, if a font file is given

    elif legend_bar_font_file:

        scalarBar.TitleFontFamily = "File"

        scalarBar.LabelFontFamily = "File"

        scalarBar.TitleFontFile = legend_bar_font_file

        scalarBar.LabelFontFile = legend_bar_font_file
        
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

    if camera_position or (zoom_factor is not None):

        # Verifies if zoom was asked for, but no camera position was gi-
        # ven

        if (zoom_factor is not None) :
        
            if camera_position is None:

                raise ValueError("'camera_position' in 'frozen_snapsho"+
                "ts' is None but 'zoom_factor' is not None. One must p"+
                "rovide acamera position to ask for zoom. Currently, '"+
                "camera_position' is: "+str(camera_position)+"; and 'z"+
                "oom_factor' is: "+str(type(zoom_factor)))
            
            # Tries to convert zoom factor to float

            try:

                zoom_factor = float(zoom_factor)

            except:

                raise ValueError("Could not convert 'zoom_factor' to f"+
                "loat in 'frozen_snapshots'. The current value is: "+str(
                zoom_factor))
            
        # Otherwise, transform zoom factor to 1 to enable the multipli-
        # cation by the camera position

        else:

            zoom_factor = 1.0

        # Verifies if camera position is a list

        if (camera_position[0]!="[" or camera_position[-1]!="]"):

            raise TypeError("'camera_position' in 'frozen_snapshots' m"+
            "ust be a list of 3 components. Currently, it is: "+str(
            camera_position)+"; whose type is: "+str(type(
            camera_position)))
        
        # Converts the argument to a list

        camera_position = (np.array(string_tools.string_toList(
        camera_position))*zoom_factor).tolist()

        renderView.CameraPosition = camera_position

    # Sets camera focal point

    if camera_focal_point:

        if (camera_focal_point[0]!="[" or camera_focal_point[-1]!="]"):

            raise TypeError("'camera_focal_point' in 'frozen_snapshots"+
            "' must be a list of 3 components. Currently, it is: "+str(
            camera_focal_point)+"; whose type is: "+str(type(
            camera_focal_point)))
        
        # Converts the argument to a list

        camera_focal_point = string_tools.string_toList(
        camera_focal_point)

        renderView.CameraFocalPoint = camera_focal_point

    # Sets camera upwards direction

    if camera_up_direction:

        if (camera_up_direction[0]!="[" or camera_up_direction[-1]!="]"):

            raise TypeError("'camera_up_direction' in 'frozen_snapshot"+
            "s' must be a list of 3 components. Currently, it is: "+str(
            camera_up_direction)+"; whose type is: "+str(type(
            camera_up_direction)))
        
        # Converts the argument to a list

        camera_up_direction = string_tools.string_toList(
        camera_up_direction)

        renderView.CameraViewUp = camera_up_direction

    # Sets camera rotation

    if camera_rotation:

        if (camera_rotation[0]!="[" or camera_rotation[-1]!="]"):

            raise TypeError("'camera_rotation' in 'frozen_snapshot"+
            "s' must be a list of 3 components. Currently, it is: "+str(
            camera_rotation)+"; whose type is: "+str(type(
            camera_rotation)))
        
        # Converts the argument to a list

        camera_rotation = string_tools.string_toList(camera_rotation)

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

    # Verifies if the size of the image in pixels has been provided

    if size_in_pixels:

        # Verifies if size of the image in pixels is a list

        if size_in_pixels[0]=="[" and size_in_pixels[-1]=="]":

            size_in_pixels = string_tools.string_toList(size_in_pixels)

        # Verifies if size of the image in pixels is a dictionary

        elif size_in_pixels[0]=="{" and size_in_pixels[-1]=="}":

            image_dict = string_tools.string_toDict(size_in_pixels)

            # Verifies the keys
            
            if not ('aspect ratio' in image_dict):

                raise KeyError("'size_in_pixels' is a dictionary in "+
                "'frozen_snapshots', but the key 'aspect ratio', is no"+
                "t present. This key tells the ratio of the height of "+
                "the screenshot to its width")
            
            elif not ('pixels in width' in image_dict):

                raise KeyError("'size_in_pixels' is a dictionary in "+
                "'frozen_snapshots', but the key 'pixels in width', "+
                "is not present. This key tells the number of pixels"+
                " in the width direction")

            # Transforms the size of the image in pixels information in-
            # to a list
            
            size_in_pixels = [int(round(image_dict["pixels in width"])), 
            int(round(image_dict["pixels in width"]*image_dict["aspect"+
            " ratio"]))] 

        else:

            raise TypeError("'size_in_pixels' in 'frozen_snapshots' "+
            "must be a list of 2 components or a dictionary with the k"+
            "eys 'aspect ratio' (the ratio of the height by the width "+
            "of the screenshot) and 'pixels in width' the number of pi"+
            "xels for the width direction. Currently, it is: "+str(
            size_in_pixels)+"; whose type is: "+str(type(
            size_in_pixels)))
        
        # Sets the size

        renderView.ViewSize = size_in_pixels

    # Verifies if a resolution ratio has been provided

    if resolution_ratio:

        try:

            resolution_ratio = float(resolution_ratio)

        except:

            raise ValueError("Could not convert 'resolution_ratio' to "+
            "float in 'frozen_snapshots'. The current value is: "+str(
            resolution_ratio))
        
    # Otherwise sets to 1

    else:

        resolution_ratio = 1.0

    # Verifies if transparent background is asked for

    if transparent_background=="True":

        transparent_background = 1

    else:

        transparent_background = 0

    # Computes the image resolution in pixels by multiplying the number
    # of pixels of the image by the resolution ratio

    image_resolution = [int(round(resolution_ratio*renderView.ViewSize[0
    ])), int(round(resolution_ratio*renderView.ViewSize[1]))]

    # Renders again

    Render()

    # Verifies if the output file is meant to be a pdf

    output_imageFileName, termination = path_tools.take_outFileNameTermination(
    output_imageFileName, get_termination=True)

    print("Saves the screenshot at: '"+str(output_imageFileName)+"."+str(
    termination))

    # If termination is pdf, saves as a png first

    if termination=="pdf":

        SaveScreenshot(output_imageFileName+".png", renderView, 
        ImageResolution=image_resolution, TransparentBackground=
        transparent_background)

        # Converts to pdf

        png_image = Image.open(output_imageFileName+".png")

        png_image.save(output_imageFileName+".pdf")

    else:

        # Saves normally with the original termination

        SaveScreenshot(output_imageFileName+"."+termination, renderView, 
        ImageResolution=image_resolution, TransparentBackground=
        transparent_background)

    if get_attributes_render=="True":

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