# Routine to externally control paraview and automate the extraction of
# simulation output

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

# Imports file handling tools

specifications = util.spec_from_file_location("file_tools", 
broken_path[2]/"PythonicUtilities"/"file_handling_tools.py")

file_tools = util.module_from_spec(specifications)

sys.modules["file_tools"] = file_tools

specifications.loader.exec_module(file_tools)

########################################################################
#                              Fonts class                             #
########################################################################

# Defines a class with fonts files

class FontsClass:

    def __init__(self):
        
        # Defines a dictionary with font files

        self.font_dictionary = {"latex": ["File", str(broken_path[2
        ]/"PythonicUtilities"/"fonts"/"CMU_Serif_Roman.ttf")], "Times": 
        "Times", "Arial": "Arial", "Courier": "Courier"}

    def __call__(self, font_name):
        
        # Verifies if the font name is valid

        if not (font_name in self.font_dictionary):

            available_names = ""

            for name in self.font_dictionary.keys():

                available_names += "\n'"+str(name)+"'"

            raise ValueError("The font name '"+str(font_name)+"' is no"+
            "t a valid font name. Check the available names:"+
            available_names)
        
        # Otherwise, returns the font file

        return self.font_dictionary[font_name]

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
transparent_background=None, warp_by_vector=None, warp_scale=None, 
glyph=None, glyph_scale=None, display_reference_configuration="True",
clip=None, clip_plane_origin=None, clip_plane_normal_vector=None,
set_camera_interactively=None, background_color=None, 
legend_bar_font_color=None, color_bar_min_value=None, 
color_bar_max_value=None, read_camera_settings_dictionary=None):
    
    # Instantiates the class of fonts

    font_class = FontsClass()

    # Resets session

    ResetSession()

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

    # Verify the file existence

    path_tools.verify_file_existence(input_fileName)
    
    # Loads the simulation output data

    data = XDMFReader(FileNames=[input_fileName])

    #data.UpdatePipeline()

    data.PointArrayStatus = [field_name]

    # Gets the components of the data

    data.UpdatePipeline()

    info = data.GetPointDataInformation()

    array_info = info.GetArray(field_name)

    number_of_components = 1

    if hasattr(array_info, "GetNumberOfComponents"):

        number_of_components = array_info.GetNumberOfComponents()

        print(str(number_of_components)+" components of the field were"+
        " recovered\n", flush=True)

    else:

        print("It was not possible to get components. The field must b"+
        "e a scalar\n", flush=True)

        # Makes the component to plot automatically magnitude

        component_to_plot = "Magnitude"

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

    # Creates a new view 

    renderView = GetActiveViewOrCreate('RenderView')

    # Verifies if color is to be set for the background

    if background_color:

        # Sets a list of color palettes 

        color_palettes = ['WarmGrayBackground', 'DarkGrayBackground', 
        'NeutralGrayBackground', 'LightGrayBackground', 'WhiteBackgrou'+
        'nd', 'BlackBackground', 'GradientBackground']

        # Verifies if background color is in the color paletters

        if background_color in color_palettes:

            LoadPalette(paletteName=background_color) 
        
        else:

            # Sets a dictionary of pre-made colors

            colors_dictionary = {"white": [1.0, 1.0, 1.0], "black": [0.0, 
            0.0, 0.0], "dark gray": [0.1, 0.1, 0.1]}

            # Verifies if it is a list

            if (background_color[0]=="[" and background_color[-1]=="]"):

                # Transforms into a list

                background_color = string_tools.string_toList(
                background_color)

            elif background_color in colors_dictionary:

                # Gets the values from the dictionary

                background_color = colors_dictionary[background_color]

            else:

                available_names = ""

                for name in colors_dictionary.keys():

                    available_names += "\n'"+name+"'"

                available_palettes = ""

                for name in color_palettes:

                    available_palettes += "\n'"+name+"'"

                raise TypeError("'background_color' in 'frozen_snapsho"+
                "ts' must be:\n\na list of 3 components of RGB values-"+
                "--[0.0, 0.0, 0.0] means black; [1.0, 1.0, 1.0] means "+
                "white;\n\nor it can be one of the available names:"+
                available_names+";\n\nor it can be one of the palletes"+
                ":"+available_palettes+".\n\n"+"Currently, it is: "+str(
                background_color))
            
            # Sets the color 

            renderView.UseColorPaletteForBackground = 0

            renderView.BackgroundColorMode = 'Single Color'

            renderView.Background = background_color

    # Sets a dictionary of displays to color

    display_dictionary = {}

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
        
    else:

        representation_type = "Surface"

    ####################################################################
    #                       Substitutive filters                       #
    ####################################################################

    # Creates a variable to store the data, especially if it was warped

    active_data = data

    # Otherwise displays the raw data only

    if display_reference_configuration=="True":

        # Displays the rendered data

        display = Show(active_data, renderView)

        # Adds this display object to the dictionary of displays

        display_dictionary["reference"] = {"object": display, "number "+
        "of components": number_of_components, "representation":
        representation_type, "get scalar bar from": True}

    # Verifies if a warped by vector object must be created

    if warp_by_vector=="True":

        # Verifies if the data has at least 3 components

        if number_of_components<2:

            raise TypeError("The data provided to snapshot '"+str(
            field_name)+"' field in '"+str(input_fileName)+"' has "+str(
            number_of_components)+" components. Thus, it is not possib"+
            "le to warp it by vector. The user asked to plot component"+
            " '"+str(component_to_plot)+"' over the deformed configura"+
            "tion")

        # Creates the warped object

        warp_object = WarpByVector(Input=data)

        warp_object.Vectors = ['POINTS', field_name]

        # Verifies if the scale of the warped mesh was given

        if warp_scale:

            # Tries to convert the warp scale to a float

            try:

                warp_scale = float(warp_scale)

            except:

                raise ValueError("Could not convert 'warp_scale' to fl"+
                "oat in 'frozen_snapshots'. The current value is: "+str(
                warp_scale))
            
        # Otherwise, uses 1 as default

        else:

            warp_scale = 1.0

        # Scales it

        warp_object.ScaleFactor = warp_scale

        # Hides the previously plotted data

        Hide(active_data, renderView)

        # Saves the warped data as the active data

        active_data = warp_object

        # Displays the warped object

        display = Show(warp_object, renderView)

        # Adds this display object to the dictionary of displays

        display_dictionary["warp"] = {"object": display, "number of co"+
        "mponents": number_of_components, "representation":
        representation_type, "get scalar bar from": True}

    # Verifies if a clip object must be created

    if clip=="True":

        # Creates the clip object as a clip using a plane

        clip_object = Clip(Input=active_data)

        clip_object.ClipType = 'Plane'

        # Verifies if the origin point of the clip plane was given

        if clip_plane_origin:

            # Verifies if it is a list

            if (clip_plane_origin[0]!="[" or clip_plane_origin[-1]!="]"):

                raise TypeError("'clip_plane_origin' in 'frozen_snapsh"+
                "ots' must be a list of 3 components with the (x,y,z) "+
                "coordinates of the origin of the clip plane. Currentl"+
                "y, it is: "+str(clip_plane_origin)+"; whose type is: "+
                str(type(clip_plane_origin)))
            
            # Converts the argument to a list and sets the parameter

            clip_plane_origin = string_tools.string_toList(
            clip_plane_origin)
            
        # Otherwise, throws an error

        else:

            raise ValueError("'clip_plane_origin' in 'frozen_snapshots"+
            "' must be a list of 3 components with the (x,y,z) coordin"+
            "ates of the origin of the clip plane. Currently, it was n"+
            "ot provided")

        # Verifies if the normal vector of the clip plane was given

        if clip_plane_normal_vector:

            # Verifies if it is a list

            if (clip_plane_normal_vector[0]!="[" or (
            clip_plane_normal_vector[-1]!="]")):

                raise TypeError("'clip_plane_normal_vector' in 'frozen"+
                "_snapshots' must be a list of 3 components with the ("+
                "x,y,z) values of the normal plane of the clip plane. "+
                "Currently, it is: "+str(clip_plane_normal_vector)+"; "+
                "whose type is: "+str(type(clip_plane_normal_vector)))
            
            # Converts the argument to a list and sets the parameter

            clip_plane_normal_vector = string_tools.string_toList(
            clip_plane_normal_vector)
            
        # Otherwise, throws an error

        else:

            raise ValueError("'clip_plane_normal_vector' in 'frozen_sn"+
            "apshots' must be a list of 3 components with the (x,y,z) "+
            "values of the normal vector of the clip plane. Currently,"+
            " it was not provided")

        # Sets the plane parameters

        clip_object.ClipType.Origin = clip_plane_origin

        clip_object.ClipType.Normal = clip_plane_normal_vector

        # Hides the previously plotted data

        Hide(active_data, renderView)

        # Saves the clipped data as the active data

        active_data = clip_object

        # Displays the clipped object

        display = Show(clip_object, renderView)

        # Adds this display object to the dictionary of displays

        display_dictionary["clip"] = {"object": display, "number of co"+
        "mponents": number_of_components, "representation":
        representation_type, "get scalar bar from": True}

    ####################################################################
    #                         Additive filters                         #
    ####################################################################

    # Displays glyph to show vector field

    if glyph=="True":

        # Verifies if the data has at least 3 components

        if number_of_components<2:

            raise TypeError("The data provided to snapshot '"+str(
            field_name)+"' field in '"+str(input_fileName)+"' has "+str(
            number_of_components)+" components. Thus, it is not possib"+
            "le to create a glyph")

        # Creates the glyph object

        glyph_object = Glyph(Input=active_data)

        glyph_object.OrientationArray = ['POINTS', field_name]

        glyph_object.ScaleArray = ['POINTS', field_name]

        # Verifies if the scale of the glyph was given

        if glyph_scale:

            # Tries to convert the glyph scale to a float

            try:

                glyph_scale = float(glyph_scale)

            except:

                raise ValueError("Could not convert 'glyph_scale' to f"+
                "loat in 'frozen_snapshots'. The current value is: "+
                str(glyph_scale))
            
        # Otherwise, uses 1 as default

        else:

            glyph_scale = 1.0

        # Applies the scale factor

        glyph_object.ScaleFactor = glyph_scale

        # Defines the type of the glyph

        glyph_object.GlyphType = 'Arrow'

        # Displays the rendered data

        display = Show(glyph_object, renderView)

        # Adds this display object to the dictionary of displays

        display_dictionary["glyph"] = {"object": display, "number of c"+
        "omponents": 1, "representation": "Surface", "get scalar bar f"+
        "rom": False}

    #Render()

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

        renderView.OrientationAxesLabelColor = axes_color

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
        
    # Sets the representation

    for display_object_dict in display_dictionary.values():

        display_object_dict["object"].Representation = (
        display_object_dict["representation"])

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
        
    # Sets the color table for the legend

    LookupTable = None

    # Sets color, thus, it must iterates over the display objects

    for display_object_dict in display_dictionary.values():

        # Gets the display object

        display = display_object_dict["object"]

        # Takes care of scalar fields

        if display_object_dict["number of components"]==1:

            ColorBy(display, ('POINTS', field_name))

        # Otherwise, allows for the required component

        else:

            ColorBy(display, ('POINTS', field_name, component_to_plot))

        # Rescales the color

        display.RescaleTransferFunctionToDataRange(True, True)

        # Gets the color bar

        if display_object_dict["get scalar bar from"]:

            LookupTable = GetColorTransferFunction(field_name)

            display.SetScalarBarVisibility(renderView, True)

    # If still no look-up table was created, forces with a additive fil-
    # ter, such as glyph

    if LookupTable is None:

        # Verifies if the dictionary of display is not empty

        if len(list(display_dictionary.keys()))==0:

            raise ValueError("There is no substitutive filter (display"+
            "_reference_configuration, warp_by_vector, or clip) nor ad"+
            "ditive filters (such as glyph). The user must select some"+
            "thing")
        
        # Gets the first display

        display = list(display_dictionary.values())[0]["object"]

        # Takes care of scalar fields

        if display_object_dict["number of components"]==1:

            ColorBy(display, ('POINTS', field_name))

        # Otherwise, allows for the required component

        else:

            ColorBy(display, ('POINTS', field_name, component_to_plot))

        # Rescales the color

        display.RescaleTransferFunctionToDataRange(True, True)

        LookupTable = GetColorTransferFunction(field_name)

        display.SetScalarBarVisibility(renderView, True)
        
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

    # Verifies if limits for the color bar were given

    if (color_bar_min_value is not None) or ((color_bar_max_value
    ) is not None):
        
        # Gets the automatically retrieved values

        look_up_table_range = [0.0, 0.0]

        LookupTable.GetRange(look_up_table_range)

        automatic_min_value = look_up_table_range[0]

        automatic_max_value = look_up_table_range[1]

        # Verifies if the minimum value is given

        if color_bar_min_value is not None:

            # Verifies if it is a float

            try:

                color_bar_min_value = float(color_bar_min_value)

            except:

                raise ValueError("Could not convert 'color_bar_min_val"+
                "ue to float in 'frozen_snapshots'. The current value "+
                "is: "+str(color_bar_min_value))
            
        # Otherwise, uses the automatically found value
            
        else:

            color_bar_min_value = automatic_min_value

        # Verifies if the maximum value is given

        if color_bar_max_value is not None:

            # Verifies if it is a float

            try:

                color_bar_max_value = float(color_bar_max_value)

            except:

                raise ValueError("Could not convert 'color_bar_max_val"+
                "ue to float in 'frozen_snapshots'. The current value "+
                "is: "+str(color_bar_max_value))
            
        # Otherwise, uses the automatically found value
            
        else:

            color_bar_max_value = automatic_max_value
        
        # Rescales the range

        LookupTable.RescaleTransferFunction(color_bar_min_value, 
        color_bar_max_value)

    Render()

    # Verifies if an interactive window is to be used to camera aspects

    if set_camera_interactively=="True":

        print("\nAdjust the camera using the opened window. Then, clos"+
        "e the window to get the camera settings", flush=True)

        # Creates a view of what has already been rendered so far

        view = GetActiveView()
        
        # Shows the interactive window to the user

        Interact()

        # Gets the camera info

        camera_position = str(view.CameraPosition)

        camera_focal_point = str(view.CameraFocalPoint)

        camera_up_direction = str(view.CameraViewUp)

        camera_parallel_scale = str(view.CameraParallelScale)

        camera_rotation = str(view.CenterOfRotation)

        # Gets the legend info

        scalarBar = GetScalarBar(LookupTable, view)

        legend_bar_position = str(scalarBar.Position)

        legend_bar_length = str(scalarBar.ScalarBarLength) 

        # Gets the size info

        vtk_view = view.GetRenderWindow()

        size_in_pixels = list(vtk_view.GetSize())

        # Transforms the size into a dictionary for saving

        pixels_in_width = size_in_pixels[0]

        aspect_ratio = size_in_pixels[1]/size_in_pixels[0]

        size_in_pixels_dict = {"aspect ratio": aspect_ratio, "pixels i"+
        "n width": pixels_in_width}

        # Transforms the original size in pixels to a string

        size_in_pixels = str(size_in_pixels)

        # Prints the gathered information

        print("The interactively gathered settings are:\nCameraPositio"+
        "n: "+str(camera_position)+"\nCameraFocalPoint: "+str(
        camera_focal_point)+"\nCameraViewUp: "+str(camera_up_direction)+
        "\nCameraParallelScale: "+str(camera_parallel_scale)+"\nCenter"+
        "OfRotation: "+str(camera_rotation)+"\nScale bar position: "+
        legend_bar_position+"\nScale bar length: "+str(legend_bar_length
        )+"\nViewSize: "+size_in_pixels+"\n")

        # Saves the information as a list of arguments

        arguments_list = ("camera_position="+camera_position+",\ncamer"+
        "a_focal_point="+camera_focal_point+",\ncamera_up_direction="+
        camera_up_direction+",\ncamera_parallel_scale="+
        camera_parallel_scale+",\ncamera_rotation="+camera_rotation+","+
        "\nlegend_bar_position="+legend_bar_position+",\nlegend_bar_le"+
        "ngth="+legend_bar_length+",\nsize_in_pixels="+str(
        size_in_pixels_dict))

        # Saves this as a list as well

        arguments_dict = ("{'camera_position': "+str(camera_position)+
        ", 'camera_focal_point': "+str(camera_focal_point)+", 'camera_"+
        "up_direction': "+str(camera_up_direction)+", 'camera_parallel"+
        "_scale': "+str(camera_parallel_scale)+", 'camera_rotation': "+
        str(camera_rotation)+", 'legend_bar_position': "+str(
        legend_bar_position)+", 'legend_bar_length': "+str(
        legend_bar_length)+", 'size_in_pixels': "+str(size_in_pixels_dict
        )+"}")

        # Saves the list of arguments as a string in a txt file

        print("Saves a txt file with the arguments to 'frozen_snapshot"+
        "s' to replicate the view set by the user as follows")

        file_tools.save_string_into_txt(arguments_list, "paraview_came"+
        "ra_settings.txt", add_extension=False, parent_path=output_path,
        verbose=True)

        file_tools.save_string_into_txt(str(arguments_dict), "paraview"+
        "_camera_settings_dictionary.txt", add_extension=False, 
        parent_path=output_path, verbose=True)

    # Verifies if the flag to read the dictionary of camera settings is
    # active

    if read_camera_settings_dictionary=="True":

        # Verifies if there is a file with the settings

        if not path_tools.verify_file_existence(output_path+"//paravie"+
        "w_camera_settings_dictionary.txt", do_not_raise_error=True):
            
            raise FileNotFoundError("The user asked to read the camera"+
            " settings from the dictionary of camera settings internal"+
            "ly created by paraview_tools. But the file '"+str(
            output_path+"//paraview_camera_settings_dictionary.txt")+
            "' was not found. Possibly it has not yet been created. pa"+
            "raview_tools creates it automatically by setting\nset_cam"+
            "era_interactively=True")
        
        print("Reads the dictionary of camera settings at '"+str(
        output_path)+"//paraview_camera_settings_dictionary.txt'\n")
        
        # Gets the dictionary from the txt file

        settings_dict = file_tools.txt_toDict("paraview_camera_setting"+
        "s_dictionary.txt", parent_path=output_path, 
        txt_has_list_of_keys_and_values=False)

        # Gets the individual components, but make them strings to let
        # them compatible with the next steps

        camera_position = str(settings_dict['camera_position'])

        camera_focal_point = str(settings_dict['camera_focal_point'])

        camera_up_direction = str(settings_dict['camera_up_direction'])

        camera_parallel_scale = str(settings_dict['camera_parallel_sca'+
        'le'])

        camera_rotation = str(settings_dict['camera_rotation'])

        legend_bar_position = str(settings_dict['legend_bar_position'])

        legend_bar_length = str(settings_dict['legend_bar_length'])

        size_in_pixels = str(settings_dict['size_in_pixels'])

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

        # Gets the font from the class

        retrieved_font = font_class(legend_bar_font)

        # If the retrieved font is a list, it has a font file attached
        # to it

        if isinstance(retrieved_font, list):

            scalarBar.TitleFontFamily = retrieved_font[0]

            scalarBar.LabelFontFamily = retrieved_font[0]

            # Gets the file

            scalarBar.TitleFontFile = retrieved_font[1]

            scalarBar.LabelFontFile = retrieved_font[1]

        # Otherwise, it is the name of a font already owned by ParaView

        else:

            scalarBar.TitleFontFamily = retrieved_font

            scalarBar.LabelFontFamily = retrieved_font

    # Sets the color 

    if legend_bar_font_color:

        if (legend_bar_font_color[0]!="[" or legend_bar_font_color[-1]!=
        "]"):

            raise TypeError("'legend_bar_font_color' in 'frozen_snapsh"+
            "ots' must be a list of 3 components of RGB values---[0.0,"+
            " 0.0, 0.0] means black; [1.0, 1.0, 1.0] means white. Curr"+
            "ently, it is: "+str(legend_bar_font_color)+"; whose type "+
            "is: "+str(type(legend_bar_font_color)))
        
        # Converts the argument to a list and sets the parameter

        legend_bar_font_color = string_tools.string_toList(
        legend_bar_font_color)

        scalarBar.TitleColor = legend_bar_font_color

        scalarBar.LabelColor = legend_bar_font_color

    # Otherwise, if a font file is given

    elif legend_bar_font_file:

        scalarBar.TitleFontFamily = "File"

        scalarBar.LabelFontFamily = "File"

        scalarBar.TitleFontFile = legend_bar_font_file

        scalarBar.LabelFontFile = legend_bar_font_file

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