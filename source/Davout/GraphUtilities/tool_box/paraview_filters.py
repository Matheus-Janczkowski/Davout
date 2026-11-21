# Routine to store filters from paraview

from paraview.simple import *

########################################################################
#                           Imports preamble                           #
########################################################################

from pathlib import Path

from importlib import util

import sys

# Gets the parent paths of the current 

broken_path = Path(__file__).parents

# Imports string tools

specifications = util.spec_from_file_location("string_tools", 
broken_path[2]/"PythonicUtilities"/"string_tools.py")

string_tools = util.module_from_spec(specifications)

sys.modules["string_tools"] = string_tools

specifications.loader.exec_module(string_tools)
  
# Defines a function to apply different filters from ParaView

def apply_paraview_filters(xdmf_class, renderView, display_dictionary,
list_of_active_data_objects, field_name, component_to_plot, 
number_of_components, representation_type, colors_class, 
display_reference_configuration, warp_by_vector, warp_scale, glyph, 
glyph_scale, clip, clip_plane_origin, clip_plane_normal_vector,
clip_marker_color, clip_marker_size, extract_surface):  

    # Gets the file name corresponding to this object

    input_fileName = xdmf_class.individual_input_file_name

    ####################################################################
    #                       Substitutive filters                       #
    ####################################################################

    # Creates a variable to store the data, especially if it was warped

    active_data = xdmf_class.data

    # If only the surface data is to be plotted

    if extract_surface=="True":

        # Extracts the data on the boundary of the mesh

        surface_object = ExtractSurface(Input=active_data)

        surface_object.UpdatePipeline()

        # Updates the active data to the surface data only. This way, 
        # all other filters can operate only on the surface data

        active_data = surface_object

    # Otherwise displays the raw data only

    if display_reference_configuration=="True":

        # Displays the rendered data

        display = Show(active_data, renderView)

        # Gets the number of occurences of the key for this object in
        # the dictionary of objects to be displayed

        n_occurrences = string_tools.get_strings_in_list_given_prefix(
        display_dictionary, "reference", 
        just_count_the_number_of_occurrences=True)

        # Adds this display object to the dictionary of displays

        display_dictionary["reference_"+str(n_occurrences)] = {"object": 
        display, "number of components": number_of_components, "repres"+
        "entation": representation_type, "get scalar bar from": True}

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

        warp_object = WarpByVector(Input=active_data)

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

        # Gets the number of occurences of the key for this object in
        # the dictionary of objects to be displayed

        n_occurrences = string_tools.get_strings_in_list_given_prefix(
        display_dictionary, "warp", just_count_the_number_of_occurrences=
        True)

        # Adds this display object to the dictionary of displays

        display_dictionary["warp_"+str(n_occurrences)] = {"object": 
        display, "number of components": number_of_components, "repres"+
        "entation": representation_type, "get scalar bar from": True}

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

        # Gets the number of occurences of the key for this object in
        # the dictionary of objects to be displayed

        n_occurrences = string_tools.get_strings_in_list_given_prefix(
        display_dictionary, "clip", just_count_the_number_of_occurrences=
        True)

        # Adds this display object to the dictionary of displays

        display_dictionary["clip_"+str(n_occurrences)] = {"object": 
        display, "number of components": number_of_components, "repres"+
        "entation": representation_type, "get scalar bar from": True}

        # Creates a point to highlight the plane origin point

        spheric_marker(clip_plane_origin, active_data, renderView, 
        colors_class, clip_marker_color, clip_marker_size)

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

        # Gets the number of occurences of the key for this object in
        # the dictionary of objects to be displayed

        n_occurrences = string_tools.get_strings_in_list_given_prefix(
        display_dictionary, "glyph", 
        just_count_the_number_of_occurrences=True)

        # Adds this display object to the dictionary of displays

        display_dictionary["glyph_"+str(n_occurrences)] = {"object": 
        display, "number of components": 1, "representation": "Surface", 
        "get scalar bar from": False}

    # Appends the current active data to the list of active data objects

    list_of_active_data_objects.append(active_data)

    # Returns the display dictionary and the list of active data objects

    return display_dictionary, list_of_active_data_objects

########################################################################
#                            Markers' tools                            #
########################################################################

# Defines a function to plot a sphere as a marker

def spheric_marker(marker_position, active_data, renderView, 
colors_class, clip_marker_color, clip_marker_size):

    # Verifies if marker position is a dictionary

    if isinstance(marker_position, dict):

        # Verifies if it has the color key

        if "color" in marker_position:

            # Overrides the color option

            clip_marker_color = marker_position["color"]

        # Verifies if it has the size key

        if "size" in marker_position:

            # Overrides the size option

            clip_marker_size = marker_position["size"]

        # Verifies if it has the position option

        if not ("position" in marker_position):

            raise KeyError("The key 'position' is not available in the"+
            " dictionary in the list of markers. This key must contain"+
            " a list of the coordinates of the marker. See the current"+
            " dictionary: "+str(marker_position))
        
        # Overrides the dictionary with the position

        marker_position = marker_position["position"]

    # Creates a point to highlight the plane origin point

    point = PointSource()

    point.Center = marker_position

    point.NumberOfPoints = 1

    # Applies a glyph filter to highlight it as a large sphere

    glyph = Glyph(Input=point)

    glyph.GlyphType = 'Sphere'

    # Verifies if the marker size has been given by the user

    if clip_marker_size:

        # Tries to convert the to a float

        try:

            clip_marker_size = float(clip_marker_size)

        except:

            raise ValueError("Could not convert 'clip_marker_size'"+
            " to float in 'frozen_snapshots'. The current value is"+
            ": "+str(clip_marker_size))
        
    else:

        # Verifies if active data is not a list

        active_data_list = None

        if not isinstance(active_data, list):

            active_data_list = [active_data]

        else:

            active_data_list = active_data

        # Gets the size from the bounding box

        length = 0.0

        # Iterates over the active data objects

        for active_data_object in active_data_list:

            bounds = active_data_object.GetDataInformation().GetBounds()
            
            length = max(bounds[1]-bounds[0], bounds[3]-bounds[2], 
            bounds[5]-bounds[4], length)

        clip_marker_size = 0.1*length

    # Adjust the sphere size

    glyph.GlyphType.Radius = clip_marker_size

    # Displays it

    display = Show(glyph, renderView)

    # Verifies if the color of the marker has been given

    if clip_marker_color:

        # Gets the RGB value from the colors class

        clip_marker_color = colors_class(clip_marker_color)

    else:

        # Makes it red per default

        clip_marker_color = [1.0, 0.0, 0.0]

    print("The color of the marker is: "+str(clip_marker_color))

    # Sets the color to the sphere

    display.DiffuseColor = clip_marker_color

    display.AmbientColor = clip_marker_color

    # Diminishes the shade influence

    display.Ambient = 0.5

    display.Diffuse = 0.5

    display.Specular = 0.1
    
    display.SpecularPower = 10