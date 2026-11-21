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
clip_marker_color, clip_marker_size, clip_type, clip_box_origin, 
clip_box_lengths_vector, clip_keep_outside_of_box, clip_sphere_center, 
clip_sphere_radius, extract_surface, edges_color):  

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
        "entation": representation_type, "get scalar bar from": True,
        "edges_color": edges_color}

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
        "entation": representation_type, "get scalar bar from": True, 
        "edges_color": edges_color}

    # Verifies if a clip object must be created

    if clip=="True":

        # Initializes the clip object 

        clip_object = Clip(Input=active_data)

        # Initializes the center or origin of the clip object

        clip_object_origin = None

        # Checks if the clip type was given

        if clip_type is not None:

            # Checks if it is one of the allowed options

            allowed_clip_formats = ["Plane", "Box", "Sphere"]

            if not (clip_type in allowed_clip_formats):

                allowed_clip_formats_string = ""

                for name in allowed_clip_formats:

                    allowed_clip_formats_string += "\n'"+str(name)+"'"

                raise ValueError("'clip_type' in 'frozen_snapshots' mu"+
                "st be one of the options below:"+
                allowed_clip_formats_string+"\n\nCurrently, it is: "+str(
                clip_type))

        # Otherwise, sets it as Plane

        else:

            clip_type = "Plane"

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

            clip_object_origin = string_tools.string_toList(
            clip_plane_origin)
            
        # Otherwise, throws an error

        elif clip_type=="Plane":

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

        elif clip_type=="Plane":

            raise ValueError("'clip_plane_normal_vector' in 'frozen_sn"+
            "apshots' must be a list of 3 components with the (x,y,z) "+
            "values of the normal vector of the clip plane. Currently,"+
            " it was not provided")

        # Verifies the parameters of the box

        if clip_box_origin:

            # Verifies if it is a list

            if (clip_box_origin[0]!="[" or clip_box_origin[-1]!="]"):

                raise TypeError("'clip_box_origin' in 'frozen_snapshot"+
                "s' must be a list of 3 components with the (x,y,z) co"+
                "ordinates of the origin of the clip box. Currently, i"+
                "t is: "+str(clip_box_origin)+"; whose type is: "+str(
                type(clip_box_origin)))
            
            # Converts the argument to a list and sets the parameter

            clip_object_origin = string_tools.string_toList(
            clip_box_origin)
            
        # Otherwise, throws an error

        elif clip_type=="Box":

            raise ValueError("'clip_box_origin' in 'frozen_snapshots' "+
            "must be a list of 3 components with the (x,y,z) coordinat"+
            "es of the origin of the clip box. Currently, it was not p"+
            "rovided")

        # Verifies if the vector of lengths of the box was given

        if clip_box_lengths_vector:

            # Verifies if it is a list

            if (clip_box_lengths_vector[0]!="[" or (
            clip_box_lengths_vector[-1]!="]")):

                raise TypeError("'clip_box_lengths_vector' in 'frozen_"+
                "snapshots' must be a list of 3 components with the (x"+
                ",y,z) values of the normal plane of the clip plane. C"+
                "urrently, it is: "+str(clip_box_lengths_vector)+"; wh"+
                "ose type is: "+str(type(clip_box_lengths_vector)))
            
            # Converts the argument to a list and sets the parameter

            clip_box_lengths_vector = string_tools.string_toList(
            clip_box_lengths_vector)
            
        # Otherwise, throws an error

        elif clip_type=="Box":

            raise ValueError("'clip_box_lengths_vector' in 'frozen_sna"+
            "pshots' must be a list of 3 components with the (x,y,z) v"+
            "alues of the side lengths of the clip box. Currently, it "+
            "was not provided")

        # Checks if the flag to keep the outside or the inside of the 
        # clip box was given

        if clip_keep_outside_of_box is not None:

            if clip_keep_outside_of_box=="True":

                # Sets it to zero, to keep the outside

                clip_keep_outside_of_box = 0

            elif clip_keep_outside_of_box=="False":

                # Sets it to one

                clip_keep_outside_of_box = 1

            else:

                raise ValueError("'clip_keep_outside_of_box' in 'froze"+
                "n_snapshots' must be a boolean, 'True' or 'False'. Cu"+
                "rrently, it is: "+str(clip_keep_outside_of_box))

        # In case it was not provided, set it automatically to keep the
        # outside of the box

        else:

            clip_keep_outside_of_box = 0

        # Verifies the parameters of the sphere

        if clip_sphere_center:

            # Verifies if it is a list

            if (clip_sphere_center[0]!="[" or clip_sphere_center[-1]!="]"):

                raise TypeError("'clip_sphere_center' in 'frozen_snaps"+
                "hots' must be a list of 3 components with the (x,y,z)"+
                " coordinates of the origin of the clip box. Currently"+
                ", it is: "+str(clip_sphere_center)+"; whose type is: "+
                str(type(clip_sphere_center)))
            
            # Converts the argument to a list and sets the parameter

            clip_object_origin = string_tools.string_toList(
            clip_sphere_center)
            
        # Otherwise, throws an error

        elif clip_type=="Sphere":

            raise ValueError("'clip_sphere_center' in 'frozen_snapshot"+
            "ts' must be a list of 3 components with the (x,y,z) coord"+
            "inates of the origin of the clip sphere. Currently, it wa"+
            "s not provided")

        # Verifies if the radius of the sphere was given

        if clip_sphere_radius:

            # Verifies if it is a float
            
            try:

                clip_sphere_radius = float(clip_sphere_radius)

            except:

                raise ValueError("Could not convert 'clip_sphere_radiu"+
                "s' to float in 'frozen_snapshots'. The current value "+
                "is: "+str(clip_sphere_radius))
            
        # Otherwise, throws an error

        elif clip_type=="Sphere":

            raise ValueError("'clip_sphere_radius' in 'frozen_snapshot"+
            "s' must be a float with the radius of the clip sphere. Cu"+
            "rrently, it was not provided")

        # Sets the type of the clip

        clip_object.ClipType = clip_type

        # If the clip type is a plane, sets the origin and normal vector

        if clip_type=="Plane":

            # Sets the plane parameters

            clip_object.ClipType.Origin = clip_object_origin

            clip_object.ClipType.Normal = clip_plane_normal_vector

        # If the clip is a box, sets the posiiton and the length vector

        elif clip_type=="Box":

            clip_object.ClipType.Position = clip_object_origin

            clip_object.ClipType.Length = clip_box_lengths_vector

            # Sets the side of the box that will be kept: 0 for the in-
            # side or 1 for the outside

            clip_object.Invert  = clip_keep_outside_of_box

        # If the clip is a sphere, sets the center and the radius

        elif clip_type=="Sphere":

            clip_object.ClipType.Center = clip_object_origin

            clip_object.ClipType.Radius = clip_sphere_radius

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
        "entation": representation_type, "get scalar bar from": True,
        "edges_color": edges_color}

        # Creates a point to highlight the plane origin point

        spheric_marker(clip_object_origin, active_data, renderView, 
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
        "get scalar bar from": False, "edges_color": None}

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