# Routine to set the different curves in the event function triggered by
# keyboard shortcuts

from ...PythonicUtilities.file_handling_tools import list_toTxt

from ...PythonicUtilities.string_tools import string_toList

from ..tool_box import collage_classes

def set_curve(key, arrows_and_lines_list, arrows_and_lines_file, 
input_path, depth_order, collage, points_list, general_axes):
    
    # Initializes the dictionary with the curve information

    curve_dictionary = None

    ####################################################################
    #                           Spline curve                           #
    ####################################################################

    if key=="q":

        # Allows for selection of parameters

        thickness = input("\nType the thickness and press enter: ")

        line_style = input("\nType the line style and press enter: ")

        color = input("\nType the color of the contour line and press "+
        "enter: ")

        # Converts these strings 

        thickness = ""

        line_style = ""

        color = ""

        thickness = convert_string(thickness, 0.2)

        line_style = convert_string(line_style, "solid")

        color = convert_string(color, "black")

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "spline po"+
        "ints": points_list}

    ####################################################################
    #                        Closed spline curve                       #
    ####################################################################

    elif key=="w":

        # Allows for selection of parameters

        thickness =  input("\nType the thickness and press enter: ")

        line_style =  input("\nType the line style and press enter: ")

        color = input("\nType the color of the contour line and press "+
        "enter: ")

        fill_color = input("\nType the color of filling and press ente"+
        "r: ")

        # Converts these strings 

        thickness = convert_string(thickness, 0.2)

        line_style = convert_string(line_style, "solid")

        color = convert_string(color, "black")

        fill_color = convert_string(fill_color, "transparent")

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "spline po"+
        "ints": points_list, "closed path": True, "fill path with colo"+
        "r": fill_color}

    ####################################################################
    #                      Arrow with spline stem                      #
    ####################################################################

    elif key=="e":

        # Allows for selection of parameters

        thickness =  input("\nType the thickness and press enter: ")

        line_style =  input("\nType the line style and press enter: ")

        color = input("\nType the color of the contour line and press "+
        "enter: ")

        arrow_style = input("\nType the style of the arrow head: ")

        # Converts these strings 

        thickness = convert_string(thickness, 0.2)

        line_style = convert_string(line_style, "solid")

        color = convert_string(color, "black")

        arrow_style = convert_string(arrow_style, "inkscape round back "+
        "arrow")

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": 
        arrow_style, "line style": line_style, "color": color, "spline"+
        " points": points_list}

    ####################################################################
    #                          Polygonal curve                         #
    ####################################################################

    if key=="a":

        # Allows for selection of parameters

        thickness = input("\nType the thickness and press enter: ")

        line_style = input("\nType the line style and press enter: ")

        color = input("\nType the color of the contour line and press "+
        "enter: ")

        # Converts these strings 

        thickness = ""

        line_style = ""

        color = ""

        thickness = convert_string(thickness, 0.2)

        line_style = convert_string(line_style, "solid")

        color = convert_string(color, "black")

        # Creates the dictionary to create this polygonal curve

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "polygonal"+
        " points": points_list}

    ####################################################################
    #                      Closed polygonal curve                      #
    ####################################################################

    elif key=="s":

        # Allows for selection of parameters

        thickness =  input("\nType the thickness and press enter: ")

        line_style =  input("\nType the line style and press enter: ")

        color = input("\nType the color of the contour line and press "+
        "enter: ")

        fill_color = input("\nType the color of filling and press ente"+
        "r: ")

        # Converts these strings 

        thickness = convert_string(thickness, 0.2)

        line_style = convert_string(line_style, "solid")

        color = convert_string(color, "black")

        fill_color = convert_string(fill_color, "transparent")

        # Creates the dictionary to create this polygonal curve

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "polygonal"+
        " points": points_list, "closed path": True, "fill path with c"+
        "olor": fill_color}

    ####################################################################
    #                     Arrow with polygonal stem                    #
    ####################################################################

    elif key=="d":

        # Allows for selection of parameters

        thickness =  input("\nType the thickness and press enter: ")

        line_style =  input("\nType the line style and press enter: ")

        color = input("\nType the color of the contour line and press "+
        "enter: ")

        arrow_style = input("\nType the style of the arrow head: ")

        # Converts these strings 

        thickness = convert_string(thickness, 0.2)

        line_style = convert_string(line_style, "solid")

        color = convert_string(color, "black")

        arrow_style = convert_string(arrow_style, "inkscape round back "+
        "arrow")

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": 
        arrow_style, "line style": line_style, "color": color, "polygo"+
        "nal points": points_list}

    # Appends to the list of arrows and lines and saves the later to the 
    # appropriate txt file

    if curve_dictionary:

        arrows_and_lines_list.append(curve_dictionary)

        list_toTxt(arrows_and_lines_list, arrows_and_lines_file, 
        parent_path=input_path)

    # Substitutes the X markers by square markers, and cleans the list 
    # of points

    points_list, general_axes = substitute_markers(points_list,
    general_axes, depth_order, collage)

    return points_list, general_axes, arrows_and_lines_list

########################################################################
#                               Utilities                              #
########################################################################

# Defines a function to substitute the X markers by square markers when
# points defined by the user are saved

def substitute_markers(points_list, general_axes, depth_order, collage):

    # Removes the X's and switches them by squares

    for point in points_list:

        if general_axes.lines:

            general_axes.lines[-1].remove()

    # Plots the squares

    for point in points_list:

        general_axes.plot(point[0], point[1], marker="s", color="black", 
        zorder=depth_order, markersize=
        collage_classes.milimeters_to_points(2.0), mew=2)

    collage.canvas.draw_idle()

    # Cleans the list of points

    points_list = []

    return points_list, general_axes

# Defines a function to convert strings to useful formats

def convert_string(string, default_value):

    # If the string has no length, returns the default value

    if len(string)==0:

        return default_value

    # Tries to convert to an integer

    try:

        string = int(string)

    except:

        # Tries to convert to a float

        try:

            string = float(string)

        except:

            # Tries to convert to a list

            if (string[0]=="[" and string[-1]=="]"):

                string = string_toList(string)

    # If the string is still a string and the default value is not None,
    # returns the default value

    if isinstance(string, str) and (default_value is not None):

        return default_value
    
    # Otherwise, returns the original or converted value

    return string