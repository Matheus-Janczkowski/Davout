# Routine to set the different curves in the event function triggered by
# keyboard shortcuts

from copy import deepcopy

from ...PythonicUtilities.file_handling_tools import list_toTxt

from ...PythonicUtilities.string_tools import string_toList

from ..tool_box import collage_classes

def set_curve(key, arrows_and_lines_list, arrows_and_lines_file, 
input_path, depth_order, collage, points_list, general_axes, tolerance,
vanishing_points, line_styles_class, arrow_head_styles_class, 
colors_class):
    
    # Initializes the dictionary with the curve information

    curve_dictionary = None

    # Creates a copy of the points

    points_list_copy = deepcopy(points_list)

    # Initializes the list of points to compare against

    comparison_points = []

    # Iterates through the points in the curves already created

    for curve in arrows_and_lines_list:

        # If it is a spline

        if "spline points" in curve:

            # Gets the list of point and appends to the list of points
            # to compare

            comparison_points.extend(curve["spline points"])

        # If it is a polygonal

        elif "polygonal points" in curve:

            # Gets the list of point and appends to the list of points
            # to compare

            comparison_points.extend(curve["polygonal points"])

        # If it has a start point

        if "start point" in curve:

            comparison_points.append(curve["start point"])

        # If it has an end point

        if "end point" in curve:

            comparison_points.append(curve["end point"])

    # Iterates through the perspective points to gather possible matching
    # points

    if vanishing_points is not None:

        for perspective_point in vanishing_points:

            # Appends the point

            comparison_points.append(perspective_point["coordinates"])

    # Iterates over the points given for this curve

    for index, point in enumerate(points_list_copy):

        # Verifies if this point is close to any point of the comparison
        # points

        for compared_point in comparison_points:

            if (((compared_point[0]-point[0])**2)+((compared_point[1]-
            point[1])**2))<(tolerance**2):
                
                # Substitutes the point for the compared one

                points_list_copy[index] = compared_point

                break

    ####################################################################
    #                           Spline curve                           #
    ####################################################################

    if key=="q":

        # Gets the necessary information

        thickness = input_repeater("\nType the thickness and press ent"+
        "er: ", reviewer_function=None, default_value=0.2, 
        necessary_type=float)

        line_style = input_repeater("\nType the line style and press e"+
        "nter: ", reviewer_function=line_styles_class.verify_line_name, 
        default_value="solid", necessary_type=str)

        color = input_repeater("\nType the color of the contour line a"+
        "nd press enter: ", reviewer_function=
        colors_class.verify_color_name, default_value="black", 
        necessary_type=str)

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "spline po"+
        "ints": points_list_copy}

    ####################################################################
    #                        Closed spline curve                       #
    ####################################################################

    elif key=="w":

        # Gets the necessary information

        thickness = input_repeater("\nType the thickness and press ent"+
        "er: ", reviewer_function=None, default_value=0.2, 
        necessary_type=float)

        line_style = input_repeater("\nType the line style and press e"+
        "nter: ", reviewer_function=line_styles_class.verify_line_name, 
        default_value="solid", necessary_type=str)

        color = input_repeater("\nType the color of the contour line a"+
        "nd press enter: ", reviewer_function=
        colors_class.verify_color_name, default_value="black", 
        necessary_type=str)

        fill_color = input_repeater("\nType the color of filling and p"+
        "ress enter: ", reviewer_function=colors_class.verify_color_name, 
        default_value="transparent", necessary_type=str)

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "spline po"+
        "ints": points_list_copy, "closed path": True, "fill path with"+
        " color": fill_color}

    ####################################################################
    #                      Arrow with spline stem                      #
    ####################################################################

    elif key=="e":

        # Gets the necessary information

        thickness = input_repeater("\nType the thickness and press ent"+
        "er: ", reviewer_function=None, default_value=0.2, 
        necessary_type=float)

        line_style = input_repeater("\nType the line style and press e"+
        "nter: ", reviewer_function=line_styles_class.verify_line_name, 
        default_value="solid", necessary_type=str)

        color = input_repeater("\nType the color of the contour line a"+
        "nd press enter: ", reviewer_function=
        colors_class.verify_color_name, default_value="black", 
        necessary_type=str)

        arrow_style = input_repeater("\nType the style of the arrow he"+
        "ad: ", reviewer_function=
        arrow_head_styles_class.verify_arrow_name, default_value="inks"+
        "cape round back arrow", necessary_type=str)

        arrow_head_size = input_repeater("\nType the size of the arrow"+
        " head: ", reviewer_function=None, default_value=0.5, 
        necessary_type=float)

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": 
        arrow_style, "line style": line_style, "color": color, "spline"+
        " points": points_list_copy, "arrow head size": arrow_head_size}

    ####################################################################
    #                          Polygonal curve                         #
    ####################################################################

    if key=="a":

        # Gets the necessary information

        thickness = input_repeater("\nType the thickness and press ent"+
        "er: ", reviewer_function=None, default_value=0.2, 
        necessary_type=float)

        line_style = input_repeater("\nType the line style and press e"+
        "nter: ", reviewer_function=line_styles_class.verify_line_name, 
        default_value="solid", necessary_type=str)

        color = input_repeater("\nType the color of the contour line a"+
        "nd press enter: ", reviewer_function=
        colors_class.verify_color_name, default_value="black", 
        necessary_type=str)

        # Creates the dictionary to create this polygonal curve

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "polygonal"+
        " points": points_list_copy}

    ####################################################################
    #                      Closed polygonal curve                      #
    ####################################################################

    elif key=="s":

        # Gets the necessary information

        thickness = input_repeater("\nType the thickness and press ent"+
        "er: ", reviewer_function=None, default_value=0.2, 
        necessary_type=float)

        line_style = input_repeater("\nType the line style and press e"+
        "nter: ", reviewer_function=line_styles_class.verify_line_name, 
        default_value="solid", necessary_type=str)

        color = input_repeater("\nType the color of the contour line a"+
        "nd press enter: ", reviewer_function=
        colors_class.verify_color_name, default_value="black", 
        necessary_type=str)

        fill_color = input_repeater("\nType the color of filling and p"+
        "ress enter: ", reviewer_function=colors_class.verify_color_name, 
        default_value="transparent", necessary_type=str)

        # Creates the dictionary to create this polygonal curve

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "polygonal"+
        " points": points_list_copy, "closed path": True, "fill path w"+
        "ith color": fill_color}

    ####################################################################
    #                     Arrow with polygonal stem                    #
    ####################################################################

    elif key=="d":

        # Gets the necessary information

        thickness = input_repeater("\nType the thickness and press ent"+
        "er: ", reviewer_function=None, default_value=0.2, 
        necessary_type=float)

        line_style = input_repeater("\nType the line style and press e"+
        "nter: ", reviewer_function=line_styles_class.verify_line_name, 
        default_value="solid", necessary_type=str)

        color = input_repeater("\nType the color of the contour line a"+
        "nd press enter: ", reviewer_function=
        colors_class.verify_color_name, default_value="black", 
        necessary_type=str)

        arrow_style = input_repeater("\nType the style of the arrow he"+
        "ad: ", reviewer_function=
        arrow_head_styles_class.verify_arrow_name, default_value="inks"+
        "cape round back arrow", necessary_type=str)

        arrow_head_size = input_repeater("\nType the size of the arrow"+
        " head: ", reviewer_function=None, default_value=0.5, 
        necessary_type=float)

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": 
        arrow_style, "line style": line_style, "color": color, "polygo"+
        "nal points": points_list_copy, "arrow head size": 
        arrow_head_size}

    # Appends to the list of arrows and lines and saves the later to the 
    # appropriate txt file

    if curve_dictionary:

        arrows_and_lines_list.append(curve_dictionary)

        list_toTxt(arrows_and_lines_list, arrows_and_lines_file, 
        parent_path=input_path)

        # Substitutes the X markers by square markers, and cleans the 
        # list of points

        points_list, general_axes = substitute_markers(points_list,
        general_axes, depth_order, collage)

    return points_list, general_axes, arrows_and_lines_list

########################################################################
#                               Utilities                              #
########################################################################

# Defines a function to repeat a call until the right answer is given

def input_repeater(input_text, reviewer_function=None, default_value=
None, necessary_type=None):

    # Initializes a flag to keep repeating the question

    flag_repeat = True 

    # Repeats the question until a right answer is given

    while flag_repeat:

        # Asks the user for the input

        input_string =  input(input_text)
        
        # Converts the input string to another format if possible

        input_string = convert_string(input_string, default_value,
        necessary_type=necessary_type, throw_error=False)

        # If input string has not been falsified during conversion

        if input_string is not None:

            # Verifies if a right answer has been given

            if reviewer_function is not None:

                processed_answer = reviewer_function(input_string)

                # If it is not false, returns it

                if processed_answer:

                    flag_repeat = False 

                    return processed_answer
                
            # If the reviewer function is None, returns the answer any-
            # ways

            else:

                flag_repeat = False 

                return input_string

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

def convert_string(string, default_value, necessary_type=None, 
throw_error=True):

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

    # Verifies the type of the string if needed

    if (necessary_type is not None) and (not isinstance(string, 
    necessary_type)):
        
        if throw_error:
        
            raise TypeError("'"+str(string)+"' has type '"+str(type(
            string))+"', but the necessary type is "+str(necessary_type))
        
        # Otherwise, just prints the error

        else:

            print("'"+str(string)+"' has type '"+str(type(string))+"',"+
            " but the necessary type is "+str(necessary_type))

            return None

    # If the string is still a string and the default value is not None,
    # returns the default value

    if isinstance(string, str) and (default_value is not None) and (
    not isinstance(default_value, str)):

        return default_value
    
    # Otherwise, returns the original or converted value

    return string