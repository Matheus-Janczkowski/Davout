# Routine to set the different curves in the event function triggered by
# keyboard shortcuts

from copy import deepcopy

from ...PythonicUtilities.file_handling_tools import list_toTxt

from ...PythonicUtilities.user_interaction_tools import input_repeater

from ..tool_box import collage_classes

def set_curve(key, arrows_and_lines_list, boxes_list, text_list,
arrows_and_lines_file, boxes_list_file, text_list_file, input_path, 
depth_order, collage, points_list, general_axes, tolerance, 
vanishing_points, line_styles_class, arrow_head_styles_class, 
colors_class, marker_box_class, alignment_class):
    
    # Initializes the dictionary with the curve information

    curve_dictionary = None

    # Initializes the dictionary with marker information

    marker_dictionary = None

    # Initializes the dictionary of text excerpt

    text_dictionary = None

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

        object_name = input_repeater("\nType the name of the object to"+
        " be used as reference later: ", reviewer_function=None, 
        default_value="unnominated", necessary_type=str)

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "spline po"+
        "ints": points_list_copy, "object name": object_name}

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

        fill_transparency = input_repeater("\nType a value between 0.0"+
        " and 1.0 for the transparency (opacity) of the color of the f"+
        "illing, then press enter: (the default value is 1.0, i.e. com"+
        "pletely opaque)", reviewer_function=None, default_value=1.0, 
        necessary_type=float)

        contour_transparency = input_repeater("\nType a value between "+
        "0.0 and 1.0 for the transparency (opacity) of the color of th"+
        "e contour, then press enter: (the default value is 1.0, i.e. "+
        "completely opaque)", reviewer_function=None, default_value=1.0, 
        necessary_type=float)

        object_name = input_repeater("\nType the name of the object to"+
        " be used as reference later: ", reviewer_function=None, 
        default_value="unnominated", necessary_type=str)

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "spline po"+
        "ints": points_list_copy, "closed path": True, "fill path with"+
        " color": fill_color, "contour transparency": 
        contour_transparency, "background transparency": 
        fill_transparency, "object name": object_name}

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

        object_name = input_repeater("\nType the name of the object to"+
        " be used as reference later: ", reviewer_function=None, 
        default_value="unnominated", necessary_type=str)

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": 
        arrow_style, "line style": line_style, "color": color, "spline"+
        " points": points_list_copy, "arrow head size": arrow_head_size, 
        "object name": object_name}

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

        object_name = input_repeater("\nType the name of the object to"+
        " be used as reference later: ", reviewer_function=None, 
        default_value="unnominated", necessary_type=str)

        # Creates the dictionary to create this polygonal curve

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "polygonal"+
        " points": points_list_copy, "object name": object_name}

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

        fill_transparency = input_repeater("\nType a value between 0.0"+
        " and 1.0 for the transparency (opacity) of the color of the f"+
        "illing, then press enter: (the default value is 1.0, i.e. com"+
        "pletely opaque)", reviewer_function=None, default_value=1.0, 
        necessary_type=float)

        contour_transparency = input_repeater("\nType a value between "+
        "0.0 and 1.0 for the transparency (opacity) of the color of th"+
        "e contour, then press enter: (the default value is 1.0, i.e. "+
        "completely opaque)", reviewer_function=None, default_value=1.0, 
        necessary_type=float)

        object_name = input_repeater("\nType the name of the object to"+
        " be used as reference later: ", reviewer_function=None, 
        default_value="unnominated", necessary_type=str)

        # Creates the dictionary to create this polygonal curve

        curve_dictionary = {"thickness": thickness, "arrow style": "no"+
        " arrow", "line style": line_style, "color": color, "polygonal"+
        " points": points_list_copy, "closed path": True, "fill path w"+
        "ith color": fill_color, "contour transparency": 
        contour_transparency, "background transparency": 
        fill_transparency, "object name": object_name}

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

        object_name = input_repeater("\nType the name of the object to"+
        " be used as reference later: ", reviewer_function=None, 
        default_value="unnominated", necessary_type=str)

        # Creates the dictionary to create this spline

        curve_dictionary = {"thickness": thickness, "arrow style": 
        arrow_style, "line style": line_style, "color": color, "polygo"+
        "nal points": points_list_copy, "arrow head size": 
        arrow_head_size, "object name": object_name}

    ####################################################################
    #                          Box and markers                         #
    ####################################################################

    elif key=="f":

        # Gets the necessary information

        marker_style = input_repeater("\nType the name of the marker o"+
        "r box and press enter: ", reviewer_function=
        marker_box_class.verify_marker_name, default_value="rectangle", 
        necessary_type=str)

        origin_point = input_repeater("\nType the alignment option rel"+
        "ative to the pressed point and press enter: ", reviewer_function=
        alignment_class.verify_alignment_name, default_value="centroid", 
        necessary_type=str)

        width = input_repeater("\nType width of the marker and press e"+
        "nter: ", reviewer_function=None, default_value=None, 
        necessary_type=float)

        height = input_repeater("\nType height of the marker and press"+
        " enter (the default value is the width itself): ", 
        reviewer_function=None, default_value=width*1.0, necessary_type=
        float)

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

        fill_transparency = input_repeater("\nType a value between 0.0"+
        " and 1.0 for the transparency (opacity) of the color of the f"+
        "illing, then press enter: (the default value is 1.0, i.e. com"+
        "pletely opaque)", reviewer_function=None, default_value=1.0, 
        necessary_type=float)

        contour_transparency = input_repeater("\nType a value between "+
        "0.0 and 1.0 for the transparency (opacity) of the color of th"+
        "e contour, then press enter: (the default value is 1.0, i.e. "+
        "completely opaque)", reviewer_function=None, default_value=1.0, 
        necessary_type=float)

        # If the marker is a polygon, asks for the number of sides

        number_of_sides = None 

        if marker_style=="polygon":

            number_of_sides = input_repeater("\nType the number of sid"+
            "es of the polygon and press enter: ", reviewer_function=
            None, default_value=None, necessary_type=int)

        object_name = input_repeater("\nType the name of the object to"+
        " be used as reference later: ", reviewer_function=None, 
        default_value="unnominated", necessary_type=str)

        # Creates the dictionary to create this marker. Adds the last 
        # point of the points list as position

        marker_dictionary = {"shape": marker_style, "contour thickness": 
        thickness, "contour style": line_style, "contour color": color, 
        "fill color": fill_color, "width": width, "height": height, "p"+
        "osition": points_list[-1], "origin point": origin_point, "num"+
        "ber of sides": number_of_sides, "contour transparency": 
        contour_transparency, "background transparency": 
        fill_transparency, "object name": object_name}

    ####################################################################
    #                           Text excerpts                          #
    ####################################################################

    elif key=="t":

        # Gets the necessary information

        text_excerpt = input_repeater("\nType the text you want to add"+
        " and press enter: ", reviewer_function=None, default_value=
        None, necessary_type=str)

        origin_point = input_repeater("\nType the alignment option rel"+
        "ative to the pressed point and press enter: ", reviewer_function=
        alignment_class.verify_alignment_name, default_value="bottom-left", 
        necessary_type=str)

        font_size = input_repeater("\nType the height of the letters i"+
        "n milimeters and press enter: ", reviewer_function=None, 
        default_value=5.0, necessary_type=float)

        rotation_in_degrees = input_repeater("\nType the rotation in d"+
        "egrees from the x-axis and press enter: ", reviewer_function=
        None, default_value=0.0, necessary_type=float)

        color = input_repeater("\nType the color of the font and press"+
        " enter: ", reviewer_function=colors_class.verify_color_name, 
        default_value="black", necessary_type=str)

        object_name = input_repeater("\nType the name of the object to"+
        " be used as reference later: ", reviewer_function=None, 
        default_value="unnominated", necessary_type=str)

        # Creates the dictionary to create this text excerpt. Adds the 
        # last point of the points list as position

        text_dictionary = {"text": text_excerpt, "font size": font_size, 
        "color": color, "position": points_list[-1], "origin point": 
        origin_point, "rotation in degrees": rotation_in_degrees, 
        "object name": object_name}

    print("\nThe graphic element has been successfully introduced. Pre"+
    "ss R to redraw if you want to see it immediately\n")

    # Defines a flag to tell if the markers were substituted in the 
    # points on screen

    flag_marker_substitution = False

    # Appends to the list of arrows and lines and saves the later to the 
    # appropriate txt file

    if curve_dictionary:

        # Updates the depth

        depth_order += 1

        curve_dictionary["depth order"] = depth_order

        arrows_and_lines_list.append(curve_dictionary)

        list_toTxt(arrows_and_lines_list, arrows_and_lines_file, 
        parent_path=input_path)

        # Substitutes the X markers by square markers, and cleans the 
        # list of points

        points_list, general_axes = substitute_markers(points_list,
        general_axes, depth_order, collage)

        flag_marker_substitution = True

    # Appends to the list of box and markers, and saves the later to the 
    # appropriate txt file

    if marker_dictionary:

        # Updates the depth

        depth_order += 1

        marker_dictionary["depth order"] = depth_order

        boxes_list.append(marker_dictionary)

        list_toTxt(boxes_list, boxes_list_file, parent_path=input_path)

        # Substitutes the X markers by square markers, and cleans the 
        # list of points

        if not flag_marker_substitution:

            points_list, general_axes = substitute_markers(points_list,
            general_axes, depth_order, collage)

            flag_marker_substitution = True

    # Appends to the list of text excerpts, and saves the later to the 
    # appropriate txt file

    if text_dictionary:

        # Updates the depth

        depth_order += 1

        text_dictionary["depth order"] = depth_order

        text_list.append(text_dictionary)

        list_toTxt(text_list, text_list_file, parent_path=input_path)

        # Substitutes the X markers by square markers, and cleans the 
        # list of points

        if not flag_marker_substitution:

            points_list, general_axes = substitute_markers(points_list,
            general_axes, depth_order, collage)

            flag_marker_substitution = True

    return (points_list, general_axes, arrows_and_lines_list, boxes_list,
    text_list, depth_order)

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