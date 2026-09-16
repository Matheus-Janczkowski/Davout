# Routine to store methods to plot text excerpts using matplotlib

from copy import deepcopy

from math import cos, sin, radians

from .collage_classes import milimeters_to_points

from matplotlib.textpath import TextPath

from matplotlib.patches import PathPatch

from matplotlib.transforms import Affine2D

from matplotlib.font_manager import FontProperties

# Defines a function to plot excerpts of text from a list

def plot_text_excerpts(general_axes, input_text_list, alignments_class, 
color_class, verbose, depth_order):
        
    # Sets a list of necessary keys

    necessary_keys = ["text", "position", "font size"]

    # Verifies if it is not a list

    if not isinstance(input_text_list, list):

        raise TypeError("'input_text_list' is not a list. It must be a"+
        " list where each item is a dictionary with the keys:\nObligat"+
        "ory:\n'text': string with the text excerpt\n'position': list "+
        "with position coordinates, [x,y]\n'font size': integer\n\nOpt"+
        "ional:\n'origin point': available options are 'centroid', 'bo"+
        "ttom-left', 'bottom-right', 'top-left', 'top-right'\n'rotatio"+
        "n in degrees': float with rotation angle in degrees (from x a"+
        "xis counter-clockwise)")
    
    # Iterates through the elements

    for index, input_dictionary in enumerate(input_text_list):

        # Verifies if this element is a dictionary

        if not isinstance(input_dictionary, dict):

            raise TypeError("The "+str(index+1)+"-th element of the 'i"+
            "nput_text_list' is not a dictionary. It must be a diction"+
            "ary with the keys:\nObligatory:\n'text': string with the "+
            "text excerpt\n'position': list with position coordinates,"+
            " [x,y]\n'font size': integer\n\nOptional:\n'origin point'"+
            ": available options are 'centroid', 'bottom-left', 'botto"+
            "m-right', 'top-left', 'top-right'\n'rotation in degrees':"+
            " float with rotation angle in degrees (from x axis counte"+
            "r-clockwise)")
        
        # Iterates through the necessary keys

        for key in necessary_keys:

            # Verifies the key existence

            if not (key in input_dictionary):

                names = ""

                for keys in necessary_keys:

                    names += "\n'"+str(keys)+"'"

                raise ValueError("The "+str(index+1)+"-th element 'inp"+
                "ut_text_list' does not have all the necessary keys, i"+
                "n particular '"+str(key)+"'. Check the necessary keys"+
                ":"+names+"\n\nThe current input dictionary is:\n"+str(
                input_dictionary))
            
        # Gets the text excerpt

        input_text = input_dictionary["text"]

        # Gets the position

        position = deepcopy(input_dictionary["position"])

        # Verifies if it is a list

        if not isinstance(position, list):

            raise TypeError("The "+str(index+1)+"-th element 'input_te"+
            "xt_list' has at key 'position' a value that is not a list"+
            ". It must be a list with [x, y] coordinates. Currently, i"+
            "t is:\n"+str(position))
        
        # Verifies if the origin point is prescribed

        origin_point = 'centroid'

        if "origin point" in input_dictionary:

            origin_point = input_dictionary["origin point"]

        # Updates position using the alignment

        position, ha, va = alignments_class(origin_point, position, 0.0, 
        0.0, text_alignment=True)
        
        # Gets the font size

        font_size = input_dictionary["font size"]

        # Verifies if it is an integer

        if (not isinstance(font_size, int)) and (not isinstance(
        font_size, float)):

            raise TypeError("The "+str(index+1)+"-th element 'input_te"+
            "xt_list' has at key 'font size' a value that is not an in"+
            "teger nor a float. Currently, it is:\n"+str(font_size))
        
        # Converts it from mm to points

        font_size = milimeters_to_points(font_size)
        
        # Verifies if there is any rotation

        angle = 0.0

        if "rotation in degrees" in input_dictionary:

            # Gets the angle

            angle = input_dictionary["rotation in degrees"]

            # Verifies if it is a float

            if not isinstance(angle, float):

                raise TypeError("The "+str(index+1)+"-th element 'boxe"+
                "s_list' has at key 'rotation in degrees' a value that"+
                " is not a float. Currently, 'rotation in degrees' is:"+
                " "+str(angle))
            
            if verbose:

                print("\nRotates the text except by "+str(angle)+" deg"+
                "rees")

        # Verifies if a depth number has been given

        local_depth_order = deepcopy(depth_order)

        if "depth order" in input_dictionary:

            local_depth_order = input_dictionary["depth order"]

        else:

            # Updates the depth number

            depth_order += 1

        # Gets the font color

        color = "black"

        if "color" in input_dictionary:

            color = input_dictionary["color"]

        color = color_class(color)

        # Verifies if aspect ration (width/height) was given

        aspect_ratio = 1.0

        if "aspect ratio" in input_dictionary:

            aspect_ratio = input_dictionary["aspect ratio"]

        # Checks if the rendering method was provided

        rendering_method = "matplotlib text"

        if "rendering method" in input_dictionary:

            rendering_method = input_dictionary["rendering method"]

        # Adds the text input

        if verbose:

            print("Adds text at point "+str(position)+" with 'origin p"+
            "oint' as '"+str(origin_point)+"'\n")

        # If the rendering method is the conventional text of matplotlib

        if rendering_method=="matplotlib text":

            text_artist = general_axes.text(position[0], position[1], 
            input_text, fontsize=font_size, ha=ha, va=va, rotation=angle, 
            rotation_mode="anchor", zorder=local_depth_order, transform=
            general_axes.transData, color=color)

            # Gets the position in display coordinates

            display_position = general_axes.transData.transform(position)

            # Creates a transform centered at the text anchor

            text_transform = (Affine2D().translate(-display_position[0], 
            -display_position[1]).scale(aspect_ratio, 1.0).translate(
            display_position[0], display_position[1]))

            # Sets the transform to the text excerpt

            text_artist.set_transform(general_axes.transData+
            text_transform)

        # Otherwise, if the rendering method is TextPath

        elif rendering_method=="TextPath":

            # Creates the text excerpt as a text path. Positions it at 
            # the origin first; the patch will be scaled and translated 
            # later

            text_path = TextPath((0, 0), input_text, size=font_size, 
            usetex=True)

            # gets the bounding box of the text path in points

            text_bounding_box = text_path.get_extents()

            # Checks the alignment options to set the anchor

            if ha=="left":

                anchor_x = text_bounding_box.xmin

            elif ha=="center":

                anchor_x = 0.5*(text_bounding_box.xmin+
                text_bounding_box.xmax)

            elif ha=="right":

                anchor_x = text_bounding_box.xmax

            else:

                anchor_x = text_bounding_box.xmin

            if va=="bottom":

                anchor_y = text_bounding_box.ymin

            elif va=="center":

                anchor_y = 0.5*(text_bounding_box.ymin+
                text_bounding_box.ymax)

            elif va=="top":

                anchor_y = text_bounding_box.ymax

            else:

                anchor_y = text_bounding_box.ymin

            # Gets the transformation of the text path 

            points_to_millimeters = 25.4/72

            path_transform = (Affine2D().translate(-anchor_x, -anchor_y)
            .scale(aspect_ratio, 1.0).rotate_deg(angle).scale(
            points_to_millimeters))

            # Then translates the path to the final position

            text_transform = (path_transform+Affine2D().translate(
            position[0], position[1])+general_axes.transData)

            # Creates the patch and add to the canvas

            patch_instance = PathPatch(text_path, transform=
            text_transform, facecolor=color, edgecolor="none", zorder=
            local_depth_order,clip_on=False)

            general_axes.add_patch(patch_instance)

        # Otherwise, throws an error

        else:

            raise NameError("'rendering method' in the dictionary of t"+
            "ext excerpts was selected as '"+str(rendering_method)+"'."+
            " This value is invalid; you have to choose one of the fol"+
            "lowing options:\n'matplotlib text'\n'TextPath'")

    return general_axes, depth_order