# Routine to make collages

import numpy as np

from copy import deepcopy

import matplotlib

import matplotlib.pyplot as plt

from collections import OrderedDict

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, RegularPolygon

from matplotlib.transforms import Affine2D, Bbox

from matplotlib.path import Path

from PIL import Image

from ..PythonicUtilities import path_tools

from .tool_box import collage_classes

########################################################################
#                            LaTeX preamble                            #
########################################################################

from pathlib import Path as PathPathlib

import os

# Gets the parent paths of the current file

broken_path = PathPathlib(__file__).parents

# Gets the path to LaTeXUtilities.sty

latex_utilities_path = broken_path[1]/"LaTeXUtilities"

# Adds this path to the system

os.environ["TEXINPUTS"] = (str(latex_utilities_path)+os.sep+os.pathsep+
os.environ.get("TEXINPUTS", ""))

# Sets LaTeX font and the latex utilities

matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
"font.serif": ["Computer Modern Roman"], "text.latex.preamble": 
r"\usepackage{LaTeXUtilities}"})

# Defines a function to create a collage using boxes

def create_box_collage(output_file, input_path=None, output_path=None,
no_padding=False, input_image_list=None, input_text_list=None, 
boxes_list=None, arrows_and_lines_list=None, dpi=300, verbose=False, aspect_ratio=
'auto', adjustable=None, layout_width_milimeters=210.0, 
layout_height_milimeters=297.0, add_overlaying_grid=False):
    
    # Initializes the class of colors, the class of alignments, and the 
    # class of line styles

    colors_class = collage_classes.ColorMiscellany()

    alignments_class = collage_classes.AlignmentOptions()

    line_style_class = collage_classes.LineStyles()

    arrow_style_class = collage_classes.ArrowHeadStyles()

    # Verifies the input and output paths

    if output_path:

        output_file = path_tools.verify_path(output_path, output_file)

    # If the output path is None, but the input path is given, makes the
    # former equal to the latter

    elif input_path:

        output_path = input_path

        output_file = path_tools.verify_path(output_path, output_file)

    # Initializes the collage using the given information

    collage = plt.figure(figsize=(layout_width_milimeters/25.4, 
    layout_height_milimeters/25.4))

    # Sets axes for all items

    general_axes = collage.add_axes([0, 0, 1, 1])

    general_axes.set_xlim(0, layout_width_milimeters)

    general_axes.set_ylim(0, layout_height_milimeters)

    general_axes.set_aspect(aspect_ratio, adjustable=adjustable)

    general_axes.axis("off") 

    # Initializes the counter of elements in depth

    depth_order = 0

    # Verifies if the list of boxes is not None

    if boxes_list is not None:

        print("#######################################################"+
        "#################\n#                            Making of box"+
        "es                           #\n#############################"+
        "###########################################\n")

        # Sets a list of necessary keys

        necessary_keys = ["contour color", "fill color", "contour thic"+
        "kness", "position", "width", "height"]

        # Verifies if it is not a list

        if not isinstance(boxes_list, list):

            raise TypeError("'boxes_list' is not a list. It must be a "+
            "list where each item is a dictionary with the keys:\n'con"+
            "tour color': string with the color name or a RGB list for"+
            " the contour\n'contour thickness': float with the contour"+
            " thickness\n'fill color': string with the color name or a"+
            " RGB list for the fill (face)\n'position': list [x,y] pos"+
            "ition of the centroid\n'width: width of the box\nheight: "+
            "height of the box\n'contour transparency': transparency f"+
            "actor (optional and between 0 and 1) to set transparency "+
            "of the contour line\n'corner radius': the radius of the c"+
            "ontour corners (optional)\n'contour style': 'solid', 'das"+
            "hed', or 'dotted'\n'origin point': available options are "+
            "'centroid', 'bottom-left', 'bottom-right', 'top-left', 't"+
            "op-right'\n'rotation in degrees': float with rotation ang"+
            "le in degrees (from x axis counter-clockwise)\n'shape': s"+
            "tring with the name of the box shape\n'background transpa"+
            "rency': float value betweem 0 and 1 to make the backgroun"+
            "d of the box transparent")
        
        # Iterates through the elements

        for index, input_dictionary in enumerate(boxes_list):

            # Verifies if this element is a dictionary

            if not isinstance(input_dictionary, dict):

                raise TypeError("The "+str(index+1)+"-th element of th"+
                "e 'boxes_list' is not a dictionary. It must be a dict"+
                "ionary with the keys:\n'contour color': string with t"+
                "he color name or a RGB list for the contour\n'contour"+
                " thickness': float with the contour thickness\n'fill "+
                "color': string with the color name or a RGB list for "+
                "the fill (face)\n'position': list [x,y] position of t"+
                "he centroid\n'width: width of the box\nheight: height"+
                " of the box\n'contour transparency': transparency fac"+
                "tor (optional and between 0 and 1) to set transparenc"+
                "y of the contour line\n'corner radius': the radius of"+
                " the contour corners (optional)\n'contour style': 'so"+
                "lid', 'dashed', or 'dotted'\n'origin point': availabl"+
                "e options are 'centroid', 'bottom-left', 'bottom-righ"+
                "t', 'top-left', 'top-right'\n'rotation in degrees': f"+
                "loat with rotation angle in degrees (from x axis coun"+
                "ter-clockwise)\n'shape': string with the name of the "+
                "box shape\n'background transparency': float value bet"+
                "weem 0 and 1 to make the background of the box transp"+
                "arent")
            
            # Iterates through the necessary keys

            for key in necessary_keys:

                # Verifies the key existence

                if not (key in input_dictionary):

                    names = ""

                    for keys in necessary_keys:

                        names += "\n'"+str(keys)+"'"

                    raise ValueError("The "+str(index+1)+"-th element "+
                    "'boxes_list' does not have all the necessary keys"+
                    ", in particular '"+str(key)+"'. Check the necessa"+
                    "ry keys:"+names)
                
            # Gets the colors

            contour_color = colors_class(input_dictionary["contour col"+
            "or"])

            fill_color = colors_class(input_dictionary["fill color"])

            # Gets the position

            position = deepcopy(input_dictionary["position"])

            # Verifies if it is a list

            if not isinstance(position, list):

                raise TypeError("The "+str(index+1)+"-th element 'boxe"+
                "s_list' has at key 'position' a value that is not a l"+
                "ist. It must be a list with [x, y] coordinates. Curre"+
                "ntly, it is:\n"+str(position))
            
            # Gets the size

            width = input_dictionary["width"]

            height = input_dictionary["height"]

            contour_thickness = input_dictionary["contour thickness"]

            # Verifies if they are float

            if ((not isinstance(width, float)) and (not isinstance(width, 
            int))) or ((not isinstance(height, float)) and (not isinstance(
            height, int))) or ((not isinstance(contour_thickness, float)
            ) and (not isinstance(contour_thickness, int))):

                raise TypeError("The "+str(index+1)+"-th element 'boxe"+
                "s_list' has at key 'width' a value that is not a floa"+
                "t or at key 'height', or at key 'contour thickness'. "+
                "They must be both float or integers. Currently, 'widt"+
                "h' is: "+str(width)+"\n'height' is: "+str(height)+"\n"+
                "contour thickness: "+str(contour_thickness))
            
            # Converts contour thickness from milimeters to points

            contour_thickness = contour_thickness*(72.0/25.4)
            
            # Gets the transparency if it is

            if "contour transparency" in input_dictionary:

                # Adds a 0.0 to the fourth channel of RGBA
                
                contour_color.append(input_dictionary["contour transparency"]) 
            
            # Gets the transparency if it is for the background only

            if "background transparency" in input_dictionary:

                # Adds a 0.0 to the fourth channel of RGBA
                
                fill_color.append(input_dictionary["background transpa"+
                "rency"]) 
            
            # Verifies if the origin point is prescribed

            origin_point = 'centroid'

            if "origin point" in input_dictionary:

                origin_point = input_dictionary["origin point"]

            # Updates position using the alignment

            position = alignments_class(origin_point, position, width, 
            height)

            # Verifies contour style

            line_style = '-'

            if "contour style" in input_dictionary:
                
                line_style = line_style_class(input_dictionary["contou"+
                "r style"], contour_thickness, "contour style")
                
            # Verifies if rounded corners are required

            boxstyle = "square,pad=0.0"

            if "corner radius" in input_dictionary:

                # Alters box style to insert the corner radius

                boxstyle=f"round,pad=0.0,rounding_size={input_dictionary[
                "corner radius"]}"

            # Verifies if a depth number has been given

            local_depth_order = deepcopy(depth_order)

            if "depth order" in input_dictionary:

                local_depth_order = input_dictionary["depth order"]

            else:

                # Updates the depth number

                depth_order += 1

            # Verifies if shape is in the input dictionary

            shape = "rectangle"

            if "shape" in input_dictionary:

                shape = input_dictionary["shape"]

            new_box = None

            # Initializes the rotation shift

            rotation_shift = None

            # If it is meant to be a rectangle

            if shape=="rectangle":

                new_box = FancyBboxPatch((position[0], position[1]), 
                width, height, linewidth=contour_thickness, edgecolor=
                contour_color, facecolor=fill_color, boxstyle=boxstyle, 
                linestyle=line_style, zorder=local_depth_order)

                # Sets the rotation shift as the proper position

                rotation_shift = deepcopy(input_dictionary["position"])

            # If it is meant to be an ellipse

            elif shape=="ellipse" or shape=="circle":

                # Calculates the center coordinates

                x_center = position[0]*1.0

                y_center = position[1]*1.0

                # Translates it according to position

                if origin_point=="bottom-left":

                    x_center += 0.5*width

                    y_center += 0.5*height 

                elif origin_point=="bottom-right":

                    x_center -= 0.5*width 

                    y_center += 0.5*height

                elif origin_point=="top-right":

                    x_center -= 0.5*width 

                    y_center -= 0.5*height

                elif origin_point=="top-left":

                    x_center += 0.5*width 

                    y_center -= 0.5*height

                # If circle is asked for, ignores height

                if shape=="circle":

                    height = width*1.0

                new_box = Ellipse((x_center, y_center), width, height, 
                linewidth=contour_thickness, edgecolor=contour_color, 
                facecolor=fill_color, linestyle=line_style, zorder=
                local_depth_order)

                # Sets the rotation shift as the proper position

                rotation_shift = [x_center*1.0, y_center*1.0]

            # If it is a polygon

            elif shape=="polygon":

                # Verifies if there is a key for the number of sides

                if not ("number of sides" in input_dictionary):

                    raise KeyError("'polygon' shape has been asked to "+
                    "create a box, but no 'number of sides' key has be"+
                    "en provided")
                
                number_of_sides = input_dictionary["number of sides"]

                # Verifies if the number of sides is integer

                if not isinstance(number_of_sides, int):

                    raise TypeError("'number of sides' to make a polyg"+
                    "on must be an integer. Currently, it is: "+str(
                    number_of_sides))

                # Calculates the center coordinates

                x_center = position[0]*1.0

                y_center = position[1]*1.0

                # Translates it according to position

                if origin_point=="bottom-left":

                    x_center += 0.5*width

                    y_center += 0.5*height 

                elif origin_point=="bottom-right":

                    x_center -= 0.5*width 

                    y_center += 0.5*height

                elif origin_point=="top-right":

                    x_center -= 0.5*width 

                    y_center -= 0.5*height

                elif origin_point=="top-left":

                    x_center += 0.5*width 

                    y_center -= 0.5*height
                
                # Creates the box

                new_box = RegularPolygon((x_center, y_center), radius=
                width*0.5, numVertices=number_of_sides, linewidth=
                contour_thickness, edgecolor=contour_color, facecolor=
                fill_color, linestyle=line_style, zorder=
                local_depth_order)

                # Stretches the polygon to match the required height. 
                # Firstly, translates to the origin, stretches, and, 
                # then, translates back to the correct position

                stretch_transform = Affine2D().translate(-x_center, 
                -y_center).scale(1.0, height/width).translate(x_center,
                y_center)

                new_box.set_transform(stretch_transform+
                general_axes.transData)

                # Sets the rotation shift as the proper position

                rotation_shift = [x_center*1.0, y_center*1.0]

            else:

                raise NameError("'shape' has been given as '"+str(shape)+
                "', but this shape is not available. The available sha"+
                "pes are:\n'rectangle'\n'ellipse'\n'circle'")

            # Verifies if there is any rotation

            if "rotation in degrees" in input_dictionary:

                # Gets the angle

                angle = input_dictionary["rotation in degrees"]

                # Verifies if it is a float

                if not isinstance(angle, float):

                    raise TypeError("The "+str(index+1)+"-th element '"+
                    "boxes_list' has at key 'rotation in degrees' a va"+
                    "lue that is not a float. Currently, 'rotation in "+
                    "degrees' is: "+str(angle))
                
                # Sets the rotation and transforms the box. Rotates a-
                # round the original point

                rotation = Affine2D().rotate_deg_around(rotation_shift[0
                ], rotation_shift[1], angle)

                new_box.set_transform(rotation+general_axes.transData)

            # Inserts the box into the figure

            print("Adds box with '"+str(shape)+"' at point "+str(position
            )+" with 'origin point' as '"+str(origin_point)+"'\n")

            general_axes.add_patch(new_box)

    # Verifies if the dictionary of input figures is not None

    if input_image_list is not None:

        print("#######################################################"+
        "#################\n#                         Insertion of fig"+
        "ures                         #\n#############################"+
        "###########################################\n")

        # Sets a list of necessary keys

        necessary_keys = ["file name", "position", "size"]

        # Verifies if it is not a list

        if not isinstance(input_image_list, list):

            raise TypeError("'input_image_list' is not a list. It must"+
            " be a list where each item is a dictionary with the keys:"+
            "\n'file name': string with the file name\n'position': lis"+
            "t with position coordinates, [x,y]\n'size': list with wid"+
            "th and height (fractions of the figure size)\n'origin poi"+
            "nt': available options are 'centroid', 'bottom-left', 'bo"+
            "ttom-right', 'top-left', 'top-right'\n'rotation in degree"+
            "s': float with rotation angle in degrees (from x axis cou"+
            "nter-clockwise)\n'trim transparent background': True if a"+
            "n image with transparent background is to be trimmed to g"+
            "et the non-transparent features only")
        
        # Iterates through the elements

        for index, input_dictionary in enumerate(input_image_list):

            # Verifies if this element is a dictionary

            if not isinstance(input_dictionary, dict):

                raise TypeError("The "+str(index+1)+"-th element of th"+
                "e 'input_image_list' is not a dictionary. It must be "+
                "a dictionary with the keys:\n'file name': string with"+
                " the file name\n'position': list with position coordi"+
                "nates, [x,y]\n'size': list with width and height (fra"+
                "ctions of the figure size)\n'origin point': available"+
                " options are 'centroid', 'bottom-left', 'bottom-right"+
                "', 'top-left', 'top-right'\n'rotation in degrees': fl"+
                "oat with rotation angle in degrees (from x axis count"+
                "er-clockwise)\n'trim transparent background': True if"+
                " an image with transparent background is to be trimme"+
                "d to get the non-transparent features only")
            
            # Iterates through the necessary keys

            for key in necessary_keys:

                # Verifies the key existence

                if not (key in input_dictionary):

                    names = ""

                    for keys in necessary_keys:

                        names += "\n'"+str(keys)+"'"

                    raise ValueError("The "+str(index+1)+"-th element "+
                    "'input_image_list' does not have all the necessar"+
                    "y keys, in particular '"+str(key)+"'. Check the n"+
                    "ecessary keys:"+names)
                
            # Gets the name input file name

            input_file_name = input_dictionary["file name"]

            # Adds the input path

            if input_path is not None:

                input_file_name = path_tools.verify_path(input_path, 
                input_file_name)

            # Gets the position

            position = deepcopy(input_dictionary["position"])

            # Verifies if it is a list

            if not isinstance(position, list):

                raise TypeError("The "+str(index+1)+"-th element 'inpu"+
                "t_image_list' has at key 'position' a value that is n"+
                "ot a list. It must be a list with [x, y] coordinates."+
                " Currently, it is:\n"+str(position))

            # Reads the image

            input_image = Image.open(input_file_name)

            # Verifies if a transparent picture is to be trimmed

            if (("trim transparent background" in input_dictionary) and (
            input_dictionary["trim transparent background"])):

                # Convert to RGBA to split the A channel

                if input_image.mode!='RGBA':

                    input_image = input_image.convert("RGBA")

                # Splits the alpha channel

                alpha = input_image.split()[-1]

                # Gets the bounding box of non-zero alpha

                bounding_box = alpha.getbbox()

                # If there is any, crops the image

                if bounding_box:

                    input_image = input_image.crop(bounding_box)
            
            # Gets the size

            size = input_dictionary["size"]

            # Verifies if it is an integer

            if isinstance(size, float):

                # Gets the sizes

                original_width, original_height = input_image.size

                # Sets the size accordingly

                size = [size, size*(original_height/original_width)]

            # Verifies if it is a list

            elif not isinstance(size, list):

                raise TypeError("The "+str(index+1)+"-th element 'inpu"+
                "t_image_list' has at key 'size' a value that is not a"+
                " list nor a float. It must be a list with [width_rati"+
                "o, height_ratio] (fractions of the figure size) or a "+
                " float with the size of the width only (aspect ratio "+
                "is kept). Currently, it is:\n"+str(size))
            
            # Verifies if the origin point is prescribed

            origin_point = 'centroid'

            if "origin point" in input_dictionary:

                origin_point = input_dictionary["origin point"]

            # Updates position using the alignment

            if verbose:

                print("The input image at '"+str(input_file_name)+"'\n"+
                "has a size of "+str(size)+"\n")

            position = alignments_class(origin_point, position, size[0], 
            size[1])

            # Verifies if there is any rotation

            if "rotation in degrees" in input_dictionary:

                # Gets the angle

                angle = input_dictionary["rotation in degrees"]

                # Verifies if it is a float

                if not isinstance(angle, float):

                    raise TypeError("The "+str(index+1)+"-th element '"+
                    "boxes_list' has at key 'rotation in degrees' a va"+
                    "lue that is not a float. Currently, 'rotation in "+
                    "degrees' is: "+str(angle))
                
                # Sets the rotation and transforms the box. Rotates a-
                # round the original point

                rotation = Affine2D().rotate_deg_around(*deepcopy(
                input_dictionary["position"]), angle)

                # Rotates the image

                input_image = input_image.rotate(angle, expand=True)

            # Verifies if a depth number has been given

            local_depth_order = deepcopy(depth_order)

            if "depth order" in input_dictionary:

                local_depth_order = input_dictionary["depth order"]

            else:

                # Updates the depth number

                depth_order += 1

            # Adds image panel

            if verbose:

                print("Adds figure at point "+str(position)+" with 'or"+
                "igin point' as '"+str(origin_point)+"'\n")

            general_axes.imshow(input_image, extent=[position[0], 
            position[0]+size[0], position[1], position[1]+size[1]], 
            origin='upper', zorder=local_depth_order)

    # Verifies if the list of input text excerpts is not None

    if input_text_list is not None:

        print("#######################################################"+
        "#################\n#                        Making of text ex"+
        "cerpts                       #\n#############################"+
        "###########################################\n")

        # Sets a list of necessary keys

        necessary_keys = ["text", "position", "font size"]

        # Verifies if it is not a list

        if not isinstance(input_text_list, list):

            raise TypeError("'input_text_list' is not a list. It must "+
            "be a list where each item is a dictionary with the keys:"+
            "\n'text': string with the text excerpt\n'position': list "+
            "with position coordinates, [x,y]\n'font size': integer\n'"+
            "origin point': available options are 'centroid', 'bottom-"+
            "left', 'bottom-right', 'top-left', 'top-right'\n'rotation"+
            " in degrees': float with rotation angle in degrees (from "+
            "x axis counter-clockwise)")
        
        # Iterates through the elements

        for index, input_dictionary in enumerate(input_text_list):

            # Verifies if this element is a dictionary

            if not isinstance(input_dictionary, dict):

                raise TypeError("The "+str(index+1)+"-th element of th"+
                "e 'input_text_list' is not a dictionary. It must be a"+
                " dictionary with the keys:\n'text': string with the t"+
                "ext excerpt\n'position': list with position coordinat"+
                "es, [x,y]\n'font size': integer\n'rotation in degrees"+
                "': float with rotation angle in degrees (from x axis "+
                "counter-clockwise)")
            
            # Iterates through the necessary keys

            for key in necessary_keys:

                # Verifies the key existence

                if not (key in input_dictionary):

                    names = ""

                    for keys in necessary_keys:

                        names += "\n'"+str(keys)+"'"

                    raise ValueError("The "+str(index+1)+"-th element "+
                    "'input_text_list' does not have all the necessary"+
                    " keys, in particular '"+str(key)+"'. Check the ne"+
                    "cessary keys:"+names)
                
            # Gets the text excerpt

            input_text = input_dictionary["text"]

            # Gets the position

            position = deepcopy(input_dictionary["position"])

            # Verifies if it is a list

            if not isinstance(position, list):

                raise TypeError("The "+str(index+1)+"-th element 'inpu"+
                "t_text_list' has at key 'position' a value that is no"+
                "t a list. It must be a list with [x, y] coordinates. "+
                "Currently, it is:\n"+str(position))
            
            # Verifies if the origin point is prescribed

            origin_point = 'centroid'

            if "origin point" in input_dictionary:

                origin_point = input_dictionary["origin point"]

            # Updates position using the alignment

            position, ha, va = alignments_class(origin_point, position, 
            0.0, 0.0, text_alignment=True)
            
            # Gets the font size

            font_size = input_dictionary["font size"]

            # Verifies if it is an integer

            if not isinstance(font_size, int):

                raise TypeError("The "+str(index+1)+"-th element 'inpu"+
                "t_text_list' has at key 'font size' a value that is n"+
                "ot an integer. Currently, it is:\n"+str(font_size))
            
            # Verifies if there is any rotation

            angle = 0.0

            if "rotation in degrees" in input_dictionary:

                # Gets the angle

                angle = input_dictionary["rotation in degrees"]

                # Verifies if it is a float

                if not isinstance(angle, float):

                    raise TypeError("The "+str(index+1)+"-th element '"+
                    "boxes_list' has at key 'rotation in degrees' a va"+
                    "lue that is not a float. Currently, 'rotation in "+
                    "degrees' is: "+str(angle))

            # Verifies if a depth number has been given

            local_depth_order = deepcopy(depth_order)

            if "depth order" in input_dictionary:

                local_depth_order = input_dictionary["depth order"]

            else:

                # Updates the depth number

                depth_order += 1

            # Adds the text input

            if verbose:

                print("Adds text at point "+str(position)+" with 'orig"+
                "in point' as '"+str(origin_point)+"'\n")

            collage.text(position[0]/layout_width_milimeters, position[1
            ]/layout_height_milimeters, input_text, fontsize=font_size, 
            ha=ha, va=va, rotation=angle, rotation_mode="anchor", zorder=
            local_depth_order)

    # Verifies if the list of arrows is not None

    if arrows_and_lines_list is not None:

        print("#######################################################"+
        "#################\n#                           Making of arro"+
        "ws                           #\n#############################"+
        "###########################################\n")

        # Sets a list of necessary keys

        necessary_keys = ["start point", "end point", "thickness"]

        # Verifies if it is not a list

        if not isinstance(arrows_and_lines_list, list):

            raise TypeError("'arrows_and_lines_list' is not a list. It must be a"+
            " list where each item is a dictionary with the keys:\n'st"+
            "art point': list with [x,y] coordinates of the starting p"+
            "oint\n'end point':  list with [x,y] coordinates of the en"+
            "d point\n'thickness': 'thickness': float value with the a"+
            "rrow line thickness\n'arrow style': string with the style"+
            " of the arrow stem and edge (optional)\n'spline points': "+
            "list of lists of point coordinates for a spline path, [[x"+
            "1, y1], [x2, y2], ..., [xn, yn]] (optional)\n'arrow head "+
            "size': integer with the arrow head size (optional)\n'line"+
            " style': string with the name of the line style of the st"+
            "em (optional)")
        
        # Iterates through the elements

        for index, input_dictionary in enumerate(arrows_and_lines_list):

            # Verifies if this element is a dictionary

            if not isinstance(input_dictionary, dict):

                raise TypeError("The "+str(index+1)+"-th element of th"+
                "e 'arrows_and_lines_list' is not a dictionary. It must be a dic"+
                "tionary with the keys:\n'start point': list with [x,y"+
                "] coordinates of the starting point\n'end point':  li"+
                "st with [x,y] coordinates of the end point\n'thicknes"+
                "s': 'thickness': float value with the arrow line thic"+
                "kness\n'arrow style': string with the style of the ar"+
                "row stem and edge (optional)\n'spline points': list o"+
                "f lists of point coordinates for a spline path, [[x1,"+
                " y1], [x2, y2], ..., [xn, yn]] (optional)\n'arrow hea"+
                "d size': integer with the arrow head size (optional)"+
                "\n'line style': string with the name of the line styl"+
                "e of the stem (optional)")
            
            # Iterates through the necessary keys

            for key in necessary_keys:

                # Verifies the key existence

                if not (key in input_dictionary):

                    names = ""

                    for keys in necessary_keys:

                        names += "\n'"+str(keys)+"'"

                    raise ValueError("The "+str(index+1)+"-th element "+
                    "'arrows_and_lines_list' does not have all the nec"+
                    "essary keys, in particular '"+str(key)+"'. Check "+
                    "the necessary keys:"+names)
                
            # Gets the colors

            arrow_color = colors_class("black")

            if "color" in input_dictionary:

                arrow_color = colors_class(input_dictionary["color"])

            # Gets the start point

            start_point = input_dictionary["start point"]

            # Verifies if it is a list

            if not isinstance(start_point, list):

                raise TypeError("The "+str(index+1)+"-th element 'arro"+
                "ws_and_lines_list' has at key 'start_point' a value t"+
                "hat is not a list. It must be a list with [x, y] coor"+
                "dinates. Currently, it is:\n"+str(start_point))

            # Gets the end point

            end_point = input_dictionary["end point"]

            # Verifies if it is a list

            if not isinstance(end_point, list):

                raise TypeError("The "+str(index+1)+"-th element 'arro"+
                "ws_and_lines_list' has at key 'end_point' a value tha"+
                "t is not a list. It must be a list with [x, y] coordi"+
                "nates. Currently, it is:\n"+str(end_point))
            
            # Gets the thickness

            thickness = input_dictionary["thickness"]

            # Verifies if they are float

            if (not isinstance(thickness, float)):

                raise TypeError("The "+str(index+1)+"-th element 'arro"+
                "ws_and_lines_list' has at key 'thickness' a value tha"+
                "t is not a float. Currently, 'thickness': "+str(
                thickness))
            
            # Converts contour thickness from milimeters to points

            thickness = thickness*(72.0/25.4)

            # Verifies contour style

            line_style = '-'

            if "line style" in input_dictionary:
                
                line_style = line_style_class(input_dictionary["line s"+
                "tyle"], thickness, "line style")

            # Verifies contour style

            arrow_style = '-|>'

            if "arrow style" in input_dictionary:

                # Gets the arrow style
                
                arrow_style = arrow_style_class(input_dictionary["arro"+
                "w style"])
            
            # Gets the arrow head size

            arrow_head_size = int(thickness*15)

            if "arrow head size" in input_dictionary:

                arrow_head_size = input_dictionary["arrow head size"]

                # Verifies if it is an integer

                if (not isinstance(arrow_head_size, float)) and (
                not isinstance(arrow_head_size, int)):

                    raise TypeError("The "+str(index+1)+"-th element '"+
                    "arrows_and_lines_list' has at key 'arrow head siz"+
                    "e' a value that is not an integer nor a float. Cu"+
                    "rrently, 'arrow head size': "+str(arrow_head_size))
                
                # Converts the size from milimeters to points

                arrow_head_size = arrow_head_size*(72/25.4)
                
            # Verifies if the arrow stem follows a spline curve

            codes = [Path.MOVETO] 

            vertices = [start_point]

            if "spline points" in input_dictionary:

                # Gets the spline points and verifies if they are a list

                spline_points = input_dictionary["spline points"]

                if not isinstance(spline_points, list):

                    raise TypeError("'spline points' in 'arrows_and_li"+
                    "nes_list' must be a list of lists [[x1, y1], [x2,"+
                    " y2], ..., [xn, yn]]. Currently, it is:\n"+str(
                    spline_points))

                # Otherwise, uses cubic splines

                else:

                    print("Inserts arrow with cubic Bézier curve\n")
                
                    # Adds the starting point to the spline points

                    spline_points = [start_point, *spline_points, 
                    end_point]
                
                    # Iterates through the points to verify consistency

                    for index_point in range(len(spline_points)-1):

                        # Gets the point and verifies it

                        point_1 = spline_points[index_point]

                        if not isinstance(point_1, list) or len(point_1
                        )!=2:

                            raise ValueError("The "+str(index_point+1)+
                            "-th point of 'spline points' is not a lis"+
                            "t or its length is not 2. Check it out:\n"+
                            str(point_1))
                        
                        # Gets the four points for the Catmull-Rom con-
                        # version and converts them to numpy

                        point_0 = None

                        if index_point>0:

                            point_0 = np.array(spline_points[index_point
                            -1])

                        else:

                            point_0 = np.array(spline_points[index_point
                            ])

                        point_1 = np.array(point_1)

                        point_2 = np.array(spline_points[index_point+1])

                        point_3 = None

                        if index_point+2<len(spline_points):

                            point_3 = np.array(spline_points[index_point
                            +2])

                        else:

                            point_3 = np.array(spline_points[index_point
                            +1])

                        # Subtracts them

                        b1 = point_1+((point_2-point_0)/6)

                        b2 = point_2-((point_3-point_1)/6)

                        b3 = point_2

                        # Adds these points to the vertices

                        vertices.extend([b1, b2, b3])

                        # Appends the instrictions to code

                        codes.extend([Path.CURVE4, Path.CURVE4, 
                        Path.CURVE4])

            else:

                print("Inserts arrow with linear stem\n")

                # Appends the code instruction to create the last segment
                
                codes.append(Path.LINETO)
                
                # Appends the end point and creates the path of the arrow

                vertices.append(end_point)

            # Creates the arrow head path and for the arrow stem separa-
            # tely

            code_head = [Path.MOVETO, Path.LINETO]

            arrow_head_path = Path(np.array(vertices[-2:len(vertices)]), 
            codes=code_head)

            arrow_stem_path = Path(np.array(vertices), codes=codes)

            # Verifies if a depth number has been given

            local_depth_order = deepcopy(depth_order)

            if "depth order" in input_dictionary:

                local_depth_order = input_dictionary["depth order"]

            else:

                # Updates the depth number

                depth_order += 1

            # Adds the arrow stem
            
            general_axes.add_patch(FancyArrowPatch(path=arrow_stem_path, 
            arrowstyle='-', color=arrow_color, linewidth=thickness, 
            zorder=local_depth_order, linestyle=line_style))

            # Adds the arrow head

            if arrow_style!='-':
            
                general_axes.add_patch(FancyArrowPatch(path=
                arrow_head_path, arrowstyle=arrow_style, color=
                arrow_color, mutation_scale=arrow_head_size, zorder=
                local_depth_order, linestyle='-', linewidth=max(
                thickness/3, 1), joinstyle='miter'))

    # Verifies if an overlaying grid is to be added

    if add_overlaying_grid:

        # Sets the lines thickness. The keys are the distance in milime-
        # ters from the last line to the next of the same kind. The va-
        # lues are the line thickness

        lines_thickness = OrderedDict([(1, 0.05*(72/25.4)), (10, 0.18*(
        72/25.4)), (50, 0.3*(72/25.4))])

        # Plots the vertical lines

        for x in range(int(round(layout_width_milimeters))+1):

            # Initializes the thickness

            thickness = 0.1

            # Gets the thickness of the line

            for distance, thickness_trial in lines_thickness.items():

                # Verifies if the division by the distance has no rest

                if x%distance==0:

                    thickness = deepcopy(thickness_trial)

            # Plots the line

            general_axes.plot([x, x], [0, layout_height_milimeters], 
            color="black", linewidth=thickness, zorder=depth_order)

        # Plots the horizontal lines

        for y in range(int(round(layout_height_milimeters))+1):

            # Initializes the thickness

            thickness = 0.1

            # Gets the thickness of the line

            for distance, thickness_trial in lines_thickness.items():

                # Verifies if the division by the distance has no rest

                if y%distance==0:

                    thickness = deepcopy(thickness_trial)

            # Plots the line

            general_axes.plot([0, layout_width_milimeters], [y, y], 
            color="black", linewidth=thickness, zorder=depth_order)

    # Saves the figure

    print("###########################################################"+
    "#############\n#                                Saving           "+
    "                     #\n#########################################"+
    "###############################\n")

    if no_padding:

        # Gets what has been drawn only

        collage.canvas.draw()

        renderer = collage.canvas.get_renderer()

        bounding_box = []

        # Iterates through the elements of the drawing

        for artist in general_axes.get_children():

            # Skips non visible objects

            if not artist.get_visible():

                continue

            # Skips axes background patch

            if artist==general_axes.patch:

                continue

            # Skip spines

            if artist in general_axes.spines.values():

                continue

            # Tries to get the window

            try:

                local_bounding_box = artist.get_window_extent(renderer)

                # Ignores empty bounding boxes

                if local_bounding_box.width>0 and (
                local_bounding_box.height>0):
                    
                    bounding_box.append(local_bounding_box)

            except Exception:

                pass

        # Makes the union of the bounding boxes

        if bounding_box:

            print("There is a bounding box to shrink and pad.\nSaves a"+
            "t "+str(output_file)+"\n")

            bounding_box = Bbox.union(bounding_box)

            # Converts to inches

            bbox_inches = bounding_box.transformed(
            collage.dpi_scale_trans.inverted())

            # Saves the figure with this bounding box in inches

            plt.savefig(output_file, bbox_inches=bbox_inches, pad_inches=
            0, dpi=dpi)

        # Otherwise, saves it plainly

        else:

            print("There is no bounding box to shrink and pad.\nSaves "+
            "at "+str(output_file)+"\n")

            plt.savefig(output_file, dpi=dpi)

    else:

        print("Saves as it is at "+str(output_file)+"\n")

        plt.savefig(output_file, dpi=dpi)

    plt.close()