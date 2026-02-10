# Routine to make collages

import numpy as np

from copy import deepcopy

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from matplotlib.transforms import Affine2D

from matplotlib.path import Path

from matplotlib.backends.backend_pdf import PdfPages

from PIL import Image

from ..PythonicUtilities import path_tools

# Sets LaTeX font

matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif",
"font.serif": ["Computer Modern Roman"]})

# Defines a class with colors

class ColorMiscellany:

    def __init__(self):

        # Sets names as keys and values as RGB lists
    
        self.color_dictionary = {"white": [1.0, 1.0, 1.0], "black": [0.0,
        0.0, 0.0], "red 1": [1.0, 0.835, 0.835], "red 2": [1.0, 0.667, 
        0.667], "red 3": [1.0, 0.502, 0.502], "red 4": [1.0, 0.333, 0.333
        ], "red 5": [1.0, 0.165, 0.165], "greyish red 1": [1.0, 0.843, 
        0.843], "greyish red 2": [0.914, 0.686, 0.686], "greyish red 3":
        [0.871, 0.529, 0.529], "greyish red 4": [0.827, 0.373, 0.373],
        "greyish red 5": [0.784, 0.216, 0.216], "grey 1": [0.89, 0.859, 
        0.859], "grey 2": [0.784, 0.718, 0.718], "grey 3": [0.675, 0.576, 
        0.576], "grey 4": [0.569, 0.435, 0.435], "grey 5": [0.424, 0.325, 
        0.325], "yellow 1": [1.0, 0.902, 0.835], "yellow 2": [1.0, 0.8, 
        0.667], "yellow 3": [1.0, 0.702, 0.502], "yellow 4": [1.0, 0.6, 
        0.333], "yellow 5": [1.0, 0.498, 0.165]}

    # Defines a function to get the color

    def __call__(self, key):
        
        # Verifies if it is one of the keys

        if key in self.color_dictionary:

            return self.color_dictionary[key]
        
        # Otherwise, verifies if it is a list

        elif isinstance(key, list):

            # Verifies if it has 3 elements

            if len(key)!=3:

                raise IndexError("'"+str(key)+"' does not have 3 eleme"+
                "nts. It must have 3 for they are the RGB values")
            
            return key 
        
        # Otherwise, throws an error

        else:

            available_colors = ""

            for color in self.color_dictionary:

                available_colors += "\n'"+str(color)+"'"

            raise ValueError("'"+str(key)+"' is not a key of the dicti"+
            "onary of colors nor is a list with RGB values (3 componen"+
            "ts). Check the valid color names:"+available_colors)
        
# Defines a class to store alignment options

class AlignmentOptions:

    def __init__(self):

        # Sets a dictionary with keys representing the position origin
        # of a generic element, and the corresponding values are lists
        # with the coefficients to multiply the element sizes 
         
        self.alignments = {'centroid': [-0.5, -0.5], 'bottom-left': [0.0, 
        0.0], 'bottom-right': [-1.0, 0.0], 'top-right': [-1.0, -1.0], 
        "top-left": [0.0, -1.0]}

        # Sets a dictionary with keys representing text alignments
         
        self.text_alignments = {'centroid': ("center", "center"), "bott"+
        "om-left": ("left", "bottom"), "bottom-right": ("right", "bott"+
        "om"), "top-left": ("left", "top"), "top-right": ("right", "to"+
        "p")}

    def __call__(self, alignment, position, width, height, 
    text_alignment=False):
        
        # If text alignment is to be given

        if text_alignment:

            # Verifies alignment consistency

            if not (alignment in self.text_alignments):

                alignments_options = ""

                for name in self.text_alignments:

                    alignments_options += "\n'"+str(name)+"'"

                raise NameError("There is no '"+str(alignment)+"' opti"+
                "on for alignment. Check the available options:"+
                alignments_options)
            
            # Returns the matplotlib native settings

            return (position, self.text_alignments[alignment][0], 
            self.text_alignments[alignment][1])

        # Verifies the alignment option

        if not (alignment in self.alignments):

            alignments_options = ""

            for name in self.alignments:

                alignments_options += "\n'"+str(name)+"'"

            raise NameError("There is no '"+str(alignment)+"' option f"+
            "or alignment. Check the available options:"+
            alignments_options)
        
        # Gets the coefficients

        c_width, c_height = self.alignments[alignment]

        # Updates it

        return [position[0]+(c_width*width), position[1]+(c_height*
        height)]

# Defines a function to create a collage using boxes

def create_box_collage(output_file, input_path=None, output_path=None,
no_padding=False, input_image_list=None, input_text_list=None, 
boxes_list=None, arrows_list=None, dpi=300):
    
    # Initializes the class of colors and the class of alignments

    colors_class = ColorMiscellany()

    alignments_class = AlignmentOptions()

    # Verifies the input and output paths

    if output_path:

        output_file = path_tools.verify_path(output_path, output_file)

    # If the output path is None, but the input path is given, makes the
    # former equal to the latter

    elif input_path:

        output_path = input_path

        output_file = path_tools.verify_path(output_path, output_file)

    # Initializes a A4 collage

    collage = plt.figure(figsize=(8.27, 11.69))

    # Verifies if the list of boxes is not None

    if boxes_list is not None:

        # Sets axes for all boxes

        box_axes = collage.add_axes([0, 0, 1, 1])

        box_axes.set_xlim(0, 1)

        box_axes.set_ylim(0, 1)

        box_axes.set_aspect('equal', adjustable='box')

        box_axes.axis("off") 

        # Sets a list of necessary keys

        necessary_keys = ["contour color", "fill color", "contour thic"+
        "kness", "position", "width", "height"]

        # Verifies if it is not a list

        if not isinstance(boxes_list, list):

            raise TypeError("'boxes_list' is not a list. It must be a "+
            "list where each item is a dictionary with the keys:\n'con"+
            "our color': string with the color name or a RGB list for "+
            "the contour\n'contour thickness': float with the contour "+
            "thickness\n'fill color': string with the color name or a "+
            "RGB list for the fill (face)\n'position': list [x,y] posi"+
            "tion of the centroid\n'width: width of the box\nheight: h"+
            "eight of the box\n'transparency': transparency factor (op"+
            "tional and between 0 and 1)\n'corner radius': the radius "+
            "of the contour corners (optional)\n'contour style': 'soli"+
            "d', 'dashed', or 'dotted'\n'origin point': available opti"+
            "ons are 'centroid', 'bottom-left', 'bottom-right', 'top-l"+
            "eft', 'top-right'\n'rotation in degrees': float with rota"+
            "tion angle in degrees (from x axis counter-clockwise)")
        
        # Iterates through the elements

        for index, input_dictionary in enumerate(boxes_list):

            # Verifies if this element is a dictionary

            if not isinstance(input_dictionary, dict):

                raise TypeError("The "+str(index+1)+"-th element of th"+
                "e 'boxes_list' is not a dictionary. It must be a dict"+
                "ionary with the keys:\n'conour color': string with th"+
                "e color name or a RGB list for the contour\n'contour "+
                "thickness': float with the contour thickness\n'fill c"+
                "olor': string with the color name or a RGB list for t"+
                "he fill (face)\n'position': list [x,y] position of th"+
                "e centroid\n'width: width of the box\nheight: height "+
                "of the box\n'origin point': available options are 'ce"+
                "ntroid', 'bottom-left', 'bottom-right', 'top-left', '"+
                "top-right'\n'rotation in degrees': float with rotatio"+
                "n angle in degrees (from x axis counter-clockwise)")
            
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

            if (not isinstance(width, float)) or (not isinstance(height, 
            float)) or (not isinstance(contour_thickness, float)):

                raise TypeError("The "+str(index+1)+"-th element 'boxe"+
                "s_list' has at key 'width' a value that is not a floa"+
                "t or at key 'height', or at key 'contour thickness'. "+
                "They must be both float. Currently, 'width' is: "+str(
                width)+"\n'height' is: "+str(height)+"\ncontour thickn"+
                "ess: "+str(contour_thickness))
            
            # Gets the transparency if it is

            alpha = 1.0

            if "transparency" in input_dictionary:

                alpha = input_dictionary["transparency"]
            
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

                # Sets a dctionary of available styles

                available_line_styles = {"solid": '-', "dashed": (0, (
                int(round(10*contour_thickness)), int(round(6*
                contour_thickness)))), "dotted": (0, (int(round(1*
                contour_thickness)), int(round(5*contour_thickness)))),
                "dash-dotted": (0, (int(round(8*contour_thickness)), int(
                round(4*contour_thickness)), int(round(2*
                contour_thickness)), int(round(4*contour_thickness))))}

                if input_dictionary["contour style"] in available_line_styles:

                    line_style = available_line_styles[input_dictionary[
                    "contour style"]]

                else:

                    available_contour_styles = ""

                    for contour in available_line_styles:

                        available_contour_styles += "\n'"+str(contour)+"'"

                    raise NameError("The only 'contour style' availabl"+
                    "e are:"+available_contour_styles)
                
            # Verifies if rounded corners are required

            boxstyle = "square,pad=0.0"

            if "corner radius" in input_dictionary:

                # Alters box style to insert the corner radius

                boxstyle=f"round,pad=0.0,rounding_size={input_dictionary[
                "corner radius"]}"

            # Creates the rectangle

            new_box = FancyBboxPatch((position[0], position[1]), width, 
            height, linewidth=contour_thickness, edgecolor=contour_color, 
            facecolor=fill_color, alpha=alpha, boxstyle=boxstyle, 
            linestyle=line_style)

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

                new_box.set_transform(rotation+box_axes.transData)

            # Inserts the box into the figure

            box_axes.add_patch(new_box)

    # Verifies if the dictionary of input figures is not None

    if input_image_list is not None:

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
            "ttom-right', 'top-left', 'top-right'")
        
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
                "', 'top-left', 'top-right'")
            
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

            position = alignments_class(origin_point, position, size[0], 
            size[1])

            # Adds image panel

            image_canvas = collage.add_axes([position[0], position[1], 
            size[0], size[1]])

            # inserts the image in the canvas

            image_canvas.imshow(input_image, interpolation='nearest')

            # Removes the scaffolding of the canvas

            image_canvas.axis("off")

    # Verifies if the list of input text excerpts is not None

    if input_text_list is not None:

        # Sets a list of necessary keys

        necessary_keys = ["text", "position", "font size"]

        # Verifies if it is not a list

        if not isinstance(input_text_list, list):

            raise TypeError("'input_text_list' is not a list. It must "+
            "be a list where each item is a dictionary with the keys:"+
            "\n'text': string with the text excerpt\n'position': list "+
            "with position coordinates, [x,y]\n'font size': integer\n'"+
            "origin point': available options are 'centroid', 'bottom-"+
            "left', 'bottom-right', 'top-left', 'top-right'")
        
        # Iterates through the elements

        for index, input_dictionary in enumerate(input_text_list):

            # Verifies if this element is a dictionary

            if not isinstance(input_dictionary, dict):

                raise TypeError("The "+str(index+1)+"-th element of th"+
                "e 'input_text_list' is not a dictionary. It must be a"+
                " dictionary with the keys:\n'text': string with the t"+
                "ext excerpt\n'position': list with position coordinat"+
                "es, [x,y]\n'font size': integer")
            
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

            # Adds the text input

            collage.text(*position, input_text, fontsize=font_size, ha=
            ha, va=va)

    # Verifies if the list of arrows is not None

    if arrows_list is not None:

        # Sets axes for all boxes

        arrow_axes = collage.add_axes([0, 0, 1, 1])

        arrow_axes.set_xlim(0, 1)

        arrow_axes.set_ylim(0, 1)

        arrow_axes.axis("off") 

        # Sets a list of necessary keys

        necessary_keys = ["start point", "end point", "thickness"]

        # Verifies if it is not a list

        if not isinstance(arrows_list, list):

            raise TypeError("'arrows_list' is not a list. It must be a"+
            " list where each item is a dictionary with the keys:\n'st"+
            "art point': list with [x,y] coordinates of the starting p"+
            "oint\n'end point':  list with [x,y] coordinates of the en"+
            "d point\n'thickness': 'thickness': float value with the a"+
            "rrow line thickness\n'arrow style': string with the style"+
            " of the arrow stem and edge (optional)\n'spline points': "+
            "list of lists of point coordinates for a spline path, [[x"+
            "1, y1], [x2, y2], ..., [xn, yn]] (optional)\n'arrow head "+
            "size': integer with the arrow head size (optional)")
        
        # Iterates through the elements

        for index, input_dictionary in enumerate(arrows_list):

            # Verifies if this element is a dictionary

            if not isinstance(input_dictionary, dict):

                raise TypeError("The "+str(index+1)+"-th element of th"+
                "e 'arrows_list' is not a dictionary. It must be a dic"+
                "tionary with the keys:\n'start point': list with [x,y"+
                "] coordinates of the starting point\n'end point':  li"+
                "st with [x,y] coordinates of the end point\n'thicknes"+
                "s': 'thickness': float value with the arrow line thic"+
                "kness\n'arrow style': string with the style of the ar"+
                "row stem and edge (optional)\n'spline points': list o"+
                "f lists of point coordinates for a spline path, [[x1,"+
                " y1], [x2, y2], ..., [xn, yn]] (optional)\n'arrow hea"+
                "d size': integer with the arrow head size (optional)")
            
            # Iterates through the necessary keys

            for key in necessary_keys:

                # Verifies the key existence

                if not (key in input_dictionary):

                    names = ""

                    for keys in necessary_keys:

                        names += "\n'"+str(keys)+"'"

                    raise ValueError("The "+str(index+1)+"-th element "+
                    "'arrows_list' does not have all the necessary key"+
                    "s, in particular '"+str(key)+"'. Check the necess"+
                    "ary keys:"+names)
                
            # Gets the colors

            arrow_color = colors_class("black")

            if "color" in input_dictionary:

                arrow_color = colors_class(input_dictionary["color"])

            # Gets the start point

            start_point = input_dictionary["start point"]

            # Verifies if it is a list

            if not isinstance(start_point, list):

                raise TypeError("The "+str(index+1)+"-th element 'arro"+
                "ws_list' has at key 'start_point' a value that is not"+
                " a list. It must be a list with [x, y] coordinates. C"+
                "urrently, it is:\n"+str(start_point))

            # Gets the end point

            end_point = input_dictionary["end point"]

            # Verifies if it is a list

            if not isinstance(end_point, list):

                raise TypeError("The "+str(index+1)+"-th element 'arro"+
                "ws_list' has at key 'end_point' a value that is not a"+
                " list. It must be a list with [x, y] coordinates. Cur"+
                "rently, it is:\n"+str(end_point))
            
            # Gets the thickness

            thickness = input_dictionary["thickness"]

            # Verifies if they are float

            if (not isinstance(thickness, float)):

                raise TypeError("The "+str(index+1)+"-th element 'arro"+
                "ws_list' has at key 'thickness' a value that is not a"+
                " float. Currently, 'thickness': "+str(thickness))

            # Verifies contour style

            arrow_style = '-|>'

            if "arrow style" in input_dictionary:

                # Gets the arrow style

                arrow_style = input_dictionary["arrow style"]

                # Verifies if it is in the list of admissible style

                arrow_styles = ["-", "->", "-|>", "-[", "<-", "<->", 
                "fancy", "simple", "wedge", "arc"]

                if not (arrow_style in arrow_styles):

                    available_arrows = ""

                    for arrow in arrow_styles:

                        available_arrows += "\n'"+str(arrow)+"'"

                    raise NameError("The only 'arrow style' available "+
                    " are:"+available_arrows)
            
            # Gets the arrow head size

            arrow_head_size = 15

            if "arrow head size" in input_dictionary:

                arrow_head_size = input_dictionary["arrow head size"]

                # Verifies if it is an integer

                if (not isinstance(arrow_head_size, int)):

                    raise TypeError("The "+str(index+1)+"-th element '"+
                    "arrows_list' has at key 'arrow head size' a value"+
                    " that is not an integer. Currently, 'arrow head s"+
                    "ize': "+str(arrow_head_size))
                
            # Verifies if the arrow stem follows a spline curve

            codes = [Path.MOVETO] 

            vertices = [start_point]

            if "spline points" in input_dictionary:

                # Gets the spline points and verifies if they are a list

                spline_points = input_dictionary["spline points"]

                if not isinstance(spline_points, list):

                    raise TypeError("'spline points' in 'arrows_list' "+
                    "must be a list of lists [[x1, y1], [x2, y2], ...,"+
                    " [xn, yn]]. Currently, it is:\n"+str(spline_points))
                
                # Adds the starting point to the spline points

                spline_points = [start_point, *spline_points, end_point]
                
                # Iterates through the points to verify consistency

                for index_point in range(len(spline_points)-1):

                    # Gets the point and verifies it

                    point_1 = spline_points[index_point]

                    if not isinstance(point_1, list) or len(point_1)!=2:

                        raise ValueError("The "+str(index_point+1)+"-t"+
                        "h point of 'spline points' is not a list or i"+
                        "ts length is not 2. Check it out:\n"+str(
                        point_1))
                    
                    # Gets the four points for the Catmull-Rom conver-
                    # sion and converts them to numpy

                    point_0 = None

                    if index_point>0:

                        point_0 = np.array(spline_points[index_point-1])

                    else:

                        point_0 = np.array(spline_points[index_point])

                    point_1 = np.array(point_1)

                    point_2 = np.array(spline_points[index_point+1])

                    point_3 = None

                    if index_point+2<len(spline_points):

                        point_3 = np.array(spline_points[index_point+2])

                    else:

                        point_3 = np.array(spline_points[index_point+1])

                    # Subtracts them

                    b1 = point_1+((point_2-point_0)/6)

                    b2 = point_2-((point_3-point_1)/6)

                    b3 = point_2

                    # Adds these points to the vertices

                    vertices.extend([b1, b2, b3])

                    # Appends the instrictions to code

                    codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

            else:

                # Appends the code instruction to create the last segment
                
                codes.append(Path.LINETO)
                
                # Appends the end point and creates the path of the arrow

                vertices.append(end_point)

            arrow_path = Path(np.array(vertices), codes=codes)

            # The arrow
            
            arrow_axes.add_patch(FancyArrowPatch(path=arrow_path, 
            arrowstyle=arrow_style, color=arrow_color, linewidth=
            thickness, mutation_scale=arrow_head_size))

    # Saves the figure

    if no_padding:

        plt.savefig(output_file, bbox_inches="tight", pad_inches=0, dpi=
        dpi)

    else:

        plt.savefig(output_file, dpi=dpi)

    plt.close()