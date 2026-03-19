# Routine to store classes and methods for the collage tools

import numpy as np

from matplotlib.patches import ArrowStyle

from matplotlib.transforms import Bbox

from pathlib import Path as Path

from importlib import util

import sys

########################################################################
#                           Imports preamble                           #
########################################################################

# Gets the parent paths of the current 

broken_path = Path(__file__).parents

# Imports path tools

specifications = util.spec_from_file_location("dictionary_tools", 
broken_path[2]/"PythonicUtilities"/"dictionary_tools.py")

dictionary_tools = util.module_from_spec(specifications)

sys.modules["dictionary_tools"] = dictionary_tools

specifications.loader.exec_module(dictionary_tools)

from matplotlib.path import Path

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
        0.333], "yellow 5": [1.0, 0.498, 0.165], "transparent": [1.0, 
        0.835, 0.835, 0.01], "dark gray": [0.1, 0.1, 0.1]}

    # Defines a function to get the color

    def __call__(self, key, throw_error=True):
        
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

            error_message = ("'"+str(key)+"' is not a key of the dicti"+
            "onary of colors nor is a list with RGB values (3 componen"+
            "ts). Check the valid color names:"+available_colors)

            # If an error is to be thrown

            if throw_error:

                raise ValueError(error_message)
            
            # Otherwise, just returns False

            return False

    # Defines a function to verify if the color is there

    def verify_color_name(self, key, throw_error=False):
        
        # Verifies if it is one of the keys

        if key in self.color_dictionary:

            return key
        
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

            error_message = ("'"+str(key)+"' is not a key of the dicti"+
            "onary of colors nor is a list with RGB values (3 componen"+
            "ts). Check the valid color names:"+available_colors)

            # If an error is to be thrown

            if throw_error:

                raise ValueError(error_message)
            
            # Otherwise, prints on screen

            else:

                print(error_message)
            
            # Otherwise, just returns False

            return False

# Defines a class with ready-to-use layout sizes

class TemplateSizes:

    def __init__(self):

        # Sets names as keys and values as lists of width and height in 
        # milimeters
    
        self.sizes_dictionary = {"A4": [210.0, 297.0], "beamer full sl"+
        "ide": [128.0, 96.0], "beamer slide with logo": [76.0, 54.0], 
        "beamer slide without logo": [76.0, 67.0]}

    # Defines a function to get the color

    def __call__(self, key, throw_error=True):
        
        # Verifies if it is one of the keys

        if key in self.sizes_dictionary:

            return self.sizes_dictionary[key]
        
        # Otherwise, verifies if it is a list

        elif isinstance(key, list):

            # Verifies if it has 2 elements

            if len(key)!=2:

                raise IndexError("'"+str(key)+"' does not have 2 eleme"+
                "nts. It must have 2 values only to select the layout "+
                "dimensions---width and height")
            
            return key 
        
        # Otherwise, throws an error

        else:

            available_sizes = ""

            for size in self.sizes_dictionary:

                available_sizes += "\n'"+str(size)+"'"

            error_message = ("'"+str(key)+"' is not a key of the dicti"+
            "onary of sizes nor is a list with layout sizes (2 compone"+
            "nts---width and height). Check the valid size names:"+
            available_sizes)

            # If an error is to be thrown

            if error_message:

                raise NameError(error_message)
            
            # Otherwise, just returns False

            return False
        
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
    text_alignment=False, throw_error=True):
        
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

            error_message = ("There is no '"+str(alignment)+"' option "+
            "for alignment. Check the available options:"+
            alignments_options)

            # If error is to be thrown

            if throw_error:

                raise NameError(error_message)
            
            # Otherwise, returns False

            return False
        
        # Gets the coefficients

        c_width, c_height = self.alignments[alignment]

        # Updates it

        return [position[0]+(c_width*width), position[1]+(c_height*
        height)]
        
    # Defines a function simply to verify the existence of the alignment
    # option

    def verify_alignment_name(self, alignment, throw_error=False):
        
        # Verifies if the asked alignment style is available

        if alignment in self.alignments:

            # Returns the alignment name

            return alignment

        else:

            available_alignment_styles = ""

            for alignment_name in self.alignments:

                available_alignment_styles += "\n'"+str(alignment_name
                )+"'"

            error_message = ("'"+str(alignment)+"' is not a valid alig"+
            "nment style. The only available options are: "+
            available_alignment_styles)

            # If error is to be thrown

            if throw_error:

                raise NameError(error_message)
            
            # Otherwise, returns false

            return False
    
# Defines a class to set line styles

class LineStyles:

    def __init__(self):

        # Defines a dictionary of line styles
        
        self.available_line_styles = {"solid": '-', "dashed": [0, [10,6]
        ], "dotted": [0, [1, 5]], "dash-dotted": [0, [8, 4, 2, 4]], "d"+
        "ashed 7x3": [0, [7, 3]], "dashed 7x7": [0, [7, 7]], "simple d"+
        "ash": "--"}

    # Defines a function to get an integer value greater or equal to 1

    def get_int(self, number):

        return max(int(round(number)), 1)

    def __call__(self, line_style, base_thickness, style_key_name, 
    throw_error=True):
        
        # Verifies if the asked line style is available

        if line_style in self.available_line_styles:

            retrieved_line_style = self.available_line_styles[line_style]

            # Verifies if the retrieved style is a list

            if isinstance(retrieved_line_style, list):

                # Updates the second list using the given base thickness

                for i in range(len(retrieved_line_style[1])):

                    retrieved_line_style[1][i] = self.get_int(
                    retrieved_line_style[1][i]*base_thickness)

                # Transforms the list in a tuple

                return (retrieved_line_style[0], tuple(
                retrieved_line_style[1]))
            
            # Otherwise, returns the plain style

            return retrieved_line_style

        else:

            available_contour_styles = ""

            for contour in self.available_line_styles:

                available_contour_styles += "\n'"+str(contour)+"'"

            error_message = ("'"+str(line_style)+"' is not a valid '"+
            str(style_key_name)+"'. The only '"+str(style_key_name)+"'"+
            " available are:"+available_contour_styles)

            # If error is to be thrown

            if throw_error:

                raise NameError(error_message)
            
            # Otherwise, returns false

            return False

    def verify_line_name(self, line_style, throw_error=False):
        
        # Verifies if the asked line style is available

        if line_style in self.available_line_styles:

            # Returns the plain key

            return line_style

        else:

            available_contour_styles = ""

            for contour in self.available_line_styles:

                available_contour_styles += "\n'"+str(contour)+"'"

            error_message = ("'"+str(line_style)+"' is not a valid 'li"+
            "ne style'. The only 'line style' availables are:"+
            available_contour_styles)

            # If error is to be thrown

            if throw_error:

                raise NameError(error_message)
            
            # Otherwise, prints on screen

            else:

                print(error_message)
            
            # Otherwise, returns false

            return False
        
# Defines a class to store arrow head styles

class ArrowHeadStyles:

    def __init__(self):
        
        # Defines a dictionary with the available arrow head styles

        self.arrow_head_styles = {"-": "-", "->": "->", "-|>": "-|>", 
        "-[": "-[", "<-": "<-", "<->": "<->", "fancy": "fancy", "simpl"+
        "e": "simple", "wedge": "wedge", "no arrow": "-", "inkscape an"+
        "gular arrow": ArrowHeadStyles.InkscapeAngularArrow, "inkscape"+
        " round back arrow": ArrowHeadStyles.InkscapeCurvedBackArrow}

    def __call__(self, arrow_head_style, throw_error=True):
        
        # Verifies if the asked arrow_head style is available

        if arrow_head_style in self.arrow_head_styles:

            retrieved_arrow_head_style = self.arrow_head_styles[
            arrow_head_style]
            
            # Verifies if the retrieved style is a string. If so, returns
            # it, since a string is the finished style

            if isinstance(retrieved_arrow_head_style, str):

                return retrieved_arrow_head_style
            
            # Otherwise, it is a class, and it must be instantiated

            else:

                return retrieved_arrow_head_style()

        else:

            available_arrow_styles = ""

            for arrow in self.arrow_head_styles:

                available_arrow_styles += "\n'"+str(arrow)+"'"

            error_message = ("The only 'arrow style' available are:"+
            available_arrow_styles)

            # If error is to be thrown

            if throw_error:

                raise NameError(error_message)
            
            # Otherwise, returns False

            return False

    def verify_arrow_name(self, arrow_head_style, throw_error=False):
        
        # Verifies if the asked arrow_head style is available

        if arrow_head_style in self.arrow_head_styles:

            return arrow_head_style

        else:

            available_arrow_styles = ""

            for arrow in self.arrow_head_styles:

                available_arrow_styles += "\n'"+str(arrow)+"'"

            error_message = ("The only 'arrow style' available are:"+
            available_arrow_styles)

            # If error is to be thrown

            if throw_error:

                raise NameError(error_message)
            
            # Otherwise, prints on screen

            else:

                print(error_message)
            
            # Otherwise, returns False

            return False
    
    ####################################################################
    #                           Custom arrows                          #
    ####################################################################

    # Defines a class for the angular arrow head

    class InkscapeAngularArrow(ArrowStyle._Base):

        def __init__(self, scale=1.0):

            self.scale = scale

            super().__init__()

        def transmute(self, path, mutation_size, linewidth):

            # Gets the scale for the points

            scale = mutation_size*self.scale*0.5

            # Gets the endpoint of the path to insert the arrow there
            
            x, y = path.vertices[-1]

            x_2, y_2 = path.vertices[-2]

            # Gets the directional vector opposite to the path

            direction = np.array([x_2-x, y_2-y])

            direction = direction*(1/(np.linalg.norm(direction)+1E-6)
            )*scale

            # Rotates the direction vector by 30 degrees to get the upper
            # edge

            theta_up = -30*(np.pi/180)

            R = np.array([[np.cos(theta_up), -np.sin(theta_up)], [np.sin(
            theta_up),  np.cos(theta_up)]])

            upper_edge = R @ direction

            # And the lower edge

            theta_low = 30*(np.pi/180)

            R = np.array([[np.cos(theta_low), -np.sin(theta_low)], [
            np.sin(theta_low),  np.cos(theta_low)]])

            lower_edge = R @ direction

            # Creates the vertices. The order is: the point, the upper
            # edge, the back edge, the lower edge, and again to the point
            
            vertices = [(x-0.6*direction[0], y-0.6*direction[1]), (x+
            upper_edge[0]-0.6*direction[0], y+upper_edge[1]-0.6*
            direction[1]), (x, y), (x+lower_edge[0]-0.6*direction[0], y+
            lower_edge[1]-0.6*direction[1]), (x-0.6*direction[0], y-0.6*
            direction[1])]

            # Creates the codes for the curve segments

            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
            Path.CLOSEPOLY]

            # Creates a path

            arrow_path = Path(vertices, codes)

            return arrow_path, True
        
    # Defines a class for the arrow head with curved back side

    class InkscapeCurvedBackArrow(ArrowStyle._Base):

        def __init__(self, scale=1.0):

            self.scale = scale

            super().__init__()

        def transmute(self, path, mutation_size, linewidth):

            # Gets the scale for the points

            scale = mutation_size*self.scale*0.5

            # Gets the endpoint of the path to insert the arrow there
            
            x, y = path.vertices[-1]

            x_2, y_2 = path.vertices[-2]

            # Gets the directional vector opposite to the path

            direction = np.array([x_2-x, y_2-y])

            direction = direction*(1/(np.linalg.norm(direction)+1E-6)
            )*scale

            # Rotates the direction vector by 30 degrees to get the upper
            # edge

            theta_up = -30*(np.pi/180)

            R = np.array([[np.cos(theta_up), -np.sin(theta_up)], [np.sin(
            theta_up),  np.cos(theta_up)]])

            upper_edge = R @ direction

            # And the lower edge

            theta_low = 30*(np.pi/180)

            R = np.array([[np.cos(theta_low), -np.sin(theta_low)], [
            np.sin(theta_low),  np.cos(theta_low)]])

            lower_edge = R @ direction

            # Creates the vertices. The order is: the point, the upper
            # edge, the lower edge, and again to the point
            
            vertices = [(x-0.6*direction[0], y-0.6*direction[1]), (x+
            upper_edge[0]-0.6*direction[0], y+upper_edge[1]-0.6*
            direction[1]), (x, y), (x+lower_edge[0]-0.6*direction[0], y+
            lower_edge[1]-0.6*direction[1]), (x-0.6*direction[0], y-0.6*
            direction[1])]

            # Creates the codes for the curve segments

            codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
            Path.CLOSEPOLY]

            # Creates a path

            arrow_path = Path(vertices, codes)

            return arrow_path, True
        
########################################################################
#                             Bounding box                             #
########################################################################

# Defines a function to get a export selection and transform it into a
# bounding box

def get_export_selection(export_selection, verbose=False):

    # Verifies if export_selection is a dictionary and the keys

    dictionary_tools.verify_obligatory_and_optional_keys(export_selection, {"origin poi"+
    "nt": {"type": str, "description": "alignment of the selection box"+
    " with respect to the 'position' vector. It can be either 'top-lef"+
    "t', 'top-right', 'bottom-right', or 'bottom-left'"}, "position": {
    "type": list, "description": "list with two components [x, y] in m"+
    "ilimeters, that tells the position of the corner that sets the al"+
    "ignment from"}, "width": {"type": float, "description": "width of"+
    " the selection box in milimeters"}, "height": {"type": float, "de"+
    "scription": "height of the selection box in milimeters"}}, {}, "e"+
    "xport_selection", "create_box_collage")

    # Gets the alignment and the measurements

    alignment = export_selection["origin point"]

    position = export_selection["position"]

    width = export_selection["width"]

    height = export_selection["height"]

    # Initializes the bounding box bounds

    x_min = None 

    x_max = None 

    y_min = None 

    y_max = None

    # Computes the bounds for each alignment case

    if alignment=="bottom-right":

        x_min = position[0]-width
        
        y_min = position[1]
        
        x_max = position[0]
        
        y_max = position[1]+height

    elif alignment=="top-right":

        x_min = position[0]-width
        
        y_min = position[1]-height
        
        x_max = position[0]
        
        y_max = position[1]

    elif alignment=="top-left":

        x_min = position[0]
        
        y_min = position[1]-height
        
        x_max = position[0]+width
        
        y_max = position[1]

    elif alignment=="bottom-left":

        x_min = position[0]
        
        y_min = position[1]
        
        x_max = position[0]+width
        
        y_max = position[1]+height

    else:

        raise NameError("'origin-point' in 'export_selection' at 'crea"+
        "te_box_collage' is '"+str(alignment)+"'. But it can be only o"+
        "ne of the following options:\n'top-left'\n'top-right'\n'botto"+
        "m-right'\n'bottom-left'")
    
    if verbose:
    
        print("\nCreates a bounding box from an export selection. The "+
        "bounds are:\n"+str(x_min)+" <= x <= "+str(x_max)+"\n"+str(y_min
        )+" <= y <= "+str(y_max))

    # Makes the bounding box in inches

    factor = 1/25.4

    return Bbox.from_extents(x_min*factor, y_min*factor, x_max*factor, 
    y_max*factor)

########################################################################
#                              Utilities                               #
########################################################################

# Defines a function to convert mm to points

def milimeters_to_points(x):

    return (x/25.4)*72

# Defines a function to convert mm to points

def points_to_milimeters(x):

    return (x/72)*25.4