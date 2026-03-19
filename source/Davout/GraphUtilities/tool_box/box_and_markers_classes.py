# Routine to store the class of box and markers

import numpy as np

from matplotlib.patches import FancyBboxPatch, Ellipse, RegularPolygon

from matplotlib.transforms import Affine2D

from pathlib import Path as Path

from importlib import util

import sys

from copy import deepcopy

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

# Defines a class to set marker and box styles

class MarkerBoxStyles:

    def __init__(self):

        # Defines a dictionary of line styles
        
        self.available_marker_styles = {"rectangle": self.rectangle, 
        "ellipse": self.ellipse, "circle": self.ellipse, "polygon": 
        self.polygon}

    # Defines a function to get the style and retrieve the result based
    # on the pre-defined method

    def __call__(self, box_style, throw_error=True):
        
        # Verifies if the asked line style is available

        if box_style in self.available_marker_styles:

            # Saves the style

            self.shape = box_style

            # And gets the function to build the marker

            retrieved_marker_function = self.available_marker_styles[
            box_style]

            # Returns the function that builds the marker

            return retrieved_marker_function

        else:

            available_marker_styles = ""

            for marker in self.available_marker_styles:

                available_marker_styles += "\n'"+str(marker)+"'"

            error_message = ("'"+str(box_style)+"' is not a valid box "+
            "style or marker style. The only available options are: "+
            available_marker_styles)

            # If error is to be thrown

            if throw_error:

                raise NameError(error_message)
            
            # Otherwise, returns false

            return False
        
    # Defines a function simply to verify the existence of the box method

    def verify_marker_name(self, box_style, throw_error=False):
        
        # Verifies if the asked line style is available

        if box_style in self.available_marker_styles:

            # Returns the box style name

            return box_style

        else:

            available_marker_styles = ""

            for marker in self.available_marker_styles:

                available_marker_styles += "\n'"+str(marker)+"'"

            error_message = ("'"+str(box_style)+"' is not a valid box "+
            "style or marker style. The only available options are: "+
            available_marker_styles)

            # If error is to be thrown

            if throw_error:

                raise NameError(error_message)
            
            # Otherwise, returns false

            return False
        
    ####################################################################
    #                          Rectangular box                         #
    ####################################################################

    def rectangle(self, input_dictionary, position, width, height, 
    contour_thickness, contour_color, fill_color, boxstyle, line_style, 
    local_depth_order, origin_point, general_axes):

        new_box = FancyBboxPatch((position[0], position[1]), width, 
        height, linewidth=contour_thickness, edgecolor=contour_color, 
        facecolor=fill_color, boxstyle=boxstyle, linestyle=line_style, 
        zorder=local_depth_order)

        # Sets the rotation shift as the proper position

        rotation_shift = deepcopy(input_dictionary["position"])

        return new_box, rotation_shift
    
    ####################################################################
    #                           Elliptic box                           #
    ####################################################################
    
    def ellipse(self, input_dictionary, position, width, height, 
    contour_thickness, contour_color, fill_color, boxstyle, line_style, 
    local_depth_order, origin_point, general_axes):

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

        if self.shape=="circle":

            height = width*1.0

        new_box = Ellipse((x_center, y_center), width, height, 
        linewidth=contour_thickness, edgecolor=contour_color, 
        facecolor=fill_color, linestyle=line_style, zorder=
        local_depth_order)

        # Sets the rotation shift as the proper position

        rotation_shift = [x_center*1.0, y_center*1.0]

        return new_box, rotation_shift
        
    ####################################################################
    #                           Polygonal box                          #
    ####################################################################
    
    def polygon(self, input_dictionary, position, width, height, 
    contour_thickness, contour_color, fill_color, boxstyle, line_style, 
    local_depth_order, origin_point, general_axes):

        # Verifies if there is a key for the number of sides

        if not ("number of sides" in input_dictionary):

            raise KeyError("'polygon' shape has been asked to crea"+
            "te a box, but no 'number of sides' key has been provi"+
            "ded")
        
        number_of_sides = input_dictionary["number of sides"]

        # Verifies if the number of sides is integer

        if not isinstance(number_of_sides, int):

            raise TypeError("'number of sides' to make a polygon m"+
            "ust be an integer. Currently, it is: "+str(
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

        new_box = RegularPolygon((x_center, y_center), radius=width*
        0.5, numVertices=number_of_sides, linewidth=
        contour_thickness, edgecolor=contour_color, facecolor=
        fill_color, linestyle=line_style, zorder=local_depth_order)

        # Stretches the polygon to match the required height. Firs-
        # tly, translates to the origin, stretches, and, then, trans-
        # lates back to the correct position

        stretch_transform = Affine2D().translate(-x_center, -y_center
        ).scale(1.0, height/width).translate(x_center, y_center)

        new_box.set_transform(stretch_transform+
        general_axes.transData)

        # Sets the rotation shift as the proper position

        rotation_shift = [x_center*1.0, y_center*1.0]

        return new_box, rotation_shift