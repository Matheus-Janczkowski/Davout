# Routine to store methods to plot arrows and lines using matplotlib

from copy import deepcopy

from matplotlib.patches import FancyBboxPatch, Ellipse, RegularPolygon

from matplotlib.transforms import Affine2D

# Defines a function to plot boxes from a list of instructions

def plot_boxes(general_axes, boxes_list, colors_class, line_style_class, 
alignments_class, depth_order, verbose=False):
    
    # Sets a list of necessary keys

    necessary_keys = ["contour color", "fill color", "contour thicknes"+
    "s", "position", "width", "height"]

    # Verifies if it is not a list

    if not isinstance(boxes_list, list):

        raise TypeError("'boxes_list' is not a list. It must be a list"+
        " where each item is a dictionary with the keys:\nObligatory:"+
        "\n'contour color': string with the color name or a RGB list f"+
        "or the contour\n'contour thickness': float with the contour t"+
        "hickness\n'fill color': string with the color name or a RGB l"+
        "ist for the fill (face)\n'position': list [x,y] position of t"+
        "he centroid\n'width: width of the box\nheight: height of the "+
        "box\n\nOptional:\n'contour transparency': transparency factor"+
        " (between 0 and 1) to set transparency of the contour line\n'"+
        "corner radius': the radius of the contour corners\n'contour s"+
        "tyle': 'solid', 'dashed', or 'dotted'\n'origin point': availa"+
        "ble options are 'centroid', 'bottom-left', 'bottom-right', 't"+
        "op-left', 'top-right'\n'rotation in degrees': float with rota"+
        "tion angle in degrees (from x axis counter-clockwise)\n'shape"+
        "': string with the name of the box shape\n'background transpa"+
        "rency': float value betweem 0 and 1 to make the background of"+
        " the box transparent")
    
    # Iterates through the elements

    for index, input_dictionary in enumerate(boxes_list):

        # Verifies if this element is a dictionary

        if not isinstance(input_dictionary, dict):

            raise TypeError("The "+str(index+1)+"-th element of the 'b"+
            "oxes_list' is not a dictionary. It must be a dictionary w"+
            "ith the keys:\nObligatory:\n'contour color': string with "+
            "the color name or a RGB list for the contour\n'contour th"+
            "ickness': float with the contour thickness\n'fill color':"+
            " string with the color name or a RGB list for the fill (f"+
            "ace)\n'position': list [x,y] position of the centroid\n'w"+
            "idth: width of the box\nheight: height of the box\n\nOpti"+
            "onal:\n'contour transparency': transparency factor (betwe"+
            "en 0 and 1) to set transparency of the contour line\n'cor"+
            "ner radius': the radius of the contour corners\n'contour "+
            "style': 'solid', 'dashed', or 'dotted'\n'origin point': a"+
            "vailable options are 'centroid', 'bottom-left', 'bottom-r"+
            "ight', 'top-left', 'top-right'\n'rotation in degrees': fl"+
            "oat with rotation angle in degrees (from x axis counter-c"+
            "lockwise)\n'shape': string with the name of the box shape"+
            "\n'background transparency': float value betweem 0 and 1 "+
            "to make the background of the box transparent")
        
        # Iterates through the necessary keys

        for key in necessary_keys:

            # Verifies the key existence

            if not (key in input_dictionary):

                names = ""

                for keys in necessary_keys:

                    names += "\n'"+str(keys)+"'"

                raise ValueError("The "+str(index+1)+"-th element 'box"+
                "es_list' does not have all the necessary keys, in par"+
                "ticular '"+str(key)+"'. Check the necessary keys:"+
                names)
            
        # Gets the colors

        contour_color = colors_class(input_dictionary["contour color"])

        fill_color = colors_class(input_dictionary["fill color"])

        # Gets the position

        position = deepcopy(input_dictionary["position"])

        # Verifies if it is a list

        if not isinstance(position, list):

            raise TypeError("The "+str(index+1)+"-th element 'boxes_li"+
            "st' has at key 'position' a value that is not a list. It "+
            "must be a list with [x, y] coordinates. Currently, it is:"+
            "\n"+str(position))
        
        # Gets the size

        width = input_dictionary["width"]

        height = input_dictionary["height"]

        contour_thickness = input_dictionary["contour thickness"]

        # Verifies if they are float

        if ((not isinstance(width, float)) and (not isinstance(width, 
        int))) or ((not isinstance(height, float)) and (not isinstance(
        height, int))) or ((not isinstance(contour_thickness, float)
        ) and (not isinstance(contour_thickness, int))):

            raise TypeError("The "+str(index+1)+"-th element 'boxes_li"+
            "st' has at key 'width' a value that is not a float or at "+
            "key 'height', or at key 'contour thickness'. They must be"+
            " both float or integers. Currently, 'width' is: "+str(width
            )+"\n'height' is: "+str(height)+"\ncontour thickness: "+str(
            contour_thickness))
        
        # Converts contour thickness from milimeters to points

        contour_thickness = contour_thickness*(72.0/25.4)
        
        # Gets the transparency if it is

        if "contour transparency" in input_dictionary:

            # Adds a 0.0 to the fourth channel of RGBA
            
            contour_color.append(input_dictionary["contour transparency"]) 
        
        # Gets the transparency if it is for the background only

        if "background transparency" in input_dictionary:

            # Adds a 0.0 to the fourth channel of RGBA
            
            fill_color.append(input_dictionary["background transparency"]) 
        
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
            
            line_style = line_style_class(input_dictionary["contour st"+
            "yle"], contour_thickness, "contour style")
            
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

            new_box = FancyBboxPatch((position[0], position[1]), width, 
            height, linewidth=contour_thickness, edgecolor=
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

        else:

            raise NameError("'shape' has been given as '"+str(shape)+
            "', but this shape is not available. The available shapes "+
            "are:\n'rectangle'\n'ellipse'\n'circle'")

        # Verifies if there is any rotation

        if "rotation in degrees" in input_dictionary:

            # Gets the angle

            angle = input_dictionary["rotation in degrees"]

            # Verifies if it is a float

            if not isinstance(angle, float):

                raise TypeError("The "+str(index+1)+"-th element 'boxe"+
                "s_list' has at key 'rotation in degrees' a value that"+
                " is not a float. Currently, 'rotation in degrees' is:"+
                " "+str(angle))
            
            # Sets the rotation and transforms the box. Rotates around 
            # the original point

            rotation = Affine2D().rotate_deg_around(rotation_shift[0], 
            rotation_shift[1], angle)

            new_box.set_transform(rotation+general_axes.transData)

        # Inserts the box into the figure

        if verbose:

            print("Adds box with '"+str(shape)+"' at point "+str(position
            )+" with 'origin point' as '"+str(origin_point)+"'\n")

        general_axes.add_patch(new_box)
    
    return general_axes, depth_order