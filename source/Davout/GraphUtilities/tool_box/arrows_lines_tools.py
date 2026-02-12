# Routine to store methods to plot arrows and lines using matplotlib

import numpy as np

from copy import deepcopy

from matplotlib.patches import FancyArrowPatch, PathPatch

from matplotlib.path import Path

# Defines a function to plot arrows and lines from a list of instructions

def plot_arrows_and_lines(general_axes, arrows_and_lines_list, 
colors_class, line_style_class, arrow_style_class, tolerance, 
depth_order):

    # Sets a list of necessary keys

    necessary_keys = ["start point", "end point", "thickness"]

    # Verifies if it is not a list

    if not isinstance(arrows_and_lines_list, list):

        raise TypeError("'arrows_and_lines_list' is not a list. It mus"+
        "t be a list where each item is a dictionary with the keys:\nO"+
        "bligatory:\n'start point': list with [x,y] coordinates of the"+
        " starting point\n'end point':  list with [x,y] coordinates of"+
        " the end point\n'thickness': 'thickness': float value with th"+
        "e arrow line thickness\n\nOptional:\n'arrow style': string wi"+
        "th the style of the arrow stem and edge\n'spline points': lis"+
        "t of lists of point coordinates for a spline path, [[x1, y1],"+
        " [x2, y2], ..., [xn, yn]]\n'arrow head size': integer with th"+
        "e arrow head size\n'line style': string with the name of the "+
        "line style of the stem\n'polygonal points': list of lists of "+
        "point coordinates for a polygonal path, [[x1, y1], [x2, y2], "+
        "..., [xn, yn]]\n'closed path': True if the line path must be "+
        "closed")
    
    # Iterates through the elements

    for index, input_dictionary in enumerate(arrows_and_lines_list):

        # Verifies if this element is a dictionary

        if not isinstance(input_dictionary, dict):

            raise TypeError("The "+str(index+1)+"-th element of the 'a"+
            "rrows_and_lines_list' is not a dictionary. It must be a d"+
            "ictionary with the keys:\nObligatory:\n'start point': lis"+
            "t with [x,y] coordinates of the starting point\n'end poin"+
            "t':  list with [x,y] coordinates of the end point\n'thick"+
            "ness': 'thickness': float value with the arrow line thick"+
            "ness\n\nOptional:\n'arrow style': string with the style o"+
            "f the arrow stem and edge\n'spline points': list of lists"+
            " of point coordinates for a spline path, [[x1, y1], [x2, "+
            "y2], ..., [xn, yn]]\n'arrow head size': integer with the "+
            "arrow head size\n'line style': string with the name of th"+
            "e line style of the stem\n'polygonal points': list of lis"+
            "ts of point coordinates for a polygonal path, [[x1, y1], "+
            "[x2, y2], ..., [xn, yn]]\n'closed path': True if the line"+
            " path must be closed")
        
        # Iterates through the necessary keys

        for key in necessary_keys:

            # Verifies the key existence

            if not (key in input_dictionary):

                names = ""

                for keys in necessary_keys:

                    names += "\n'"+str(keys)+"'"

                raise ValueError("The "+str(index+1)+"-th element 'arr"+
                "ows_and_lines_list' does not have all the necessary k"+
                "eys, in particular '"+str(key)+"'. Check the necessar"+
                "y keys:"+names)
            
        # Gets the colors

        arrow_color = colors_class("black")

        if "color" in input_dictionary:

            arrow_color = colors_class(input_dictionary["color"])

        # Gets the start point

        start_point = input_dictionary["start point"]

        # Verifies if it is a list

        if not isinstance(start_point, list):

            raise TypeError("The "+str(index+1)+"-th element 'arrows_a"+
            "nd_lines_list' has at key 'start_point' a value that is n"+
            "ot a list. It must be a list with [x, y] coordinates. Cur"+
            "rently, it is:\n"+str(start_point))

        # Gets the end point

        end_point = input_dictionary["end point"]

        # Verifies if it is a list

        if not isinstance(end_point, list):

            raise TypeError("The "+str(index+1)+"-th element 'arrows_a"+
            "nd_lines_list' has at key 'end_point' a value that is not"+
            " a list. It must be a list with [x, y] coordinates. Curre"+
            "ntly, it is:\n"+str(end_point))
        
        # Gets the thickness

        thickness = input_dictionary["thickness"]

        # Verifies if they are float

        if (not isinstance(thickness, float)):

            raise TypeError("The "+str(index+1)+"-th element 'arrows_a"+
            "nd_lines_list' has at key 'thickness' a value that is not"+
            " a float. Currently, 'thickness': "+str(thickness))
        
        # Converts contour thickness from milimeters to points

        thickness = thickness*(72.0/25.4)

        # Verifies contour style

        line_style = '-'

        if "line style" in input_dictionary:
            
            line_style = line_style_class(input_dictionary["line style"
            ], thickness, "line style")

        # Verifies contour style

        arrow_style = '-|>'

        if "arrow style" in input_dictionary:

            # Gets the arrow style
            
            arrow_style = arrow_style_class(input_dictionary["arrow st"+
            "yle"])
        
        # Gets the arrow head size

        arrow_head_size = int(thickness*15)

        if "arrow head size" in input_dictionary:

            arrow_head_size = input_dictionary["arrow head size"]

            # Verifies if it is an integer

            if (not isinstance(arrow_head_size, float)) and (
            not isinstance(arrow_head_size, int)):

                raise TypeError("The "+str(index+1)+"-th element 'arro"+
                "ws_and_lines_list' has at key 'arrow head size' a val"+
                "ue that is not an integer nor a float. Currently, 'ar"+
                "row head size': "+str(arrow_head_size))
            
            # Converts the size from milimeters to points

            arrow_head_size = arrow_head_size*(72/25.4)

        # Verifies if the line must be a closed path

        closed_path = False

        if ("closed path" in input_dictionary) and input_dictionary["c"+
                "losed path"]:
            
            # Updates the flag that tells if the path is closed

            closed_path = True

            # Verifies if the start and end points match

            if np.linalg.norm(np.array(start_point)-np.array(
            end_point))>tolerance:
                
                raise ValueError("'closed path' is True, but the 'star"+
                "t point' and 'end point' are not the same given a tol"+
                "erance of "+str(tolerance)+". They must be the same p"+
                "oint, at least numerically. Check the:\nstart point: "+
                str(start_point)+"\nend point: "+str(end_point))
            
        # Verifies if the arrow stem follows a spline curve

        codes = [Path.MOVETO] 

        vertices = [start_point]

        if "spline points" in input_dictionary:

            # Gets the spline points and verifies if they are a list

            spline_points = input_dictionary["spline points"]

            if not isinstance(spline_points, list):

                raise TypeError("'spline points' in 'arrows_and_lines_"+
                "list' must be a list of lists [[x1, y1], [x2, y2], .."+
                "., [xn, yn]]. Currently, it is:\n"+str(spline_points))

            # Otherwise, uses cubic splines

            else:

                print("Inserts arrow with cubic Bézier curve\n")
            
                # Adds the starting point to the spline points

                full_spline_points = [start_point, *spline_points, 
                end_point]
            
                # Iterates through the points to verify consistency

                for index_point in range(len(full_spline_points)-1):

                    # Gets the point and verifies it

                    point_1 = full_spline_points[index_point]

                    if not isinstance(point_1, list) or len(point_1
                    )!=2:

                        raise ValueError("The "+str(index_point+1)+"-t"+
                        "h point of 'spline points' is not a list or i"+
                        "ts length is not 2. Check it out:\n"+str(
                        point_1))
                    
                    # Gets the four points for the Catmull-Rom conversion 
                    # and converts them to numpy

                    point_0 = None

                    # If the path is closed, gets the end point as refe-
                    # rence. Or, if it is one of the other points, re-
                    # trieves the last

                    if index_point>0:

                        point_0 = np.array(full_spline_points[
                        index_point-1])

                    elif closed_path:

                        point_0 = np.array(full_spline_points[
                        -2])

                    else:

                        point_0 = np.array(full_spline_points[
                        index_point])

                    point_1 = np.array(point_1)

                    point_2 = np.array(full_spline_points[
                    index_point+1])

                    point_3 = None

                    if index_point+2<len(full_spline_points):

                        point_3 = np.array(full_spline_points[
                        index_point+2])

                    # If the path is closed, gets the start point as re-
                    # ference

                    elif closed_path:

                        point_3 = np.array(full_spline_points[1])

                    # Or, if it is one of the other points, retrieves 
                    # the next one

                    else:

                        point_3 = np.array(full_spline_points[
                        index_point+1])

                    # Subtracts them

                    b1 = point_1+((point_2-point_0)/6)

                    b2 = point_2-((point_3-point_1)/6)

                    b3 = point_2

                    # Adds these points to the vertices

                    vertices.extend([b1, b2, b3])

                    # Appends the instrictions to code

                    codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

        # Verifies if there are points to draw a polygonal line

        elif "polygonal points" in input_dictionary:

            # Gets the spline points and verifies if they are a list

            polygonal_points = input_dictionary["polygonal points"]

            if not isinstance(polygonal_points, list):

                raise TypeError("'polygonal points' in 'arrows_and_lin"+
                "es_list' must be a list of lists [[x1, y1], [x2, y2],"+
                " ..., [xn, yn]]. Currently, it is:\n"+str(
                polygonal_points))

            # Otherwise, uses a polygonal path

            else:

                print("Inserts arrow with polygonal curve\n")
            
                # Adds the starting point to the spline points

                vertices = [start_point, *polygonal_points, end_point]

                # Iterates through the points

                for point in range(len(vertices)-1):

                    codes.append(Path.LINETO)

        # Otherwise, just finish the path

        else:

            print("Inserts arrow with linear stem\n")

            # Appends the code instruction to create the last segment
            
            codes.append(Path.LINETO)
            
            # Appends the end point and creates the path of the arrow

            vertices.append(end_point)

        # If the path is closed, adds the proper command

        if closed_path:

            # Repeats the first point as the end

            vertices.append(vertices[-1])

            # Adds the instruction to close the path

            codes.append(Path.CLOSEPOLY)

        # Creates the arrow head path and for the arrow stem separately

        code_head = [Path.MOVETO, Path.LINETO]

        arrow_head_path = Path(np.array(vertices[-2:len(vertices)]), 
        codes=code_head)

        arrow_stem_path = Path(np.array(vertices), codes=codes)

        # Verifies if the path is to be filled with color]

        if ("fill path with color" in input_dictionary) and (
        input_dictionary["fill path with color"]):
            
            # Verifies if the path is closed

            if not closed_path:

                raise ValueError("'fill path with color' was given as "+
                str(input_dictionary["fill path with color"])+". But t"+
                "he path is not closed. Make sure to type'closed path'"+
                "=True")
            
            # Takes the color

            fill_path_color = colors_class(input_dictionary["fill path"+
            " with color"])

            # Adds the path to the canvas as a patch

            general_axes.add_patch(PathPatch(arrow_stem_path, facecolor=
            fill_path_color, edgecolor=arrow_color, zorder=
            local_depth_order, linewidth=0))

        # Verifies if a depth number has been given

        local_depth_order = deepcopy(depth_order)

        if "depth order" in input_dictionary:

            local_depth_order = input_dictionary["depth order"]

        else:

            # Updates the depth number

            depth_order += 1

        # Adds the arrow stem
        
        general_axes.add_patch(FancyArrowPatch(path=arrow_stem_path, 
        arrowstyle='-', color=arrow_color, linewidth=thickness, zorder=
        local_depth_order, linestyle=line_style))

        # Adds the arrow head

        if arrow_style!='-':
        
            general_axes.add_patch(FancyArrowPatch(path=arrow_head_path, 
            arrowstyle=arrow_style, color=arrow_color, mutation_scale=
            arrow_head_size, zorder=local_depth_order, linestyle='-', 
            linewidth=max(thickness/3, 1), joinstyle='miter'))

    # Returns the axes canvas

    return general_axes, depth_order