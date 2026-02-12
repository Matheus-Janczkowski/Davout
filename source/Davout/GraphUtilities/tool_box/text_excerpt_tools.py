# Routine to store methods to plot text excerpts using matplotlib

from copy import deepcopy

# Defines a function to plot excerpts of text from a list

def plot_text_excerpts(collage, input_text_list, alignments_class, 
layout_width_milimeters, layout_height_milimeters, verbose, depth_order):
        
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
                ":"+names)
            
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

        if not isinstance(font_size, int):

            raise TypeError("The "+str(index+1)+"-th element 'input_te"+
            "xt_list' has at key 'font size' a value that is not an in"+
            "teger. Currently, it is:\n"+str(font_size))
        
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

        # Verifies if a depth number has been given

        local_depth_order = deepcopy(depth_order)

        if "depth order" in input_dictionary:

            local_depth_order = input_dictionary["depth order"]

        else:

            # Updates the depth number

            depth_order += 1

        # Adds the text input

        if verbose:

            print("Adds text at point "+str(position)+" with 'origin p"+
            "oint' as '"+str(origin_point)+"'\n")

        collage.text(position[0]/layout_width_milimeters, position[1
        ]/layout_height_milimeters, input_text, fontsize=font_size, 
        ha=ha, va=va, rotation=angle, rotation_mode="anchor", zorder=
        local_depth_order)

    return collage, depth_order