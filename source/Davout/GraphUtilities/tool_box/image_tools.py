# Routine to store methods to plot figures using matplotlib

from copy import deepcopy

from matplotlib.transforms import Affine2D

from PIL import Image

from ...PythonicUtilities import path_tools

# Defines a function to plot figures within a matplotlib canvas

def plot_images(general_axes, input_image_list, alignments_class, 
input_path, verbose, depth_order):

    # Sets a list of necessary keys

    necessary_keys = ["file name", "position", "size"]

    # Verifies if it is not a list

    if not isinstance(input_image_list, list):

        raise TypeError("'input_image_list' is not a list. It must be "+
        "a list where each item is a dictionary with the keys:\nObliga"+
        "tory keys\n'file name': string with the file name\n'position'"+
        ": list with position coordinates, [x,y]\n'size': list with wi"+
        "dth and height\n\nOptional keys:\n'origin point': available o"+
        "ptions are 'centroid', 'bottom-left', 'bottom-right', 'top-le"+
        "ft', 'top-right'\n'rotation in degrees': float with rotation "+
        "angle in degrees (from x axis counter-clockwise)\n'trim trans"+
        "parent background': True if an image with transparent backgro"+
        "und is to be trimmed to get the non-transparent features only")
    
    # Iterates through the elements

    for index, input_dictionary in enumerate(input_image_list):

        # Verifies if this element is a dictionary

        if not isinstance(input_dictionary, dict):

            raise TypeError("The "+str(index+1)+"-th element of the 'i"+
            "nput_image_list' is not a dictionary. It must be a dictio"+
            "nary with the keys:\nObligatory keys\n'file name': string"+
            " with the file name\n'position': list with position coord"+
            "inates, [x,y]\n'size': list with width and height\n\nOpti"+
            "onal keys:\n'origin point': available options are 'centro"+
            "id', 'bottom-left', 'bottom-right', 'top-left', 'top-righ"+
            "t'\n'rotation in degrees': float with rotation angle in d"+
            "egrees (from x axis counter-clockwise)\n'trim transparent"+
            " background': True if an image with transparent backgroun"+
            "d is to be trimmed to get the non-transparent features on"+
            "ly")
        
        # Iterates through the necessary keys

        for key in necessary_keys:

            # Verifies the key existence

            if not (key in input_dictionary):

                names = ""

                for keys in necessary_keys:

                    names += "\n'"+str(keys)+"'"

                raise ValueError("The "+str(index+1)+"-th element 'inp"+
                "ut_image_list' does not have all the necessary keys, "+
                "in particular '"+str(key)+"'. Check the necessary key"+
                "s:"+names)
            
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

            raise TypeError("The "+str(index+1)+"-th element 'input_im"+
            "age_list' has at key 'position' a value that is not a lis"+
            "t. It must be a list with [x, y] coordinates. Currently, "+
            "it is:\n"+str(position))

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

            raise TypeError("The "+str(index+1)+"-th element 'input_im"+
            "age_list' has at key 'size' a value that is not a list no"+
            "r a float. It must be a list with [width_ratio, height_ra"+
            "tio] (fractions of the figure size) or a  float with the "+
            "size of the width only (aspect ratio is kept). Currently,"+
            " it is:\n"+str(size))
        
        # Verifies if the origin point is prescribed

        origin_point = 'centroid'

        if "origin point" in input_dictionary:

            origin_point = input_dictionary["origin point"]

        # Updates position using the alignment

        if verbose:

            print("The input image at '"+str(input_file_name)+"'\nhas "+
            "a size of "+str(size)+"\n")

        position = alignments_class(origin_point, position, size[0], 
        size[1])

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

            print("Adds figure at point "+str(position)+" with 'origin"+
            " point' as '"+str(origin_point)+"'\n")

        general_axes.imshow(input_image, extent=[position[0], position[0
        ]+size[0], position[1], position[1]+size[1]], origin='upper', 
        zorder=local_depth_order)

    # Returns the axes

    return general_axes, depth_order