# Routine to make collages

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, FancyBboxPatch

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

# Defines a function to create a collage using boxes

def create_box_collage(output_file, input_path=None, output_path=None,
no_padding=False, input_image_list=None, input_text_list=None, 
boxes_list=None, dpi=300):
    
    # Initializes the class of colors

    colors_class = ColorMiscellany()

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
            "d', 'dashed', or 'dotted'")
        
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
                "of the box")
            
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

            position = input_dictionary["position"]

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
            
            # Translates the position by the size

            position[0] -= 0.5*width

            position[1] -= 0.5*height

            # Verifies contour style

            line_style = '-'

            if "contour style" in input_dictionary:

                if input_dictionary["contour style"]=="solid":

                    line_style = '-'

                elif input_dictionary["contour style"]=="dashed":

                    line_style = '--'

                elif input_dictionary["contour style"]=="dotted":

                    line_style = ':'

                else:

                    raise NameError("The only 'contour style' availabl"+
                    "e are: 'solid', 'dashed', or 'dotted'")

            # Creates the rectangle

            if "corner radius" in input_dictionary:

                box_axes.add_patch(FancyBboxPatch((position[0], position[
                1]), width, height, linewidth=contour_thickness, edgecolor=
                contour_color, facecolor=fill_color, alpha=alpha,
                boxstyle=f"round,pad=0.0,rounding_size={input_dictionary[
                "corner radius"]}", linestyle=line_style))

            else:

                box_axes.add_patch(Rectangle((position[0], position[1]), 
                width, height, linewidth=contour_thickness, edgecolor=
                contour_color, facecolor=fill_color, alpha=alpha, 
                linestyle=line_style))

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
            "th and height (fractions of the figure size)")
        
        # Iterates through the elements

        for index, input_dictionary in enumerate(input_image_list):

            # Verifies if this element is a dictionary

            if not isinstance(input_dictionary, dict):

                raise TypeError("The "+str(index+1)+"-th element of th"+
                "e 'input_image_list' is not a dictionary. It must be "+
                "a dictionary with the keys:\n'file name': string with"+
                " the file name\n'position': list with position coordi"+
                "nates, [x,y]\n'size': list with width and height (fra"+
                "ctions of the figure size)")
            
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

            position = input_dictionary["position"]

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

            # Adds image panel

            image_canvas = collage.add_axes([position[0]-(0.5*size[0]), 
            position[1]-(0.5*size[1]), size[0], size[1]])

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

            raise TypeError("'input_text_list' is not a list. It must"+
            " be a list where each item is a dictionary with the keys:"+
            "\n'text': string with the text excerpt\n'position': list "+
            "with position coordinates, [x,y]\n'font size': integer")
        
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

            position = input_dictionary["position"]

            # Verifies if it is a list

            if not isinstance(position, list):

                raise TypeError("The "+str(index+1)+"-th element 'inpu"+
                "t_text_list' has at key 'position' a value that is no"+
                "t a list. It must be a list with [x, y] coordinates. "+
                "Currently, it is:\n"+str(position))
            
            # Gets the font size

            font_size = input_dictionary["font size"]

            # Verifies if it is an integer

            if not isinstance(font_size, int):

                raise TypeError("The "+str(index+1)+"-th element 'inpu"+
                "t_text_list' has at key 'font size' a value that is n"+
                "ot an integer. Currently, it is:\n"+str(font_size))

            # Adds the text input

            collage.text(*position, input_text, fontsize=font_size)

    # Saves the figure

    if no_padding:

        plt.savefig(output_file, bbox_inches="tight", pad_inches=0, dpi=
        dpi)

    else:

        plt.savefig(output_file, dpi=dpi)

    plt.close()