# Routine to make collages

from copy import deepcopy

import matplotlib

import matplotlib.pyplot as plt

from collections import OrderedDict

from matplotlib.transforms import Bbox

from ..PythonicUtilities import path_tools

from .tool_box import collage_classes

from .tool_box.arrows_lines_tools import plot_arrows_and_lines

from .tool_box.text_excerpt_tools import plot_text_excerpts

from .tool_box.image_tools import plot_images

from .tool_box.box_tools import plot_boxes

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
layout_height_milimeters=297.0, add_overlaying_grid=False, tolerance=1E-5):
    
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

        general_axes, depth_order = plot_boxes(general_axes, boxes_list, 
        colors_class, line_style_class, alignments_class, depth_order)

    # Verifies if the dictionary of input figures is not None

    if input_image_list is not None:

        print("#######################################################"+
        "#################\n#                         Insertion of fig"+
        "ures                         #\n#############################"+
        "###########################################\n")

        general_axes, depth_order = plot_images(general_axes, 
        input_image_list, alignments_class, input_path, verbose, 
        depth_order)

    # Verifies if the list of input text excerpts is not None

    if input_text_list is not None:

        print("#######################################################"+
        "#################\n#                        Making of text ex"+
        "cerpts                       #\n#############################"+
        "###########################################\n")

        # Plots the text excerpts

        collage, depth_order = plot_text_excerpts(collage,
        input_text_list, alignments_class, layout_width_milimeters, 
        layout_height_milimeters, verbose, depth_order)

    # Verifies if the list of arrows is not None

    if arrows_and_lines_list is not None:

        print("#######################################################"+
        "#################\n#                      Making of arrows an"+
        "d lines                      #\n#############################"+
        "###########################################\n")

        # Plots the list of lines or arrows

        general_axes, depth_order = plot_arrows_and_lines(general_axes, 
        arrows_and_lines_list, colors_class, line_style_class, 
        arrow_style_class, tolerance, depth_order)

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