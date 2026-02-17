# Routine to make collages

import numpy as np

from copy import deepcopy

import matplotlib

import matplotlib.pyplot as plt

from collections import OrderedDict

from matplotlib.transforms import Bbox

from ..PythonicUtilities import path_tools

from ..PythonicUtilities.file_handling_tools import txt_toList, list_toTxt

from .tool_box import collage_classes

from .tool_box.arrows_lines_tools import plot_arrows_and_lines

from .tool_box.text_excerpt_tools import plot_text_excerpts

from .tool_box.image_tools import plot_images

from .tool_box.box_tools import plot_boxes

from .tool_box.perspective_tools import perspective_lines_from_vanishing_points

from .tool_box.interactivity_tools import create_interactive_window

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
boxes_list=None, arrows_and_lines_list=None, dpi=300, verbose=False, 
aspect_ratio='auto', adjustable=None, layout_width_milimeters=210.0, 
layout_height_milimeters=297.0, add_overlaying_grid=False, tolerance=
1E-5, grid_annotation_length=10, rule_fontsize=6, rule_number_offset=0.5,
vanishing_points_list=None, save_lists_to_txt=False, interactive_preview=
False, arrows_and_lines_file="arrows_and_lines_list"):
    
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

    # Initializes a flag to tell if the process of redrawing is to be
    # carried out iteratively

    flag_redraw = True 

    while flag_redraw:

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

            if verbose:

                print("###############################################"+
                "#########################\n#                         "+
                "   Making of boxes                           #\n#####"+
                "#####################################################"+
                "##############\n")

            # Verifies if boxes list is a string. If so, it is a file 
            # path with the list to be read

            if isinstance(boxes_list, str):

                read_boxes_list = txt_toList(boxes_list, parent_path=
                input_path)

                general_axes, depth_order = plot_boxes(general_axes, 
                read_boxes_list, colors_class, line_style_class, 
                alignments_class, depth_order, verbose=verbose)

            # Otherwise, uses it plainly

            else:

                # If the given list is to be saved as a txt file

                if save_lists_to_txt:

                    list_toTxt(boxes_list, "boxes_list", parent_path=
                    input_path)

                # Plots the boxes

                general_axes, depth_order = plot_boxes(general_axes, 
                boxes_list, colors_class, line_style_class, 
                alignments_class, depth_order, verbose=verbose)

        # Verifies if the dictionary of input figures is not None

        if input_image_list is not None:

            if verbose:

                print("###############################################"+
                "#########################\n#                         "+
                "Insertion of figures                         #\n#####"+
                "#####################################################"+
                "##############\n")

            # Verifies if images list is a string. If so, it is a file 
            # path with the list to be read

            if isinstance(input_image_list, str):

                read_input_image_list = txt_toList(input_image_list, 
                parent_path=input_path)

                general_axes, depth_order = plot_images(general_axes, 
                read_input_image_list, alignments_class, input_path, 
                verbose, depth_order)

            else:

                # If the given list is to be saved as a txt file

                if save_lists_to_txt:

                    list_toTxt(input_image_list, "input_image_list", 
                    parent_path=input_path)

                # Plots the images

                general_axes, depth_order = plot_images(general_axes, 
                input_image_list, alignments_class, input_path, verbose, 
                depth_order)

        # Verifies if the list of input text excerpts is not None

        if input_text_list is not None:

            if verbose:

                print("###############################################"+
                "#########################\n#                        M"+
                "aking of text excerpts                       #\n#####"+
                "#####################################################"+
                "##############\n")

            # Verifies if text list is a string. If so, it is a file 
            # path with the list to be read

            if isinstance(input_text_list, str):

                read_input_text_list = txt_toList(input_text_list, 
                parent_path=input_path)

                # Plots the text excerpts

                general_axes, depth_order = plot_text_excerpts(
                general_axes, read_input_text_list, alignments_class, 
                verbose, depth_order)

            else:

                # If the given list is to be saved as a txt file

                if save_lists_to_txt:

                    list_toTxt(input_text_list, "input_text_list", 
                    parent_path=input_path)

                # Plots the text excerpts

                general_axes, depth_order = plot_text_excerpts(
                general_axes, input_text_list, alignments_class, verbose, 
                depth_order)

        # Verifies if the list of arrows is not None

        if arrows_and_lines_list is not None:

            if verbose:

                print("###############################################"+
                "#########################\n#                      Mak"+
                "ing of arrows and lines                      #\n#####"+
                "#####################################################"+
                "##############\n")

            # Verifies if arrows list is a string. If so, it is a file 
            # path with the list to be read

            if isinstance(arrows_and_lines_list, str):

                read_arrows_and_lines_list = txt_toList(
                arrows_and_lines_list, parent_path=input_path)

                # Plots the list of lines or arrows

                general_axes, depth_order = plot_arrows_and_lines(
                general_axes, read_arrows_and_lines_list, colors_class, 
                line_style_class, arrow_style_class, tolerance, 
                depth_order, verbose=verbose)

            else:

                # If the given list is to be saved as a txt file

                if save_lists_to_txt:

                    list_toTxt(arrows_and_lines_list, 
                    arrows_and_lines_file, parent_path=input_path)

                # Plots the list of lines or arrows

                general_axes, depth_order = plot_arrows_and_lines(
                general_axes, arrows_and_lines_list, colors_class,
                line_style_class, arrow_style_class, tolerance, 
                depth_order, verbose=verbose)

        # Verifies if an overlaying grid is to be added

        if add_overlaying_grid:

            # Sets the lines thickness. The keys are the distance in mi-
            # limeters from the last line to the next of the same kind. 
            # The values are the line thickness

            lines_thickness = OrderedDict([(1, 
            collage_classes.milimeters_to_points(0.05)), (10, 
            collage_classes.milimeters_to_points(0.18)), (50, 
            collage_classes.milimeters_to_points(0.3))])

            # Plots the vertical lines

            for x in range(int(round(layout_width_milimeters))+1):

                # Initializes the thickness

                thickness = 0.1

                # Gets the thickness of the line

                for distance, thickness_trial in lines_thickness.items():

                    # Verifies if the division by the distance has no 
                    # rest

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

                    # Verifies if the division by the distance has no 
                    # rest

                    if y%distance==0:

                        thickness = deepcopy(thickness_trial)

                # Plots the line

                general_axes.plot([0, layout_width_milimeters], [y, y], 
                color="black", linewidth=thickness, zorder=depth_order)

        # Saves the figure

        print("#######################################################"+
        "#################\n#                                Saving   "+
        "                             #\n#############################"+
        "###########################################\n")

        if no_padding:

            # Gets what has been drawn only

            collage.canvas.draw_idle()

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

                print("There is a bounding box to shrink and pad.\nSav"+
                "es at "+str(output_file)+"\n")

                bounding_box = Bbox.union(bounding_box)

                # Converts to inches

                bbox_inches = bounding_box.transformed(
                collage.dpi_scale_trans.inverted())

                # Gets the limits of the bounding box and converts to 
                # milimeters

                factor = 25.4

                x_min, y_min = bbox_inches.x0*factor, (bbox_inches.y0*
                factor)

                x_max, y_max = bbox_inches.x1*factor, (bbox_inches.y1*
                factor)

                print("The limits of the bounding box are:\n"+str(x_min
                )+" <= x <= "+str(x_max)+"\n"+str(y_min)+" <= y <= "+str(
                y_max)+"\n")

                # If the grid is to be plotted, annotates the grid ruler

                if add_overlaying_grid:

                    # Adds the x ticks

                    for x in range(int(np.floor(x_min/grid_annotation_length
                    )*grid_annotation_length), int(np.ceil(x_max/
                    grid_annotation_length)*grid_annotation_length)+1, 
                    grid_annotation_length):
                        
                        general_axes.text(x, y_min-rule_number_offset, f"{x}", 
                        ha='right', va='top', fontsize=rule_fontsize)

                    # Adds the y ticks

                    for y in range(int(np.floor(y_min/grid_annotation_length
                    )*grid_annotation_length), int(np.ceil(y_max/
                    grid_annotation_length)*grid_annotation_length)+1, 
                    grid_annotation_length):
                        
                        general_axes.text(x_min-rule_number_offset, y, f"{y}", 
                        ha='right', va='top', fontsize=rule_fontsize)

                    # Updates the bounding box to include the numbers

                    extension_length = (collage_classes.points_to_milimeters(
                    rule_fontsize)+rule_number_offset)

                    factor = 1/25.4

                    bbox_inches = Bbox.from_extents((x_min-(extension_length
                    *2))*factor, (y_min-extension_length)*factor, x_max*
                    factor, y_max*factor)

                # Verifies if a list of vanishing points has been given to 
                # construct perspective lines

                if vanishing_points_list:

                    # Calls the function to add the perspective lines

                    general_axes = perspective_lines_from_vanishing_points(
                    general_axes, vanishing_points_list, 0.1, depth_order, 
                    x_min, x_max, y_min, y_max, verbose=verbose)

                # Saves the figure with this bounding box in inches

                plt.savefig(output_file, bbox_inches=bbox_inches, 
                pad_inches=0, dpi=dpi)

                if verbose:

                    print("Finishes saving the figure at "+str(
                    output_file)+"\n")

                # Verifies if an interactive window is to be shown

                if interactive_preview:

                    flag_redraw = create_interactive_window(general_axes, 
                    collage, 0.0, layout_width_milimeters, 0.0, 
                    layout_height_milimeters, x_min, x_max, y_min, y_max, 
                    input_path, depth_order, arrows_and_lines_file, 
                    flag_redraw, verbose=verbose)

                # Otherwise, just makes the flag for redrawing False

                else:

                    flag_redraw = False

                    plt.close()

            # Otherwise, saves it plainly

            else:

                # Sets the minima and maxima of the dimensions

                x_min = 0.0
                
                y_min = 0.0

                x_max = layout_width_milimeters

                y_max = layout_height_milimeters

                # If the grid is to be plotted, annotates the grid ruler

                if add_overlaying_grid:

                    # Updates the number offset to make the number appear

                    rule_number_offset = (collage_classes.points_to_milimeters(
                    rule_fontsize)+rule_number_offset)

                    # Adds the x ticks

                    for x in range(int(np.floor(x_min/grid_annotation_length
                    )*grid_annotation_length), int(np.ceil(x_max/
                    grid_annotation_length)*grid_annotation_length)+1, 
                    grid_annotation_length):
                        
                        general_axes.text(x, min(y_min+rule_number_offset,
                        y_max), f"{x}", ha='right', va='top', fontsize=
                        rule_fontsize)

                    # Adds the y ticks

                    for y in range(int(np.floor(y_min/grid_annotation_length
                    )*grid_annotation_length), int(np.ceil(y_max/
                    grid_annotation_length)*grid_annotation_length)+1, 
                    grid_annotation_length):
                        
                        general_axes.text(min(x_min+rule_number_offset, x_max
                        ), y, f"{y}", ha='right', va='top', fontsize=
                        rule_fontsize)

                # Verifies if a list of vanishing points has been given to 
                # construct perspective lines

                if vanishing_points_list:

                    # Calls the function to add the perspective lines

                    general_axes = perspective_lines_from_vanishing_points(
                    general_axes, vanishing_points_list, 0.1, depth_order, 
                    x_min, x_max, y_min, y_max)

                # Saves the figure

                print("There is no bounding box to shrink and pad.\nSaves "+
                "at "+str(output_file)+"\n")

                plt.savefig(output_file, dpi=dpi)

                # Verifies if an interactive window is to be shown

                if interactive_preview:

                    flag_redraw = create_interactive_window(general_axes, 
                    collage, 0.0, layout_width_milimeters, 0.0, 
                    layout_height_milimeters, x_min, x_max, y_min, y_max, 
                    input_path, depth_order, arrows_and_lines_file, 
                    flag_redraw, verbose=verbose)

                # Otherwise, just makes the flag for redrawing False

                else:

                    flag_redraw = False

                    plt.close()

        else:

            # Sets the minima and maxima of the dimensions

            x_min = 0.0
            
            y_min = 0.0

            x_max = layout_width_milimeters

            y_max = layout_height_milimeters

            # If the grid is to be plotted, annotates the grid ruler

            if add_overlaying_grid:

                # Updates the number offset to make the number appear

                rule_number_offset = (collage_classes.points_to_milimeters(
                rule_fontsize)+rule_number_offset)

                # Adds the x ticks

                for x in range(int(np.floor(x_min/grid_annotation_length)*
                grid_annotation_length), int(np.ceil(x_max/
                grid_annotation_length)*grid_annotation_length)+1, 
                grid_annotation_length):
                    
                    general_axes.text(x, min(y_min+rule_number_offset, y_max
                    ), f"{x}", ha='right', va='top', fontsize=rule_fontsize)

                # Adds the y ticks

                for y in range(int(np.floor(y_min/grid_annotation_length)*
                grid_annotation_length), int(np.ceil(y_max/
                grid_annotation_length)*grid_annotation_length)+1, 
                grid_annotation_length):
                    
                    general_axes.text(min(x_min+(2*rule_number_offset), 
                    x_max), y, f"{y}", ha='right', va='top', fontsize=
                    rule_fontsize)

            # Verifies if a list of vanishing points has been given to cons-
            # truct perspective lines

            if vanishing_points_list:

                # Calls the function to add the perspective lines

                general_axes = perspective_lines_from_vanishing_points(
                general_axes, vanishing_points_list, 0.1, depth_order, x_min, 
                x_max, y_min, y_max)

            # Saves the figure

            print("Saves as it is at "+str(output_file)+"\n")

            plt.savefig(output_file, dpi=dpi)

            # Verifies if an interactive window is to be shown

            if interactive_preview:

                flag_redraw = create_interactive_window(general_axes, 
                collage, 0.0, layout_width_milimeters, 0.0, 
                layout_height_milimeters, x_min, x_max, y_min, y_max, 
                input_path, depth_order, arrows_and_lines_file, 
                flag_redraw, verbose=verbose)

            # Otherwise, just makes the flag for redrawing False

            else:

                flag_redraw = False

                plt.close()