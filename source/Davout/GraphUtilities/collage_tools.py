# Routine to make collages

import numpy as np

from copy import deepcopy

import matplotlib

import matplotlib.pyplot as plt

from collections import OrderedDict

from matplotlib.transforms import Bbox

from ..PythonicUtilities import path_tools

from ..PythonicUtilities.file_handling_tools import txt_toList, list_toTxt, save_string_into_txt

from .tool_box import collage_classes

from .tool_box.arrows_lines_tools import plot_arrows_and_lines

from .tool_box.text_excerpt_tools import plot_text_excerpts

from .tool_box.image_tools import plot_images

from .tool_box.box_tools import plot_boxes

from .tool_box.perspective_tools import perspective_lines_from_vanishing_points

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
False):
    
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

        # Verifies if boxes list is a string. If so, it is a file path 
        # with the list to be read

        if isinstance(boxes_list, str):

            read_boxes_list = txt_toList(boxes_list, parent_path=
            input_path)

            general_axes, depth_order = plot_boxes(general_axes, 
            read_boxes_list, colors_class, line_style_class, 
            alignments_class, depth_order)

        # Otherwise, uses it plainly

        else:

            # If the given list is to be saved as a txt file

            if save_lists_to_txt:

                list_toTxt(boxes_list, "boxes_list", parent_path=
                input_path)

            # Plots the boxes

            general_axes, depth_order = plot_boxes(general_axes, 
            boxes_list, colors_class, line_style_class, 
            alignments_class, depth_order)

    # Verifies if the dictionary of input figures is not None

    if input_image_list is not None:

        print("#######################################################"+
        "#################\n#                         Insertion of fig"+
        "ures                         #\n#############################"+
        "###########################################\n")

        # Verifies if images list is a string. If so, it is a file path 
        # with the list to be read

        if isinstance(input_image_list, str):

            read_input_image_list = txt_toList(input_image_list, 
            parent_path=input_path)

            general_axes, depth_order = plot_images(general_axes, 
            read_input_image_list, alignments_class, input_path, verbose, 
            depth_order)

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

        print("#######################################################"+
        "#################\n#                        Making of text ex"+
        "cerpts                       #\n#############################"+
        "###########################################\n")

        # Verifies if text list is a string. If so, it is a file path 
        # with the list to be read

        if isinstance(input_text_list, str):

            read_input_text_list = txt_toList(input_text_list, 
            parent_path=input_path)

            # Plots the text excerpts

            collage, depth_order = plot_text_excerpts(collage,
            read_input_text_list, alignments_class, 
            layout_width_milimeters, layout_height_milimeters, verbose, 
            depth_order)

        else:

            # If the given list is to be saved as a txt file

            if save_lists_to_txt:

                list_toTxt(input_text_list, "input_text_list", 
                parent_path=input_path)

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

        # Verifies if arrows list is a string. If so, it is a file path 
        # with the list to be read

        if isinstance(arrows_and_lines_list, str):

            read_arrows_and_lines_list = txt_toList(
            arrows_and_lines_list, parent_path=input_path)

            # Plots the list of lines or arrows

            general_axes, depth_order = plot_arrows_and_lines(
            general_axes, read_arrows_and_lines_list, colors_class, 
            line_style_class, arrow_style_class, tolerance, depth_order)

        else:

            # If the given list is to be saved as a txt file

            if save_lists_to_txt:

                list_toTxt(arrows_and_lines_list, "arrows_and_lines_li"+
                "st", parent_path=input_path)

            # Plots the list of lines or arrows

            general_axes, depth_order = plot_arrows_and_lines(
            general_axes, arrows_and_lines_list, colors_class,
            line_style_class, arrow_style_class, tolerance, depth_order)

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

            # Gets the limits of the bounding box and converts to mili-
            # meters

            factor = 25.4

            x_min, y_min = bbox_inches.x0*factor, bbox_inches.y0*factor

            x_max, y_max = bbox_inches.x1*factor, bbox_inches.y1*factor

            print("The limits of the bounding box are:\n"+str(x_min)+
            " <= x <= "+str(x_max)+"\n"+str(y_min)+" <= y <= "+str(y_max
            )+"\n")

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

                extension_length = ((rule_fontsize/72)*25.4+
                rule_number_offset)

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
                x_min, x_max, y_min, y_max)

            # Verifies if an interactive window is to be shown

            if interactive_preview:

                create_interactive_window(general_axes, collage, 0.0, 
                layout_width_milimeters, 0.0, layout_height_milimeters, 
                x_min, x_max, y_min, y_max, input_path)

            # Saves the figure with this bounding box in inches

            plt.savefig(output_file, bbox_inches=bbox_inches, pad_inches=
            0, dpi=dpi)

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

                rule_number_offset = ((rule_fontsize/72)*25.4+
                rule_number_offset)

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

            # Verifies if an interactive window is to be shown

            if interactive_preview:

                create_interactive_window(general_axes, collage, 0.0, 
                layout_width_milimeters, 0.0, layout_height_milimeters, 
                x_min, x_max, y_min, y_max, input_path)

            print("There is no bounding box to shrink and pad.\nSaves "+
            "at "+str(output_file)+"\n")

            plt.savefig(output_file, dpi=dpi)

    else:

        # Sets the minima and maxima of the dimensions

        x_min = 0.0
        
        y_min = 0.0

        x_max = layout_width_milimeters

        y_max = layout_height_milimeters

        # If the grid is to be plotted, annotates the grid ruler

        if add_overlaying_grid:

            # Updates the number offset to make the number appear

            rule_number_offset = ((rule_fontsize/72)*25.4+
            rule_number_offset)

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

        # Verifies if an interactive window is to be shown

        if interactive_preview:

            create_interactive_window(general_axes, collage, 0.0, 
            layout_width_milimeters, 0.0, layout_height_milimeters, 
            x_min, x_max, y_min, y_max, input_path)

        print("Saves as it is at "+str(output_file)+"\n")

        plt.savefig(output_file, dpi=dpi)

    plt.close()

# Defines a function to create an interactive window

def create_interactive_window(general_axes, collage, old_x_min, 
old_x_max, old_y_min, old_y_max, new_x_min, new_x_max, new_y_min, 
new_y_max, input_path):

    # Zoom axes to the bounding box

    general_axes.set_xlim(new_x_min, new_x_max)

    general_axes.set_ylim(new_y_min, new_y_max)

    # Redraw before opening interactive window

    collage.canvas.draw_idle()

    # Initializes a list of points

    points_list = []

    # Defines a function to detect the mouse motion

    def on_move(event):

        if event.inaxes==general_axes and (event.xdata is not None):

            # Prints coordinates to terminal continuously

            print(f"[MOVE] x = {event.xdata:.2f}, y = {event.ydata:.2f}", 
            end="\r")

    # Defines a function to detect clicking of the mouse

    def on_click(event):

        if event.inaxes == general_axes and event.xdata is not None:

            print(f"[CLICK] x = {event.xdata:.4f}, y = {event.ydata:.4f}")

            # Saves the point into the list

            points_list.append([event.xdata, event.ydata])

    # Defines a function to detect the press of the ENTER key

    def on_key(event):

        nonlocal points_list

        if event.key == "enter":

            plt.close(collage)

        # Uses the key t to get string from the user

        if event.key == "t":

            list_name = input("\nType the name of this list: ")

            print("The user selected the name '"+str(list_name)+"'")

            # Concatenates it to the list of points

            data_list = list_name+"="+str(points_list)

            # Reads the already saved data

            saved_string = ""

            try:

                with open(input_path+"//preview_data.txt", "r", encoding=
                "utf-8") as infile:

                        saved_string = infile.read()

            except:

                pass

            # Adds the data

            saved_string += "\n\n"+data_list

            # Rewrites it

            save_string_into_txt(saved_string, "preview_data.txt", 
            parent_path=input_path)

            # Cleans the list of points

            points_list = []

    # Accesses the collage

    collage.canvas.mpl_connect("motion_notify_event", on_move)

    collage.canvas.mpl_connect("button_press_event", on_click)

    collage.canvas.mpl_connect("key_press_event", on_key)

    print("\nInteractive preview enabled (terminal output).")

    print("Move mouse → see coordinates")

    print("Click → print coordinates")

    print("Press key T → type the name of the list of points")

    print("Press ENTER → continue and save image\n")

    # Shows the image

    plt.show()

    # After closing preview, restore original limits

    general_axes.set_xlim(old_x_min, old_x_max)

    general_axes.set_ylim(old_y_min, old_y_max)