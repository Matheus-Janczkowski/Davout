# Routine to store methods for interactivity with the user using mat-
# plotlib functionalities

import matplotlib as mpl

mpl.use('TkAgg')

from time import time

# Overrules some commands to free them for the curves defined here

mpl.rcParams['keymap.quit'] = []

mpl.rcParams['keymap.home'] = []

mpl.rcParams['keymap.save'] = []

mpl.rcParams['keymap.fullscreen'] = []

mpl.rcParams['keymap.grid'] = []

mpl.rcParams['keymap.back'] = []

mpl.rcParams['keymap.forward'] = []

import matplotlib.pyplot as plt

from ..tool_box import collage_classes

from ..tool_box import curves_setter

from ...PythonicUtilities.file_handling_tools import save_string_into_txt, txt_toList, list_toTxt

from ...PythonicUtilities.path_tools import verify_file_existence

# Sets a class of data that is communicated from and to the interactive
# window

class InteractiveWindowInfo:

    def __init__(self, flag_redraw, add_overlaying_grid, 
    vanishing_points_list, tolerance, perspective_lines_visualization):
        
        self.flag_redraw = flag_redraw
        
        self.add_overlaying_grid = add_overlaying_grid

        self.vanishing_points_list = vanishing_points_list

        self.tolerance = tolerance

        self.perspective_lines_visualization = (
        perspective_lines_visualization)

# Defines a function to create an interactive window

def create_interactive_window(general_axes, collage, old_x_min, 
old_x_max, old_y_min, old_y_max, new_x_min, new_x_max, new_y_min, 
new_y_max, input_path, depth_order, arrows_and_lines_file,
interactive_window_info, verbose=False):

    # Zoom axes to the bounding box

    general_axes.set_xlim(new_x_min, new_x_max)

    general_axes.set_ylim(new_y_min, new_y_max)

    # Redraw before opening interactive window

    collage.canvas.draw_idle()

    # Initializes a list of points

    points_list = []

    # Reads the list of arrows and lines

    arrows_and_lines_list = txt_toList(arrows_and_lines_file, input_path,
    do_not_raise_error=True)

    # If no arrows and lines has been read, makes it an empty list

    if arrows_and_lines_list is None:

        print("\nThe list of arrows and lines was not found at:\n"+str(
        input_path+"//"+arrows_and_lines_file)+"\nAn empty list was au"+
        "tomatically created\n")

        arrows_and_lines_list = []

    elif verbose:

        print("\nThe list of arrows and lines was successfully read.\n"+
        str(arrows_and_lines_list)+"\n")

    # Sets some commands for clicking and panning (dragging)

    is_panning = False

    # Sets the initial point and the initial limits

    last_mouse_position = (0, 0)

    # Function to handle the closing of the window by the X symbol. This
    # function assures that the flag for redrawing is deactivated

    manual_close = True

    def on_close(event):
        
        nonlocal interactive_window_info, points_list, manual_close

        # Treats the case where the event-loop is closed by the a key-
        # board shortcut

        if manual_close:

            print("[CLOSE] Window closed by user.", flush=True)
            
            interactive_window_info.flag_redraw = False
            
            points_list = []

        # If the user has indeed set to close by clicking on X

        else:

            # Resets the flag for the next iteration

            manual_close = True

    # When pressing the scroll

    def on_press(event):

        nonlocal is_panning, last_mouse_position

        # If the scroll (midlle button of the mouse is pressed)

        if event.inaxes==general_axes and event.button==2:

            is_panning = True

            last_mouse_position = (event.xdata, event.ydata)

        # Defines a function to detect clicking of the mouse (left click
        # only)

        elif event.button==1 and event.inaxes==general_axes and (
        event.xdata is not None):

            # Gets the clicked point coordinates

            x_coordinate = event.xdata

            y_coordinate = event.ydata

            print(f"[CLICK] x = {x_coordinate:.4f}, y = {(y_coordinate
            ):.4f}", flush=True)

            # Saves the point into the list

            points_list.append([x_coordinate, y_coordinate])

            # Marks the click point on the screen 

            general_axes.plot(x_coordinate, y_coordinate, marker="x",
            color="black", zorder=depth_order, markersize=
            collage_classes.milimeters_to_points(2.0), mew=2)

            # Refreshes the figure on the open screen

            collage.canvas.draw_idle()

    # If the scroll is released

    def on_release(event):

        nonlocal is_panning

        if event.button==2:

            # Makes the panning flag inactive

            is_panning = False
        
    # If the flag panning is active, drags the plot along

    last_zoom_time = 0

    flag_input = False

    def on_motion(event):

        nonlocal last_mouse_position, last_zoom_time, flag_input

        # Controls time to not fire the event at each time. This pre-
        # vents event flooding

        current_time = time()

        if (current_time-last_zoom_time)<0.03:

            # Ignores at each 30 miliseconds

            return
        
        last_zoom_time = current_time

        # Zooms in and out altering the limits of the axes

        if is_panning and event.inaxes==general_axes and (
        event.xdata is not None):
            
            # Gets the movement direction

            dx = last_mouse_position[0] - event.xdata

            dy = last_mouse_position[1] - event.ydata

            # Gets the current original limits of the plot bounding box

            x0, x1 = general_axes.get_xlim()

            y0, y1 = general_axes.get_ylim()

            # Updates the limits using the direction

            general_axes.set_xlim(x0+dx, x1+dx)

            general_axes.set_ylim(y0+dy, y1+dy)

            # Updates the mouse last position and redraws the canvas

            last_mouse_position = (event.xdata, event.ydata)

            collage.canvas.draw_idle()
            
        # Otherwise, simply detect the mouse position
        
        elif event.inaxes==general_axes and (event.xdata is not None
        ) and (not flag_input):

            # Prints coordinates to terminal continuously

            print(f"[MOVE] x = {event.xdata:.2f}, y = {event.ydata:.2f}", 
            end="\r", flush=True)

    # Defines a function to detect the press of the ENTER key

    def on_key(event):

        nonlocal points_list, general_axes, manual_close, arrows_and_lines_list, flag_input, interactive_window_info

        # If enter is pressed, saves the image and do not resume redraw-
        # ing

        if event.key=="enter":

            interactive_window_info.flag_redraw = False

            manual_close = False

            plt.close(collage)

        # If C is pressed, toggle the addition of grid

        elif event.key=="c":

            if interactive_window_info.add_overlaying_grid:

                print("Disables the overlaying grid\n")

                interactive_window_info.add_overlaying_grid = False 

            else:

                print("Enables the overlaying grid\n")

                interactive_window_info.add_overlaying_grid = True

        # If B is pressed, toggle the addition of the perspective lines

        elif event.key=="b":

            if interactive_window_info.perspective_lines_visualization:

                print("Disables the overlaying perspective lines\n")

                interactive_window_info.perspective_lines_visualization = False 

            else:

                print("Enables the overlaying perspective lines\n")

                interactive_window_info.perspective_lines_visualization = True

        # If V is pressed, adds a new point with lines connecting to the
        # vanishing points

        elif event.key=="v":

            if points_list:

                print("Adds new rays from the vanishing points to a se"+
                "lected point\nPress R to redraw the image if you want"+
                " to see the perspective lines immediatly")

                for new_point in points_list:

                    # Initializes a list of directions from the last ad-
                    # ded point to the provided vanishing points

                    rays_directions = []

                    # Iterates through the vanishing points

                    for vanishing_point in interactive_window_info.vanishing_points_list:

                        # Verifies if this is not another perspective 
                        # point

                        flag_get_ray = True

                        if ("type" in vanishing_point) and vanishing_point[
                        "type"]=="perspective point":
                            
                            flag_get_ray = False 

                        # Updates only if the point is a proper vanishing 
                        # point

                        if flag_get_ray:

                            # Gets the coordinates

                            ray_coordinates = vanishing_point["coordin"+
                            "ates"]

                            # Gets the directions

                            rays_directions.append([ray_coordinates[0]-
                            new_point[0], ray_coordinates[1]-new_point[1
                            ]])

                    # Assembles the dictionary of this ray point

                    perspective_point = {"coordinates": new_point, "ra"+
                    "ys central direction": rays_directions, "angle am"+
                    "plitude": 0.0, "number of rays": 2,"color": "black", 
                    "type": "perspective point"}

                    # Adds this dictionary to the list of vanishing 
                    # points

                    interactive_window_info.vanishing_points_list.append(
                    perspective_point)

                # Substitutes the X markers by square markers, and cleans 
                # the list of points

                points_list, general_axes = curves_setter.substitute_markers(
                points_list, general_axes, depth_order, collage)

            else:

                print("Tries to add new rays from the vanishing points"+
                " to a selected point. But there is no selected point\n")

        # If CTRL+V is pressed, deletes the last point with lines connec-
        # ting to the vanishing points

        elif event.key=="ctrl+v":

            if interactive_window_info.vanishing_points_list:

                removed = interactive_window_info.vanishing_points_list[-1]

                # Takes input

                flag_input = True 

                response = input("\nAre you sure you want to remove th"+
                "e last vanishing or perspective point? Type 'y' if so"+
                ": ")

                # If it is different than y, moves forward

                if response!="y":

                    print("\nCancels removing the following element:\n"+
                    str(removed))

                    flag_input = False 

                    return None
                
                # Removes the last point and disables the input flag

                interactive_window_info.vanishing_points_list = (
                interactive_window_info.vanishing_points_list[0:-1])

                print("Type R if you want to redraw the image immediat"+
                "ly")
                
                flag_input = False 

            else:

                print("Tries to remove vanishing points or perspective"+
                "points. But there is no such point to be deleted\n")

        # Otherwise, if 'r' is pressed, saves the drawing and redraws a-
        # gain

        elif event.key=="r":

            interactive_window_info.flag_redraw = True

            manual_close = False

            # Sets a variable informing to close the figure

            plt.close()

        # Uses the key t to get string from the user

        elif event.key=="t":

            list_name = input("\nType the name of this list: ")

            print("The user selected the name '"+str(list_name)+"'", 
            flush=True)

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

            output_file_name = "preview_data.txt"

            save_string_into_txt(saved_string, output_file_name, 
            parent_path=input_path)

            print("Saved the data points at '"+str(input_path)+"//"+
            output_file_name+"\n")

            # Substitutes the X markers by square markers, and cleans the
            # list of points

            points_list, general_axes = curves_setter.substitute_markers(
            points_list, general_axes, depth_order, collage)

        # Detects Ctrl+Z to erase the last point
         
        elif event.key=="ctrl+z":

            if points_list:

                removed = points_list.pop()
                
                print(f"[UNDO] removed {removed}", flush=True)

                # Removes the last plotted X marker
                
                if general_axes.lines:

                    general_axes.lines[-1].remove()

                collage.canvas.draw_idle()

            else:

                print("[UNDO] no points left", flush=True)

                # Removes the last plotted X marker
                
                if general_axes.lines:

                    print("But removes already saved markers", flush=
                    True)

                    general_axes.lines[-1].remove()

                    collage.canvas.draw_idle()

                else:

                    print("No markers left to delete", flush=True)

                print("", flush=True)

        # Detects Ctrl+X to erase the last curve
         
        elif event.key=="ctrl+x":

            if arrows_and_lines_list:

                removed = arrows_and_lines_list[-1]

                # Takes input

                flag_input = True 

                response = input("\nAre you sure you want to remove th"+
                "e last line? Type 'y' if so: ")

                # If it is different than y, moves forward

                if response!="y":

                    print("\nCancels removing the following element:\n"+
                    str(removed))

                    flag_input = False 

                    return None
                
                # Removes the last line and disables the input flag

                arrows_and_lines_list = arrows_and_lines_list[0:-1]
                
                flag_input = False 
                
                print(f"[UNDO] removed {removed}", flush= True)

                # Rewrites the arrows list

                list_toTxt(arrows_and_lines_list, arrows_and_lines_file,
                parent_path=input_path)

                # Triggers redrawing
                
                interactive_window_info.flag_redraw = True

                manual_close = False

                plt.close()

            else:

                print("[UNDO] no curves left", flush=True)

                print("", flush=True)

        # Verifies if one of the pre-fabricated curves were asked

        else:

            # Enables the flag input to disable the printing of MOVE

            flag_input = True

            points_list, general_axes, arrows_and_lines_list = curves_setter.set_curve(
            event.key, arrows_and_lines_list, arrows_and_lines_file, 
            input_path, depth_order, collage, points_list, general_axes,
            interactive_window_info.tolerance, 
            interactive_window_info.vanishing_points_list)

            # Disables the flag input again

            flag_input = False

    # Function to zoom in and out using mouse scroll

    def on_scroll(event):

        if event.inaxes != general_axes:

            return
        
        # Ctrl key must be pressed

        if not (event.key in ('control', 'ctrl')):

            return
        
        scale_factor = 1.2

        x_min, x_max = general_axes.get_xlim()

        y_min, y_max = general_axes.get_ylim()

        xdata, ydata = event.xdata, event.ydata

        # If zoom is in

        if event.button == 'up':

            factor = 1/scale_factor

        # Otherwise, if it is out

        elif event.button == 'down':

            factor = scale_factor

        else:

            factor = 1.0

        # Rescales the canvas

        general_axes.set_xlim([xdata-(xdata-x_min)*factor, xdata+(x_max-
        xdata)*factor])

        general_axes.set_ylim([ydata-(ydata-y_min)*factor, ydata+(y_max-
        ydata)*factor])

        collage.canvas.draw_idle()

    # Accesses the collage and states the events

    collage.canvas.mpl_connect("key_press_event", on_key)

    collage.canvas.mpl_connect("scroll_event", on_scroll)

    collage.canvas.mpl_connect("button_press_event", on_press)

    collage.canvas.mpl_connect("button_release_event", on_release)

    collage.canvas.mpl_connect("motion_notify_event", on_motion)

    collage.canvas.mpl_connect("close_event", on_close)

    print("\n1. Interactive preview enabled (terminal output).", flush=True)

    print("Move mouse ............................... => see coordinat"+
    "es", flush=True)

    print("Click .................................... => print coordin"+
    "ates", flush=True)

    print("\n2. Deletion:")

    print("Press CTRL+Z ............................. => delete the la"+
    "st point", flush=True)

    print("Press CTRL+X ............................. => delete the la"+
    "st line", flush=True)

    print("Press CTRL+V ............................. => delete the la"+
    "st perspective point with lines connecting to the vanishing point"+
    "s, or a vanishing point itself", flush=True)

    print("\n3. Zoom:")

    print("Press CTRL and scroll with the mouse ..... => zoom in and o"+
    "ut", flush=True)

    print("Press the mouse scroll and hold it pressed => drag the figu"+
    "re around", flush=True)

    print("\n4. Redrawing and saving:")

    print("Press key R .............................. => redraw and sa"+
    "ve the figure", flush=True)

    print("Press ENTER .............................. => save figure a"+
    "nd stops the drawing loop", flush=True)

    print("\n5. Data query:")

    print("Press key T .............................. => type the name"+
    " of the list of points", flush=
    True)

    print("\n6. Visibility:")

    print("Press key C .............................. => toggle the ad"+
    "dition or not of the grid", flush=True)

    print("Press key V .............................. => add a new per"+
    "spective point with lines connecting to the vanishing points", 
    flush=True)

    print("Press key B .............................. => toggle the ad"+
    "dition or not of the perspective lines", flush=True)

    print("\n7.1. Spline curves:")

    print("Press key Q .............................. => transform the"+
    " last points in a spline curve", flush=True)

    print("Press key W .............................. => transform the"+
    " last points in a closed spline curve", flush=True)

    print("Press key E .............................. => transform the"+
    " last points in an arrow with a spline stem", flush=True)

    print("\n7.2. Polygonal curves:")

    print("Press key A .............................. => transform the"+
    " last points in a polygonal curve", flush=True)

    print("Press key S .............................. => transform the"+
    " last points in a closed polygonal curve", flush=True)

    print("Press key D .............................. => transform the"+
    " last points in an arrow with a polygonal stem", flush=True)

    # Shows the image

    plt.show()

    # After closing preview, restore original limits

    general_axes.set_xlim(old_x_min, old_x_max)

    general_axes.set_ylim(old_y_min, old_y_max)

    return interactive_window_info