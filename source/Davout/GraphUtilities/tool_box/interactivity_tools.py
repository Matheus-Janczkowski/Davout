# Routine to store methods for interactivity with the user using mat-
# plotlib functionalities

import matplotlib.pyplot as plt

from ..tool_box import collage_classes

from ...PythonicUtilities.file_handling_tools import save_string_into_txt

# Defines a function to create an interactive window

def create_interactive_window(general_axes, collage, old_x_min, 
old_x_max, old_y_min, old_y_max, new_x_min, new_x_max, new_y_min, 
new_y_max, input_path, depth_order, dpi):

    # Zoom axes to the bounding box

    general_axes.set_xlim(new_x_min, new_x_max)

    general_axes.set_ylim(new_y_min, new_y_max)

    # Redraw before opening interactive window

    collage.canvas.draw()

    plt.pause(0.05)

    collage.canvas.draw()

    # Initializes a list of points

    points_list = []

    # Sets some commands for clicking and panning (dragging)

    is_panning = False

    # Sets the initial point and the initial limits

    pan_start = [0, 0]

    x_lim_start = [0, 0]
    
    y_lim_start = [0, 0]

    # When pressing the scroll

    def on_press(event):

        global is_panning, pan_start, x_lim_start, y_lim_start

        if event.inaxes != general_axes:

            return
        
        if event.button == 1:  # left mouse button for pan
            is_panning = True
            pan_start = [event.xdata, event.ydata]
            x_lim_start = list(general_axes.get_xlim())
            y_lim_start = list(general_axes.get_ylim())

    def on_release(event):
        global is_panning
        is_panning = False

    def on_motion(event):
        if event.inaxes != general_axes or not is_panning:
            return
        dx = pan_start[0] - event.xdata
        dy = pan_start[1] - event.ydata
        # shift limits
        general_axes.set_xlim(x_lim_start[0] + dx, x_lim_start[1] + dx)
        general_axes.set_ylim(y_lim_start[0] + dy, y_lim_start[1] + dy)
        collage.canvas.draw_idle()

    # Defines a function to detect the mouse motion

    def on_move(event):

        if event.inaxes==general_axes and (event.xdata is not None):

            # Prints coordinates to terminal continuously

            print(f"[MOVE] x = {event.xdata:.2f}, y = {event.ydata:.2f}", 
            end="\r")

    # Defines a function to detect clicking of the mouse

    def on_click(event):

        if event.inaxes==general_axes and event.xdata is not None:

            # Gets the clicked point coordinates

            x_coordinate = event.xdata

            y_coordinate = event.ydata

            print(f"[CLICK] x = {x_coordinate:.4f}, y = {(y_coordinate
            ):.4f}")

            # Saves the point into the list

            points_list.append([x_coordinate, y_coordinate])

            # Marks the click point on the screen 

            general_axes.plot(x_coordinate, y_coordinate, marker="x",
            color="black", zorder=depth_order, markersize=
            collage_classes.milimeters_to_points(2.0), mew=2)

            # Refreshes the figure on the open screen

            collage.canvas.draw()

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

    # Function to zoom in and out using mouse scroll

    def on_scroll(event):

        if event.inaxes != general_axes:

            return
        
        # Ctrl key must be pressed

        if not event.key == 'control':

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

    # Accesses the collage

    collage.canvas.mpl_connect("motion_notify_event", on_move)

    collage.canvas.mpl_connect("button_press_event", on_click)

    collage.canvas.mpl_connect("key_press_event", on_key)

    collage.canvas.mpl_connect("scroll_event", on_scroll)

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