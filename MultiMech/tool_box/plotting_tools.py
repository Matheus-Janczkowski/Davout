# Routine to store methods to plot

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as plt_colors

import matplotlib.ticker as ticker

from ..tool_box import file_handling_tools as file_tools

from ..tool_box import programming_tools

########################################################################
#                          Bidimensional plots                         #
########################################################################

# Defines a function to plot curves in the XY plane

def plane_plot(file_name=None, data=None, x_data=None, y_data=None, 
label=None, x_label=None, y_label=None, title=None, flag_grid=True, 
highlight_points=False, color=None, flag_scientificNotation=False,
element_style='-', element_size=1.5,  legend_position='upper left', 
plot_type="line", color_map=False, flag_noTicks=False, aspect_ratio='a'+
'uto', x_grid=None, y_grid=None, color_bar=False, color_barMaximum=None, 
color_barMinimum=None, color_barTicks=None, color_barTitle=None, 
color_barIntegerTicks=False, color_barNumberOfTicks=5, 
color_barIncludeMinMaxTicks=False, x_ticksLabels=None, y_ticksLabels=
None, ticks_fontsize=12, label_fontsize=14, legend_fontsize=12,
highlight_pointsColors='black', parent_path=None):
    
    """
    You can provide an array of data, where the first column will be in
    terpreted as the x points, and the remaining columns will be diffe
    rent curves with corresponding y points. You can provide x and y da
    ta separately as well. The keyword arguments are:

    label                                 - label for the curve

    x_label and y_label                   - labels for the x and y axes

    title                                 - title for the whole plot

    flag_grid                             - True (default) if a grid is 
    to be shown

    highlight_points                      - True if the truly provided
    data points are to be highlighted with X markers; False (default) o
    therwise

    color                                 - can be a string with the name
    or a list of values or names

    flag_scientificNotation               - True if the axes are to be 
    numbered using scientific notation; False (default) otherwise

    element_style                         - string with the type of line 
    or marker, the default value is '-' (line)

    element_size                          - float with the width of the 
    line or the area of the marker. 1.5 as default

    legend_position                       - position for the legend. 'upper
    right' is the default value

    plot_type                             - informs if the plot is with 
    lines ("line"), or scattered ("scatter")

    color_map                             - informs the color map, can 
    be a string or a matplotlib color map function

    flag_noTicks                          - True if no ticks are to be 
    shown on the axes. False (default) otherwise

    aspect_ratio                          - string or float with the as
    pect ratio of the frame. 'auto' is the default value

    x_grid and y_grid                     - list of coordinates where to 
    cross the grid lines. The default values are None

    color_bar                             - True if a color bar is to be 
    shown. False (default) otherwise

    color_barMaximum and color_barMinimum - maximum and minimum values 
    for the color bar

    color_barTicks                        - list with the ticks for the 
    color bar. None is the default value

    color_barTitle                        - title of the color bar

    color_barIntegerTicks                 - True if the color bar ticks 
    must be integer. False (default) otherwise

    color_barNumberOfTicks                - number of ticks to be shown 
    on the color bar. 5 is the default value

    color_barIncludeMinMaxTicks           - True if the maximum and min
    imum values of the color bar are to be put in. False (default) o
    therwise
    """

    print("Starts plotting")

    # Initializes a flag to inform if mutliple curves are supplied

    multiple_curves = False

    # Initializes a flag to inform whether each curve has a different
    # number of points

    different_nPoints = False

    # Checks if all variables data, x_data, and y_data are None

    if (data is None) and (x_data is None) and (y_data is None):

        raise ValueError("The list of lists of data has not been provi"+
        "ded nor the x_data list neither the y_data. Thus, no data can"+
        " be plotted")

    # If the data was given

    elif not (data is None):

        if not isinstance(data, list):

            # Verifies if does not have the to list method

            if not hasattr(data, "tolist"):

                raise TypeError("The data list is not a list nor a num"+
                "py array neither a tensorflow tensor. No data can be "+
                "retrieved for plotting")
            
            else:

                data = data.tolist()

        elif len(data)==0:

            raise IndexError("The data list is empty. No data can be r"+
            "etrieved for plotting")

        # Iterates through the data point

        x_data = []

        y_data = []

        # Verifies if each point of data has more than two values. If so,
        # this means more curves are provided

        if len(data[0])>2:

            multiple_curves = len(data[0])-1

            for i in range(len(data[0])-1):

                y_data.append([])

        # If the plot type is catter, treats each point as a separate 
        # curve

        elif plot_type=="scatter":

            multiple_curves = len(data)

        for point in data:

            # Verifies if this data point is a list

            if not isinstance(point, list):

                raise TypeError("Each data point in the data list must"+
                " be a list, for plotting them")
            
            elif len(point)<2:

                raise IndexError("The data point has length less than "+
                "2, thus, a pair of data cannot be retrieved for plott"+
                "ing")

            x_data.append(point[0])

            if multiple_curves:

                for i in range(len(point)-1):

                    y_data[i].append(point[i+1])

            else:

                y_data.append(point[1])

    # Checks the x and y lists

    else:

        if not isinstance(x_data, list):

            # Verifies if does not have the to list method

            if not hasattr(x_data, "tolist"):

                raise TypeError("The x data is not a list nor a numpy "+
                "array neither a tensorflow tensor, thus, cannot be us"+
                "ed for plotting")
            
            else:

                x_data = x_data.tolist()

        elif not isinstance(y_data, list):

            # Verifies if does not have the to list method

            if not hasattr(y_data, "tolist"):

                raise TypeError("The y data is not a list nor a numpy "+
                "array neither a tensorflow tensor, thus, cannot be us"+
                "ed for plotting")
            
            else:

                y_data = y_data.tolist()

        elif isinstance(y_data[0], list):

            multiple_curves = len(y_data)

            if (not isinstance(x_data[0], list)) and (not isinstance(
            x_data[0], np.ndarray)):

                if len(x_data)!=len(y_data[0]):

                    raise IndexError("The x data and y data are lists "+
                    "of different sizes. Length(x_data)="+str(len(x_data
                    ))+", length(y_data[0])="+str(len(y_data[0]))+" Mu"+
                    "ltiple curves are to be plotted, but, different n"+
                    "umber of points have been given for each curve. T"+
                    "hus, cannot be used for plotting")
                
            elif len(x_data)!=len(y_data):

                raise IndexError("Multiple curves have been given, and"+
                " the number of x lists is also more than one, thus, e"+
                "ach curve is supposed to have its own set of x values"+
                ". But the number of x lists is different thant those "+
                "of y lists")
            
            else:

                # Updates the flag that informs whether each curve has
                # different numbers of points

                different_nPoints = True

                # Checks each curve

                for i in range(len(y_data)):

                    if (not isinstance(x_data[i], list)) and (
                    not isinstance(x_data[i], np.ndarray)):

                        raise TypeError("The "+str(i+1)+"-th curve doe"+
                        "s not have the x_data as a list.")

                    elif (not isinstance(y_data[i], list)) and (
                    not isinstance(y_data[i], np.ndarray)):

                        raise TypeError("The "+str(i+1)+"-th curve doe"+
                        "s not have the y_data as a list.")

                    elif len(x_data[i])!=len(y_data[i]):

                        raise IndexError("The "+str(i+1)+"-th curve do"+
                        "es not have the same length for the sublists "+
                        "x_data and of y_data")

        elif len(x_data)!=len(y_data):

            raise IndexError("The x data and y data are lists of diffe"+
            "rent sizes. Thus, cannot be used for plotting")
        
        # If the plot type is catter, treats each point as a separate 
        # curve

        elif plot_type=="scatter":

            multiple_curves = len(x_data)

    # Sets the graph to be plotted in LaTeX style

    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], "axes.labelsize": 14, "fo"+
    "nt.size": 14, "legend.fontsize": 12, "xtick.labelsize": 12, "ytic"+
    "k.labelsize": 12, "text.latex.preamble": r"\usepackage{amsmath}"})

    # Gets the color map

    if color_map:

        if isinstance(color_map, str):

            try:

                color_map = plt.get_cmap(color_map)

            except Exception as error_message:

                print("Error Message:"+str(error_message)+"\nProbably "+
                "this color map does not exist")

    # Creates the figure and the subplots

    figure, subplots_tuple = plt.subplots()

    # Verifies the nature of the line styles and of the color vector

    if multiple_curves:

        # Verifies line styles

        if isinstance(element_style, str):

            element_style = [element_style for i in range(
            multiple_curves)]

        elif not isinstance(element_style, list):

            raise TypeError("Multiple curves were given to be plotted,"+
            " but the element_style variables is neither a string nor "+
            "a list")
        
        elif len(element_style)!=multiple_curves:

            raise IndexError(str(multiple_curves)+" curves were given "+
            "to be plotted but "+str(len(element_style))+" line styles"+
            " were given")

        # Verifies line thickness

        if isinstance(element_size, float) or isinstance(element_size, 
        int):

            element_size = [element_size for i in range(
            multiple_curves)]

        elif not isinstance(element_size, list):

            raise TypeError("Multiple curves were given to be plotted,"+
            " but the element_size variables is neither a float or an "+
            "integer nor a list")
        
        elif len(element_size)!=multiple_curves:

            raise IndexError(str(multiple_curves)+" curves were given "+
            "to be plotted but "+str(len(element_size))+" line thickne"+
            "sses were given")
        
        # Verifies the color vector

        if color is None:

            if color_map:

                color = [color_map(i) for i in np.linspace(0, 1,
                multiple_curves)]

            else:

                color = [plt.get_cmap('viridis')(i) for i in np.linspace(0,
                1, multiple_curves)]

        elif isinstance(color, str):

            color = [color for i in range(multiple_curves)]

        elif isinstance(color, float) or isinstance(color, int):

            if color_map:

                color = [color_map(color) for i in range(multiple_curves
                )]

            else:

                color = [color for i in range(multiple_curves)]

        elif not isinstance(color, list):

            raise TypeError("Multiple curves were given to be plotted,"+
            " but the color variables is neither a string nor a list")
        
        elif len(color)!=multiple_curves:

            raise IndexError(str(multiple_curves)+" curves were given "+
            "to be plotted but "+str(len(color))+" colors were given")
        
        elif color_map:

            # Gets the extrema values of the colors

            color_min = None

            color_max = None

            try:

                color_min = min(color)

                color_max = max(color)

                # If the minimum value is not given

                if color_barMinimum is None:

                    color_barMinimum = color_min*1.0

                elif color_min<color_barMinimum:

                    color_barMinimum = color_min*1.0

                # If the maximum is not given

                if color_barMaximum is None:

                    color_barMaximum = color_max*1.0

                elif color_max>color_barMaximum:

                    color_barMaximum = color_max*1.0

            except:

                pass

            # Iterates through the color values

            for i in range(len(color)):

                if isinstance(color[i], float) or isinstance(color[i], 
                int):
                    
                    color[i] = color_map((color[i]-color_barMinimum)/(
                    color_barMaximum-color_barMinimum))

            # Updates the color map variable to account for the maximum
            # and minimum values

            if (not (color_min is None)) and (not (color_max is None)):

                color_map = [color_map, color_barMinimum, 
                color_barMaximum]
            
    else:

        # Verifies the line styles
        
        if isinstance(element_style, list):

            if len(element_style)>1:

                raise IndexError(str(len(element_style))+" line styles"+
                " were given, but there is only one curve to be plotte"+
                "d")
            
            else:

                element_style = element_style[0]

        # Verifies the line thicknesses
        
        if isinstance(element_size, list):

            if len(element_size)>1:

                raise IndexError(str(len(element_size))+" line thickne"+
                "sses were given, but there is only one curve to be pl"+
                "otted")
            
            else:

                element_size = element_size[0]

        # And verifies the line colors
        
        if isinstance(color, list):

            if len(color)>1:

                raise IndexError(str(len(color))+" colors were given, "+
                "but there is only one curve to be plotted")
            
            else:

                if color_map and (isinstance(color[0], float) or (
                isinstance(color[0], int))):

                    color = color_map(color[0])

                else:

                    color = color[0]

        elif color is None:

            color = 'orange'

    # Sets the aspect ratio of the plot

    subplots_tuple.set_aspect(aspect_ratio)

    # Inititalizes the plotted entities

    plotted_entities = None

    # Plots it

    if label is None:

        if multiple_curves:

            for i in range(multiple_curves):

                if plot_type=="line":

                    if different_nPoints:

                        plotted_entities = subplots_tuple.plot(x_data[i], 
                        y_data[i], linestyle=element_style[i], linewidth=
                        element_size[i], color=color[i])

                    else:

                        plotted_entities = subplots_tuple.plot(x_data, 
                        y_data[i], linestyle=element_style[i], linewidth=
                        element_size[i], color=color[i])

                elif plot_type=="scatter":

                    # If multiple curves are plotted at once

                    if isinstance(y_data[i], list):

                        if different_nPoints:

                            plotted_entities = subplots_tuple.scatter(
                            x_data[i], y_data[i], marker=element_style[i
                            ], s=element_size[i]**2, color=color[i], 
                            zorder=3)

                        else:

                            plotted_entities = subplots_tuple.scatter(
                            x_data, y_data[i], marker=element_style[i], 
                            s=element_size[i]**2, color=color[i], zorder=
                            3)

                    # If it is just the default treatment of scatter 
                    # plots

                    else:

                        plotted_entities = subplots_tuple.scatter(x_data[
                        i], y_data[i], marker=element_style[i], s=
                        element_size[i]**2, color=color[i], zorder=3)

                else:

                    raise ValueError("There are two types of plot: 'li"+
                    "ne' for data points that are orderly joined by li"+
                    "nes, and 'scatter' for points that are plotted sc"+
                    "atter fashion")

        else:

            if plot_type=="line":

                plotted_entities = subplots_tuple.plot(x_data, y_data, 
                linestyle=element_style, linewidth=element_size, color=
                color)

            elif plot_type=="scatter":

                plotted_entities = subplots_tuple.scatter(x_data, y_data, 
                marker=element_style, s=element_size**2, color=color, 
                zorder=3)

            else:

                raise ValueError("There are two types of plot: 'line' "+
                "for data points that are orderly joined by lines, and"+
                " 'scatter' for points that are plotted scatter fashio"+
                "n")

    else:

        if multiple_curves:

            if not isinstance(label, list):

                raise TypeError("The label is not a list, but there ar"+
                "e multiple curves to be plotted.")

            if len(label)!=multiple_curves:

                raise IndexError("The length of the label list, "+str(
                len(label))+", is not equal to the number of curves, "+
                str(multiple_curves))

            for i in range(multiple_curves):
                
                if plot_type=="line":

                    if different_nPoints:

                        plotted_entities = subplots_tuple.plot(x_data[i], 
                        y_data[i], linestyle=element_style[i], linewidth=
                        element_size[i], color=color[i], label=label[i])

                    else:

                        plotted_entities = subplots_tuple.plot(x_data, 
                        y_data[i], linestyle=element_style[i], linewidth=
                        element_size[i], color=color[i], label=label[i])

                elif plot_type=="scatter":

                    # If multiple curves are plotted at once

                    if isinstance(y_data[i], list):

                        if different_nPoints:

                            plotted_entities = subplots_tuple.scatter(
                            x_data[i], y_data[i], marker=element_style[i
                            ], s=element_size[i]**2, color=color[i], 
                            zorder=3)

                        else:

                            plotted_entities = subplots_tuple.scatter(
                            x_data, y_data[i], marker=element_style[i], 
                            s=element_size[i]**2, color=color[i], 
                            zorder=3)

                    # If it is just the default treatment of scatter 
                    # plots

                    else:

                        plotted_entities = subplots_tuple.scatter(x_data[
                        i], y_data[i], marker=element_style[i], s=
                        element_size[i]**2, color=color[i], zorder=3)

                else:

                    raise ValueError("There are two types of plot: 'li"+
                    "ne' for data points that are orderly joined by li"+
                    "nes, and 'scatter' for points that are plotted sc"+
                    "atter fashion")

        else:

            if plot_type=="line":

                plotted_entities = subplots_tuple.plot(x_data, y_data, 
                linestyle=element_style, linewidth=element_size, color=
                color, label=label)

            elif plot_type=="scatter":

                plotted_entities = subplots_tuple.scatter(x_data, y_data, 
                marker=element_style, s=element_size**2, color=color, 
                label=label, zorder=3)

            else:

                raise ValueError("There are two types of plot: 'line' "+
                "for data points that are orderly joined by lines, and"+
                " 'scatter' for points that are plotted scatter fashio"+
                "n")

    plt.grid(flag_grid)

    # Plots the set of points as scattered markers

    if highlight_points:

        if highlight_points==True:

            highlight_points = 'x'

        if multiple_curves:

            for i in range(multiple_curves):

                if isinstance(highlight_pointsColors, str):

                    if different_nPoints:

                        subplots_tuple.scatter(x_data[i], y_data[i], 
                        color=highlight_pointsColors, marker=
                        highlight_points, zorder=3)

                    else:

                        subplots_tuple.scatter(x_data, y_data[i], color=
                        highlight_pointsColors, marker=highlight_points, 
                        zorder=3)

                else:

                    if different_nPoints:

                        subplots_tuple.scatter(x_data[i], y_data[i], 
                        color=color[i], marker=highlight_points, zorder=
                        3)

                    else:

                        subplots_tuple.scatter(x_data, y_data[i], color=
                        color[i], marker=highlight_points, zorder=3)

        else:

            if isinstance(highlight_pointsColors, str):

                subplots_tuple.scatter(x_data, y_data, color=
                highlight_pointsColors, marker=highlight_points, zorder=
                3)

            else:

                subplots_tuple.scatter(x_data, y_data, color='black', 
                marker=highlight_points, zorder=3)

    # Verifies if a color bar is asked for

    if color_bar:
        
        # Verifies if the color map has minimum and maximum values

        if not isinstance(color_map, list):

            raise TypeError("The color map does not have minimum and m"+
            "aximum values. Either the data has just one curve and, th"+
            "us cannot possibly have a color map, or a color map has n"+
            "ot been provided")

        # Creates two artificial points to add the color gradient

        if multiple_curves:

            # Verifies the purely scattered case

            if not isinstance(y_data[0], list):

                plotted_entities = subplots_tuple.scatter(x_data[0:2], 
                [y_data[0], y_data[1]], c=[color_map[1], color_map[2]], 
                cmap=color_map[0], vmin=color_map[1], vmax=color_map[2], 
                marker='x', zorder=3, s=0.001)

            # Otherwise, plots elements of the first curve

            else:

                if different_nPoints:

                    plotted_entities = subplots_tuple.scatter(x_data[0][
                    0:2], y_data[0][0:2], c=[color_map[1], color_map[2]], 
                    cmap=color_map[0], vmin=color_map[1], vmax=color_map[
                    2], marker='x', zorder=3, s=0.001)

                else:

                    plotted_entities = subplots_tuple.scatter(x_data[0:2
                    ], y_data[0][0:2], c=[color_map[1], color_map[2]], 
                    cmap=color_map[0], vmin=color_map[1], vmax=color_map[
                    2], marker='x', zorder=3, s=0.001)

        else:

            plotted_entities = subplots_tuple.scatter(x_data[0:2], 
            y_data[0:2], c=[color_map[1], color_map[2]], cmap=color_map[
            0], vmin=color_map[1], vmax=color_map[2], marker='x', 
            zorder=3, s=0.001)

        # Creates the color bar

        color_bar = plt.colorbar(plotted_entities)

        if color_barTitle:

            color_bar.set_label(color_barTitle)

        else:

            color_bar.set_label("Magnitude")

        if isinstance(color_barTicks, list):

            color_bar.set_ticks(color_barTicks)

        else:

            # Verifies if the ticks must be integers

            if color_barIntegerTicks:

                initial_tick = int(np.ceil(color_barMinimum)+1)

                final_tick = int(np.floor(color_barMaximum))

                tick_step = ((final_tick-initial_tick)/(
                color_barNumberOfTicks+1))

                color_barTicks = []

                if color_barIncludeMinMaxTicks:

                    color_barTicks.append(color_barMinimum)

                # Iterates through the ticks

                for i in range(color_barNumberOfTicks):

                    current_tick = int(initial_tick+np.floor((i+1)*
                    tick_step))

                    if not (current_tick in color_barTicks):

                        color_barTicks.append(current_tick)

                # If the maximum value is to be added as a tick

                if color_barIncludeMinMaxTicks:

                    color_barTicks.append(color_barMaximum)

                # Sets the ticks

                color_bar.set_ticks(color_barTicks)

            else:

                # Creates the ticks list using linspace

                color_bar.set_ticks(np.linspace(color_barMinimum, 
                color_barMaximum, color_barNumberOfTicks))

        # Sets a formatter object to ensure that integer ticks are shown
        # as integer

        def formatter_function(x, _):

            if flag_scientificNotation:

                # Tests for null input

                if x==0:

                    return r"$ 0$"
                
                # Otherwise, evaluates the exponent and the component
                
                exponent = int(np.floor(np.log10(abs(x))))

                coefficient = x/(10**exponent)

                # Adds padding for positive coefficients

                formatted_base = f"{coefficient:{12}.{3}f}"

                return fr"${formatted_base} \times 10^{{{exponent}}}$"

            if abs(x-int(x)) < 1e-6:

                return f"{int(x):>3}"  # format as integer, right-aligned to 3 spaces
            
            # Right-align float, width 5
            
            else:

                return f"{x:>5.2f}" 

        color_barFormatter = ticker.FuncFormatter(formatter_function)

        color_bar.ax.yaxis.set_major_formatter(color_barFormatter)

    # Applies scientific notation to the ticks

    if flag_scientificNotation and (not flag_noTicks):

        subplots_tuple.yaxis.set_major_formatter(ticker.ScalarFormatter(
        useMathText=True))

        subplots_tuple.xaxis.set_major_formatter(ticker.ScalarFormatter(
        useMathText=True))

        subplots_tuple.ticklabel_format(style='sci', axis='both', 
        scilimits=(0,0))

        # Sets the ticks' font size

        subplots_tuple.tick_params(axis='both', which='major', labelsize=
        ticks_fontsize)

    elif flag_noTicks:

        subplots_tuple.tick_params(axis='both', which='both', length=0, 
        labelbottom=False, labelleft=False)

    # Sets the grid

    if not (x_grid is None):

        subplots_tuple.set_xticks(x_grid)

        subplots_tuple.set_xticklabels([])

        subplots_tuple.grid(True, axis='x')

    if not (y_grid is None):

        subplots_tuple.set_yticks(y_grid)

        subplots_tuple.set_yticklabels([])

        subplots_tuple.grid(True, axis='y')

    # Sets the tick labels

    if isinstance(x_ticksLabels, list):

        # Gets the tick values and the location. Do not use minor ticks
        # when they are given as lists

        subplots_tuple.set_xticks(x_ticksLabels)

        subplots_tuple.set_xticklabels(x_ticksLabels)

        # Sets the font size of the x ticks

        for tick_label in subplots_tuple.get_xminorticklabels():

            tick_label.set_fontsize(ticks_fontsize)

    elif isinstance(x_ticksLabels, dict):

        # Gets the tick values and the location

        tick_location = list(x_ticksLabels.keys())

        tick_names = list(x_ticksLabels.values())

        subplots_tuple.set_xticks(tick_location, minor=True)

        subplots_tuple.set_xticklabels(tick_names, minor=True)

        # Sets the font size of the x ticks

        for tick_label in subplots_tuple.get_xminorticklabels():

            tick_label.set_fontsize(ticks_fontsize)

    if isinstance(y_ticksLabels, list):

        # Gets the tick values and the location. Do not use minor ticks
        # when they are given as lists

        subplots_tuple.set_yticks(y_ticksLabels)

        subplots_tuple.set_yticklabels(y_ticksLabels)

        # Sets the font size of the y ticks

        for tick_label in subplots_tuple.get_yminorticklabels():

            tick_label.set_fontsize(ticks_fontsize)

    elif isinstance(y_ticksLabels, dict):

        # Gets the tick values and the location

        tick_location = list(y_ticksLabels.keys())

        tick_names = list(y_ticksLabels.values())

        subplots_tuple.set_yticks(tick_location, minor=True)

        subplots_tuple.set_yticklabels(tick_names, minor=True)

        # Sets the font size of the y ticks

        for tick_label in subplots_tuple.get_yminorticklabels():

            tick_label.set_fontsize(ticks_fontsize)

    # Verifies and uses if necessary other optional attributes

    if not (x_label is None):

        plt.xlabel(x_label, fontsize=label_fontsize)

    if not (y_label is None):

        plt.ylabel(y_label, fontsize=label_fontsize)

    if not (title is None):

        plt.title(title)

    if not (label is None):

        plt.legend(loc=legend_position, bbox_to_anchor=(1.0, 1.0), 
        fontsize=legend_fontsize)

    # Adjust the size of the plot to contain ticks and labels

    plt.tight_layout()

    # Saves the plot or shows it

    if file_name is None:

        plt.show()

    else:

        # Verifies the file name

        file_name, termination = file_tools.take_outFileNameTermination(
        file_name, get_termination=True)

        # Verifies if the termination is empty

        if len(termination)==0:

            # Adds pdf

            termination = "pdf"

        # Adds the termination again

        file_name = file_tools.verify_path(parent_path, file_name+"."+
        termination)

        try:

            plt.savefig(file_name)

        except:

            raise InterruptedError("The file is probably open on a vis"+
            "ualizer. Close it and try again")

    print("Finishes plotting\n")

########################################################################
#                          Matrices plotting                           #
########################################################################

# Defines a function to make a grid plot of the values of a matrix

def plot_matrix(list_of_matrices, parent_path, base_file_name,
include_time=True, scaling_function="linear", color_map="seismic", title=
None, flag_scientificNotation=False, scaling_functionAdditionalParams=
None, max_ticksColorBar=3, x_grid=None, y_grid=None, element_size=12,
x_ticksLabels=None, y_ticksLabels=None):

    """
    Function to plot matrices relative components.
    
    Arguments:
    
    list_of_matrices: list of matrices on the format [[t0, matrix0], [
    t1, matrix1], ..., [tn, matrixn]], or on format [matrix0, matrix1,
    ..., matrixn]

    parent_path: path to the directory where the files will be stored

    base_file_name: basic name for the files without any termination, li
    ke .pdf or .png. This routine will automatically add the time step
    and the termination
    
    include_time: flag for informing whether the time value is supplied
    within each list, like t0 in [t0, matrix0]. The default value is True

    scaling_function: string with the name of the chosen scaling func
    tion

    scaling_functionAdditionalParams: dictionary with the additional pa
    rameters for the scaling function
    """

    # Scales the list of matrices

    (scaled_matrices, min_component, max_component, scaling_function,
    scaling_functionTitle) = scale_matricesForPlotting(list_of_matrices, 
    include_time=include_time, scaling_function=scaling_function,
    scaling_functionAdditionalParams=scaling_functionAdditionalParams)

    # Gets the minimum and maximum values for the color bar

    color_barMaximum = max_component*1.0 

    color_barMinimum = min_component*1.0 

    if max_component>0 and min_component<0:

        color_barMaximum = max(max_component, abs(min_component))

        color_barMinimum = -max(max_component, abs(min_component))

    # Iterates through the list of matrices to get each matrix

    for t in range(len(scaled_matrices)):

        # Initializes the x data and the y data, which are the position 
        # of each component of the matrices

        x_data = []

        y_data = []

        # Initializes the color values, which are the components of the 
        # matrix

        colors = []

        # Gets the matrix itself and the time value

        time = t+1

        matrix = None

        if include_time:

            time = scaled_matrices[t][0]

            matrix = scaled_matrices[t][1]

        else:

            matrix = scaled_matrices[t]

        # Iterates through the matrix

        for i in range(len(matrix)):

            for j in range(len(matrix[i])):

                x_data.append(j+1)

                y_data.append(len(matrix)-i)

                colors.append(matrix[i][j])

        # Converts the time value to integer

        time = file_tools.float_toString(time)

        # Plots and saves the figure

        plane_plot(file_name=parent_path+"//"+base_file_name+"_t_"+time
        +".pdf", x_data=x_data, y_data=y_data, element_style="s", 
        element_size=element_size, color=colors, color_map=
        color_mapBuilder(color_map, max_ticksColorBar=max_ticksColorBar
        ), plot_type="scatter", flag_grid=True, flag_noTicks=False, 
        aspect_ratio='equal', x_grid=x_grid, y_grid=y_grid, color_bar=
        True, color_barMaximum=color_barMaximum, color_barMinimum=
        color_barMinimum, color_barTitle=scaling_functionTitle, title=
        title, color_barIncludeMinMaxTicks=True, color_barIntegerTicks=
        False, color_barNumberOfTicks=max_ticksColorBar, 
        flag_scientificNotation=flag_scientificNotation, x_ticksLabels=
        x_ticksLabels, y_ticksLabels=y_ticksLabels)

########################################################################
#                              Color maps                              #
########################################################################

# Defines a function to get a fully built color map from a string

def color_mapBuilder(color_map_name, max_ticksColorBar=3):

    if color_map_name=="blue orange green white purple brown pink":

        # Sets the colors list

        colors = ["#1f77b4","#ff7f0e","#2ca02c","#ffffff", 
        "#ffffff","#9467bd","#8c564b","#e377c2"]

        # Sets their divisions along the [0,1] interval

        discrete_values = np.concatenate([np.linspace(0, 0.45, int(len(
        colors)*0.5)), np.linspace(0.55, 1, int(len(colors)*0.5))])

        # Makes the custom color map

        return plt_colors.LinearSegmentedColormap.from_list(
        color_map_name, list(zip(discrete_values, colors)), N=
        max_ticksColorBar-1)
    
    # If it is not one of the custom color maps, returns the name to try
    # and pick up one from matplotlib
    
    else:

        return color_map_name

########################################################################
#                              Utilities                               #
########################################################################

# Defines a function to scale and prepare tensors for plotting. It must
# receive a list of matrices

def scale_matricesForPlotting(list_of_matrices, include_time=True,
scaling_function="linear", scaling_functionAdditionalParams=None):

    """
    Function to prepare and scale matrices for plotting.
    
    Arguments:
    
    list_of_matrices: list of matrices on the format [[t0, matrix0], [
    t1, matrix1], ..., [tn, matrixn]], or on format [matrix0, matrix1,
    ..., matrixn]
    
    include_time: flag for informing whether the time value is supplied
    within each list, like t0 in [t0, matrix0]. The default value is True

    scaling_function: string with the name of the chosen scaling func
    tion

    scaling_functionAdditionalParams: dictionary with the additional pa
    rameters for the scaling function
    """

    print("Starts scaling the list of matrices\n")

    # If the additional parameters are "default"

    if scaling_functionAdditionalParams=="default":

        scaling_functionAdditionalParams = None

    # Gets the scaling function

    scaling_function, scaling_functionTitle = generate_scalingFunctions(
    scaling_function, additional_parameters=
    scaling_functionAdditionalParams)

    # Iterates through the list of matrices to get the mininum and maxi-
    # mum components

    min_component = 0.0

    max_component = 0.0

    for i in range(len(list_of_matrices)):

        # Takes the list from the step information

        list_array = None

        if include_time:

            list_array = list_of_matrices[i][1]
            
        else:

            list_array = list_of_matrices[i]

        # Converts the list to a numpy array

        try:

            list_array = np.array(list_array)

        except:

            raise ValueError("The list "+str(list_array)+" has not a p"+
            "roper format to be converted into a numpy array")
        
        # Checks for the minimum and maximum values

        min_component = min(np.min(list_array), min_component)

        max_component = max(np.max(list_array), max_component)

        # Scales each component of the matrix using the vectorize func-
        # tionality of numpy. Then, allocates it into the list of matri-
        # ces

        if include_time:

            list_of_matrices[i][1] = np.vectorize(scaling_function)(
            list_array)

        else:

            list_of_matrices[i] = np.vectorize(scaling_function)(
            list_array)

    print("Finishes scaling the list of matrices\n")

    # Returns the scaled list of matrices and scales the minimum and ma-
    # ximum components

    return (list_of_matrices, scaling_function(min_component), 
    scaling_function(max_component), scaling_function, 
    scaling_functionTitle)
    
########################################################################
#                           Scaling functions                          #
########################################################################

# Defines a function to give different options of scaling functions u-
# sing numpy functions. These scaling functions will be used for plot-
# ting for instance

@programming_tools.optional_argumentsInitializer({('additional_paramet'+
'ers'): lambda: dict()})

def generate_scalingFunctions(curve_name, additional_parameters=
None, verify_curveNameExistence=False):
    
    # The flag verify_curveNameExistence is True when the interest is 
    # just to point out if the curve name is in the scope of the imple-
    # mented functions 

    # Tests if it is linear

    if curve_name=="linear":

        if verify_curveNameExistence:

            return True
        
        # Check out if additional parameters have been been given

        default_parameters = check_additionalParameters(
        additional_parameters, {"end_point": [1.0, 1.0], "starting_poi"+
        "nt": [0.0, 0.0]})

        a1 = ((default_parameters["end_point"][1]-default_parameters[
        "starting_point"][1])/(default_parameters["end_point"][0]-
        default_parameters["starting_point"][0]))

        a0 = (default_parameters["starting_point"][1]-(a1*
        default_parameters["starting_point"][1]))

        return lambda x: (a1*x)+a0, "$\\left(\\cdot\\right)$"

    # Tests if the function is the logarithmic filter

    elif curve_name=="logarithmic filter":

        if verify_curveNameExistence:

            return True
        
        # Check out if additional parameters have been been given

        default_parameters = check_additionalParameters(
        additional_parameters, {"alpha": 3})

        def log_filter(x):

            value = np.log10(np.abs(x)+1.0)

            if value>default_parameters["alpha"]:

                return np.sign(x)*(value-default_parameters["alpha"]) 
            
            else:

                return 0.0
            
        alpha = str(default_parameters["alpha"])

        return log_filter, ("$sgn\\left(\\cdot\\right)argmax\\left(log"+
        "\\left(\\|\\cdot\\|+1\\right)-"+alpha+",0\\right)$")

    else:

        if verify_curveNameExistence:

            return False
        
        else:

            raise NameError("The scaling function '"+str(curve_name)+
            "' has not yet been implemented")
        
# Defines a function to check additional parameters to each loading cur-
# ve

def check_additionalParameters(additional_parameters, default_parameters):

    if additional_parameters is None:

        return default_parameters 
    
    elif not isinstance(additional_parameters, dict):

        raise TypeError("The additional_parameters must be a dictionar"+
        "y to get the simple generators of load curves. Whereas it cur"+
        "rently is: "+str(additional_parameters))
    
    else:

        # Iterates through the dictionary of default parameters

        for name in additional_parameters:

            if name in default_parameters:

                # Checks if they have the same type

                if not (type(default_parameters[name])==type(
                additional_parameters[name])):
                    
                    raise TypeError("The '"+str(name)+"' additional pa"+
                    "rameter to create a loading curve has not the sam"+
                    "e type as of the default one. Look at the default"+
                    " parameter: "+str(default_parameters[name])+"\nan"+
                    "d the given parameter: "+str(additional_parameters[
                    name]))

                default_parameters[name] = additional_parameters[name]

            else:

                raise KeyError("The dictionary of additional parameter"+
                "s to get a simple generator of load curves has the ke"+
                "y '"+str(name)+"', but this key is not a valid additi"+
                "onal information. Check out the valid ones and their "+
                "respective default values: "+str(default_parameters))
            
        return default_parameters