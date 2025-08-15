# Routine to plot the tangent operators

import os

import numpy as np

import matplotlib.colors as plt_colors

from ....tool_box import file_handling_tools as file_tools

from ....tool_box import plotting_tools

# Defines a function to run the process

def plot_operators():

    # Sets the scaling function to show the data in a meaningful way

    x_c = 1E6

    kappa = 0.8

    kappa = np.sqrt(kappa)

    delta_y = (2*x_c)/np.log((1+kappa)/(1-kappa))

    def scaling_filter(x):

        if abs(x)>x_c:

            return x
        
        else:

            return 0.0

        #return x-(delta_y*np.tanh(x/delta_y))

    # Sets a logarithmic scaling with a high-pass filter

    cut_out = 5
    
    def scaling_log(x):

        value = np.log10(np.abs(x)+1.0)

        if value >=cut_out:

            return np.sign(x)*(value-cut_out) 
        
        else:

            return 0.0

        return 
    
    scaling = scaling_log

    # Sets if the color bar will be in scientific notation or not

    flag_scientificNotation = False

    # Sets if the color bar ticks will be integers or not

    color_barIntegerTicks = False
    
    # Sets the title of the color bar

    color_barTitle = ("$sgn\\left(\\|\\cdot\\|\\right)\\left(log\\left"+
    "(\\|\\cdot\\|+1\\right)-\\alpha\\right)$")
    
    # Defines the number of ticks in the color bar

    max_ticksColorBar = 21

    # Sets the color map for the components' plot

    color_map = 'seismic'

    # Sets a custom color 
    
    custom_map = True

    if custom_map:

        colors = [(0.2298057, 0.29871797, 0.75368315), (0.127568, 0.566949, 0.550556), (
        1, 1, 1), (0.993248, 0.906157, 0.143936), (0.70567316, 0.01555616, 0.15023281)]

        colors = ["#003f5c", "#2f4b7c", "#a0a0c0", "#ffffff", "#ffffff", 
        "#fbb282", "#f95d6a", "#d62728"]

        colors = ["#1f77b4","#ff7f0e","#2ca02c","#ffffff", "#ffffff","#9467bd",
        "#8c564b","#e377c2"]

        discrete_values = np.concatenate([np.linspace(0, 0.45, int(len(
        colors)*0.5)), np.linspace(0.55, 1, int(len(colors)*0.5))])

        print(discrete_values)

        """# Gets discrete values for the color mapping

        discrete_values = np.linspace(0, 1, 17)

        original_colorMap = plt.get_cmap(color_map)

        # Samples the corresponding discrete colors

        discrete_colors = original_colorMap(discrete_values)

        # Creates a new discrete colormap

        color_map = plt_colors.ListedColormap(discrete_colors)"""

        color_map = plt_colors.LinearSegmentedColormap.from_list(
        'rainbow_white', list(zip(discrete_values, colors)), N=
        max_ticksColorBar-1)

    # Creates a dictionary to convert the current indices to the origi-
    # nal indices of the tensor

    voigt_conversion = {0: 0, 1: 4, 2: 8, 3: 1, 4: 5, 5: 2, 6: 3, 7: 7,
    8: 6}

    #voigt_conversion = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    #8: 8}

    # Sets the basic size for the component representative circle

    basic_size = 12.0

    # Sets the multiscale boundary conditions for each one of the fields

    multiscale_BCsSets = [["MinimallyConstrainedFirstOrderBC", "Minima"+
    "llyConstrainedFirstOrderBC", "Minimally Constrained"], ["Periodic"+
    "FirstOrderBC", "PeriodicFirstOrderBC", "Periodic"], ["LinearFirst"+
    "OrderBC", "LinearFirstOrderBC", "Linear"]] 

    # Sets the basic path

    base_paths = [(os.getcwd()+"//tests//micropolar//tangent_operators"+
    "//characteristic_length_1//results_eps_1E_6"), (os.getcwd()+"//te"+
    "sts//micropolar//tangent_operators//characteristic_length_1//resu"+
    "lts_eps_1E_5")]

    base_paths = [(os.getcwd()+"//tests//micropolar//tangent_operators"+
    "//cauchy_hyperelastic//results_eps_1E_6")]

    # Sets a list of names for each set of parameters, which will yield
    # different simulations

    simulations_names = ["simulation_11", "simulation_12", "simulation"+
    "_13", "simulation_21", "simulation_22", "simulation_23", "simulat"+
    "ion_31", "simulation_32", "simulation_33"]

    # Iterates through the different perturbation step sizes

    for base_path in base_paths:

        # Iterates through the multiscale boundary conditions

        for multiscale_BCs in multiscale_BCsSets:

            # Iterates through the simulations

            for i in range(len(simulations_names)):

                # Gets the folder name

                folder_name = (base_path+"//"+simulations_names[i]+"//"+
                multiscale_BCs[0]+"_"+multiscale_BCs[1])

                # Calls the function to read the derivatives of the 
                # stress tensors and plot the dispersion of the compo-
                # nents

                plot_dispersion(folder_name, basic_size, 
                voigt_conversion, color_map, multiscale_BCs[2], scaling,
                max_ticksColorBar, color_barTitle, 
                flag_scientificNotation, color_barIntegerTicks)  

# Defines a function to read the derivatives of the stress tensors, and 
# plot the dispersion of the components

def plot_dispersion(folder_name, basic_size, voigt_conversion, color_map,
title, scaling, max_ticksColorBar, color_barTitle, 
flag_scientificNotation, color_barIntegerTicks):

    # Reads the derivative files

    dP_dGradU = file_tools.txt_toList("dP_dGradU", parent_path=
    folder_name)

    dP_dGradPhi = file_tools.txt_toList("dP_dGradPhi", parent_path=
    folder_name)

    dPcouple_dGradU = file_tools.txt_toList("dPcouple_dGradU",
    parent_path=folder_name)

    dPcouple_dGradPhi = file_tools.txt_toList("dPcouple_dGradPhi", 
    parent_path=folder_name)

    # Gets the maximum and minimum values of components

    min_component, max_component = get_maximumMagnitude(dP_dGradU)

    min_component, max_component = get_maximumMagnitude(dP_dGradPhi, 
    max_component=max_component, min_component=min_component)

    min_component, max_component = get_maximumMagnitude(dPcouple_dGradU, 
    max_component=max_component, min_component=min_component)

    min_component, max_component = get_maximumMagnitude(
    dPcouple_dGradPhi, max_component=max_component, min_component=
    min_component)

    # Iterates through the time points

    for i in range(len(dP_dGradU)):

        # Gets the time point and the file name

        time = dP_dGradU[i][0]

        file_name = ("C_voigt_t_"+str(int(np.floor(time)))+"_"+str(time-
        np.floor(time))[2:]+".pdf")

        # Plots the tensors

        plot_tensor(dP_dGradU[i][1], dP_dGradPhi[i][1], dPcouple_dGradU[
        i][1], dPcouple_dGradPhi[i][1], folder_name, file_name, title, 
        basic_size, voigt_conversion, min_component, max_component, 
        color_map, scaling, max_ticksColorBar, color_barTitle,
        flag_scientificNotation, color_barIntegerTicks)

# Defines a function to get the maximum value of magnitude of a list of
# tensors

def get_maximumMagnitude(tensors_list, max_component=0.0, min_component=
0.0):

    # Iterates through the time values

    for time in tensors_list:

        # iterates through the components of the matrix

        for i in range(len(time[1])):

            for j in range(len(time[1][i])):

                max_component = max(max_component, time[1][i][j])

                min_component = min(min_component, time[1][i][j])

    return min_component, max_component

# Defines a function to normalize the values of the tensors by the maxi-
# mum magnitude found

def normalize_tensors(tensors_list, maximum_magnitude):

    # Iterates through the time values

    for time in tensors_list:

        # iterates through the components of the matrix

        for i in range(len(time[1])):

            for j in range(len(time[1][i])):

                time[1][i][j] = time[1][i][j]/maximum_magnitude

    return tensors_list

# Deines a function to plot the tensor

def plot_tensor(dP_dGradU, dP_dGradPhi, dPcouple_dGradU, 
dPcouple_dGradPhi, parent_path, file_name, title, basic_size,
voigt_conversion, min_component, max_component, color_map, scaling,
max_ticksColorBar, color_barTitle, flag_scientificNotation,
color_barIntegerTicks):

    # Initializes the color list and the marker size list

    color_list = []

    marker_sizeList = []

    # Initializes the plot lists

    x_data = []

    y_data = []

    # Creates a function for the conversion of indices

    def indices_function(i,j):

        x_data.append(j+1)

        y_data.append(18-i)

        marker_sizeList.append(basic_size)

        # Verifies if the indices are in the derivative of the first Pi-
        # ola-Kirchhoff stress tensor w.r.t. the displacement gradient

        if i<9 and j<9:

            color_list.append(scaling(dP_dGradU[voigt_conversion[i]][
            voigt_conversion[j]]))

        # Verifies if the indices are in the derivative of the first Pi-
        # ola-Kirchhoff stress tensor w.r.t. the microrotation gradient

        if i<9 and j>8:

            color_list.append(scaling(dP_dGradPhi[voigt_conversion[i]][
            voigt_conversion[j-9]]))

        # Verifies if the indices are in the derivative of the first Pi-
        # ola-Kirchhoff couple stress tensor w.r.t. the displacement 
        # gradient

        if i>8 and j<9:

            color_list.append(scaling(dPcouple_dGradU[voigt_conversion[i
            -9]][voigt_conversion[j]]))

        # Verifies if the indices are in the derivative of the first Pi-
        # ola-Kirchhoff couple stress tensor w.r.t. the microrotation 
        # gradient

        if i>8 and j>8:

            color_list.append(scaling(dPcouple_dGradPhi[voigt_conversion[
            i-9]][voigt_conversion[j-9]]))

    print("Initializes the assembly and scaling of the tensor colors")

    for i in range(18):

        for j in range(18):

            indices_function(i,j)

    print("Finishes the assembly and scaling of the tensor colors")

    # Gets the extreme values for the color bar

    max_magnitude = max(abs(max_component), abs(min_component))

    # Plots and saves the figure

    plotting_tools.plane_plot(file_name=parent_path+"//"+file_name, 
    x_data=x_data, y_data=y_data, element_style="s", element_size=
    marker_sizeList, color=color_list, color_map=color_map, plot_type=
    "scatter", flag_grid=True, flag_noTicks=True, aspect_ratio='equal', 
    x_grid=[3.5, 6.5, 9.5, 12.5, 15.5], y_grid=[3.5, 6.5, 9.5, 12.5, 
    15.5], color_bar=True, color_barMaximum=scaling(max_magnitude), 
    color_barMinimum=scaling(max_magnitude*np.sign(min_component)), 
    color_barTitle=color_barTitle, title=title, 
    color_barIncludeMinMaxTicks=True, color_barIntegerTicks=
    color_barIntegerTicks, color_barNumberOfTicks=max_ticksColorBar, 
    flag_scientificNotation=flag_scientificNotation)

def test():

    t = np.linspace(-50.0, 50.0, 200)

    x_c = 30

    kappa = 0.8

    kappa = np.sqrt(kappa)

    delta_y = (2*x_c)/np.log((1+kappa)/(1-kappa))

    def scaling(x):

        return x-(delta_y*np.tanh(x/delta_y))
    
    y_data = np.zeros((len(t), 2))
    
    for i in range(len(t)):

        y_data[i,0] = t[i]

        y_data[i,1] = scaling(t[i])

    plotting_tools.plane_plot(x_data=t, y_data=y_data)

#test()

plot_operators()