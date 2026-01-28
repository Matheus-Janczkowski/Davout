# Routine to plot the homogenization of the stress tensors

import numpy as np

import matplotlib.pyplot as plt

from dolfin import *

from ....PythonicUtilities import plotting_tools

from ....PythonicUtilities import file_handling_tools as file_tools

from ....PythonicUtilities import path_tools

# Defines a function to plot the homogenized stress tensors

def plot_stress():

    # Gets the current path

    base_path = path_tools.get_parent_path_of_file(file=__file__)

    # Defines the paths to the stress tensor files

    homogenized_files = []

    # Defines a list of simulations to be considered for torsion

    simulations = ["simulation_48", "simulation_33"]

    # Defines a list of lists to each plot to be made. Each sublist con-
    # tains respectively: the name of the stress file for the macroscale,
    # the name of the file for the macroscale displacement gradient, the
    # name of the file for the microscale stress, name of the plot file,
    # component of the deformation gradient to be used as x axis, compo-
    # nents of the stress tensor to be plotted, the labels of the curves,
    # and the symbol in LaTeX for the stress tensor

    files_list = [["homogenized_first_piola", "homogenized_displacemen"+
    "t_gradient", "MinimallyConstrainedFirstOrderBC_MinimallyConstrain"+
    "edFirstOrderBC//homogenized_first_piola_microscale_fluctuation", 
    "homogenized_first_piola_minimally_constrained_BC.pdf", "2,3", ["1"+
    ",1", "2,3", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_couple_first_piola", "homogenized_displacement_gradi"+
    "ent", "MinimallyConstrainedFirstOrderBC_MinimallyConstrainedFirst"+
    "OrderBC//homogenized_couple_first_piola_microscale", "homogenized"+
    "_couple_first_piola_minimally_constrained_BC.pdf", "2,3", ["1,1", 
    "2,1", "3,1"], "\\boldsymbol{P}^{c}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", 
    "LinearFirstOrderBC_LinearFirstOrderBC//homogenized_first_piola_mi"+
    "croscale_fluctuation", "homogenized_first_piola_linear_BC.pdf", 
    "2,3", ["1,1", "2,3", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", 
    "PeriodicFirstOrderBC_PeriodicFirstOrderBC//homogenized_first_piol"+
    "a_microscale_fluctuation", "homogenized_first_piola_periodic_BC.p"+
    "df", "2,3", ["1,1", "2,3", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_couple_first_piola", "homogenized_displacement_gradi"+
    "ent", "PeriodicFirstOrderBC_PeriodicFirstOrderBC//homogenized_cou"+
    "ple_first_piola_microscale", "homogenized_couple_first_piola_peri"+
    "odic_BC.pdf", "2,3", ["1,1", "2,1", "3,1"], "\\boldsymbol{P}^{c}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", [
    "LinearFirstOrderBC_LinearFirstOrderBC//homogenized_first_piola_mi"+
    "croscale_fluctuation", "PeriodicFirstOrderBC_PeriodicFirstOrderBC"+
    "//homogenized_first_piola_microscale_fluctuation", "MinimallyCons"+
    "trainedFirstOrderBC_MinimallyConstrainedFirstOrderBC//homogenized"+
    "_first_piola_microscale_fluctuation"], "homogenized_first_piola_B"+
    "Cs_comparison.pdf", "2,3", [], ["Linear", "Periodic", "Min. C."], 
    "\\boldsymbol{P}"],
    ["homogenized_couple_first_piola", "homogenized_displacement_gradi"+
    "ent", ["LinearFirstOrderBC_LinearFirstOrderBC//homogenized_couple"+
    "_first_piola_microscale", "PeriodicFirstOrderBC_PeriodicFirstOrde"+
    "rBC//homogenized_couple_first_piola_microscale", "MinimallyConstr"+
    "ainedFirstOrderBC_MinimallyConstrainedFirstOrderBC//homogenized_c"+
    "ouple_first_piola_microscale"], "homogenized_couple_first_piola_B"+
    "Cs_comparison.pdf", "2,3", [], ["Linear", "Periodic", "Min. C."], 
    "\\boldsymbol{P}^{c}"]]

    """files_list = [["homogenized_first_piola", "homogenized_displacemen"+
    "t_gradient", "MinimallyConstrainedFirstOrderBC_MinimallyConstrain"+
    "edFirstOrderBC//homogenized_first_piola_microscale_fluctuation", 
    "homogenized_first_piola_minimally_constrained_BC.pdf", "2,3", ["1"+
    ",1", "2,3", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_couple_first_piola", "homogenized_displacement_gradi"+
    "ent", "MinimallyConstrainedFirstOrderBC_MinimallyConstrainedFirst"+
    "OrderBC//homogenized_couple_first_piola_microscale", "homogenized"+
    "_couple_first_piola_minimally_constrained_BC.pdf", "2,3", ["1,1", 
    "2,1", "3,1"], "\\boldsymbol{P}^{c}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", 
    "LinearFirstOrderBC_LinearFirstOrderBC//homogenized_first_piola_mi"+
    "croscale_fluctuation", "homogenized_first_piola_linear_BC.pdf", 
    "2,3", ["1,1", "2,3", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", 
    "PeriodicFirstOrderBC_PeriodicFirstOrderBC//homogenized_first_piol"+
    "a_microscale_fluctuation", "homogenized_first_piola_periodic_BC.p"+
    "df", "2,3", ["1,1", "2,3", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_couple_first_piola", "homogenized_displacement_gradi"+
    "ent", "PeriodicFirstOrderBC_PeriodicFirstOrderBC//homogenized_cou"+
    "ple_first_piola_microscale", "homogenized_couple_first_piola_peri"+
    "odic_BC.pdf", "2,3", ["1,1", "2,1", "3,1"], "\\boldsymbol{P}^{c}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", [
    "PeriodicFirstOrderBC_PeriodicFirstOrderBC//homogenized"+
    "_first_piola_microscale_fluctuation"], "homogenized_first_piola_B"+
    "Cs_comparison.pdf", "2,3", [], ["RVE"], 
    "\\boldsymbol{P}"],
    ["homogenized_couple_first_piola", "homogenized_displacement_gradi"+
    "ent", ["PeriodicFirstOrderBC_PeriodicFirstOrderBC//homogenized_c"+
    "ouple_first_piola_microscale"], "homogenized_couple_first_piola_B"+
    "Cs_comparison.pdf", "2,3", [], ["RVE"], 
    "\\boldsymbol{P}^{c}"]]"""

    # Adds the paths to the torsion simulations

    base_pathTorsion = base_path+"//torsion_case//results//text"

    for simulation in simulations:

        for file_name in files_list:

            micro_files = []

            if isinstance(file_name[2], list):

                for micro_file in file_name[2]:

                    micro_files.append(base_pathTorsion+"//"+simulation+
                    "//"+micro_file)

            else:

                micro_files = (base_pathTorsion+"//"+simulation+"//"+
                file_name[2])

            homogenized_files.append([base_pathTorsion+"//"+simulation+
            "//"+file_name[0], base_pathTorsion+"//"+simulation+"//"+
            file_name[1], micro_files, base_pathTorsion+"//"+simulation+
            "//"+file_name[3], *file_name[4:len(file_name)]])

    # Defines a list of simulations to be considered for bending

    simulations = ["simulation_31", "simulation_33"]

    # Defines a list of lists to each plot to be made. Each sublist con-
    # tains respectively: the name of the stress file for the macroscale,
    # the name of the file for the macroscale displacement gradient, the
    # name of the file for the microscale stress, name of the plot file,
    # component of the deformation gradient to be used as x axis, compo-
    # nents of the stress tensor to be plotted, the labels of the curves

    files_list = [["homogenized_first_piola", "homogenized_displacemen"+
    "t_gradient", "MinimallyConstrainedFirstOrderBC_MinimallyConstrain"+
    "edFirstOrderBC//homogenized_first_piola_microscale_fluctuation", 
    "homogenized_first_piola_minimally_constrained_BC.pdf", "2,3", ["2"+
    ",2", "2,3", "3,2", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_couple_first_piola", "homogenized_displacement_gradi"+
    "ent", "MinimallyConstrainedFirstOrderBC_MinimallyConstrainedFirst"+
    "OrderBC//homogenized_couple_first_piola_microscale", "homogenized"+
    "_couple_first_piola_minimally_constrained_BC.pdf", "2,3", ["1,1", 
    "1,2", "1,3"], "\\boldsymbol{P}^{c}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", 
    "LinearFirstOrderBC_LinearFirstOrderBC//homogenized_first_piola_mi"+
    "croscale_fluctuation", "homogenized_first_piola_linear_BC.pdf", 
    "2,3", ["2,2", "2,3", "3,2", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", 
    "PeriodicFirstOrderBC_PeriodicFirstOrderBC//homogenized_first_piol"+
    "a_microscale_fluctuation", "homogenized_first_piola_periodic_BC.p"+
    "df", "2,3", ["2,2", "2,3", "3,2", "3,3"], "\\boldsymbol{P}"], 
    ["homogenized_first_piola", "homogenized_displacement_gradient", [
    "LinearFirstOrderBC_LinearFirstOrderBC//homogenized_first_piola_mi"+
    "croscale_fluctuation", "PeriodicFirstOrderBC_PeriodicFirstOrderBC"+
    "//homogenized_first_piola_microscale_fluctuation", "MinimallyCons"+
    "trainedFirstOrderBC_MinimallyConstrainedFirstOrderBC//homogenized"+
    "_first_piola_microscale_fluctuation"], "homogenized_first_piola_B"+
    "Cs_comparison.pdf", "2,3", [], ["Linear", "Periodic", "Min. C."], 
    "\\boldsymbol{P}"],
    ["homogenized_couple_first_piola", "homogenized_displacement_gradi"+
    "ent", ["LinearFirstOrderBC_LinearFirstOrderBC//homogenized_couple"+
    "_first_piola_microscale", "PeriodicFirstOrderBC_PeriodicFirstOrde"+
    "rBC//homogenized_couple_first_piola_microscale", "MinimallyConstr"+
    "ainedFirstOrderBC_MinimallyConstrainedFirstOrderBC//homogenized_c"+
    "ouple_first_piola_microscale"], "homogenized_couple_first_piola_B"+
    "Cs_comparison.pdf", "2,3", [], ["Linear", "Periodic", "Min. C."], 
    "\\boldsymbol{P}^{c}"]]

    # Adds the paths to the bending simulations

    base_pathBending = base_path+"//bending_case//results//text"

    for simulation in simulations:

        for file_name in files_list:

            micro_files = []

            if isinstance(file_name[2], list):

                for micro_file in file_name[2]:

                    micro_files.append(base_pathBending+"//"+simulation+
                    "//"+micro_file)

            else:

                micro_files = (base_pathBending+"//"+simulation+"//"+
                file_name[2])

            homogenized_files.append([base_pathBending+"//"+simulation+
            "//"+file_name[0], base_pathBending+"//"+simulation+"//"+
            file_name[1], micro_files, base_pathBending+"//"+simulation+
            "//"+file_name[3], *file_name[4:len(file_name)]])

    # Reads the files

    for i in range(len(homogenized_files)):

        # Reads the stress file and the displacement gradient
        
        read_stress = file_tools.txt_toList(homogenized_files[i][0])
        
        read_displacementGradient = file_tools.txt_toList(
        homogenized_files[i][1])

        # Reads the stress file of the microscale

        read_stressMicroscale = []

        if isinstance(homogenized_files[i][2], list):

            for micro_file in homogenized_files[i][2]:

                read_stressMicroscale.append(file_tools.txt_toList(
                micro_file))

        else:

            read_stressMicroscale = file_tools.txt_toList(
            homogenized_files[i][2])

        # Separates each component of the displacement gradient and cor-
        # rects it to the deformation gradient

        F_dict = dict()

        F_dict["1,1"] = [(F[1][0][0]+1.0) for F in read_displacementGradient]

        F_dict["1,2"] = [F[1][0][1] for F in read_displacementGradient]

        F_dict["1,3"] = [F[1][0][2] for F in read_displacementGradient]

        F_dict["2,1"] = [F[1][1][0] for F in read_displacementGradient]

        F_dict["2,2"] = [(F[1][1][1]+1.0) for F in read_displacementGradient]

        F_dict["2,3"] = [F[1][1][2] for F in read_displacementGradient]

        F_dict["3,1"] = [F[1][2][0] for F in read_displacementGradient]

        F_dict["3,2"] = [F[1][2][1] for F in read_displacementGradient]

        F_dict["3,3"] = [(F[1][2][2]+1.0) for F in read_displacementGradient]

        # Separates each component of the stress

        stress_dict = dict()

        stress_dict["1,1"] = [stress[1][0][0] for stress in read_stress]

        stress_dict["1,2"] = [stress[1][0][1] for stress in read_stress]

        stress_dict["1,3"] = [stress[1][0][2] for stress in read_stress]

        stress_dict["2,1"] = [stress[1][1][0] for stress in read_stress]

        stress_dict["2,2"] = [stress[1][1][1] for stress in read_stress]

        stress_dict["2,3"] = [stress[1][1][2] for stress in read_stress]

        stress_dict["3,1"] = [stress[1][2][0] for stress in read_stress]

        stress_dict["3,2"] = [stress[1][2][1] for stress in read_stress]

        stress_dict["3,3"] = [stress[1][2][2] for stress in read_stress]

        # Separates each component of the microscale stress

        microscale_stressDict = []

        if isinstance(homogenized_files[i][2], list):

            for j in range(len(homogenized_files[i][2])):

                microscale_stressDict.append(dict())

                microscale_stressDict[-1]["1,1"] = [stress[1][0][0] for (
                stress) in read_stressMicroscale[j]]

                microscale_stressDict[-1]["1,2"] = [stress[1][0][1] for (
                stress) in read_stressMicroscale[j]]

                microscale_stressDict[-1]["1,3"] = [stress[1][0][2] for (
                stress) in read_stressMicroscale[j]]

                microscale_stressDict[-1]["2,1"] = [stress[1][1][0] for (
                stress) in read_stressMicroscale[j]]

                microscale_stressDict[-1]["2,2"] = [stress[1][1][1] for (
                stress) in read_stressMicroscale[j]]

                microscale_stressDict[-1]["2,3"] = [stress[1][1][2] for (
                stress) in read_stressMicroscale[j]]

                microscale_stressDict[-1]["3,1"] = [stress[1][2][0] for (
                stress) in read_stressMicroscale[j]]

                microscale_stressDict[-1]["3,2"] = [stress[1][2][1] for (
                stress) in read_stressMicroscale[j]]

                microscale_stressDict[-1]["3,3"] = [stress[1][2][2] for (
                stress) in read_stressMicroscale[j]]

        else:

            microscale_stressDict = dict()

            microscale_stressDict["1,1"] = [stress[1][0][0] for (stress
            ) in read_stressMicroscale]

            microscale_stressDict["1,2"] = [stress[1][0][1] for (stress
            ) in read_stressMicroscale]

            microscale_stressDict["1,3"] = [stress[1][0][2] for (stress
            ) in read_stressMicroscale]

            microscale_stressDict["2,1"] = [stress[1][1][0] for (stress
            ) in read_stressMicroscale]

            microscale_stressDict["2,2"] = [stress[1][1][1] for (stress
            ) in read_stressMicroscale]

            microscale_stressDict["2,3"] = [stress[1][1][2] for (stress
            ) in read_stressMicroscale]

            microscale_stressDict["3,1"] = [stress[1][2][0] for (stress
            ) in read_stressMicroscale]

            microscale_stressDict["3,2"] = [stress[1][2][1] for (stress
            ) in read_stressMicroscale]

            microscale_stressDict["3,3"] = [stress[1][2][2] for (stress
            ) in read_stressMicroscale]

        # Builds up the y data

        y_data = []

        x_data = []

        x_label = ""

        if homogenized_files[i][4] in F_dict.keys():

            x_data = F_dict[homogenized_files[i][4]]

            x_label = "$\\boldsymbol{F}_{"+homogenized_files[i][4]+"}$"

        else:

            x_label = "$t$"

            # If the component is not in the keys of the deformation 
            # gradient, puts the time values

            for instant in read_displacementGradient:

                x_data.append(instant[0])

        y_label = "$"+homogenized_files[i][-1]+"$"

        label = []

        line_styles = []

        color_map = plt.get_cmap('coolwarm')

        delta_color = 0.0

        color = []

        if isinstance(homogenized_files[i][2], list):
            
            if len(homogenized_files[i][5])>0:
            
                raise ValueError("When multiple microscale BCs are giv"+
                "en, stress components cannot be individually plotted")
        
        elif len(homogenized_files[i][5])>0:
            
            delta_color = 1/(len(homogenized_files[i][5]))

        for j in range(len(homogenized_files[i][5])):

            component = homogenized_files[i][5][j]

            y_data.append(stress_dict[component])

            y_data.append(microscale_stressDict[component])

            label.append("$"+homogenized_files[i][-1]+"_{M,"+component+
            "}$")

            label.append("$"+homogenized_files[i][-1]+"_{H,"+component+
            "}$")

            line_styles.extend(["solid", "dashed"])

            color_number = color_map(j*delta_color)

            color.extend([color_number, color_number])

        # Adds the Frobenius norm data

        stress_norm = []

        for component_list in stress_dict.values():

            if len(stress_norm)==0:

                for j in range(len(component_list)):

                    stress_norm.append(component_list[j]**2)

            else:

                for j in range(len(component_list)):

                    stress_norm[j] += component_list[j]**2

        for j in range(len(stress_norm)):

            stress_norm[j] = np.sqrt(stress_norm[j])

        if isinstance(homogenized_files[i][2], list):

            y_data.append(stress_norm)

            y_label = "$\\|"+homogenized_files[i][-1]+"\\|_{F}$"

            label.append("DNS")

            line_styles.append("solid")

            color.append("black")

            if len(homogenized_files[i][2])>1:

                delta_color = 1/(len(homogenized_files[i][2])-1)

            for k in range(len(homogenized_files[i][2])):

                micro_scaleStressNorm = []

                for component_list in microscale_stressDict[k].values():

                    if len(micro_scaleStressNorm)==0:

                        for j in range(len(component_list)):

                            micro_scaleStressNorm.append(component_list[
                            j]**2)

                    else:

                        for j in range(len(component_list)):

                            micro_scaleStressNorm[j] += component_list[j
                            ]**2

                for j in range(len(micro_scaleStressNorm)):

                    micro_scaleStressNorm[j] = np.sqrt(
                    micro_scaleStressNorm[j])
                            
                y_data.append(micro_scaleStressNorm)

                label.append(str(homogenized_files[i][6][k]))

                line_styles.append("dashed")

                color.append(color_map(1.0-(k*delta_color)))

        else:

            micro_scaleStressNorm = []

            for component_list in microscale_stressDict.values():

                if len(micro_scaleStressNorm)==0:

                    for j in range(len(component_list)):

                        micro_scaleStressNorm.append(component_list[j
                        ]**2)

                else:

                    for j in range(len(component_list)):

                        micro_scaleStressNorm[j] += component_list[j
                        ]**2

            for j in range(len(micro_scaleStressNorm)):

                micro_scaleStressNorm[j] = np.sqrt(
                micro_scaleStressNorm[j])

            y_data.append(stress_norm)
                        
            y_data.append(micro_scaleStressNorm)

            label.append("$\\|"+homogenized_files[i][-1]+"_{M}\\|_{F}$")

            label.append("$\\|"+homogenized_files[i][-1]+"_{H}\\|_{F}$")

            line_styles.extend(["solid", "dashed"])

            color.extend([color_map(1.0), color_map(1.0)])

        # Plots and saves the figure
        
        """print(homogenized_files[i][3])

        print(label)

        print(y_data)

        print("\n")"""

        plotting_tools.plane_plot(homogenized_files[i][3], x_data=x_data, 
        y_data=y_data, x_label=x_label, y_label=y_label, 
        highlight_points=True, flag_scientificNotation=True, label=label, 
        element_style=line_styles, color=color, legend_position='upper'+
        ' left', ticks_fontsize=20, label_fontsize=20, legend_fontsize=
        16)

plot_stress()