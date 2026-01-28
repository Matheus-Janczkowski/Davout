# Routine to numerically differentiate the tangent operators

import os

from dolfin import *

from mshr import *

from ....constitutive_models.hyperelasticity import micropolar_hyperelasticity as micropolar_constitutiveModels

from ....constitutive_models.hyperelasticity import isotropic_hyperelasticity as cauchy_constitutiveModels

from ....multiscale import multiscale_micropolar as variational_framework

from ....multiscale import multiscale_hyperelasticity as cauchy_variationalFramework

from .....PythonicUtilities import file_handling_tools as file_tools

from .....CuboidGmsh.aa_tests.micropolar_meshes import beam_micropolar_case_1 as beam_gmsh

# Defines a function to try multiple parameters

def evaluate_tangentOperators(flag_newMesh=False):

    # Sets the problem to be solved (BVP_solution for micropolar or 
    # Cauchy_BVPsolution for Cauchy hyperelasticity)

    BVP_function = Cauchy_BVPsolution

    # Sets the perturbation step

    pertubation_step = -6

    # Sets the mesh refinement

    transfinite_directions = [12, 12, 12, 6, 6]

    transfinite_directions = [6,6,6]

    bias_directions = {"cylinder radial": 1.5, "box radial": 1.1}

    # Sets the multiscale boundary conditions for each one of the fields

    #multiscale_BCsSets = [["MinimallyConstrainedFirstOrderBC", "Minima"+
    #"llyConstrainedFirstOrderBC"], ["PeriodicFirstOrderBC", "PeriodicF"+
    #"irstOrderBC"], ["LinearFirstOrderBC", "LinearFirstOrderBC"]]

    multiscale_BCsSets = [["MinimallyConstrainedFirstOrderBC", "Minima"+
    "llyConstrainedFirstOrderBC"]]

    # Defines a flag to use the fluctuation of the field instead of the
    # field proper in the BVP

    fluctuation_field = True

    # Reads the parameters set

    base_paths = [(os.getcwd()+"//tests//micropolar//tangent_operators"+
    "//cauchy_hyperelastic//results_eps_1E_"+str(abs(
    pertubation_step)))]

    # Sets a list of names for each set of parameters, which will yield
    # different simulations

    simulations_names = ["simulation_11", "simulation_12", "simulation"+
    "_13", "simulation_21", "simulation_22", "simulation_23", "simulat"+
    "ion_31", "simulation_32", "simulation_33"]

    # Iterates through the simulation paths

    for base_path in base_paths:

        parameters_sets = file_tools.txt_toList("parameters_sets", 
        parent_path=base_path)

        # Iterates through the multiscale boundary conditions

        counter = 0

        for multiscale_BCs in multiscale_BCsSets:

            # Iterates through the simulations

            for i in range(min([len(parameters_sets),len(
            simulations_names)])):

                # Makes a new mesh just for the first test and if a new 
                # mesh is asked for

                flag_mesh = False

                if flag_newMesh and counter==0:

                    flag_mesh = True

                counter += 1

                # Calls the numerical derivation scheme

                subfolder_name = [simulations_names[i], multiscale_BCs[0
                ]+"_"+multiscale_BCs[1]]

                BVP_arguments = [multiscale_BCs[0], multiscale_BCs[1],
                base_path, *parameters_sets[i][0:10]]

                BVP_keywordArguments = [*parameters_sets[i][10:18], 
                flag_mesh, subfolder_name, fluctuation_field, 
                transfinite_directions, bias_directions]

                central_differencesTangentOperators(base_path, 
                subfolder_name, 10**(pertubation_step), BVP_arguments, 
                BVP_keywordArguments, BVP_function)

# Defines a function to perturbate the macroscale gradients to get the
# tangent operators numerically evaluated using central finite differen-
# ces

def central_differencesTangentOperators(base_path, subfolder_name, 
pertubation_step, BVP_arguments, BVP_keywordArguments, BVP_function):
    
    # Gets the path to the simulation data

    results_pathText = ""
    
    results_pathText += base_path

    for name in subfolder_name:

        results_pathText += "//"+name
    
    # Initializes the fourth order tensors for the derivatives of the
    # stress tensors with respect to the displacement and microrotation
    # gradients

    dP_dGradU = []

    dP_dGradPhi = []

    dPcouple_dGradU = []

    dPcouple_dGradPhi = []

    # Differentiates with respect to the displacement gradient

    # Iterates through the components of the displacement gradient ten-
    # sor

    for k in range(3):

        for l in range(3):

            # Solves the boundary value problem and gets the stress ten-
            # sor perturbating ahead

            P_ahead, P_coupleAhead = gradients_perturbation(base_path, 
            results_pathText, subfolder_name, [k,l], "Displacement", 
            pertubation_step, BVP_arguments, BVP_keywordArguments,
            BVP_function)

            # Again, but perturbating backwards

            P_abaft, P_coupleAbaft = gradients_perturbation(base_path, 
            results_pathText, subfolder_name, [k,l], "Displacement", 
            -pertubation_step, BVP_arguments, BVP_keywordArguments,
            BVP_function)

            # Subtracts and divides them by the step to get the finite
            # differences

            P_diff = subtract_dividesTensorLists(P_ahead, P_abaft, 2*
            pertubation_step)

            P_coupleDiff = subtract_dividesTensorLists(P_coupleAhead, 
            P_coupleAbaft, 2*pertubation_step)

            # Verifies if the fourth order tensor is empty

            if len(dP_dGradU)==0:

                # Initializes a list of tensors for each required tensor
                # using Voigt notation

                for m in range(len(P_ahead)):

                    dP_dGradU.append([P_ahead[m][0], [([0.0 for p in (
                    range(9))]) for n in range(9)]])

                    dP_dGradPhi.append([P_ahead[m][0], [([0.0 for p in (
                    range(9))]) for n in range(9)]])

                    dPcouple_dGradU.append([P_ahead[m][0], [([0.0 for (p
                    ) in range(9)]) for n in range(9)]])

                    dPcouple_dGradPhi.append([P_ahead[m][0], [([0.0 for (
                    p) in range(9)]) for n in range(9)]])

            # Iterates through the time points

            for t_step in range(len(P_diff)):

                # Iterates through the stress indices

                for i in range(3):

                    for j in range(3):

                        dP_dGradU[t_step][1][int((3*i)+j)][int((3*k)+l)
                        ] += P_diff[t_step][1][i][j]

                        dPcouple_dGradU[t_step][1][int((3*i)+j)][int((3*
                        k)+l)] += P_coupleDiff[t_step][1][i][j]

            # Writes the tensors

            file_tools.list_toTxt(dP_dGradU, "dP_dGradU", parent_path=
            results_pathText)

            file_tools.list_toTxt(dPcouple_dGradU, "dPcouple_dGradU", 
            parent_path=results_pathText)

    # Differentiates with respect to the microrotation gradient

    # Iterates through the components of the microrotation gradient ten-
    # sor

    for k in range(3):

        for l in range(3):

            # Solves the boundary value problem and gets the stress ten-
            # sor perturbating ahead

            P_ahead, P_coupleAhead = gradients_perturbation(base_path, 
            results_pathText, subfolder_name, [k,l], "Microrotation", 
            pertubation_step, BVP_arguments, BVP_keywordArguments,
            BVP_function)

            # Again, but perturbating backwards

            P_abaft, P_coupleAbaft = gradients_perturbation(base_path, 
            results_pathText, subfolder_name, [k,l], "Microrotation", 
            -pertubation_step, BVP_arguments, BVP_keywordArguments,
            BVP_function)

            # Subtracts and divides them by the step to get the finite
            # differences

            P_diff = subtract_dividesTensorLists(P_ahead, P_abaft, 2*
            pertubation_step)

            P_coupleDiff = subtract_dividesTensorLists(P_coupleAhead, 
            P_coupleAbaft, 2*pertubation_step)

            # Iterates through the time points

            for t_step in range(len(P_diff)):

                # Iterates through the stress indices

                for i in range(3):

                    for j in range(3):

                        dP_dGradPhi[t_step][1][int((3*i)+j)][int((3*k)+l
                        )] += P_diff[t_step][1][i][j]

                        dPcouple_dGradPhi[t_step][1][int((3*i)+j)][int((
                        3*k)+l)] += (P_coupleDiff[t_step][1][i][j])

            # Writes the tensors

            file_tools.list_toTxt(dP_dGradPhi, "dP_dGradPhi", parent_path=
            results_pathText)

            file_tools.list_toTxt(dPcouple_dGradPhi, "dPcouple_dGradPhi", 
            parent_path=results_pathText)

# Defines a function to perturbate a quantity

def gradients_perturbation(base_path, results_pathText, subfolder_name, 
perturbed_indices, perturbed_field, pertubation_step, BVP_arguments, 
BVP_keywordArguments, BVP_function):
    
    # Reads the displacement gradient

    displacement_gradient = file_tools.txt_toList("original_homogenize"+
    "d_displacement_gradient", parent_path=base_path+"//"+
    subfolder_name[0])

    # Reads the microrotation gradient

    microrotation_gradient = file_tools.txt_toList("original_homogeniz"+
    "ed_microrotation_grad", parent_path=base_path+"//"+subfolder_name[0
    ])

    # Perturbates the fields
    
    if perturbed_field=="Displacement":

        # Iterates through the time steps

        for time_step in displacement_gradient:

            # Adds the perturbation to the index of the displacement 
            # gradient

            time_step[1][perturbed_indices[0]][perturbed_indices[1]] += (
            pertubation_step)

    elif perturbed_field=="Microrotation":

        # Iterates through the time steps

        for time_step in microrotation_gradient:

            # Adds the perturbation to the index of the microrotation 
            # gradient

            time_step[1][perturbed_indices[0]][perturbed_indices[1]] += (
            pertubation_step)

    else:

        raise NameError("The field '"+str(perturbed_field)+"' is not a"+
        "vailable for perturbation")
    
    # Writes the displacement gradient

    file_tools.list_toTxt(displacement_gradient, "homogenized_displace"+
    "ment_gradient", parent_path=base_path+"//"+subfolder_name[0])
    
    # Writes the microrotation gradient

    file_tools.list_toTxt(microrotation_gradient, "homogenized_microro"+
    "tation_grad", parent_path=base_path+"//"+subfolder_name[0])

    # Calls the BVP solution

    print(BVP_keywordArguments)

    BVP_function(*BVP_arguments, gamma_matrix=BVP_keywordArguments[0], 
    gamma_fiber=BVP_keywordArguments[1], RVE_width=BVP_keywordArguments[
    2], RVE_length=BVP_keywordArguments[3], fiber_radius=
    BVP_keywordArguments[4], RVE_localizationX=BVP_keywordArguments[5], 
    RVE_localizationY=BVP_keywordArguments[6], RVE_localizationZ=
    BVP_keywordArguments[7], flag_newMesh=BVP_keywordArguments[8], 
    subfolder_name=BVP_keywordArguments[9], fluctuation_field=
    BVP_keywordArguments[10], transfinite_directions=
    BVP_keywordArguments[11], bias_directions=BVP_keywordArguments[12])

    # Reads the homogenized first Piola-Kirchhoff stress tensor

    first_piola = file_tools.txt_toList("homogenized_first_piola_micro"+
    "scale", parent_path=results_pathText)

    couple_firstPiola = file_tools.txt_toList("homogenized_couple_firs"+
    "t_piola_microscale", parent_path=results_pathText)

    # Returns the stress tensors

    return first_piola, couple_firstPiola

# Defines a function to subtract two lists of tensors one from the other
# and divides by a scalar

def subtract_dividesTensorLists(list1, list2, scalar):

    for t in range(len(list1)):

        for i in range(3):

            for j in range(3):

                list1[t][1][i][j] = ((list1[t][1][i][j]-list2[t][1][i][j
                ])/scalar)

    return list1

# Defines a function to solve the BVP in the microscale

def BVP_solution(displacement_multiscaleBC, microrotation_multiscaleBC,
base_path, E_matrix, E_fiber, nu_matrix, nu_fiber, N_micropolarMatrix, 
N_micropolarFiber, characteristic_lengthMatrix, 
characteristic_lengthFiber, flag_bending, load_case, gamma_matrix=0.0, 
gamma_fiber=0.0, RVE_width=1.0, RVE_length=1.0, fiber_radius=0.25, 
n_RVEsX=1, n_RVEsY=1, n_RVEsZ=1, RVE_localizationX=1, RVE_localizationY=
1, RVE_localizationZ=3, flag_newMesh=True, subfolder_name=["simulation"],
fluctuation_field=False, transfinite_directions=[6, 6, 3, 4, 3], 
bias_directions={"cylinder radial": 1.5, "box radial": 1.5}):

    ####################################################################
    #                        Simulation results                        #
    ####################################################################

    results_pathText = ""
    
    results_pathText += base_path

    for name in subfolder_name:

        results_pathText += "//"+name
    
    # Sets the data of the simulation in a txt file for further verifi-
    # cation

    file_tools.list_toTxt(file_tools.named_list({"E_matrix:": E_matrix, 
    "E_fiber:": E_fiber, "nu_matrix:": nu_matrix, "nu_fiber:": nu_fiber, 
    "N_micropolarMatrix:": N_micropolarMatrix, "N_micropolarFiber:": 
    N_micropolarFiber, "characteristic_lengthMatrix:": 
    characteristic_lengthMatrix, "characteristic_lengthFiber:": 
    characteristic_lengthFiber, "flag_bending:": flag_bending, "gamma_"+
    "matrix:": gamma_matrix, "gamma_fiber:": gamma_fiber, "RVE_width:": 
    RVE_width, "RVE_length:": RVE_length, "fiber_radius:": fiber_radius, 
    "RVE_localizationX": RVE_localizationX, "RVE_localizationY": 
    RVE_localizationY, "RVE_localizationZ": RVE_localizationZ}), "00_p"+
    "arameters", parent_path=results_pathText)

    ####################################################################
    ####################################################################
    ##                    User defined parameters                     ##
    ####################################################################
    ####################################################################
    
    homogenized_piolaFileName = ["homogenized_first_piola_microscale.t"+
    "xt", "homogenized_couple_first_piola_microscale.txt"]

    post_processes = [["Displacement", dict()]]

    post_processes[-1][-1]["HomogenizeFirstPiola"] = {"directo"+
    "ry path": results_pathText, "file name": 
    homogenized_piolaFileName[0], "subdomain":""}

    post_processes[-1][-1]["HomogenizeCoupleFirstPiola"] = {"d"+
    "irectory path": results_pathText, "file name": 
    homogenized_piolaFileName[1], "subdomain":""}

    ####################################################################
    #                       Material properties                        #
    ####################################################################

    # Sets a dictionary of properties

    alpha_matrix = 0.0

    alpha_fiber = 0.0

    # Saves the properties into a dictionary for the matrix

    material_propertiesMatrix = dict()

    material_propertiesMatrix["E"] = E_matrix

    material_propertiesMatrix["nu"] = nu_matrix

    material_propertiesMatrix["N"] = N_micropolarMatrix

    material_propertiesMatrix["alpha"] = alpha_matrix

    material_propertiesMatrix["flag bending"] = flag_bending

    material_propertiesMatrix["characteristic length"] = (
    characteristic_lengthMatrix)

    material_propertiesMatrix["gamma"] = gamma_matrix

    # And for the fiber

    material_propertiesFiber = dict()

    material_propertiesFiber["E"] = E_fiber

    material_propertiesFiber["nu"] = nu_fiber

    material_propertiesFiber["N"] = N_micropolarFiber

    material_propertiesFiber["alpha"] = alpha_fiber

    material_propertiesFiber["flag bending"] = flag_bending

    material_propertiesFiber["characteristic length"] = (
    characteristic_lengthFiber)

    material_propertiesFiber["gamma"] = gamma_fiber

    # Sets the material as a HGO material

    constitutive_model = dict()

    constitutive_model["RVE matrix"] = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
    material_propertiesMatrix)

    constitutive_model["RVE fiber"] = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
    material_propertiesFiber)

    ####################################################################
    #                               Mesh                               #
    ####################################################################

    # Defines the name of the file to save the mesh in. Do not write the 
    # file termination, e.g. .msh or .xdmf; both options will be saved 
    # automatically

    file_directory = os.getcwd()+"//tests//test_meshes"

    mesh_fileName = "micropolar_beam_with_fibers_microscale"

    if flag_newMesh:

        beam_gmsh.case_1(RVE_width, RVE_length, fiber_radius, n_RVEsX, 
        n_RVEsY, n_RVEsZ, RVE_localizationX, RVE_localizationY, 
        RVE_localizationZ, mesh_fileName=mesh_fileName, file_directory=
        file_directory, transfinite_directions=transfinite_directions,
        translation=[RVE_length*(RVE_localizationX-1), RVE_width*(
        RVE_localizationY-1), RVE_width*(RVE_localizationZ-1)],
        bias_directions=bias_directions)

    ####################################################################
    #                          Function space                          #
    ####################################################################

    # Defines the shape functions degree

    polynomial_degreeDisplacement = 2

    polynomial_degreeMicrorotation = 1

    ####################################################################
    #                         Solver parameters                        #
    ####################################################################

    # Sets the solver parameters in a dictionary

    solver_parameters = dict()

    solver_parameters["linear_solver"] = "mumps"

    solver_parameters["newton_relative_tolerance"] = 1e-6

    solver_parameters["newton_absolute_tolerance"] = 1e-4

    solver_parameters["newton_maximum_iterations"] = 30

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Defines the paths to the macro quantities files

    macro_displacementName = (base_path+"//"+subfolder_name[0]+"//homo"+
    "genized_displacement")

    macro_gradDisplacementName= (base_path+"//"+subfolder_name[0]+"//h"+
    "omogenized_displacement_gradient") 

    macro_microrotationName = (base_path+"//"+subfolder_name[0]+"//hom"+
    "ogenized_microrotation") 

    macro_gradMicrorotationName = (base_path+"//"+subfolder_name[0]+"/"+
    "/homogenized_microrotation_grad")

    ####################################################################
    ####################################################################
    ##                    Calculation and solution                    ##
    ####################################################################
    ####################################################################

    # Defines a flag to print every step

    verbose = True

    # Solves the variational problem

    variational_framework.micropolar_microscale(
    displacement_multiscaleBC, microrotation_multiscaleBC,
    macro_displacementName, macro_gradDisplacementName, 
    macro_microrotationName, macro_gradMicrorotationName, 
    constitutive_model, post_processes, file_directory+"//"+
    mesh_fileName, solver_parameters, polynomial_degreeDisplacement=
    polynomial_degreeDisplacement, polynomial_degreeMicrorotation=
    polynomial_degreeMicrorotation, verbose=verbose, fluctuation_field=
    fluctuation_field)

# Defines a function to test a purely Cauchy hyperelastic problem

def Cauchy_BVPsolution(displacement_multiscaleBC, microrotation_multiscaleBC,
base_path, E_matrix, E_fiber, nu_matrix, nu_fiber, N_micropolarMatrix, 
N_micropolarFiber, characteristic_lengthMatrix, 
characteristic_lengthFiber, flag_bending, load_case, gamma_matrix=0.0, 
gamma_fiber=0.0, RVE_width=1.0, RVE_length=1.0, fiber_radius=0.25, 
n_RVEsX=1, n_RVEsY=1, n_RVEsZ=1, RVE_localizationX=1, RVE_localizationY=
1, RVE_localizationZ=3, flag_newMesh=True, subfolder_name=["simulation"],
fluctuation_field=False, transfinite_directions=[6, 6, 3, 4, 3], 
bias_directions={"cylinder radial": 1.5, "box radial": 1.5}):

    ####################################################################
    #                        Simulation results                        #
    ####################################################################

    results_pathText = ""
    
    results_pathText += base_path

    for name in subfolder_name:

        results_pathText += "//"+name
    
    # Sets the data of the simulation in a txt file for further verifi-
    # cation

    file_tools.list_toTxt(file_tools.named_list({"E_matrix:": E_matrix, 
    "E_fiber:": E_fiber, "nu_matrix:": nu_matrix, "nu_fiber:": nu_fiber, 
    "RVE_width:": RVE_width, "RVE_length:": RVE_length, "fiber_radius:": 
    fiber_radius, "RVE_localizationX": RVE_localizationX, "RVE_localiz"+
    "ationY": RVE_localizationY, "RVE_localizationZ": RVE_localizationZ}
    ), "00_parameters_purely_cauchy", parent_path=results_pathText)

    ####################################################################
    ####################################################################
    ##                    User defined parameters                     ##
    ####################################################################
    ####################################################################
    
    homogenized_piolaFileName = ["homogenized_first_piola_microscale.t"+
    "xt"]

    post_processes = [["Displacement", dict()]]

    post_processes[-1][-1]["HomogenizeFirstPiola"] = {"directo"+
    "ry path": results_pathText, "file name": 
    homogenized_piolaFileName[0], "subdomain":""}

    ####################################################################
    #                       Material properties                        #
    ####################################################################

    # Saves the properties into a dictionary for the matrix

    material_propertiesMatrix = dict()

    material_propertiesMatrix["E"] = E_matrix

    material_propertiesMatrix["nu"] = nu_matrix

    # And for the fiber

    material_propertiesFiber = dict()

    material_propertiesFiber["E"] = E_fiber

    material_propertiesFiber["nu"] = nu_fiber

    # Sets the material as a HGO material

    constitutive_model = dict()

    """constitutive_model["RVE matrix"] = cauchy_constitutiveModels.Neo_Hookean(
    material_propertiesMatrix)

    constitutive_model["RVE fiber"] = cauchy_constitutiveModels.Neo_Hookean(
    material_propertiesFiber)"""

    constitutive_model["Matrix"] = cauchy_constitutiveModels.Neo_Hookean(
    material_propertiesMatrix)

    ####################################################################
    #                               Mesh                               #
    ####################################################################

    # Defines the name of the file to save the mesh in. Do not write the 
    # file termination, e.g. .msh or .xdmf; both options will be saved 
    # automatically

    file_directory = os.getcwd()+"//tests//test_meshes"

    mesh_fileName = "micropolar_beam_with_fibers_microscale"

    if flag_newMesh:

        beam_gmsh.case_1(RVE_width, RVE_length, fiber_radius, n_RVEsX, 
        n_RVEsY, n_RVEsZ, RVE_localizationX, RVE_localizationY, 
        RVE_localizationZ, mesh_fileName=mesh_fileName, file_directory=
        file_directory, transfinite_directions=transfinite_directions,
        translation=[RVE_length*(RVE_localizationX-1), RVE_width*(
        RVE_localizationY-1), RVE_width*(RVE_localizationZ-1)],
        bias_directions=bias_directions)

    ####################################################################
    #                          Function space                          #
    ####################################################################

    # Defines the shape functions degree

    polynomial_degreeDisplacement = 2

    ####################################################################
    #                         Solver parameters                        #
    ####################################################################

    # Sets the solver parameters in a dictionary

    solver_parameters = dict()

    solver_parameters["linear_solver"] = "mumps"

    solver_parameters["newton_relative_tolerance"] = 1e-6

    solver_parameters["newton_absolute_tolerance"] = 1e-4

    solver_parameters["newton_maximum_iterations"] = 30

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Defines the paths to the macro quantities files

    macro_displacementName = (base_path+"//"+subfolder_name[0]+"//homo"+
    "genized_displacement")

    macro_gradDisplacementName= (base_path+"//"+subfolder_name[0]+"//h"+
    "omogenized_displacement_gradient")

    ####################################################################
    ####################################################################
    ##                    Calculation and solution                    ##
    ####################################################################
    ####################################################################

    # Defines a flag to print every step

    verbose = True

    # Solves the variational 
    
    cauchy_variationalFramework.hyperelastic_microscale(
    displacement_multiscaleBC, macro_displacementName, 
    macro_gradDisplacementName, constitutive_model, post_processes, 
    file_directory+"//"+mesh_fileName, solver_parameters, 
    polynomial_degree=polynomial_degreeDisplacement, verbose=verbose, 
    fluctuation_field=fluctuation_field)

evaluate_tangentOperators(flag_newMesh=False)