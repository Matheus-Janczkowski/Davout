# Routine to test a hyperelastic disc

import os

import traceback

import numpy as np

from dolfin import *

from mshr import *

from ....constitutive_models.hyperelasticity import micropolar_hyperelasticity as micropolar_constitutiveModels

from ....multiscale import multiscale_micropolar as variational_framework

from ....tool_box import file_handling_tools as file_tools

from .....CuboidGmsh.aa_tests.micropolar_meshes import beam_micropolar_case_1 as beam_gmsh

# Defines a function to try multiple parameters

def case_3(flag_newMesh=False):

    # Sets the mesh file directory path

    mesh_file_directory = os.path.join(file_tools.get_parent_path_of_file(
    file=__file__, path_bits_to_be_excluded=3),"test_meshes")

    # Sets the mesh refinement

    transfinite_directions = [16, 16, 12, 6, 8]

    bias_directions = {"cylinder radial": 1.5, "box radial": 1.1}

    # Sets the multiscale boundary conditions for each one of the fields

    multiscale_BCsSets = [["MinimallyConstrainedFirstOrderBC", "Minima"+
    "llyConstrainedFirstOrderBC"], ["LinearFirstOrderBC", "LinearFirst"+
    "OrderBC"], ["PeriodicFirstOrderBC", "PeriodicFirstOrderBC"]]

    multiscale_BCsSets = [["MinimallyConstrainedFirstOrderBC", "Minima"+
    "llyConstrainedFirstOrderBC"]]

    # Defines a flag to use the fluctuation of the field instead of the
    # field proper in the BVP

    fluctuation_field = True

    # Reads the parameters set

    base_path = os.path.join(file_tools.get_parent_path_of_file(file=
    __file__), "results")

    # Sets the number of time steps

    n_steps = 5

    # Defines the RVE overall parameters

    RVE_width = 1.0

    RVE_length = 1.0

    # Defines the fiber radius

    fiber_radius = 0.25

    # Sets the Young modulus and the Poisson ration from psi to MPa

    nu_matrix = 0.4

    nu_fiber = 0.4

    flag_bending = True

    gamma_matrix = 0.0

    gamma_fiber = 0.0

    characteristic_lengthMatrix = fiber_radius*0.5
    
    characteristic_lengthFiber = fiber_radius*0.5

    E_matrix = 100E6

    E_fiber = 100.0*E_matrix

    N_matrix = 0.001#0.2

    N_fiber = 0.001#0.2

    parameters_set = [[E_matrix, E_fiber, nu_matrix, nu_fiber, N_matrix,
    N_fiber, characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, gamma_matrix, gamma_fiber, RVE_width, RVE_length, 
    fiber_radius]]

    """N_matrix = 0.08

    N_fiber = 0.08
    
    parameters_set.append([E_matrix, E_fiber, nu_matrix, nu_fiber, 
    N_matrix, N_fiber, characteristic_lengthMatrix, 
    characteristic_lengthFiber, flag_bending, gamma_matrix, gamma_fiber, 
    RVE_width, RVE_length, fiber_radius])"""

    # Sets the list of final values of displacement gradient. The gradi-
    # ent will be set as null

    null_vector = [0.0, 0.0, 0.0]

    null_tensor = [null_vector, null_vector, null_vector]

    displacement_gradients = [null_tensor, null_tensor, null_tensor, 
    null_tensor, [[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[
    0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.1, 0.0], 
    [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.1], [0.0, 0.1, 0.0
    ], [0.1, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 
    0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.1, 0.0]], 
    null_tensor] 
    
    displacement_gradients = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]],
    null_tensor]

    displacement_gradients = [[[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 
    0.0, 0.0]], [[0.0,0.0,0.0],[0.0,0.0,0.1],[0.0,0.0,0.0]], [[0.0,0.0,
    0.0],[0.0,0.0,0.1],[0.0,0.1,0.0]], null_tensor, null_tensor, 
    null_tensor]

    # Sets the same for the microrotation gradient

    phi_component = 0.35

    microrotation_gradients = [[[0.0, phi_component, 0.0], [0.0, 0.0, 
    0.0], [0.0, 0.0, 0.0]], [[phi_component, 0.0, 0.0], [0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0]], [[0.0, 0.0, phi_component], [0.0, phi_component, 
    0.0], [phi_component, 0.0, 0.0]], [[0.0, phi_component, 0.0], [
    phi_component, 0.0, 0.0], [0.0, 0.0, 0.0]], null_tensor, null_tensor,
    null_tensor, null_tensor, null_tensor, null_tensor, [[0.0, 0.0, 0.0
    ], [0.0, 0.0, phi_component], [0.0, phi_component, 0.0]]]

    microrotation_gradients = [null_tensor, [[0.0, 0.0, 0.0
    ], [0.0, 0.0, phi_component], [0.0, 0.0, 0.0]]]

    microrotation_gradients = [null_tensor, null_tensor, null_tensor, [[
    phi_component, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 
    0.0, 0.0], [0.0, 0.0, phi_component], [0.0, 0.0, 0.0]], [[0.0, 0.0,
    0.0], [0.0, 0.0, phi_component], [0.0, phi_component, 0.0]]]

    # Sets the displacements and microrotations as null vectors

    displacement_macro = interpolate_macroQuantities(null_vector, 
    n_steps)

    microrotation_macro = interpolate_macroQuantities(null_vector, 
    n_steps)

    # Iterates through the multiscale boundary conditions

    counter = 23

    # Gets the number of simulations

    n_simulations = min([len(displacement_gradients), len(
    microrotation_gradients)])

    # Iterates through the base parameters

    for base_parameters in parameters_set:

        # Iterates through the simulations

        for i in range(n_simulations):

            # Gets the number of this simulation

            counter += 1

            simulation_number = str(counter)

            for j in range(len(str(int(n_simulations*len(parameters_set)
            )))-len(simulation_number)):

                simulation_number = "0"+simulation_number

            # Creates the directory to this result

            subfolder_name = "simulation_"+simulation_number

            # Gets the list of displacement gradient and of microrotation
            # gradient

            displacement_gradient = interpolate_macroQuantities(
            displacement_gradients[i], n_steps)

            microrotation_gradient = interpolate_macroQuantities(
            microrotation_gradients[i], n_steps)

            # Saves the quantities
            
            file_tools.list_toTxt(displacement_macro, "homogenized_dis"+
            "placement", parent_path=base_path+"//text//"+subfolder_name, 
            add_extension=True)
            
            file_tools.list_toTxt(microrotation_macro, "homogenized_mi"+
            "crorotation", parent_path=base_path+"//text//"+
            subfolder_name, add_extension=True)
            
            file_tools.list_toTxt(displacement_gradient, "homogenized_"+
            "displacement_gradient", parent_path=base_path+"//text"+"/"+
            "/"+subfolder_name, add_extension=True)
            
            file_tools.list_toTxt(microrotation_gradient, "homogenized"+
            "_microrotation_grad", parent_path=base_path+"//text//"+
            subfolder_name, add_extension=True)

            # Iterates through the multiscale boundary conditions

            for multiscale_BCs in multiscale_BCsSets:

                # Makes a new mesh just for the first test and if a new 
                # mesh is asked for

                flag_mesh = False

                if flag_newMesh and counter==1:

                    flag_mesh = True

                # Calls the simulation for bending 

                try:

                    beam_case_3(multiscale_BCs[0], multiscale_BCs[1], 
                    base_path, *base_parameters[0:9], gamma_matrix=
                    base_parameters[9], gamma_fiber=base_parameters[10], 
                    RVE_width=base_parameters[11], RVE_length=
                    base_parameters[12], fiber_radius=base_parameters[13
                    ], flag_newMesh=flag_mesh, subfolder_name=[
                    subfolder_name, multiscale_BCs[0]+"_"+multiscale_BCs[
                    1]], fluctuation_field=fluctuation_field, 
                    transfinite_directions=transfinite_directions, 
                    bias_directions=bias_directions, mesh_file_directory=
                    mesh_file_directory)

                except Exception as error_message:

                    print("Error Message:", str(error_message))

                    print("Full Traceback:\n")

                    traceback.print_exc()

                    print("\n\n")

                    print("\n\nThe simulation "+str(counter)+" did not"+
                    " converge")

                    print("\n\n")

                    raise KeyboardInterrupt

# Defines a function to linearly interpolate the final values

def interpolate_macroQuantities(final_value, n_steps, t_initial=0.0, 
t_final=1.0):
    
    # Initializes the macro quantity as a list

    macro_quantity = []

    # Gets the final value as a numpy array

    final_value = np.array(final_value)

    # Iterates through the steps

    for i in range(n_steps):

        # Gets the current time

        t_current = t_initial+((t_final-t_initial)*(i/(n_steps-1)))

        # Multiplies the current time by the numpy array and adds to the
        # list o macro quantities

        macro_quantity.append([t_current, (final_value*t_current).tolist(
        )])

    # Returns the macro quantity list

    return macro_quantity

# Defines a function to try different parameters

def beam_case_3(displacement_multiscaleBC, microrotation_multiscaleBC,
base_path, E_matrix, E_fiber, nu_matrix, nu_fiber, N_micropolarMatrix, 
N_micropolarFiber, characteristic_lengthMatrix, 
characteristic_lengthFiber, flag_bending, gamma_matrix=0.0, gamma_fiber=
0.0, RVE_width=1.0, RVE_length=1.0, fiber_radius=0.25, n_RVEsX=1, 
n_RVEsY=1, n_RVEsZ=1, RVE_localizationX=1, RVE_localizationY=1, 
RVE_localizationZ=1, flag_newMesh=True, subfolder_name=["simulation"],
fluctuation_field=False, transfinite_directions=[6, 6, 3, 4, 3], 
bias_directions={"cylinder radial": 1.5, "box radial": 1.5},
mesh_file_directory=os.getcwd()+"//aa_tests//test_meshes"):

    # Sets the data of the simulation in a txt file

    parent_path = [base_path+"//graphics", base_path+"//text"]

    for name in subfolder_name:

        parent_path[0] += "//"+name

        parent_path[1] += "//"+name

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
    "arameters", parent_path=parent_path[0])

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
    "arameters", parent_path=parent_path[1])

    ####################################################################
    ####################################################################
    ##                    User defined parameters                     ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                        Simulation results                        #
    ####################################################################

    # Defines the path to the results directory 

    results_pathGraphics = base_path+"//graphics"

    results_pathText = base_path+"//text"

    for name in subfolder_name:

        results_pathGraphics += "//"+name

        results_pathText += "//"+name

    if fluctuation_field:

        saving_fileNames = ["displacement_microscale_fluctuation.xdmf", 
        "microrotation_microscale_fluctuation.xdmf", "lambda_displacem"+
        "ent.xdmf", "lambda_grad_displacement.xdmf", "lambda_microrota"+
        "tion.xdmf", "lambda_grad_microrotation.xdmf"]

        homogenized_displacementFileName = ["homogenized_displacement_"+
        "microscale_fluctuation.txt", "homogenized_microrotation_micro"+
        "scale.txt"]

        homogenized_gradDisplacementFileName = ["homogenized_displacem"+
        "ent_gradient_microscale_fluctuation.txt", "homogenized_micror"+
        "otation_grad_microscale.txt"]

        homogenized_piolaFileName = ["homogenized_first_piola_microsca"+
        "le_fluctuation.txt", "homogenized_couple_first_piola_microsca"+
        "le.txt"]

        homogenized_cauchyFileName = ["homogenized_cauchy_microscale_f"+
        "luctuation.txt", "homogenized_couple_cauchy_microscale.txt"]

    else:

        saving_fileNames = ["displacement_microscale.xdmf", "microrota"+
        "tion_microscale.xdmf", "lambda_displacement.xdmf", "lambda_gr"+
        "ad_displacement.xdmf", "lambda_microrotation.xdmf", "lambda_g"+
        "rad_microrotation.xdmf"]

        homogenized_displacementFileName = ["homogenized_displacement_"+
        "microscale.txt", "homogenized_microrotation_microscale.txt"]

        homogenized_gradDisplacementFileName = ["homogenized_displacem"+
        "ent_gradient_microscale.txt", "homogenized_microrotation_grad"+
        "_microscale.txt"]

        homogenized_piolaFileName = ["homogenized_first_piola_microsca"+
        "le.txt", "homogenized_couple_first_piola_microscale.txt"]

        homogenized_cauchyFileName = ["homogenized_cauchy_microscale.t"+
        "xt", "homogenized_couple_cauchy_microscale.txt"]

    if fluctuation_field:

        stress_fieldFileName = ["cauchy_stress_microscale_fluctuation."+
        "xdmf", "couple_cauchy_stress_microscale_fluctuation.xdmf", "f"+
        "irst_piola_stress_microscale.xdmf", "couple_first_piola_stres"+
        "s_microscale.xdmf"]

    else:

        stress_fieldFileName = ["cauchy_stress_microscale.xdmf", "coup"+
        "le_cauchy_stress_microscale.xdmf", "first_piola_stress_micros"+
        "cale.xdmf", "couple_first_piola_stress_microscale.xdmf"]

    post_processes = []

    fields_names = ["Displacement", "Microrotation"]
    
    if (displacement_multiscaleBC=="MinimallyConstrainedFirstOrderBC"
    and microrotation_multiscaleBC!="PeriodicFirstOrderBC"):
    
        fields_names.extend(["Displacement_lagrange_multiplier", "Disp"+
        "lacement_gradient_lagrange_multiplier"]) 

    if (microrotation_multiscaleBC=="MinimallyConstrainedFirstOrderBC"
    and displacement_multiscaleBC!="PeriodicFirstOrderBC"):
    
        fields_names.extend(["Microrotation_lagrange_multiplier", "Mic"+
        "rorotation_gradient_lagrange_multiplier"])

    # Iterates through the fields (displacement and microrotation)

    for i in range(len(fields_names)):

        post_processes.append([fields_names[i], dict()])

        post_processes[-1][-1]["SaveField"] = {"directory path":
        results_pathGraphics, "file name":saving_fileNames[i]}

        if i==0 or i==1:

            # Put "" in the subdomain to integrate over the entire do-
            # main

            post_processes[-1][-1]["HomogenizeField"] = {"directory pa"+
            "th": results_pathText, "file name":
            homogenized_displacementFileName[i], "subdomain":""}

            # Put "" in the subdomain to integrate over the entire do-
            # main

            post_processes[-1][-1]["HomogenizeFieldsGradient"] = {"dir"+
            "ectory path":results_pathText, "file name":
            homogenized_gradDisplacementFileName[i], "subdomain":""}

        # Adds the stress field to the displacement field even though
        # it can be evaluated with any field, since it takes all fields
        # simultaneously

        if i==0:

            """post_processes[-1][-1]["SaveCauchyStressField"] = {"direct"+
            "ory path": results_pathGraphics, "file name": 
            stress_fieldFileName[0], "polynomial degree": 1}

            post_processes[-1][-1]["SaveCoupleCauchyStressField"] = {
            "directory path": results_pathGraphics, "file name": 
            stress_fieldFileName[1], "polynomial degree": 1}"""

            post_processes[-1][-1]["HomogenizeFirstPiola"] = {"directo"+
            "ry path": results_pathText, "file name": 
            homogenized_piolaFileName[0], "subdomain":""}

            post_processes[-1][-1]["HomogenizeCoupleFirstPiola"] = {"d"+
            "irectory path": results_pathText, "file name": 
            homogenized_piolaFileName[1], "subdomain":""}

            """post_processes[-1][-1]["HomogenizeCauchy"] = {"directo"+
            "ry path": results_pathText, "file name": 
            homogenized_cauchyFileName[0], "subdomain":""}

            post_processes[-1][-1]["HomogenizeCoupleCauchy"] = {
            "directory path": results_pathText, "file name": 
            homogenized_cauchyFileName[1], "subdomain":""}"""

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

    mesh_fileName = ("micropolar_beam_with_fibers_microscale_deformati"+
    "on_lab")

    if flag_newMesh:

        beam_gmsh.case_1(RVE_width, RVE_length, fiber_radius, n_RVEsX, 
        n_RVEsY, n_RVEsZ, RVE_localizationX, RVE_localizationY, 
        RVE_localizationZ, mesh_fileName=mesh_fileName, file_directory=
        mesh_file_directory, transfinite_directions=transfinite_directions,
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

    solver_parameters["linear_solver"] = "mumps"#"mumps"

    solver_parameters["newton_relative_tolerance"] = 1e-6#1e-8

    solver_parameters["newton_absolute_tolerance"] = 1e-4#1e-8

    solver_parameters["newton_maximum_iterations"] = 30

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Defines the paths to the macro quantities files

    macro_displacementName = (base_path+"//text//"+subfolder_name[0]+
    "//homogenized_displacement")

    macro_gradDisplacementName= (base_path+"//text//"+subfolder_name[0]+
    "//homogenized_displacement_gradient") 

    macro_microrotationName = (base_path+"//text//"+subfolder_name[0]+
    "//homogenized_microrotation") 

    macro_gradMicrorotationName = (base_path+"//text//"+subfolder_name[0
    ]+"//homogenized_microrotation_grad")

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
    constitutive_model, post_processes, mesh_file_directory+"//"+
    mesh_fileName, solver_parameters, polynomial_degreeDisplacement=
    polynomial_degreeDisplacement, polynomial_degreeMicrorotation=
    polynomial_degreeMicrorotation, verbose=verbose, fluctuation_field=
    fluctuation_field)

case_3(flag_newMesh=True)