# Routine to test a hyperelastic disc

import os

from dolfin import *

from mshr import *

from ....constitutive_models.hyperelasticity import micropolar_hyperelasticity as micropolar_constitutiveModels

from ....multiscale import multiscale_micropolar as variational_framework

from ....tool_box import file_handling_tools as file_tools

from .....CuboidGmsh.aa_tests.micropolar_meshes import beam_micropolar_case_1 as beam_gmsh

# Defines a function to try multiple parameters

def case1_varyingMicropolarNumber(flag_newMesh=False):

    # Sets the mesh refinement

    transfinite_directions = [12, 12, 12, 6, 8]

    bias_directions = {"cylinder radial": 1.5, "box radial": 1.1}

    # Sets the multiscale boundary conditions for each one of the fields

    multiscale_BCsSets = [["PeriodicFirstOrderBC", "PeriodicFirstOrder"+
    "BC"], ["LinearFirstOrderBC", "LinearFirstOrderBC"], ["MinimallyCo"+
    "nstrainedFirstOrderBC", "MinimallyConstrainedFirstOrderBC"]]

    # Defines a flag to use the fluctuation of the field instead of the
    # field proper in the BVP

    fluctuation_field = True

    # Reads the parameters set

    base_path = os.getcwd()+"//tests//micropolar//traction_case//results"

    parameters_sets = file_tools.txt_toList(base_path+"//parameters_se"+
    "ts")

    # Sets a list of names for each set of parameters, which will yield
    # different simulations

    simulations_names = ["simulation_11", "simulation_12", "simulation"+
    "_13", "simulation_21", "simulation_22", "simulation_23", "simulat"+
    "ion_31", "simulation_32", "simulation_33"]

    # Defines a list of lists, each list is a set of material parameters:
    # 0.  Young modulus of the matrix
    # 1.  Young modulus of the fiber
    # 2.  Poisson ratio of the matrix
    # 3.  Poisson ratio of the fiber
    # 4.  micropolar number of the matrix
    # 5.  micropolar number of the fiber
    # 6.  characteristic length of the matrix
    # 7.  characteristic length of the fiber
    # 8.  flag for bending
    # 9.  load factor
    # 10. gamma of the matrix
    # 11. gamma of the fiber
    # 12. RVE width
    # 13. RVE length
    # 14. radius of the fiber

    counter = 0

    for multiscale_BCs in multiscale_BCsSets:

        # Iterates through the simulations

        for i in range(min([len(parameters_sets),len(simulations_names)]
        )):

            # Makes a new mesh just for the first test and if a new mesh 
            # is asked for

            flag_mesh = False

            if flag_newMesh and counter==0:

                flag_mesh = True

            counter += 1

            # Calls the simulation for traction 

            beam_case_1(multiscale_BCs[0], multiscale_BCs[1], base_path, 
            *parameters_sets[i][0:10], gamma_matrix=parameters_sets[i][
            10], gamma_fiber=parameters_sets[i][11], RVE_width=
            parameters_sets[i][12], RVE_length=parameters_sets[i][13], 
            fiber_radius=parameters_sets[i][14], flag_newMesh=flag_mesh, 
            subfolder_name=[simulations_names[i], multiscale_BCs[0]+"_"+
            multiscale_BCs[1]], fluctuation_field=fluctuation_field, 
            transfinite_directions=transfinite_directions,
            RVE_localizationX=parameters_sets[i][15], RVE_localizationY=
            parameters_sets[i][16], RVE_localizationZ=parameters_sets[i
            ][17], bias_directions=bias_directions)

# Defines a function to try different parameters

def beam_case_1(displacement_multiscaleBC, microrotation_multiscaleBC,
base_path, E_matrix, E_fiber, nu_matrix, nu_fiber, N_micropolarMatrix, 
N_micropolarFiber, characteristic_lengthMatrix, 
characteristic_lengthFiber, flag_bending, load_factor, gamma_matrix=0.0, 
gamma_fiber=0.0, RVE_width=1.0, RVE_length=1.0, fiber_radius=0.25, 
n_RVEsX=1, n_RVEsY=1, n_RVEsZ=1, RVE_localizationX=1, RVE_localizationY=
1, RVE_localizationZ=3, flag_newMesh=True, subfolder_name=["simulation"],
fluctuation_field=False, transfinite_directions=[6, 6, 3, 4, 3], 
bias_directions={"cylinder radial": 1.5, "box radial": 1.5}):

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

        saving_fileNames = ["displacement_microscale_fluctuation.xdmf", "microrotation"+
        "_microscale_fluctuation.xdmf", "lambda_displacement.xdmf", "lambda_grad_displ"+
        "acement.xdmf", "lambda_microrotation.xdmf", "lambda_grad_microrot"+
        "ation.xdmf"]

        homogenized_displacementFileName = ["homogenized_displacement_micr"+
        "oscale_fluctuation.txt", "homogenized_microrotation_microscale.txt"]

        homogenized_gradDisplacementFileName = ["homogenized_displacement_"+
        "gradient_microscale_fluctuation.txt", "homogenized_microrotation_grad_microsc"+
        "ale.txt"]

        homogenized_piolaFileName = ["homogenized_first_piola_microscale_fluctuation.t"+
        "xt", "homogenized_couple_first_piola_microscale.txt"]

        homogenized_cauchyFileName = ["homogenized_cauchy_microscale_fluctuation.t"+
        "xt", "homogenized_couple_cauchy_microscale.txt"]

    else:

        saving_fileNames = ["displacement_microscale.xdmf", "microrotation"+
        "_microscale.xdmf", "lambda_displacement.xdmf", "lambda_grad_displ"+
        "acement.xdmf", "lambda_microrotation.xdmf", "lambda_grad_microrot"+
        "ation.xdmf"]

        homogenized_displacementFileName = ["homogenized_displacement_micr"+
        "oscale.txt", "homogenized_microrotation_microscale.txt"]

        homogenized_gradDisplacementFileName = ["homogenized_displacement_"+
        "gradient_microscale.txt", "homogenized_microrotation_grad_microsc"+
        "ale.txt"]

        homogenized_piolaFileName = ["homogenized_first_piola_microscale.t"+
        "xt", "homogenized_couple_first_piola_microscale.txt"]

        homogenized_cauchyFileName = ["homogenized_cauchy_microscale.t"+
        "xt", "homogenized_couple_cauchy_microscale.txt"]

    if fluctuation_field:

        stress_fieldFileName = ["cauchy_stress_microscale_fluctuation.xdmf", "couple_c"+
        "auchy_stress_microscale_fluctuation.xdmf", "first_piola_stress_microscale.xdmf",
        "couple_first_piola_stress_microscale.xdmf"]

    else:

        stress_fieldFileName = ["cauchy_stress_microscale.xdmf", "couple_c"+
        "auchy_stress_microscale.xdmf", "first_piola_stress_microscale.xdmf",
        "couple_first_piola_stress_microscale.xdmf"]

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

            post_processes[-1][-1]["SaveCauchyStressField"] = {"direct"+
            "ory path": results_pathGraphics, "file name": 
            stress_fieldFileName[0], "polynomial degree": 1}

            post_processes[-1][-1]["SaveCoupleCauchyStressField"] = {
            "directory path": results_pathGraphics, "file name": 
            stress_fieldFileName[1], "polynomial degree": 1}

            post_processes[-1][-1]["HomogenizeFirstPiola"] = {"directo"+
            "ry path": results_pathText, "file name": 
            homogenized_piolaFileName[0], "subdomain":""}

            post_processes[-1][-1]["HomogenizeCoupleFirstPiola"] = {"d"+
            "irectory path": results_pathText, "file name": 
            homogenized_piolaFileName[1], "subdomain":""}

            post_processes[-1][-1]["HomogenizeCauchy"] = {"directo"+
            "ry path": results_pathText, "file name": 
            homogenized_cauchyFileName[0], "subdomain":""}

            post_processes[-1][-1]["HomogenizeCoupleCauchy"] = {
            "directory path": results_pathText, "file name": 
            homogenized_cauchyFileName[1], "subdomain":""}

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

    solver_parameters["linear_solver"] = "mumps"#"mumps"

    solver_parameters["newton_relative_tolerance"] = 1e-6#1e-8

    solver_parameters["newton_absolute_tolerance"] = 1e-4#1e-8

    solver_parameters["newton_maximum_iterations"] = 30

    """

    solver_parameters["preconditioner"] = "petsc_amg"

    solver_parameters["krylov_absolute_tolerance"] = 1e-5

    solver_parameters["krylov_relative_tolerance"] = 1e-5

    solver_parameters["krylov_maximum_iterations"] = 15000

    solver_parameters["krylov_monitor_convergence"] = False"""

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
    constitutive_model, post_processes, file_directory+"//"+
    mesh_fileName, solver_parameters, polynomial_degreeDisplacement=
    polynomial_degreeDisplacement, polynomial_degreeMicrorotation=
    polynomial_degreeMicrorotation, verbose=verbose, fluctuation_field=
    fluctuation_field)

case1_varyingMicropolarNumber(flag_newMesh=True)