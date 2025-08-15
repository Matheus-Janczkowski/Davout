# Routine to test a hyperelastic disc

import os

import traceback

import numpy as np

from ....constitutive_models.hyperelasticity import micropolar_hyperelasticity as micropolar_constitutiveModels

from ....physics import hyperelastic_micropolar_continuum as variational_framework

from ....tool_box import file_handling_tools as file_tools

from .....CuboidGmsh.aa_tests.micropolar_meshes import beam_micropolar_case_1 as beam_gmsh

# Defines a function to try multiple parameters

def case1_varyingMicropolarNumber(flag_newMesh=False):

    # Sets the mesh file directory path

    mesh_file_directory = os.path.join(file_tools.get_parent_path_of_file(
    file=__file__, path_bits_to_be_excluded=3),"test_meshes")

    # Defines the load factor

    load_factor = 1.0

    # Defines the RVE overall parameters

    RVE_width = 1.0

    RVE_length = 1.0

    # Defines the fiber radius

    fiber_radius = 0.25

    # Defines the number of RVEs at each direction

    n_RVEsX = 1

    n_RVEsY = 7

    n_RVEsZ = 25

    # Sets the Young modulus and the Poisson ration from psi to MPa

    nu_matrix = 0.4

    nu_fiber = 0.4

    flag_bending = True

    gamma_matrix = 0.0

    gamma_fiber = 0.0

    characteristic_lengthMatrix = RVE_width*1.0
    
    characteristic_lengthFiber = RVE_width*1.0

    # Sets the x, y, and z indices of the RVE to be selected for homoge-
    # nization. These indices begin with 1

    RVE_localizationX = int(np.ceil(0.5*n_RVEsX))

    RVE_localizationY = 6#np.ceil(0.5*n_RVEsY)

    RVE_localizationZ = int(np.ceil(0.5*n_RVEsZ))

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

    E_matrix = 100E6

    E_fiber = 1.0*E_matrix

    test11 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.001, 0.001, 
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]
    
    test12 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.01, 0.01, 
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]
    
    test13 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.08, 0.08, 
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]

    E_matrix = 100E6

    E_fiber = 10.0*E_matrix

    test21 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.001, 0.001,
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]
    
    test22 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.01, 0.01, 
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]
    
    test23 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.08, 0.08, 
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]

    E_matrix = 100E6

    E_fiber = 100.0*E_matrix

    test31 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.001, 0.001, 
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]
    
    test32 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.01, 0.01, 
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]
    
    test33 = [E_matrix, E_fiber, nu_matrix, nu_fiber, 0.08, 0.08, 
    characteristic_lengthMatrix, characteristic_lengthFiber, 
    flag_bending, load_factor, gamma_matrix, gamma_fiber, RVE_width, 
    RVE_length, fiber_radius, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ]

    # Sets a dictionary with names and parameters sets

    parameters_sets = {"simulation_11": test11, "simulation_13": test13, 
    "simulation_31": test31, "simulation_33": test33}

    # Saves the parameters set

    base_path = os.path.join(file_tools.get_parent_path_of_file(file=
    __file__), "results")

    file_tools.list_toTxt(file_tools.named_list(parameters_sets), "par"+
    "ameters_sets", parent_path=base_path)

    # Iterates through the simulations

    counter = 0

    for simulation_name, parameters_set in parameters_sets.items():

        # Makes a new mesh just for the first test and if a new mesh is
        # asked for

        flag_mesh = False

        if flag_newMesh and counter==0:

            flag_mesh = True

        counter += 1

        # Calls the simulation for bending

        try:

            beam_case_1(base_path, *parameters_set[0:10], gamma_matrix=
            parameters_set[10], gamma_fiber=parameters_set[11], 
            RVE_width=parameters_set[12], RVE_length=parameters_set[13], 
            fiber_radius=parameters_set[14], n_RVEsX=n_RVEsX, n_RVEsY=
            n_RVEsY, n_RVEsZ=n_RVEsZ, RVE_localizationX=
            RVE_localizationX, RVE_localizationY=RVE_localizationY, 
            RVE_localizationZ=RVE_localizationZ, flag_newMesh=flag_mesh, 
            subfolder_name=simulation_name, mesh_file_directory=
            mesh_file_directory)

        # Shows the exception and keeps going

        except Exception as error_message:

            print("\n\nSimulation "+simulation_name+" did not converge")

            print("Error Message:", str(error_message))

            print("Full Traceback:\n")

            traceback.print_exc()

            print("\n\n")

# Defines a function to try different parameters

def beam_case_1(base_path, E_matrix, E_fiber, nu_matrix, nu_fiber, 
N_micropolarMatrix, N_micropolarFiber, characteristic_lengthMatrix, 
characteristic_lengthFiber, flag_bending, load_factor, gamma_matrix=0.0, 
gamma_fiber=0.0, RVE_width=1.0, RVE_length=1.0, fiber_radius=0.25, 
n_RVEsX=1, n_RVEsY=1, n_RVEsZ=5, RVE_localizationX=1, RVE_localizationY=
1, RVE_localizationZ=3, flag_newMesh=True, subfolder_name="simulation",
mesh_file_directory=os.getcwd()+"//aa_tests//test_meshes"):

    # Evaluates the fiber ratio

    fiber_ratio = (np.pi*(fiber_radius**2))/(RVE_length**2)

    # Mixes them using the lower bound of the rule of mixtures to give a
    # conservative load

    E_mixture = 1.0/((fiber_ratio/E_fiber)+((1-fiber_ratio)/E_matrix))

    # Defines the maximum load
    
    maximum_load = ((0.5*load_factor*E_mixture*((RVE_length*n_RVEsX*((
    n_RVEsY*RVE_width)**3))/12))/((n_RVEsZ*RVE_width)**3))

    # Sets the data of the simulation in a txt file

    file_tools.list_toTxt(file_tools.named_list({"E_matrix:": E_matrix, 
    "E_fiber:": E_fiber, "nu_matrix:": nu_matrix, "nu_fiber:": nu_fiber, 
    "N_micropolarMatrix:": N_micropolarMatrix, "N_micropolarFiber:": 
    N_micropolarFiber, "characteristic_lengthMatrix:": 
    characteristic_lengthMatrix, "characteristic_lengthFiber:": 
    characteristic_lengthFiber, "flag_bending:": flag_bending, "load_f"+
    "actor:": load_factor, "gamma_matrix:": gamma_matrix, "gamma_fiber:": 
    gamma_fiber, "RVE_width:": RVE_width, "RVE_length:": RVE_length, 
    "fiber_radius:": fiber_radius, "RVE_localizationX": 
    RVE_localizationX, "RVE_localizationY": RVE_localizationY, "RVE_lo"+
    "calizationZ": RVE_localizationZ, "E_mixture": E_mixture, "maximum"+
    "_load": maximum_load}), "00_parameters", parent_path=
    base_path+"//graphics//"+subfolder_name)

    file_tools.list_toTxt(file_tools.named_list({"E_matrix:": E_matrix, 
    "E_fiber:": E_fiber, "nu_matrix:": nu_matrix, "nu_fiber:": nu_fiber, 
    "N_micropolarMatrix:": N_micropolarMatrix, "N_micropolarFiber:": 
    N_micropolarFiber, "characteristic_lengthMatrix:": 
    characteristic_lengthMatrix, "characteristic_lengthFiber:": 
    characteristic_lengthFiber, "flag_bending:": flag_bending, "load_f"+
    "actor:": load_factor, "gamma_matrix:": gamma_matrix, "gamma_fiber:": 
    gamma_fiber, "RVE_width:": RVE_width, "RVE_length:": RVE_length, 
    "fiber_radius:": fiber_radius, "RVE_localizationX": 
    RVE_localizationX, "RVE_localizationY": RVE_localizationY, "RVE_lo"+
    "calizationZ": RVE_localizationZ, "E_mixture": E_mixture, "maximum"+
    "_load": maximum_load}), "00_parameters", parent_path=
    base_path+"//text//"+subfolder_name)

    ####################################################################
    #                        Simulation results                        #
    ####################################################################

    # Defines the path to the results directory 

    results_pathGraphics = base_path+"//graphics//"+subfolder_name

    results_pathText = base_path+"//text//"+subfolder_name

    displacement_fileName = ["displacement.xdmf", "microrotation.xdmf"]

    displacement_fileNameSubmesh = ["displacement_submesh.xdmf", "micr"+
    "orotation_submesh.xdmf"]

    homogenized_displacementFileName = ["homogenized_displacement.txt", 
    "homogenized_microrotation.txt"]

    homogenized_gradDisplacementFileName = ["homogenized_displacement_"+
    "gradient.txt", "homogenized_microrotation_grad.txt"]

    homogenized_piolaFileName = ["homogenized_first_piola.txt", "homog"+
    "enized_couple_first_piola.txt"]

    homogenized_cauchyFileName = ["homogenized_cauchy.txt", "homogeniz"+
    "ed_couple_cauchy.txt"]

    stress_fieldFileName = ["cauchy_stress.xdmf", "couple_cauchy_stres"+
    "s.xdmf", "first_piola_stress.xdmf", "couple_first_piola_stress.xd"+
    "mf"]

    stress_fieldFileNameSubmesh = ["cauchy_stress_submesh.xdmf", "coup"+
    "le_cauchy_stress_submesh.xdmf", "first_piola_stress_submesh.xdmf", 
    "couple_first_piola_stress_submesh.xdmf"]

    post_processes = []

    post_processesSubmesh = []

    fields_names = ["Displacement", "Microrotation"]

    # Iterates through the fields (displacement and microrotation)

    for i in range(2):

        # Adds a pair of field number following the variational conven-
        # tion and the dictionary for post processes

        post_processes.append([fields_names[i], dict()])

        post_processes[-1][-1]["SaveField"] = {"directory path":
        results_pathGraphics, "file name":displacement_fileName[i]}

        # Put "" in the subdomain to integrate over the entire domain

        post_processes[-1][-1]["HomogenizeField"] = {"directory path":
        results_pathText, "file name":homogenized_displacementFileName[i
        ], "subdomain": ["RVE matrix", "RVE fiber"]}

        # Put "" in the subdomain to integrate over the entire domain

        post_processes[-1][-1]["HomogenizeFieldsGradient"] = {"directo"+
        "ry path":results_pathText, "file name":
        homogenized_gradDisplacementFileName[i], "subdomain":["RVE mat"+
        "rix", "RVE fiber"]}

        # Adds the stress field to the displacement field even though
        # it can be evaluated with any field, since it takes all fields
        # simultaneously

        if i==0:

            post_processes[-1][-1]["HomogenizeFirstPiola"] = {"directo"+
            "ry path": results_pathText, "file name": 
            homogenized_piolaFileName[0], "subdomain": ["RVE matrix", 
            "RVE fiber"]}

            post_processes[-1][-1]["HomogenizeCoupleFirstPiola"] = {"d"+
            "irectory path": results_pathText, "file name": 
            homogenized_piolaFileName[1], "subdomain": ["RVE matrix", 
            "RVE fiber"]}

            post_processes[-1][-1]["HomogenizeCauchy"] = {"directo"+
            "ry path": results_pathText, "file name": 
            homogenized_cauchyFileName[0], "subdomain": ["RVE matrix", 
            "RVE fiber"]}

            post_processes[-1][-1]["HomogenizeCoupleCauchy"] = {
            "directory path": results_pathText, "file name": 
            homogenized_cauchyFileName[1], "subdomain": ["RVE matrix", 
            "RVE fiber"]}

            """post_processes[-1][-1]["FirstElasticityTensorAtPoint"] = {
            "directory path": results_pathText, "file name": 
            "first_elasticity_tensor_dP_dF", "polynomial degree": 1,
            "point coordinates": [RVE_length*(RVE_localizationX-0.5), 
            RVE_width*(RVE_localizationY-0.5), RVE_width*(
            RVE_localizationZ-0.5)], "flag plotting": True, "voigt not"+
            "ation": "conventional", "plotting arguments": {"scaling f"+
            "unction additional parameters": {"alpha":5}}}"""

        # Adds a pair of field number following the variational conven-
        # tion and the dictionary for post processes

        post_processesSubmesh.append([fields_names[i], dict()])

        post_processesSubmesh[-1][-1]["SaveField"] = {"directory path":
        results_pathGraphics, "file name":displacement_fileNameSubmesh[i
        ]}

        if i==0:

            post_processesSubmesh[-1][-1]["SaveCauchyStressField"] = {
            "directory path": results_pathGraphics, "file name": 
            stress_fieldFileNameSubmesh[0], "polynomial degree": 1}

            post_processesSubmesh[-1][-1]["SaveCoupleCauchyStressField"
            ] = {"directory path": results_pathGraphics, "file name": 
            stress_fieldFileNameSubmesh[1], "polynomial degree": 1}

            post_processesSubmesh[-1][-1]["SaveFirstPiolaStressField"]={
            "directory path": results_pathGraphics, "file name": 
            stress_fieldFileNameSubmesh[2], "polynomial degree": 1}

            post_processesSubmesh[-1][-1]["SaveCoupleFirstPiolaStressF"+
            "ield"] = {"directory path": results_pathGraphics, "file n"+
            "ame": stress_fieldFileNameSubmesh[3], "polynomial degree": 
            1}

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

    constitutive_model[("Matrix","RVE matrix")] = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
    material_propertiesMatrix)

    constitutive_model[("Fiber","RVE fiber")] = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
    material_propertiesFiber)

    ####################################################################
    #                               Mesh                               #
    ####################################################################

    # Defines the name of the file to save the mesh in. Do not write the 
    # file termination, e.g. .msh or .xdmf; both options will be saved 
    # automatically

    mesh_fileName = "micropolar_beam_with_fibers_bending"

    if flag_newMesh:

        beam_gmsh.case_1(RVE_width, RVE_length, fiber_radius, n_RVEsX, 
        n_RVEsY, n_RVEsZ, RVE_localizationX, RVE_localizationY, 
        RVE_localizationZ, mesh_fileName=mesh_fileName, file_directory=
        mesh_file_directory)

    # Defines a set of physical groups to create a submesh

    volume_physGroupsSubmesh = ["RVE matrix", "RVE fiber"]

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

    # Sets the initial time

    t = 0.0

    # Sets the final pseudotime of the simulation

    t_final = 1.0

    # Sets the maximum number of steps of loading

    maximum_loadingSteps = 5

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Assemble the traction vector using this load expression
    
    traction_boundary = {"load case": "UniformReferentialTraction", "a"+
    "mplitude_tractionX": 0.0, "amplitude_tractionY": maximum_load, "a"+
    "mplitude_tractionZ": 0.0, "parametric_load_curve": "linear", "t": 
    t, "t_final": t_final}

    # Defines a dictionary of tractions

    traction_dictionary = dict()

    traction_dictionary["upper"] = traction_boundary

    traction_dictionary["lower"] = traction_boundary

    # Defines a dictionary of moments on the boundary

    moment_boundary = {"load case": "UniformReferentialTraction", "amp"+
    "litude_tractionX": 0.0, "amplitude_tractionY": 0.0, "amplitude_tr"+
    "actionZ": 0.0, "parametric_load_curve": "linear", "t": t, "t_final": 
    t_final}

    moment_dictionary = dict()

    moment_dictionary["lower"] = moment_boundary

    # Defines a dictionary of boundary conditions. Each key is a physi-
    # cal group and each value is another dictionary or a list of dic-
    # tionaries with the boundary conditions' information. Each of these 
    # dictionaries must have the key "BC case", which carries the name 
    # of the function that builds this boundary condition

    bcs_dictionary = dict()

    bcs_dictionary["back"] = {"BC case": "FixedSupportDirichletBC", 
    "sub_fieldsToApplyBC": ["Displacement", "Microrotation"]}

    ####################################################################
    ####################################################################
    ##                    Calculation and solution                    ##
    ####################################################################
    ####################################################################

    # Defines a flag to print every step

    verbose = True

    # Solves the variational problem

    variational_framework.hyperelasticity_displacementMicrorotationBased(
    constitutive_model, traction_dictionary, moment_dictionary, 
    maximum_loadingSteps, t_final, post_processes, mesh_file_directory+"//"+
    mesh_fileName, solver_parameters, polynomial_degreeDisplacement=
    polynomial_degreeDisplacement, polynomial_degreeMicrorotation=
    polynomial_degreeMicrorotation, t=t, solution_name=[["Displacement",
    "DNS"], ["Microrotation", "DNS"]], volume_physGroupsSubmesh=
    volume_physGroupsSubmesh, post_processesSubmesh=
    post_processesSubmesh, verbose=verbose, dirichlet_boundaryConditions=
    bcs_dictionary)

case1_varyingMicropolarNumber(flag_newMesh=False)