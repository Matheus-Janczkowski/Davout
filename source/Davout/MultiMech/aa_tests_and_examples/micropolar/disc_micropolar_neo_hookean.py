# Routine to test a hyperelastic disc

import os

from .....Grante.MultiMech.constitutive_models.hyperelasticity import micropolar_hyperelasticity as micropolar_constitutiveModels

from .....Grante.MultiMech.physics import hyperelastic_micropolar_continuum as variational_framework

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

########################################################################
########################################################################
##                      User defined parameters                       ##
########################################################################
########################################################################

########################################################################
#                          Simulation results                          #
########################################################################

# Defines the path to the results directory 

results_pathGraphics = os.getcwd()+"//tests//micropolar//results//graphics"

results_pathText = os.getcwd()+"//tests//micropolar//results//text"

displacement_fileName = ["displacement.xdmf", "microrotation.xdmf"]

homogenized_displacementFileName = ["homogenized_displacement.txt", "h"+
"omogenized_microrotation.txt"]

homogenized_gradDisplacementFileName = ["homogenized_displacement_grad"+
"ient.txt", "homogenized_microrotation_grad.txt"]

post_processes = []

fields_names = ["Displacement", "Microrotation"]

# Iterates through the fields (displacement and microrotation)

for i in range(2):

    post_processes.append([fields_names[i], dict()])

    post_processes[-1][-1]["SaveField"] = {"directory path":results_pathGraphics, 
    "file name":displacement_fileName[i]}

    # Put "" in the subdomain to integrate over the entire domain

    post_processes[-1][-1]["HomogenizeField"] = {"directory path":
    results_pathText, "file name":homogenized_displacementFileName[i], 
    "subdomain":""}

    # Put "" in the subdomain to integrate over the entire domain

    post_processes[-1][-1]["HomogenizeFieldsGradient"] = {"directory path":
    results_pathText, "file name":homogenized_gradDisplacementFileName[i], 
    "subdomain":""}

post_processesSubmesh = []

########################################################################
#                         Material properties                          #
########################################################################

# Sets a dictionary of properties

material_properties = dict()

E = 100E6

nu = 0.4

flag_bending = True

characteristic_length = 0.1

gamma = 0.0

alpha = 0.0

material_properties["E"] = E

material_properties["nu"] = nu

material_properties["flag bending"] = flag_bending

material_properties["characteristic length"] = characteristic_length

material_properties["N"] = 0.1

material_properties["alpha"] = alpha

material_properties["gamma"] = gamma

# Sets the material as a HGO material

constitutive_model = dict()

constitutive_model[tuple([1,2,3])] = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
material_properties)

#constitutive_model[tuple(["annulus fibrosus", "nucleus pulposus", ("en"+
#"d plate")])] = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
#material_properties)

#constitutive_model = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
#material_properties)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

file_directory = (get_parent_path_of_file(path_bits_to_be_excluded=2)+
"//test_meshes")

mesh_fileName = "intervertebral_disc"

# Defines a set of physical groups to create a submesh

volume_physGroupsSubmesh = []

########################################################################
#                            Function space                            #
########################################################################

# Defines the shape functions degree

polynomial_degreeDisplacement = 2

polynomial_degreeMicrorotation = 1

########################################################################
#                           Solver parameters                          #
########################################################################

# Sets the solver parameters in a dictionary

solver_parameters = dict()

solver_parameters["linear_solver"] = "mumps"#"minres"

solver_parameters["newton_relative_tolerance"] = 1e-5#1e-3

solver_parameters["newton_absolute_tolerance"] = 1e-5#1e-3

solver_parameters["newton_maximum_iterations"] = 10#50

# Sets the initial time

t = 0.0

# Sets the final pseudotime of the simulation

t_final = 1.0

# Sets the maximum number of steps of loading

maximum_loadingSteps = 5

########################################################################
#                          Boundary conditions                         #
########################################################################

# Defines a load expression

maximum_load = 4E1

# Assemble the traction vector using this load expression

traction_boundary = {"load case": "NormalReferentialTorsion", "amp"+
"litude_torsion": 100*maximum_load, "parametric_load_curve": "squa"+
"re root", "t": t, "t_final": t_final}#, "influence_radius": 0.10}

# Defines a dictionary of tractions

traction_dictionary = dict()

# Defines a dictionary of moments on the boundary

moment_boundary = {"load case": "UniformReferentialTraction", "amp"+
"litude_tractionX": 0.0, "amplitude_tractionY": 0.0, "amplitude_tr"+
"actionZ": 0.0, "parametric_load_curve": "linear", "t": t, "t_final": 
t_final}

moment_dictionary = dict()

# Defines a dictionary of boundary conditions. Each key is a physi-
# cal group and each value is another dictionary or a list of dic-
# tionaries with the boundary conditions' information. Each of these 
# dictionaries must have the key "BC case", which carries the name 
# of the function that builds this boundary condition

bcs_dictionary = dict()

bcs_dictionary["bottom"] = {"BC case": "FixedSupportDirichletBC", 
"sub_fieldsToApplyBC": ["Displacement", "Microrotation"]}

bcs_dictionary["top"] = {"BC case": "PrescribedDirichletBC", "bc_infor"+
"mationsDict": {"load_function": "linear", "degrees_ofFreedomList": 2,
"end_point": [1.0, 5E-2], "sub_fieldsToApplyBC": "Displacement"}}

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Defines a flag to print every step

verbose = True

# Solves the variational problem

variational_framework.hyperelasticity_displacementMicrorotationBased(
constitutive_model, traction_dictionary, moment_dictionary, 
maximum_loadingSteps, t_final, post_processes, file_directory+"//"+
mesh_fileName, solver_parameters, polynomial_degreeDisplacement=
polynomial_degreeDisplacement, polynomial_degreeMicrorotation=
polynomial_degreeMicrorotation, t=t, solution_name=[["Displacement", 
"DNS"], ["Microrotation", "DNS"]], volume_physGroupsSubmesh=
volume_physGroupsSubmesh, post_processesSubmesh=
post_processesSubmesh, verbose=verbose, dirichlet_boundaryConditions=
bcs_dictionary)