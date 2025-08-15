# Routine to test a hyperelastic disc

import os

import source.constitutive_models.hyperelasticity.isotropic_hyperelasticity as constitutive_models

import source.physics.hyperelastic_cauchy_continuum as variational_framework

########################################################################
########################################################################
##                      User defined parameters                       ##
########################################################################
########################################################################

########################################################################
#                          Simulation results                          #
########################################################################

# Defines the path to the results directory 

results_path = os.getcwd()+"//tests//hyperelasticity//results"

displacement_fileName = "displacement.xdmf"

pressure_fileName = "pressure_points.txt"

traction_fileName = "traction.xdmf"

post_processes = dict()

post_processes["SaveField"] = {"directory path":results_path, 
"file name":displacement_fileName}

post_processes["SavePressureAtPoint"] = {"directory path":results_path, 
"file name":pressure_fileName, "polynomial degree": 1, "point coordina"+
"tes": [0.0, 0.0, 0.0], "flag plotting": True}

post_processes["SaveReferentialTractionField"] = {"directory path":
results_path, "file name": traction_fileName, "polynomial degree":1}

########################################################################
#                         Material properties                          #
########################################################################

# Sets the Young modulus and the Poisson ratio

E = 100E6

v = 0.4

# Sets a dictionary of properties

material_properties = dict()

material_properties["E"] = E

material_properties["nu"] = v

# Sets the material as a neo-hookean material using the corresponding
# class

constitutive_model = constitutive_models.Neo_Hookean(material_properties)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = "tests//test_meshes//intervertebral_disc"

########################################################################
#                            Function space                            #
########################################################################

# Defines the shape functions degree

polynomial_degree = 2

########################################################################
#                           Solver parameters                          #
########################################################################

# Sets the solver parameters in a dictionary

solver_parameters = dict()

solver_parameters["newton_relative_tolerance"] = 1e-4

solver_parameters["newton_absolute_tolerance"] = 1e-4

solver_parameters["newton_maximum_iterations"] = 15

"""

solver_parameters["linear_solver"] = "minres"

solver_parameters["preconditioner"] = "petsc_amg"

solver_parameters["krylov_absolute_tolerance"] = 1e-7

solver_parameters["krylov_relative_tolerance"] = 1e-7

solver_parameters["krylov_maximum_iterations"] = 15000

solver_parameters["krylov_monitor_convergence"] = False"""

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

maximum_load = 2E7

# Assemble the traction vector using this load expression

#"""
traction_boundary = {"load case": "UniformReferentialTraction", "ampli"+
"tude_tractionX": 0.0, "amplitude_tractionY": 0.0, "amplitude_tractionZ": 
maximum_load, "parametric_load_curve": "square root", "t": t, "t_final":
t_final}#"""

"""
traction_boundary = {"load case": "NormalUniformFollowerTraction", "am"+
"plitude_traction": 3.6*maximum_load, "parametric_load_curve": "square"+
" root", "t": t, "t_final": t_final}"""

"""
traction_boundary = {"load case": "NormalFollowerTorsion", "amplitude_"+
"torsion": 0.0045*maximum_load, "parametric_load_curve": "square root", 
"t": t, "t_final": t_final, "influence_radius": 0.10}#"""

"""
traction_boundary = {"load case": "NormalFollowerMoment", "amplitude_b"+
"endingMoment": 0.015*maximum_load, "parametric_load_curve": "square r"+
"oot", "t": t, "t_final": t_final, "influence_radius": 0.1, "bending"+
"_axis": [1.0, 0.0, 0.0], "no_parasiticForcesCorrection": False}"""

# Defines a dictionary of tractions

traction_dictionary = dict()

traction_dictionary["top"] = traction_boundary

# Defines a dictionary of boundary conditions. Each key is a physical
# group and each value is another dictionary or a list of dictionaries 
# with the boundary conditions' information. Each of these dictionaries
# must have the key "BC case", which carries the name of the function 
# that builds this boundary condition

bcs_dictionary = dict()

bcs_dictionary["bottom"] = {"BC case": "FixedSupportDirichletBC"}

"""
bcs_dictionary["top"] = {"BC case": "PrescribedDirichletBC", "bc_infor"+
"mationsDict": {"load_function": "linear", "degrees_ofFreedomList": 2,
"end_point": [1.0, 5E-2]}}"""

"""
bcs_dictionary["top"] = {"BC case": "PrescribedDirichletBC", "bc_infor"+
"mationsDict": {"load_function": "SurfaceTranslationAndRotation", "tra"+
"nslation": [0.0, 0.0, 0.05], "in_planeSpinDirection": [1.0, 0.0, 0.0], 
"in_planeSpin": 15, "normal_toPlaneSpin": 45.0}}"""

# Defines a dictionary of body forces

body_forcesDict = dict()

body_forcesDict[""] = {"load case": "UniformReferentialBodyForce", "am"+
"plitude_bodyX":0.0, "amplitude_bodyY": 0.0, "amplitude_bodyZ": 2E5}

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Solves the variational problem

variational_framework.hyperelasticity_displacementBased(
constitutive_model, traction_dictionary, maximum_loadingSteps, t_final, 
post_processes, mesh_fileName, solver_parameters, polynomial_degree=
polynomial_degree, t=t, dirichlet_boundaryConditions=bcs_dictionary, 
verbose=True, body_forcesDict=body_forcesDict)