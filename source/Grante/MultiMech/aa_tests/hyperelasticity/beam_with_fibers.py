# Routine to test a hyperelastic disc

import os

#import periodic_structure as mesher

from .....Grante.PythonicUtilities import file_handling_tools

from .....Grante.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as constitutive_models

from .....Grante.MultiMech.physics import hyperelastic_cauchy_continuum as variational_framework

from .....Grante.MultiMech.aa_tests_and_examples import test_meshes_paths

########################################################################
########################################################################
##                      User defined parameters                       ##
########################################################################
########################################################################

########################################################################
#                          Simulation results                          #
########################################################################

# Defines the path to the results directory 

results_path = os.path.join(file_handling_tools.get_parent_path_of_file(
), "results")

displacement_fileName = "displacement.xdmf"

post_processes = dict()

post_processes["SaveField"] = {"directory path":results_path, 
"file name":displacement_fileName}

post_processes["SaveCauchyStressField"] = {"directory path": 
results_path, "file name": "cauchy_stress.xdmf", "polynomia"+
"l degree": 1}

"""post_processes["FirstElasticityTensorAtPoint"] = {"directory path": 
results_path, "file name": "first_elasticity_tensor_dP_dF", "polynomia"+
"l degree": 1, "point coordinates": [0.5, 0.5, 1.0], "flag plotting": 
False, "voigt notation": "conventional", "plotting arguments": {"scalin"+
"g function additional parameters": {"alpha":5}}}

post_processes["SecondElasticityTensorAtPoint"] = {"directory path": 
results_path, "file name": "second_elasticity_tensor_dS_dC", "polynomi"+
"al degree": 1, "point coordinates": [0.5, 0.5, 1.0], "flag plotting": 
False, "voigt notation": "conventional", "plotting arguments": {"scalin"+
"g function additional parameters": {"alpha":5}}}

post_processes["ThirdElasticityTensorAtPoint"] = {"directory path": 
results_path, "file name": "third_elasticity_tensor_dsigma_db", "polyn"+
"omial degree": 1, "point coordinates": [0.5, 0.5, 1.0], "flag plottin"+
"g": False, "voigt notation": "natural", "plotting arguments": {"s"+
"caling function additional parameters": {"alpha":5}}}"""

########################################################################
#                         Material properties                          #
########################################################################

# Sets the Young modulus and the Poisson ratio

E = 100E6

nu = 0.4

# Sets a dictionary of properties

material_properties = dict()

material_properties["E"] = E

material_properties["nu"] = nu

# Sets the material as a neo-hookean material using the corresponding
# class

constitutive_model = constitutive_models.Neo_Hookean(material_properties)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = test_meshes_paths.get_mesh_path("micropolar_beam_with_"+
"fibers_bending")

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

solver_parameters["linear_solver"] = "mumps"

solver_parameters["newton_relative_tolerance"] = 1e-4

solver_parameters["newton_absolute_tolerance"] = 1e-4

solver_parameters["newton_maximum_iterations"] = 10

"""solver_parameters["preconditioner"] = "hypre_amg"

solver_parameters["krylov_absolute_tolerance"] = 1e-9

solver_parameters["krylov_relative_tolerance"] = 1e-9

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

maximum_load = -2E5#1E-1

traction_boundary = {"load case": "UniformReferentialTraction", "ampli"+
"tude_tractionX": 0.0, "amplitude_tractionY": maximum_load, "amplitude"+
"_tractionZ": 0.0, "t": t, "t_final": t_final}#"""

# Defines a dictionary of tractions

traction_dictionary = dict()

traction_dictionary["upper"] = traction_boundary

# Defines a dictionary of boundary conditions. Each key is a physical
# group and each value is another dictionary or a list of dictionaries 
# with the boundary conditions' information. Each of these dictionaries
# must have the key "BC case", which carries the name of the function 
# that builds this boundary condition

bcs_dictionary = dict()

bcs_dictionary["back"] = {"BC case": "FixedSupportDirichletBC"}

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
verbose=True)