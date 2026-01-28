# Routine to test a hyperelastic disc

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Grante.MultiMech.constitutive_models.hyperelasticity import anisotropic_hyperelasticity as anisotropic_constitutiveModels

from .....Grante.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as isotropic_constitutiveModels

from .....Grante.MultiMech.physics import hyperelastic_cauchy_continuum as variational_framework

########################################################################
########################################################################
##                      User defined parameters                       ##
########################################################################
########################################################################

########################################################################
#                          Simulation results                          #
########################################################################

# Defines the path to the results directory 

results_path = get_parent_path_of_file()

displacement_fileName = "displacement_HGO.xdmf"

stress_fileName = ["cauchy_stress.xdmf", "cauchy_stress_submesh.xdmf"]

homogenized_displacementFileName = "homogenized_displacement.txt"

homogenized_gradDisplacementFileName = ("homogenized_displacement_grad"+
"ient.txt")

homogenized_dispRVEFileName = "homogenized_displacement_RVE.txt"

post_processes = dict()

post_processes["SaveField"] = {"directory path":results_path, 
"file name":displacement_fileName}

post_processes["SaveCauchyStressField"] = {"directory path":results_path,
"file name":stress_fileName[0], "polynomial degree":1}

########################################################################
#                         Material properties                          #
########################################################################

# Sets a dictionary of properties

material_properties1 = dict()

# Half shearing modulus

material_properties1["c"] = 10E6

# k1 is the fiber modulus and k2 is the exponential coefficient

material_properties1["k1"] = 10E4

material_properties1["k2"] = 5.0

# Kappa is the fiber dispersion and it is bounded between 0 and 1/3. A 
# third is an isotropic material

material_properties1["kappa"] = 0.2

# Gamma is the fiber angle in degrees

material_properties1["gamma"] = 45.0

# k is the matrix bulk modulus

material_properties1["k"] = 15E6

# The vectors ahead form a plane where the fiber is locally present

material_properties1["a direction"] = ([1.0, 0.0, 0.0])

material_properties1["d direction"] = ([0.0, 0.0, 1.0])

material_properties2 = dict()

material_properties2["E"] = 1E7

material_properties2["nu"] = 0.4

material_properties3 = dict()

material_properties3["c1"] = 1E6

material_properties3["c2"] = 2E6

material_properties3["bulk modulus"] = 3E6

# Sets the material as a HGO material

constitutive_model = dict()

constitutive_model["volume 1"] = anisotropic_constitutiveModels.HolzapfelGasserOgdenUnconstrained(
material_properties1)

constitutive_model["volume 2"] = isotropic_constitutiveModels.NeoHookean(
material_properties2)

constitutive_model["volume 3"] = isotropic_constitutiveModels.MooneyRivlin(
material_properties3)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = {"length x": 0.3, "length y": 0.2, "length z": 1.0, "n"+
"umber of divisions in x": 5, "number of divisions in y": 5, "number o"+
"f divisions in z": 25, "verbose": False, "mesh file name": "box_mesh", 
"mesh file directory": get_parent_path_of_file(), "number of subdomain"+
"s in z direction": 3}#"tests//test_meshes//intervertebral_disc"

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

#solver_parameters["linear_solver"] = "minres"

solver_parameters["newton_relative_tolerance"] = 1e-4

solver_parameters["newton_absolute_tolerance"] = 1e-4

solver_parameters["newton_maximum_iterations"] = 15

"""solver_parameters["preconditioner"] = "petsc_amg"

solver_parameters["krylov_absolute_tolerance"] = 1e-6

solver_parameters["krylov_relative_tolerance"] = 1e-6

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

maximum_load = 9E5

# Assemble the traction vector using this load expression

traction_boundary = {"load case": "UniformReferentialTraction", "ampli"+
"tude_tractionX": 0.0, "amplitude_tractionY": 0.0, "amplitude_tractionZ": 
maximum_load, "parametric_load_curve": "square_root", "t": t, "t_final":
t_final}

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