# Routine to test a hyperelastic disc

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Grante.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as constitutive_models

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

results_path = get_parent_path_of_file() #os.getcwd()+"//aa_tests//hyperelasticity//results"

displacement_fileName = "displacement.xdmf"

post_processes = dict()

post_processes["SaveField"] = {"directory path":results_path, 
"file name":displacement_fileName}

post_processes["SaveForcesAndMomentsOnSurface"] = {"directory path": 
results_path, "file name": "forces_and_moments.txt", "surface physical"+
" group name": "right"}

post_processes["SaveStrainEnergy"] = {"directory path": results_path, 
"file name": "strain_energy.txt"}

########################################################################
#                         Material properties                          #
########################################################################

# Sets the Young modulus and the Poisson ratio

E = 1E6

poisson = 0.3

# Sets a dictionary of properties

material_properties = dict()

material_properties["E"] = E

material_properties["nu"] = poisson

# Sets the material as a neo-hookean material using the corresponding
# class

constitutive_model = constitutive_models.NeoHookean(material_properties)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = {"length x": 0.3, "length y": 1.0, "length z": 0.2, "n"+
"umber of divisions in x": 5, "number of divisions in y": 25, "number o"+
"f divisions in z": 5, "verbose": False, "mesh file name": "box_mesh", 
"mesh file directory": get_parent_path_of_file(), "number of subdomain"+
"s in z direction": 2}

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

maximum_load = 5E5

# Assemble the traction vector using this load expression

traction_boundary = {"load case": "UniformReferentialTraction", "ampli"+
"tude_tractionX": 0.0, "amplitude_tractionY": maximum_load, "amplitude_tractionZ": 
0.0, "parametric_load_curve": "square_root", "t": t, "t_final":
t_final}

# Defines a dictionary of tractions

traction_dictionary = dict()

traction_dictionary["right"] = traction_boundary

# Defines a dictionary of boundary conditions. Each key is a physical
# group and each value is another dictionary or a list of dictionaries 
# with the boundary conditions' information. Each of these dictionaries
# must have the key "BC case", which carries the name of the function 
# that builds this boundary condition

bcs_dictionary = dict()

bcs_dictionary["left"] = {"BC case": "FixedSupportDirichletBC"}

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