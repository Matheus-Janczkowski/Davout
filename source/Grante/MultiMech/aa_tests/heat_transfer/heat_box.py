# Routine to test a hyperelastic disc

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Grante.MultiMech.constitutive_models.heat_transfer import isotropic_heat_conduction as constitutive_models

from .....Grante.MultiMech.physics import steady_state_heat_transfer as variational_framework

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

temperature_file_name = "temperature.xdmf"

post_processes = dict()

post_processes["SaveField"] = {"directory path":results_path, 
"file name":temperature_file_name}

########################################################################
#                         Material properties                          #
########################################################################

# Sets the conductivity parameters

k = 10.0

# Sets a dictionary of properties

material_properties = dict()

material_properties["k"] = k

# Sets the material as a Fourier material using the corresponding class

constitutive_model = constitutive_models.Fourier(material_properties)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = {"length x": 1.0, "length y": 1.5, "length z": 5.0, "n"+
"umber of divisions in x": 5, "number of divisions in y": 7, "number o"+
"f divisions in z": 25, "verbose": False, "mesh file name": "box_mesh", 
"mesh file directory": get_parent_path_of_file()}

########################################################################
#                            Function space                            #
########################################################################

# Defines the shape functions degree

polynomial_degree = 1

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

maximum_loadingSteps = 1

########################################################################
#                          Boundary conditions                         #
########################################################################

# Defines a dictionary of outwards boundary heat fluxes

boundary_heat_flux = dict()

#boundary_heat_flux["top"] = 1E2

# Defines a dictionary for volumetric heat generation

heat_generation_dict = {"volume": 100.0}

# Defines a dictionary of boundary conditions. Each key is a physical
# group and each value is another dictionary or a list of dictionaries 
# with the boundary conditions' information. Each of these dictionaries
# must have the key "BC case", which carries the name of the function 
# that builds this boundary condition

bcs_dictionary = dict()

bcs_dictionary["bottom"] = 200.0

#bcs_dictionary["top"] = 500.0

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Solves the variational problem

variational_framework.steady_state_heat_transfer_temperature_based(
constitutive_model, boundary_heat_flux, maximum_loadingSteps, t_final, 
post_processes, mesh_fileName, solver_parameters, 
polynomial_degree=polynomial_degree, t=t, 
dirichlet_boundaryConditions=bcs_dictionary, verbose=True,
heat_generation_dict=heat_generation_dict)