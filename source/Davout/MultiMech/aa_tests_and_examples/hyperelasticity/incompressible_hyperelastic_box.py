# Routine to test a hyperelastic disc

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Grante.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as constitutive_models

from .....Grante.MultiMech.physics import hyperelastic_incompressible_cauchy_continuum as variational_framework

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

# The post processes list is a list of lists: each list corresponds to a
# field of the problem. Each list has two components: the name of the 
# field as defined in the physics file; a dictionary with the post pro-
# cesses. The post processes dictionary has keys with the names of the
# post processes and values which are dictionaries themselves with fur-
# ther information

post_processes = [["Displacement", dict()], ["Pressure", dict()]]

# The flag "readable xdmf file" makes the file able to be read into a 
# function afterwards

post_processes[0][1]["SaveField"] = {"directory path": results_path, 
"file name": "displacement_incompressible_box.xdmf", "readable xdmf fi"+
"le": True, "visualization copy for readable xdmf": True}

post_processes[0][1]["SaveMeshVolumeRatioToReferenceVolume"] = {"director"+
"y path": results_path, "file name": "volume_ratio_incompressible_box.txt"}

post_processes[0][1]["SaveStrainEnergy"] = {"directory path": 
results_path, "file name": "strain_energy_incompressible_box.txt"}

post_processes[0][1]["SaveForcesAndMomentsOnSurface"] = {"directory path": 
results_path, "file name": "forces_and_moments_incompressible_box.txt", 
"surface physical group name": "top"}

post_processes[0][1]["SaveFirstPiolaStressField"] = {"directory path":
results_path, "file name": "first_piola_stress_incompressible_box"}

post_processes[0][1]["SaveCauchyStressField"] = {"directory path":
results_path, "file name": "cauchy_stress_incompressible_box"}

post_processes[0][1]["SaveReferentialTractionField"] = {"directory path":
results_path, "file name": "referential_traction_incompressible_box"}

post_processes[1][1]["SaveField"] = {"directory path": results_path, 
"file name": "pressure_incompressible_box.xdmf", "readable xdmf file": 
True, "visualization copy for readable xdmf": True}

########################################################################
#                         Material properties                          #
########################################################################

# Sets the Young modulus and the Poisson ratio

E_1 = 1E6

poisson_1 = 0.3

E_2 = 1E6

poisson_2 = 0.3

# Sets the material as a neo-hookean material using the corresponding
# class. Sets the constitutive model as a dictionary: each key corres-
# ponds to the name of the volumetric physical group; and the value is
# the constitutive model for that physical group

constitutive_model = dict()

constitutive_model["volume 1"] = constitutive_models.NeoHookean({"E": 
E_1, "nu": poisson_1})

constitutive_model["volume 2"] = constitutive_models.NeoHookean({"E": 
E_2, "nu": poisson_2})

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = {"length x": 0.3, "length y": 0.2, "length z": 1.0, "n"+
"umber of divisions in x": 10, "number of divisions in y": 10, "number o"+
"f divisions in z": 10, "verbose": False, "mesh file name": "box_mesh", 
"mesh file directory": get_parent_path_of_file(), "number of subdomain"+
"s in z direction": 2, "bias x": ["Bump", 0.2], "bias y": ["Bump", 0.2]}

########################################################################
#                            Function space                            #
########################################################################

# Defines the shape functions degree

polynomial_degree_displacement = 2

polynomial_degree_pressure = 1

########################################################################
#                           Solver parameters                          #
########################################################################

# Sets the solver parameters in a dictionary

solver_parameters = dict()

solver_parameters["newton_relative_tolerance"] = 1e-4

solver_parameters["newton_absolute_tolerance"] = 1e-4

solver_parameters["newton_maximum_iterations"] = 15

solver_parameters["linear_solver"] = "mumps"

# Sets the initial time

t = 0.0

# Sets the final pseudotime of the simulation

t_final = 1.0

# Sets the maximum number of steps of loading

maximum_loadingSteps = 5

########################################################################
#                          Boundary conditions                         #
########################################################################

# Defines a dictionary of tractions

traction_dictionary = dict()

maximum_load = 5E5

traction_boundary = {"load case": "UniformReferentialTraction", "ampli"+
"tude_tractionX": 0.0, "amplitude_tractionY": 0.0, "amplitude_tractionZ": 
maximum_load, "parametric_load_curve": "square_root", "t": t, "t_final":
t_final}

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

variational_framework.hyperelasticity_two_fields(
constitutive_model, traction_dictionary, maximum_loadingSteps, t_final, 
post_processes, mesh_fileName, solver_parameters, 
polynomial_degree_displacement=polynomial_degree_displacement, 
polynomial_degree_pressure=polynomial_degree_pressure, t=t, 
dirichlet_boundaryConditions=bcs_dictionary, verbose=True)