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

results_path = get_parent_path_of_file() #os.getcwd()+"//aa_tests//hyperelasticity//results"

displacement_fileName = "displacement_parallel"

post_processes = [["Displacement", dict()], ["Pressure", dict()]]

post_processes[0][1]["SaveField"] = {"directory path": results_path, 
"file name": displacement_fileName, "readable xdmf file": False, "visua"+
"lization copy for readable xdmf": True}

post_processes[1][1]["SaveField"] = {"directory path": results_path, 
"file name": "pressure_parallel.xdmf", "readable xdmf file": True, "vi"+
"sualization copy for readable xdmf": True }

post_processes[0][1]["SaveMeshVolumeRatioToReferenceVolume"] = {"direc"+
"tory path": results_path, "file name": "volume_ratio_parallel.txt"}

post_processes[0][1]["SaveCauchyStressField"] = {"directory path":results_path,
"file name": "cauchy_stress_field.xdmf", "polynomial degree":1}

"""post_processes[0][1]["SaveFirstPiolaStressField"] = {"directory path":
results_path, "file name": "first_piola_stress_field.xdmf", "polynomial deg"+
"ree":1}

post_processes[0][1]["SaveReferentialTractionField"] = {"directory path":
results_path, "file name": "referential_traction.xdmf", "polynomial deg"+
"ree":1}

post_processes[0][1]["SavePressureAtPoint"] = {"directory path": results_path, 
"file name": "pressure_point", "polynomial degree":1, "poin"+
"t coordinates": [0.15, 0.1, 5.0]}

post_processes[0][1]["HomogenizeField"] = {"directory path": results_path, 
"file name": "homogenized_displacement.txt", "subdomain": "volume 1"}

post_processes[0][1]["HomogenizeFieldsGradient"] = {"directory path": results_path, 
"file name": "homogenized_displacement_gradient.txt", "subdomain": "volume 1"}

post_processes[0][1]["HomogenizeFirstPiola"] = {"directory path": results_path, 
"file name": "homogenized_first_piola.txt", "subdomain": "volume 1"}

post_processes[0][1]["HomogenizeCauchy"] = {"directory path": results_path, 
"file name": "homogenized_cauchy.txt", "subdomain": "volume 1"}"""

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

mesh_fileName = {"length x": 0.3, "length y": 0.2, "length z": 1.0, "n"+
"umber of divisions in x": 5, "number of divisions in y": 5, "number o"+
"f divisions in z": 25, "verbose": False, "mesh file name": "box_mesh", 
"mesh file directory": get_parent_path_of_file()}

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

# Defines a load expression

maximum_load = 5E5

# Assemble the traction vector using this load expression

traction_boundary = {"load case": "UniformReferentialTraction", "ampli"+
"tude_tractionX": 0.0, "amplitude_tractionY": 0.0, "amplitude_tractionZ": 
maximum_load, "parametric_load_curve": "square_root", "t": t, "t_final":
t_final}

# Defines a dictionary of tractions

traction_dictionary = dict()

#traction_dictionary["top"] = traction_boundary

# Defines a dictionary of boundary conditions. Each key is a physical
# group and each value is another dictionary or a list of dictionaries 
# with the boundary conditions' information. Each of these dictionaries
# must have the key "BC case", which carries the name of the function 
# that builds this boundary condition

bcs_dictionary = dict()

bcs_dictionary["bottom"] = {"BC case": "FixedSupportDirichletBC"}

bcs_dictionary["top"] = {"BC case": "PrescribedDirichletBC", "bc_infor"+
"mationsDict": {"load_function": "SurfaceTranslationAndRotation", "tra"+
"nslation": [0.0, 0.0, 0.05], "rotation_x": 45.0, "rotation_y": 0.0,
"rotation_z": 0.0}}

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Solves the variational problem

variational_framework.hyperelasticity_two_fields(constitutive_model,
traction_dictionary, maximum_loadingSteps, t_final, post_processes, 
mesh_fileName, solver_parameters, polynomial_degree_displacement=
polynomial_degree_displacement, polynomial_degree_pressure=
polynomial_degree_pressure, t=t, dirichlet_boundaryConditions=
bcs_dictionary, verbose=True, run_in_parallel=True)