# Routine to test a hyperelastic disc
#
# Parallel execution:
# mpirun -np 2 python3 -m source.Grante.MultiMech.aa_tests_and_examples.hyperelasticity.disc_neo_hookean_incompressible

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

post_processes = [["Displacement", dict()], ["Pressure", dict()]]

n_process = "incompressible_disc"

post_processes[0][1]["SaveField"] = {"directory path": results_path, 
"file name": "displacement_"+str(n_process)+".xdmf", "readable xdmf fi"+
"le": False, "visualization copy for readable xdmf": False}

post_processes[0][1]["SaveMeshVolumeRatioToReferenceVolume"] = {"direc"+
"tory path": results_path, "file name": "volume_ratio.txt"}

post_processes[0][1]["SaveFirstPiolaStressField"] = {"direc"+
"tory path": results_path, "file name": "first_piola_incompressible_disc"}

post_processes[0][1]["SaveForcesAndMomentsOnSurface"] = {"direc"+
"tory path": results_path, "file name": "forces_and_moments", "surface"+
" physical group name": "top"}

post_processes[1][1]["SaveField"] = {"directory path": results_path, 
"file name": "pressure_"+str(n_process)+".xdmf", "visualization copy f"+
"or readable xdmf": False}

########################################################################
#                         Material properties                          #
########################################################################

# Sets the Young modulus and the Poisson ratio. The Young modulus is gi-
# ven in N/(mm^2) since the mesh is in mm

E = 1.0

v = 0.3

# Sets a dictionary of properties

material_properties = dict()

material_properties["E"] = E

material_properties["nu"] = v

material_properties_nucleus = dict()

material_properties_nucleus["c1"] = 1E6

material_properties_nucleus["c2"] = 1E5

material_properties_nucleus["bulk modulus"] = 1E8

# Sets the material as a neo-hookean material using the corresponding
# class

constitutive_model = dict()

#constitutive_model["annulus"] = constitutive_models.NeoHookean(material_properties)

#constitutive_model["nucleus"] = constitutive_models.MooneyRivlin(material_properties_nucleus)

constitutive_model = constitutive_models.NeoHookean(material_properties)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = (get_parent_path_of_file(path_bits_to_be_excluded=2)+
"//test_meshes//intervertebral_disc_mesh")

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

# Defines a load expression. The traction is given in N/(mm^2) since the 
# mesh is in mm

maximum_load = 0.2

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

"""
bcs_dictionary["top"] = {"BC case": "PrescribedDirichletBC", "bc_infor"+
"mationsDict": {"load_function": "SurfaceTranslationAndRotation", "tra"+
"nslation": [0.0, 0.0, 5.0], "in_planeSpinDirection": [1.0, 0.0, 0.0], 
"in_planeSpin": 10.0, "normal_toPlaneSpin": 10.0}}#"""

"""bcs_dictionary["top"] = {"BC case": "PrescribedDirichletBC", "bc_infor"+
"mationsDict": {"load_function": "SurfaceTranslationAndRotation", "tra"+
"nslation": [0.0, 0.0, 0.05], "rotation_x": 10.0, "rotation_y": 0.0,
"rotation_z": 0.0}}"""

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Solves the variational problem

variational_framework.hyperelasticity_two_fields(
constitutive_model, traction_dictionary, maximum_loadingSteps, t_final, 
post_processes, mesh_fileName, solver_parameters, 
polynomial_degree_displacement=polynomial_degree, t=t, 
dirichlet_boundaryConditions=bcs_dictionary, verbose=True)