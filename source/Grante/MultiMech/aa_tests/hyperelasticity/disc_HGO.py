# Routine to test a hyperelastic disc

from dolfin import *

import os

#import periodic_structure as mesher

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

results_path = os.getcwd()+"//tests//hyperelasticity//results"

displacement_fileName = "displacement.xdmf"

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

# Put "" in the subdomain to integrate over the entire domain

post_processes["HomogenizeField"] = {"directory path":results_path,
"file name":homogenized_displacementFileName, "subdomain":[2]}

# Put "" in the subdomain to integrate over the entire domain

post_processes["HomogenizeFieldsGradient"] = {"directory path":
results_path, "file name":homogenized_gradDisplacementFileName, 
"subdomain":""}

# Sets the post processes of the submesh

post_processesSubmesh = dict()

# THe subdomain variable must be "" for a submesh, for there isn't sub-
# domains in a submesh

post_processesSubmesh["HomogenizeField"] = {"directory path":
results_path, "file name":homogenized_dispRVEFileName, "subdomain":""}

post_processesSubmesh["SaveCauchyStressField"] = {"directory path":
results_path, "file name": stress_fileName[1], "polynomial degree":1}

########################################################################
#                         Material properties                          #
########################################################################

# Sets a dictionary of properties

material_properties1 = dict()

# Shearing modulus

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

material_properties1["local system of coordinates: a direction"] = (
as_vector([1.0, 0.0, 0.0]))

material_properties1["local system of coordinates: d direction"] = (
as_vector([0.0, 0.0, 1.0]))

material_properties2 = dict()

material_properties2["E"] = 1E7

material_properties2["nu"] = 0.4

material_properties3 = dict()

material_properties3["E"] = 1E8

material_properties3["nu"] = 0.35

# Sets the material as a HGO material

constitutive_model = dict()

option = 1

if option==1:

    constitutive_model[1] = anisotropic_constitutiveModels.Holzapfel_Gasser_Ogden_Unconstrained(
    material_properties1)

    constitutive_model[2] = isotropic_constitutiveModels.Neo_Hookean(
    material_properties2)

    constitutive_model[3] = isotropic_constitutiveModels.Neo_Hookean(
    material_properties3)

elif option==2:

    constitutive_model[1] = anisotropic_constitutiveModels.Holzapfel_Gasser_Ogden_Unconstrained(
    material_properties1)

    constitutive_model[tuple([2,3])] = isotropic_constitutiveModels.Neo_Hookean(
    material_properties2)

elif option==3:

    constitutive_model = anisotropic_constitutiveModels.Holzapfel_Gasser_Ogden_Unconstrained(
    material_properties1)

elif option==4:

    constitutive_model[1] = isotropic_constitutiveModels.Neo_Hookean(
    material_properties3)

    constitutive_model[2] = isotropic_constitutiveModels.Neo_Hookean(
    material_properties3)

    constitutive_model[3] = isotropic_constitutiveModels.Neo_Hookean(
    material_properties3)

elif option==5:

    constitutive_model = isotropic_constitutiveModels.Neo_Hookean(
    material_properties3)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = "tests//test_meshes//intervertebral_disc"

# Defines a set of physical groups to create a submesh

volume_physGroupsSubmesh = [2]

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

solver_parameters["linear_solver"] = "minres"

solver_parameters["newton_relative_tolerance"] = 1e-4

solver_parameters["newton_absolute_tolerance"] = 1e-4

solver_parameters["newton_maximum_iterations"] = 50

solver_parameters["preconditioner"] = "petsc_amg"

solver_parameters["krylov_absolute_tolerance"] = 1e-6

solver_parameters["krylov_relative_tolerance"] = 1e-6

solver_parameters["krylov_maximum_iterations"] = 15000

solver_parameters["krylov_monitor_convergence"] = False

# Sets the initial time

t = 0.0

# Sets the final pseudotime of the simulation

t_final = 1.0

# Sets the maximum number of steps of loading

maximum_loadingSteps = 10

########################################################################
#                          Boundary conditions                         #
########################################################################

# Defines a load expression

maximum_load = -4E6

load = Expression("(t/t_final)*maximum_load", t=t, t_final=t_final,
maximum_load=maximum_load, degree=0)

# Assembles this load into the list of Neumann boundary conditions

neumann_loads = [load]

# Assemble the traction vector using this load expression

traction_boundary = as_vector([0.0, 0.0, load])

# Defines a dictionary of tractions

traction_dictionary = dict()

traction_dictionary[5] = traction_boundary

# Defines the boundary physical groups to apply fixed support boundary
# condition. This variable can be either a list of physical groups tags
# or simply a tag

fixed_supportPhysicalGroups = 4

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Solves the variational problem

variational_framework.hyperelasticity_displacementBased(
constitutive_model, traction_dictionary, maximum_loadingSteps, t_final, 
post_processes, mesh_fileName, solver_parameters, neumann_loads=
neumann_loads, polynomial_degree=polynomial_degree, t=t, 
fixed_supportPhysicalGroups=fixed_supportPhysicalGroups,
volume_physGroupsSubmesh=volume_physGroupsSubmesh, post_processesSubmesh
=post_processesSubmesh)