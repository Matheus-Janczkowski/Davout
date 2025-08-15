# Routine to test a hyperelastic disc

import os

from dolfin import *

from mshr import *

import numpy as np

import source.constitutive_models.hyperelasticity.isotropic_hyperelasticity as constitutive_models

import source.physics.hyperelastic_cauchy_continuum as variational_framework

import tests.test_meshes.beam_gmsh as beam_gmsh

########################################################################
########################################################################
##                      User defined parameters                       ##
########################################################################
########################################################################

########################################################################
#                          Simulation results                          #
########################################################################

# Defines the path to the results directory 

results_pathGraphics = (os.getcwd()+"//tests//micropolar//Bauer_et_al/"+
"/results//graphics")

results_pathText = (os.getcwd()+"//tests//micropolar//Bauer_et_al//res"+
"ults//text")

displacement_fileName = "displacement_neo_hookean.xdmf"

homogenized_displacementFileName = "homogenized_displacement.txt"

homogenized_gradDisplacementFileName = ("homogenized_displacement_grad"+
"ient.txt")

post_processes = dict()

post_processes["save field"] = {"directory path":results_pathGraphics, 
"file name":displacement_fileName}

# Put "" in the subdomain to integrate over the entire domain

post_processes["homogenize field"] = {"directory path":
results_pathText, "file name":homogenized_displacementFileName, 
"subdomain":""}

# Put "" in the subdomain to integrate over the entire domain

post_processes["homogenize field's gradient"] = {"directory path":
results_pathText, "file name":homogenized_gradDisplacementFileName, 
"subdomain":""}

post_processesSubmesh = dict()

########################################################################
#                         Material properties                          #
########################################################################

# Sets a dictionary of properties

material_properties = dict()

mu = 26.12

K_constitutive = 63.84

lmbda = K_constitutive-(2*mu/3)

E = ((mu/(lmbda+mu))*((2*mu)+(3*lmbda)))

v = lmbda/(2*(lmbda+mu))

print("E:", E, ", v:", v)

# Saves the properties into a dictionary

material_properties["E"] = E

material_properties["v"] = v

# Sets the material as a HGO material

constitutive_model = dict()

#constitutive_model["Generic volume"] = constitutive_models.Neo_Hookean(
#material_properties)

n_volumes = 2

if n_volumes==2:

    constitutive_model[1] = constitutive_models.Neo_Hookean(
    material_properties)

    constitutive_model[8] = constitutive_models.Neo_Hookean(
    material_properties)

else:

    constitutive_model = constitutive_models.Neo_Hookean(material_properties)

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = "tests//test_meshes//micropolar_beam"

# Generates the mesh

ratio_Lb = 1.5E-1

gamma = 1.18E0

beta = 0.0

beam_gmsh.generate_micropolarBeam(mu, ratio_Lb, beta, gamma, 
mesh_fileName, n_volumes, transfinite=True)

# Defines a set of physical groups to create a submesh

volume_physGroupsSubmesh = []

########################################################################
#                            Function space                            #
########################################################################

# Defines the shape functions degree

polynomial_degreeDisplacement = 2

########################################################################
#                           Solver parameters                          #
########################################################################

# Sets some parameters

parameters["form_compiler"]["representation"] = "uflacs"
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Sets the solver parameters in a dictionary

solver_parameters = dict()

solver_parameters["linear_solver"] = "mumps"#"minres"

solver_parameters["newton_relative_tolerance"] = 1e-8#1e-3

solver_parameters["newton_absolute_tolerance"] = 1e-8#1e-3

solver_parameters["newton_maximum_iterations"] = 10#50

"""

solver_parameters["preconditioner"] = "petsc_amg"

solver_parameters["krylov_absolute_tolerance"] = 1e-6

solver_parameters["krylov_relative_tolerance"] = 1e-6

solver_parameters["krylov_maximum_iterations"] = 15000

solver_parameters["krylov_monitor_convergence"] = False"""

# Sets the initial time

t = 0.0

# Sets the final pseudotime of the simulation

t_final = 1.0

# Sets the maximum number of steps of loading

maximum_loadingSteps = 11

########################################################################
#                          Boundary conditions                         #
########################################################################

# Defines a load expression

maximum_load = 1E-2

load = Expression("(t/t_final)*maximum_load", t=t, t_final=t_final,
maximum_load=maximum_load, degree=0)

# Assembles this load into the list of Neumann boundary conditions

neumann_loads = [load]

# Assemble the traction vector using this load expression

traction_boundary = as_vector([0.0, load, 0.0])#as_vector([load, 0.0, 0.0])

# Defines a dictionary of tractions

traction_dictionary = dict()

traction_dictionary[4] = traction_boundary

# Defines the boundary physical groups to apply fixed support boundary
# condition. This variable can be either a list of physical groups tags
# or simply a tag. Applies for both displacement and microrotation

fixed_supportDisplacementPhysicalGroups = "back"

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Solves the variational problem

variational_framework.hyperelasticity_displacementBased(
constitutive_model, traction_dictionary, 
maximum_loadingSteps, t_final, post_processes, mesh_fileName, 
solver_parameters, neumann_loads=neumann_loads, 
polynomial_degree=polynomial_degreeDisplacement, 
t=t, fixed_supportPhysicalGroups=fixed_supportDisplacementPhysicalGroups, 
solution_name=["displacement","DNS"], volume_physGroupsSubmesh=
volume_physGroupsSubmesh, post_processesSubmesh=post_processesSubmesh)