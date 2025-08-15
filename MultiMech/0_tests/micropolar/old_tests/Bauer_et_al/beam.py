# Routine to test a hyperelastic disc

import os

from dolfin import *

from mshr import *

import numpy as np

import source.constitutive_models.hyperelasticity.micropolar_hyperelasticity as micropolar_constitutiveModels

import source.physics.hyperelastic_micropolar_continuum as variational_framework

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

displacement_fileName = ["displacement.xdmf", "microrotation.xdmf"]

homogenized_displacementFileName = ["homogenized_displacement.txt", "h"+
"omogenized_microrotation.txt"]

homogenized_gradDisplacementFileName = ["homogenized_displacement_grad"+
"ient.txt", "homogenized_microrotation_grad.txt"]

post_processes = []

# Iterates through the fields (displacement and microrotation)

for i in range(2):

    post_processes.append(dict())

    post_processes[-1]["save field"] = {"directory path":results_pathGraphics, 
    "file name":displacement_fileName[i]}

    # Put "" in the subdomain to integrate over the entire domain

    post_processes[-1]["homogenize field"] = {"directory path":
    results_pathText, "file name":homogenized_displacementFileName[i], 
    "subdomain":""}

    # Put "" in the subdomain to integrate over the entire domain

    post_processes[-1]["homogenize field's gradient"] = {"directory path":
    results_pathText, "file name":homogenized_gradDisplacementFileName[i], 
    "subdomain":""}

post_processesSubmesh = []

########################################################################
#                         Material properties                          #
########################################################################

# Sets the Young modulus and the Poisson ration from psi to MPa

E = 100E6#1.2E4*0.00689476

nu = 0.4#0.2

L = 0.4*(2.54*10)

# Converts to Lamé parameters

mu = E/(2*(1+nu))

lmbda = (nu*E)/((1+nu)*(1-(2*nu)))

# Sets a dictionary of properties

material_properties = dict()

alpha = 0.0

beta = 2*mu*(L*L)

kappa = (1.0E-2)*mu

N_micropolar = 0.0#np.sqrt(kappa/((2*mu)+kappa))

gamma = 0.0#1.18E0

ratio_Lb = 1.5E-1

# Saves the properties into a dictionary

material_properties["mu"] = mu

material_properties["lambda"] = lmbda

material_properties["N"] = N_micropolar

material_properties["alpha"] = alpha

material_properties["beta"] = beta

material_properties["gamma"] = gamma

# Sets the material as a HGO material

constitutive_model = dict()

n_volumes = 1

if n_volumes==2:

    constitutive_model["Generic volume"] = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
    material_properties)

else:

    constitutive_model = micropolar_constitutiveModels.Micropolar_Neo_Hookean(
    material_properties)

########################################################################
#                                 Mesh                                 #
########################################################################

# Sets the dimensions of the beam from (Ramezani et al, 2009). Converts
# from inches to mm

beam_widthX = 1.0#1.0*(2.54*10.0)

beam_widthY = 1.0#1.0*(2.54*10.0)

beam_length = 11.0#10.0*(2.54*10.0)

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

mesh_fileName = "tests//test_meshes//micropolar_beam_with_fibers"
#mesh_fileName = "tests//test_meshes//micropolar_beam"

# Generates the mesh

"""beam_gmsh.generate_micropolarBeam(mu, ratio_Lb, beta, gamma, 
mesh_fileName, n_volumes, transfinite=True, transfinite_x=5, 
transfinite_y=5, transfinite_z=21, beam_widthX=beam_widthX, beam_widthY=
beam_widthY, beam_length=beam_length)"""

# Defines a set of physical groups to create a submesh

volume_physGroupsSubmesh = []

########################################################################
#                            Function space                            #
########################################################################

# Defines the shape functions degree

polynomial_degreeDisplacement = 2

polynomial_degreeMicrorotation = 1

########################################################################
#                           Solver parameters                          #
########################################################################

# Sets the solver parameters in a dictionary

solver_parameters = dict()

solver_parameters["linear_solver"] = "mumps"#"mumps"

solver_parameters["newton_relative_tolerance"] = 1e-3#1e-8

solver_parameters["newton_absolute_tolerance"] = 1e-3#1e-8

solver_parameters["newton_maximum_iterations"] = 30

"""

solver_parameters["preconditioner"] = "petsc_amg"

solver_parameters["krylov_absolute_tolerance"] = 1e-5

solver_parameters["krylov_relative_tolerance"] = 1e-5

solver_parameters["krylov_maximum_iterations"] = 15000

solver_parameters["krylov_monitor_convergence"] = False"""

# Sets the initial time

t = 0.0

# Sets the final pseudotime of the simulation

t_final = 1.0

# Sets the maximum number of steps of loading

maximum_loadingSteps = 31

########################################################################
#                          Boundary conditions                         #
########################################################################

# Defines a load expression

K = 9.3

maximum_load = 2E5#0.5*((K*E)/(beam_length**3))*((beam_widthX*(beam_widthY**3))/(12*beam_widthX))#2.0E-4

load = Expression("(t/t_final)*maximum_load", t=t, t_final=t_final,
maximum_load=maximum_load, degree=2)

# Assembles this load into the list of Neumann boundary conditions

neumann_loads = [load]

# Assemble the traction vector using this load expression

traction_boundary = as_vector([0.0, load, 0.0])

# Defines a dictionary of tractions

traction_dictionary = dict()

traction_dictionary["upper"] = traction_boundary

traction_dictionary["lower"] = traction_boundary

# Defines a dictionary of moments on the boundary

moment_boundary = as_vector([0.0, 0.0, 0.0])

moment_dictionary = dict()

moment_dictionary["lower"] = moment_boundary

# Defines the boundary physical groups to apply fixed support boundary
# condition. This variable can be either a list of physical groups tags
# or simply a tag. Applies for both displacement and microrotation

fixed_supportDisplacementPhysicalGroups = "back"

fixed_supportMicrorotationPhysicalGroups = "back"

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Defines a flag to print every step

verbose = True

# Solves the variational problem

variational_framework.hyperelasticity_displacementMicrorotationBased(
constitutive_model, traction_dictionary, moment_dictionary, 
maximum_loadingSteps, t_final, post_processes, mesh_fileName, 
solver_parameters, neumann_loads=neumann_loads, 
polynomial_degreeDisplacement=polynomial_degreeDisplacement, 
polynomial_degreeMicrorotation=polynomial_degreeMicrorotation,
t=t, fixed_supportDisplacementPhysicalGroups=
fixed_supportDisplacementPhysicalGroups, solution_name=[["displacement",
"DNS"], ["microrotation", "DNS"]], volume_physGroupsSubmesh=
volume_physGroupsSubmesh, fixed_supportMicrorotationPhysicalGroups=
fixed_supportMicrorotationPhysicalGroups, post_processesSubmesh=
post_processesSubmesh, verbose=verbose)