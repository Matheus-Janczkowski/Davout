# Routine to test a hyperelastic disc

import os

from dolfin import *

from mshr import *

from ......Grante.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as constitutive_models

from ......Grante.MultiMech.multiscale import multiscale_hyperelasticity as variational_framework

from ......Grante.CuboidGmsh.aa_tests.micropolar_meshes import beam_micropolar_case_1 as beam_gmsh

from ......Grante.PythonicUtilities import path_tools

from ......Grante.MultiMech.aa_tests_and_examples import test_meshes_paths

########################################################################
########################################################################
##                      User defined parameters                       ##
########################################################################
########################################################################
        
displacement_multiscaleBC = "MinimallyConstrainedFirstOrderBC"#"LinearFirstOrderBC"#"PeriodicFirstOrderBC"#

fluctuation_field = False

base_path = str(path_tools.get_parent_path_of_file())#os.getcwd()+"//tests//multiscale//hyperelasticity//results"

E_matrix = 1E6 

E_fiber = 1E8 

nu_matrix = 0.4

nu_fiber = 0.4

RVE_width = 1.0

RVE_length = 1.0

fiber_radius = 0.25

n_RVEsX = 1

n_RVEsY = 1

n_RVEsZ = 1

RVE_localizationX = 1

RVE_localizationY = 1

RVE_localizationZ = 3

flag_newMesh = False

########################################################################
#                          Simulation results                          #
########################################################################

# Defines the path to the results directory 

results_pathGraphics = base_path+"//results//graphics//"

results_pathText = base_path+"//results//text//"

if fluctuation_field:

    saving_fileNames = ["displacement_microscale_fluctuation.xdmf", 
    "lambda_displacement.xdmf", "lambda_grad_displacement.xdmf"]

    homogenized_displacementFileName = ["homogenized_displacement_micr"+
    "oscale_fluctuation.txt"]

    homogenized_gradDisplacementFileName = ["homogenized_displacemnt_g"+
    "radient_microscale_fluctuation.txt"]

    homogenized_piolaFileName = ["homogenized_first_piola_microscale_f"+
    "luctuation.txt"]

    homogenized_cauchyFileName = ["homogenized_cauchy_microscale_fluct"+
    "uation.txt"]

else:

    saving_fileNames = ["displacement_microscale.xdmf", "lambda_displa"+
    "cement.xdmf", "lambda_grad_displacement.xdmf"]

    homogenized_displacementFileName = ["homogenized_displacement_micr"+
    "oscale.txt"]

    homogenized_gradDisplacementFileName = ["homogenized_displacement_"+
    "gradient_microscale.txt"]

    homogenized_piolaFileName = ["homogenized_first_piola_microscale.t"+
    "xt"]

    homogenized_cauchyFileName = ["homogenized_cauchy_microscale.txt"]

if fluctuation_field:

    stress_fieldFileName = ["cauchy_stress_microscale_fluctuation.xdmf", 
    "first_piola_stress_microscale.xdmf"]

else:

    stress_fieldFileName = ["cauchy_stress_microscale.xdmf", "first_pi"+
    "ola_stress_microscale.xdmf"]

post_processes = []

fields_names = ["Displacement"]

if displacement_multiscaleBC=="MinimallyConstrainedFirstOrderBC":

    fields_names.extend(["Displacement_lagrange_multiplier", "Disp"+
    "lacement_gradient_lagrange_multiplier"]) 

# Iterates through the fields (displacement and the Lagrange multipliers)

for i in range(len(fields_names)):

    post_processes.append([fields_names[i], dict()])

    post_processes[-1][-1]["SaveField"] = {"directory path":
    results_pathGraphics, "file name":saving_fileNames[i]}

    if i==0:

        # Put "" in the subdomain to integrate over the entire do-
        # main

        post_processes[-1][-1]["HomogenizeField"] = {"directory path": 
        results_pathText, "file name": homogenized_displacementFileName[
        i], "subdomain":""}

        # Put "" in the subdomain to integrate over the entire do-
        # main

        post_processes[-1][-1]["HomogenizeFieldsGradient"] = {"directo"+
        "ry path":results_pathText, "file name":
        homogenized_gradDisplacementFileName[i], "subdomain":""}

    # Adds the stress field to the displacement field even though it can 
    # be evaluated with any field, since it takes all fields simultane-
    # ously

    if i==0:

        post_processes[-1][-1]["SaveCauchyStressField"] = {"direct"+
        "ory path": results_pathGraphics, "file name": 
        stress_fieldFileName[0], "polynomial degree": 1}

        post_processes[-1][-1]["HomogenizeFirstPiola"] = {"directo"+
        "ry path": results_pathText, "file name": 
        homogenized_piolaFileName[0], "subdomain":""}

        post_processes[-1][-1]["HomogenizeCauchy"] = {"directo"+
        "ry path": results_pathText, "file name": 
        homogenized_cauchyFileName[0], "subdomain":""}

####################################################################
#                       Material properties                        #
####################################################################

# Sets a dictionary of properties

# Saves the properties into a dictionary for the matrix

material_propertiesMatrix = dict()

material_propertiesMatrix["E"] = E_matrix

material_propertiesMatrix["nu"] = nu_matrix

# And for the fiber

material_propertiesFiber = dict()

material_propertiesFiber["E"] = E_fiber

material_propertiesFiber["nu"] = nu_fiber

# Sets the material as a HGO material

constitutive_model = dict()

constitutive_model["RVE matrix"] = constitutive_models.Neo_Hookean(
material_propertiesMatrix)

constitutive_model["RVE fiber"] = constitutive_models.Neo_Hookean(
material_propertiesFiber)

####################################################################
#                               Mesh                               #
####################################################################

# Defines the name of the file to save the mesh in. Do not write the 
# file termination, e.g. .msh or .xdmf; both options will be saved 
# automatically

#file_directory = os.getcwd()+"//tests//test_meshes"

mesh_fileName = test_meshes_paths.get_mesh_path("micropolar_beam_w"+
"ith_fibers_microscale")

print(mesh_fileName)

if flag_newMesh:

    beam_gmsh.case_1(RVE_width, RVE_length, fiber_radius, n_RVEsX, 
    n_RVEsY, n_RVEsZ, RVE_localizationX, RVE_localizationY, 
    RVE_localizationZ, mesh_fileName=mesh_fileName)#mesh_fileName)
    """, file_directory=
    file_directory)"""

####################################################################
#                          Function space                          #
####################################################################

# Defines the shape functions degree

polynomial_degree = 2

####################################################################
#                         Solver parameters                        #
####################################################################

# Sets the solver parameters in a dictionary

solver_parameters = dict()

solver_parameters["linear_solver"] = "mumps"#"mumps"

solver_parameters["newton_relative_tolerance"] = 1e-5#1e-8

solver_parameters["newton_absolute_tolerance"] = 1e-3#1e-8

solver_parameters["newton_maximum_iterations"] = 30

####################################################################
#                        Boundary conditions                       #
####################################################################

# Defines the paths to the macro quantities files

macro_displacementName = (results_pathText+"homogenized_displace"+
"ment")

macro_gradDisplacementName= (results_pathText+"homogenized_displ"+
"acement_gradient") 

####################################################################
####################################################################
##                    Calculation and solution                    ##
####################################################################
####################################################################

# Defines a flag to print every step

verbose = True

# Solves the variational problem

variational_framework.hyperelastic_microscale(displacement_multiscaleBC, 
macro_displacementName, macro_gradDisplacementName, constitutive_model, 
post_processes, mesh_fileName, solver_parameters,
polynomial_degree=polynomial_degree, verbose=verbose, fluctuation_field=
fluctuation_field)