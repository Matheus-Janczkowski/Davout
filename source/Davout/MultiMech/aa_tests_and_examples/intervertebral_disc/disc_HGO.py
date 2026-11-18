# cd /home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/simulations
# Routine to test a hyperelastic disc

from .....Davout.MultiMech.constitutive_models.hyperelasticity import anisotropic_hyperelasticity as anisotropic_constitutiveModels

from .....Davout.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as isotropic_constitutiveModels

from .....Davout.MultiMech.physics import hyperelastic_incompressible_cauchy_continuum as variational_framework

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from time import time

def simulation_disc(linear_solver:str, c_HGO:float,
kappa:float, bulk_modulus:float, c2_mooney:float, c1_mooney:float, mesh_file_name:str,
k1_file_name:str, k2_file_name:str, gamma_file_name:str, axial_direction_file_name:str,
circumferential_direction_file_name:str, translation:list, axle_vector:list,
displacement_file_name:str, pressure_file_name:str, volume_ratio_file_name:str,
strain_energy_file_name:str, forces_and_moments_file_name:str, maximum_load=0.0,
maximum_loadingSteps=None, intermediate_saving=False):
    
    start_time = time()

    ########################################################################
    ########################################################################
    ##                      User defined parameters                       ##
    ########################################################################
    ########################################################################

    ########################################################################
    #                          Simulation results                          #
    ########################################################################

    # Defines a list of lists to identify the name of the parameter and where 
    # the directory path file is

    post_processes = [["Displacement", dict()], ["Pressure", dict()]]

    post_processes[0][1]["SaveField"] = {"directory path": "", 
    "file name": displacement_file_name, "readable xdmf file": True, 
    "visualization copy for readable xdmf": True, "intermediate saving flag":
    intermediate_saving}

    post_processes[0][1]["SaveMeshVolumeRatioToReferenceVolume"] = {"director"+
    "y path": "", "file name": volume_ratio_file_name}

    post_processes[0][1]["SaveStrainEnergy"] = {"directory path": 
    "", "file name": strain_energy_file_name}

    post_processes[0][1]["SaveForcesAndMomentsOnSurface"] = {"directory path": 
    "", "file name": forces_and_moments_file_name, "surface physical"+
    " group name": "top"}

    post_processes[1][1]["SaveField"] = {"directory path":"", 
    "file name": pressure_file_name, "readable xdmf file": True, "visualization copy for readable xdmf": True}

    ########################################################################
    #                         Material properties                          #
    ########################################################################

    polynomial_degree_properties = 1

    # Observation: The material properties are been used from Nicolini's ar-
    # ticle

    # Sets a dictionary of properties

    material_properties1 = dict()

    # c is the parameter for HGO - NeoHookean (gelatin where
    # fibers are arranged)

    material_properties1["c"] = c_HGO

    # k1 is the fiber modulus and k2 is the exponential coefficient

    material_properties1["k1"] = {"field file": k1_file_name, "mesh file": mesh_file_name, "field ty"+
    "pe": "scalar", "interpolation function": "CG", "polynomial degree": polynomial_degree_properties}

    material_properties1["k2"] = {"field file": k2_file_name,
    "mesh file": mesh_file_name, "field type": "scalar", "interpolation fu"+
    "nction": "CG", "polynomial degree": polynomial_degree_properties}

    # Kappa is the fiber dispersion and it is bounded between 0 and 1/3. A 
    # third is an isotropic material

    material_properties1["kappa"] = kappa

    # Gamma is the fiber angle in degrees

    material_properties1["gamma"] = {"field file": gamma_file_name,
    "mesh file": mesh_file_name, "field type": "scalar", "interpolation function":
    "CG", "polynomial degree": polynomial_degree_properties}

    # k is the matrix bulk modulus. In this case, where the constitutive
    # model is incompressible both for AF (annulus fibrosus) and for the NP
    # (nucleus pulposus).

    material_properties1["k"] = bulk_modulus

    # The vectors ahead form a plane where the fiber is locally present

    material_properties1["a direction"] = {"field file": axial_direction_file_name,
    "mesh file": mesh_file_name, "field type": "vector", "interpolation function": "CG", "polynomial degree": polynomial_degree_properties}

    material_properties1["d direction"] = {"field file": circumferential_direction_file_name,
    "mesh file": mesh_file_name, "field type": "vector", "interpolation function": "CG", "polynomial degree": polynomial_degree_properties}

    material_properties2 = dict()

    # The "material_properties2" defines the parameters from MooneyRivlin

    # c01 = "c2" and c10 = "c1" are parameters from MooneyRivlin constitutive model
    # for nucleus

    material_properties2["c2"] = c2_mooney

    material_properties2["c1"] = c1_mooney

    material_properties2["bulk modulus"] = bulk_modulus

    # Sets the material as a HGO material

    constitutive_model = dict()

    constitutive_model["annulus"] = anisotropic_constitutiveModels.HolzapfelGasserOgdenUnconstrained(
    material_properties1)

    constitutive_model["nucleus"] = isotropic_constitutiveModels.MooneyRivlin(
    material_properties2)

    ########################################################################
    #                            Function space                            #
    ########################################################################

    # Defines the shape functions degree

    #LBB consition, justify the number of the polynomial degree

    polynomial_degree_displacement = 2
    
    polynomial_degree_pressure = 1

    ########################################################################
    #                           Solver parameters                          #
    ########################################################################

    # Sets the solver parameters in a dictionary

    solver_parameters = dict()

    solver_parameters["newton_relative_tolerance"] = 1e-3

    solver_parameters["newton_absolute_tolerance"] = 1e-3

    solver_parameters["newton_maximum_iterations"] = 25

    solver_parameters["linear_solver"] = linear_solver

    # Sets the final pseudotime of the simulation
    
    t_final = 1.0

    # Sets the maximum number of steps of loading

    if maximum_loadingSteps is None:

        maximum_loadingSteps = 5 

    # Sets the initial time

    t = t_final/maximum_loadingSteps

    ########################################################################
    #                          Boundary conditions                         #
    ########################################################################

    # Defines a dictionary of boundary conditions. Each key is a physical
    # group and each value is another dictionary or a list of dictionaries 
    # with the boundary conditions' information. Each of these dictionaries
    # must have the key "BC case", which carries the name of the function 
    # that builds this boundary condition

    traction_dictionary = dict()

    bcs_dictionary = dict()

    bcs_dictionary["bottom"] = {"BC case": "FixedSupportDirichletBC"} 

    bcs_dictionary["top"] = {"BC case": "PrescribedDirichletBC", "bc_infor"+
    "mationsDict": {"load_function": "SurfaceTranslationAndRotation", "tra"+
    "nslation": translation, "rotation_x": axle_vector[0], 
    "rotation_y": axle_vector[1], "rotation_z": axle_vector[2]}}

    ########################################################################
    ########################################################################
    ##                      Calculation and solution                      ##
    ########################################################################
    ########################################################################

    # Solves the variational problem

    variational_framework.hyperelasticity_two_fields(
    constitutive_model, traction_dictionary, maximum_loadingSteps, t_final, 
    post_processes, mesh_file_name, solver_parameters, 
    polynomial_degree_displacement=polynomial_degree_displacement, 
    polynomial_degree_pressure=polynomial_degree_pressure, t=t,
    dirichlet_boundaryConditions=bcs_dictionary, verbose=True)

    final_time = time()

    return final_time-start_time

if __name__=="__main__":

    # Gets the parent path

    parent_path = get_parent_path_of_file()

    # Sets the mesh file name

    mesh_file_name = parent_path+"//intervertebral_disc_mesh"

    # Assembles the names of the result files

    displacement_file_name = parent_path+"//displacement_test"

    pressure_file_name = parent_path+"//pressure_test"

    volume_ratio_file_name = parent_path+"//volume_ratio_test"

    strain_energy_file_name = parent_path+"///strain_energy_test"

    forces_and_moments_file_name = (parent_path+"//forces_and_moments_"+
    "test")

    # Options for linear solver: "cg", "gmres", "minres", "mumps"

    linear_solver = "mumps"

    # c for HGO

    c_HGO = 0.195E6

    # kappa value

    kappa = 0.07

    # Bulk modulus 

    bulk_modulus = 15.0E6

    # c2 = c01 for Mooney Rivlin

    c2_mooney = 0.03E6

    # c1 = c10 for Mooney Rivlin 

    c1_mooney = 0.16E6

    # Files of the material properties

    axial_direction_file_name  = parent_path+"//axial_direction_000"

    circumferential_direction_file_name = (parent_path+"//circumferent"+
    "ial_direction_000")

    k1_file_name = parent_path+"//k1_0"

    k2_file_name = parent_path+"//k2_0"

    gamma_file_name = parent_path+"//gamma_0"

    # Boundary conditions prescribed Dirichlet displacement
    #
    # Critical translations:
    # x: [-, 3.1] mm
    # y: [-3.7, 3.7] mm
    # z: [-2.1, 1.5] mm
    #
    # Critical rotations:
    # x: [-, ] degrees
    # y: [-, ] degrees
    # z: [-, ] degrees

    translation = [0.0, 0.0, 0.003]#[-0.0018501445426034196,-0.0009263909321998073,-0.00012300453951900352]

    axle_vector = [0.0, 0.0, 0.0]#[1.94471325855789, -4.349424294172992, 2.813952923034023]

    simulation_disc(linear_solver, c_HGO, kappa, bulk_modulus, 
    c2_mooney, c1_mooney, mesh_file_name, k1_file_name, k2_file_name, 
    gamma_file_name, axial_direction_file_name,
    circumferential_direction_file_name, translation, axle_vector,
    displacement_file_name, pressure_file_name, volume_ratio_file_name,
    strain_energy_file_name, forces_and_moments_file_name)