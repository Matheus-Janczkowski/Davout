# Routine to test a hyperelastic disc

from Grante.MultiMech.constitutive_models.hyperelasticity import anisotropic_hyperelasticity as anisotropic_constitutiveModels

from Grante.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as isotropic_constitutiveModels

from Grante.MultiMech.physics import hyperelastic_incompressible_cauchy_continuum as variational_framework

def simulation_disc(linear_solver:str, preconditioner:str, c_HGO:float,
kappa:float, bulk_modulus:float, c2_mooney:float, c1_mooney:float, mesh_file_name:str,
k1_file_name:str, k2_file_name:str, gamma_file_name:str, axial_direction_file_name:str,
circumferential_direction_file_name:str, translation:list, in_plane_spin_direction:list,
in_plane_spin:float, normal_to_plane_spin:float):

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

    post_processes[0][1]["SaveField"] = {"directory path": "/home/grante_"+
    "maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/results", 
    "file name": "displacement.xdmf", "readable xdmf file": True, "visualization copy for readable xdmf": True}

    post_processes[0][1]["SaveMeshVolumeRatioToReferenceVolume"] = {"director"+
    "y path": "/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/Intervert"+
    "ebralDiscANN/results", "file name": "volume_ratio.txt"}

    post_processes[1][1]["SaveField"] = {"directory path":"/home/grante_maximus/"+
    "rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/results", 
    "file name": "pressure.xdmf", "readable xdmf file": True, "visualization copy for readable xdmf": True}

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

    # TODO: ALTERAR PATH MALHA e arquivo parametros

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

    # TODO: ALTERAR PATH, COLOCAR COMO VARIÁVEL FIELD FILE

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

    solver_parameters["newton_maximum_iterations"] = 15

    solver_parameters["linear_solver"] = linear_solver

    """solver_parameters["preconditioner"] = preconditioner

    solver_parameters["krylov_absolute_tolerance"] = 1e-6 # critical line (variable)

    solver_parameters["krylov_relative_tolerance"] = 1e-6 # critical line (variable)

    solver_parameters["krylov_maximum_iterations"] = 15000

    solver_parameters["krylov_monitor_convergence"] = True"""

    # Sets the initial time

    t = 0.0

    # Sets the final pseudotime of the simulation
    
    t_final = 1.0

    # Sets the maximum number of steps of loading

    maximum_loadingSteps = 5 

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

    # TODO deixar parametros como argumentos da função 

    bcs_dictionary["top"] = {"BC case": "PrescribedDirichletBC", "bc_infor"+
    "mationsDict": {"load_function": "SurfaceTranslationAndRotation", "tra"+
    "nslation": translation, "in_planeSpinDirection": in_plane_spin_direction, 
    "in_planeSpin": in_plane_spin, "normal_toPlaneSpin": normal_to_plane_spin}}

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

if __name__=="__main__":

    mesh_file_name = ("/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/"+
    "IntervertebralDiscANN/material_properties_interpolation/intervertebral_disc_mesh")

    # Options for linear solver: "cg", "gmres", "minres", "mumps"

    linear_solver = "mumps"

    # Options for preconditioner: "ilu", "hypre_amg"

    preconditioner = "hypre_amg"

    # c for HGO

    c_HGO = 0.2E6

    # kappa value

    kappa = 0.05

    # Bulk modulus 

    bulk_modulus = 15E6

    # c2 = c01 for Mooney Rivlin

    c2_mooney = 0.03E6

    # c1 = c10 for Mooney Rivlin 

    c1_mooney = 0.18E6

    # Files of the material properties

    axial_direction_file_name  = "/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/material_properties_interpolation/axial_direction_000"

    circumferential_direction_file_name = "/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/material_properties_interpolation/circumferential_direction_000"

    k1_file_name = "/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/material_properties_interpolation/k1_000"

    k2_file_name = "/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/material_properties_interpolation/k2_000"

    gamma_file_name = "/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/material_properties_interpolation/gamma_000"

    # Boundary conditions prescribed Dirichlet displacement

    translation = [0.0, 0.0, 0.05]

    in_plane_spin_direction = [1.0, 0.0, 0.0]

    in_plane_spin = 10.0

    normal_to_plane_spin = 5.0

    simulation_disc(linear_solver, preconditioner, c_HGO, kappa, bulk_modulus, c2_mooney, c1_mooney, mesh_file_name, k1_file_name,
    k2_file_name, gamma_file_name, axial_direction_file_name, circumferential_direction_file_name, translation, in_plane_spin_direction,
    in_plane_spin, normal_to_plane_spin)