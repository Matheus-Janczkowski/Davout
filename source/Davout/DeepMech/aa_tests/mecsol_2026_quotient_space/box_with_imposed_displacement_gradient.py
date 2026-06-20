# Routine to test a hyperelastic disc

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as constitutive_models

from .....Davout.MultiMech.physics import hyperelastic_cauchy_continuum as variational_framework

from .....Davout.GraphUtilities.paraview_tools import frozen_snapshots

def solve_BVP(results_path, displacement_file_name, young_modulus_file,
mesh_file_name, displacement_gradient_components, 
lagrange_multiplier_file_name, save_snapshot=False):

    ####################################################################
    ####################################################################
    ##                    User defined parameters                     ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                        Simulation results                        #
    ####################################################################

    post_processes = [["Displacement", {}], ["Lagrange multiplier disp"+
    "lacement gradient average", {}]]

    # The flag "readable xdmf file" makes the file able to be read into 
    # a function afterwards. The flag "visualization copy for readable 
    # xdmf" makes another copy of the readable xdmf but with the tradi-
    # tional method of saving. The readable file can sometimes feature 
    # null values in ParaView

    post_processes[0][1]["SaveField"] = {"directory path": results_path, 
    "file name": displacement_file_name, "saving method": "binary", 
    "visualization copy for readable xdmf": True}

    post_processes[1][1]["SaveField"] = {"directory path": results_path, 
    "file name": lagrange_multiplier_file_name, "saving method": "bina"+
    "ry", "txt copy": True}

    ####################################################################
    #                       Material properties                        #
    ####################################################################

    # Sets the Young modulus and the Poisson ratio

    E = {"field file": young_modulus_file, "mesh file": mesh_file_name, 
    "field type": "scalar", "interpolation function": "CG", "polynomia"+
    "l degree": 1}

    poisson = 0.3

    # Sets a dictionary of properties

    material_properties = dict()

    material_properties["E"] = E

    material_properties["nu"] = poisson

    # Sets the material as a neo-hookean material using the correspon-
    # ding class

    constitutive_model = constitutive_models.NeoHookean(
    material_properties)

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

    solver_parameters["newton_relative_tolerance"] = 1e-6

    solver_parameters["newton_absolute_tolerance"] = 1e-6

    solver_parameters["newton_maximum_iterations"] = 15

    # Sets the initial time

    t = 0.2

    # Sets the final pseudotime of the simulation

    t_final = 1.0

    # Sets the maximum number of steps of loading

    maximum_loadingSteps = 5

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Defines a dictionary of tractions

    traction_dictionary = dict()

    # Defines a dictionary of boundary conditions. Each key is a physi-
    # cal group and each value is another dictionary or a list of dicti-
    # onaries with the boundary conditions' information. Each of these 
    # dictionaries must have the key "BC case", which carries the name 
    # of the function that builds this boundary condition

    bcs_dictionary = dict()

    # Applies the boundary condition to any boundary surface (the top in
    # this case), but the RemoveRigidBodyMotion is applied to the domain.
    # Hence, the information of a boundary surface is just for consis-
    # tency with the other BCs programming

    bcs_dictionary["top"] = {"BC case": "RemoveRigidBodyMotion", "cons"+
    "train_translation_only": False}

    bcs_dictionary["bottom"] = {"BC case": "VolumetricAverageOfDisplac"+
    "ementGradient", "average_gradient": 
    displacement_gradient_components}

    ####################################################################
    ####################################################################
    ##                    Calculation and solution                    ##
    ####################################################################
    ####################################################################

    # Solves the variational problem

    variational_framework.hyperelasticity_displacementBased(
    constitutive_model, traction_dictionary, maximum_loadingSteps, 
    t_final, post_processes, mesh_file_name, solver_parameters, 
    polynomial_degree=polynomial_degree, t=t, 
    dirichlet_boundaryConditions=bcs_dictionary, verbose=True, 
    post_processesSubmesh=[])

    # Saves a snapshot of the solution using the automatization of Para-
    # View

    if save_snapshot:

        frozen_snapshots(displacement_file_name+"_visualization_copy", 
        "Displacement", input_path=results_path, 
        camera_position=[3.295421262287293, -4.617612102822565, 3.7314785394692436],
        camera_focal_point=[0.3868267767658733, -0.07264176288584451, 0.4047648861287915],
        camera_up_direction=[-0.14883962531451048, 0.5202719201071424, 0.8409303746947565],
        camera_parallel_scale=1.6406698504546489,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.785724255357283, 0.10503211991434691],
        legend_bar_length=0.7389935760171309,
        size_in_pixels={'aspect ratio': 0.6768115942028986, 'pixels in width': 690},
        axes_color=[0.0, 0.0, 0.0], get_attributes_render=False, 
        output_imageFileName="RVE_displacement_imposed_gradient.png", resolution_ratio=5, 
        warp_by_vector=True, representation_type="Surface With Edges", 
        set_camera_interactively=False, time=1.0)

# Testing block

if __name__=="__main__":

    # Defines the path to the results directory 

    results_path = get_parent_path_of_file()

    displacement_file_name = "displacement_with_imposed_displacement_gradient"

    lagrange_multiplier_file_name = "lagrange_multiplier_displacement_gradient"

    # Defines the mesh file name and the name of the file of the field 
    # of Young modulus

    mesh_file_name = results_path+"//box_mesh_mecsol"

    young_modulus_file = results_path+"//E_from_dict.xdmf"

    # Defines the components of the volume average of the displacement
    # gradient

    displacement_gradient_components = [[2.5, 0.0, 0.0], [0.0, 2.5, 0.0], 
    [0.0, 0.0, 2.5]]

    # Solves the boundary value problem

    solve_BVP(results_path, displacement_file_name, young_modulus_file,
    mesh_file_name, displacement_gradient_components, 
    lagrange_multiplier_file_name, save_snapshot=True)