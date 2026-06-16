# Routine to test a hyperelastic disc

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as constitutive_models

from .....Davout.MultiMech.physics import hyperelastic_cauchy_continuum as variational_framework

from .....Davout.GraphUtilities.paraview_tools import frozen_snapshots

def solve_BVP(results_path, displacement_file_name, young_modulus_file,
mesh_file_name, displacement_gradient_components, save_snapshot=False):

    ####################################################################
    ####################################################################
    ##                    User defined parameters                     ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                        Simulation results                        #
    ####################################################################

    post_processes = {}

    # The flag "readable xdmf file" makes the file able to be read into 
    # a function afterwards. The flag "visualization copy for readable 
    # xdmf" makes another copy of the readable xdmf but with the tradi-
    # tional method of saving. The readable file can sometimes feature 
    # null values in ParaView

    post_processes["SaveField"] = {"directory path": results_path, 
    "file name": displacement_file_name, "saving method": "binary", 
    "visualization copy for readable xdmf": True}

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

    t = 0.0

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
        camera_position=[2.0330282921191993, 1.8148603901320899, 1.2731417495411312],
        camera_focal_point=[0.012529434003528184, 0.36158465784309013, 0.12076486597591955],
        camera_up_direction=[-0.3230357774017903, -0.27010620502800287, 0.9070228908488427],
        camera_parallel_scale=0.7098627676493106,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.7242157236408994, 0.11000000000000004],
        legend_bar_length=0.7600000000000029,
        size_in_pixels={'aspect ratio': 0.6791171477079796, 'pixels in width': 589},
        axes_color=[0.0, 0.0, 0.0], get_attributes_render=False, 
        output_imageFileName="RVE_displacement.png", resolution_ratio=5, 
        warp_by_vector=True, representation_type="Surface With Edges", 
        set_camera_interactively=False, time=1.0)

# Testing block

if __name__=="__main__":

    # Defines the path to the results directory 

    results_path = get_parent_path_of_file()

    displacement_file_name = "displacement_with_imposed_displacement_gradient"

    # Defines the mesh file name and the name of the file of the field 
    # of Young modulus

    mesh_file_name = results_path+"//box_mesh_mecsol"

    young_modulus_file = results_path+"//E_from_dict.xdmf"

    # Defines the components of the volume average of the displacement
    # gradient

    displacement_gradient_components = [[-0.060, 0.0, 0.0], [0.0, -0.060, 0.0], 
    [0.0, 0.0, -0.060]]

    # Solves the boundary value problem

    solve_BVP(results_path, displacement_file_name, young_modulus_file,
    mesh_file_name, displacement_gradient_components, save_snapshot=True)