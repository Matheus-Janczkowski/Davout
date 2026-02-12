# Routine to test a hyperelastic disc

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.MultiMech.constitutive_models.hyperelasticity import isotropic_hyperelasticity as constitutive_models

from .....Davout.MultiMech.physics import hyperelastic_cauchy_continuum as variational_framework

from .....Davout.GraphUtilities.paraview_tools import frozen_snapshots

from .....Davout.GraphUtilities.collage_tools import create_box_collage

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

displacement_fileName = "displacement.xdmf"

post_processes = dict()

post_processes["SaveField"] = {"directory path":results_path, 
"file name":displacement_fileName}

post_processes["SaveForcesAndMomentsOnSurface"] = {"directory path": 
results_path, "file name": "forces_and_moments.txt", "surface physical"+
" group name": "right"}

post_processes["SaveStrainEnergy"] = {"directory path": results_path, 
"file name": "strain_energy.txt"}

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

mesh_fileName = {"length x": 0.3, "length y": 1.0, "length z": 0.2, "n"+
"umber of divisions in x": 5, "number of divisions in y": 25, "number o"+
"f divisions in z": 5, "verbose": False, "mesh file name": "box_mesh", 
"mesh file directory": get_parent_path_of_file(), "number of subdomain"+
"s in z direction": 2}

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
"tude_tractionX": 0.0, "amplitude_tractionY": maximum_load, "amplitude_tractionZ": 
0.0, "parametric_load_curve": "square_root", "t": t, "t_final":
t_final}

# Defines a dictionary of tractions

traction_dictionary = dict()

traction_dictionary["right"] = traction_boundary

# Defines a dictionary of boundary conditions. Each key is a physical
# group and each value is another dictionary or a list of dictionaries 
# with the boundary conditions' information. Each of these dictionaries
# must have the key "BC case", which carries the name of the function 
# that builds this boundary condition

bcs_dictionary = dict()

bcs_dictionary["left"] = {"BC case": "FixedSupportDirichletBC"}

########################################################################
########################################################################
##                      Calculation and solution                      ##
########################################################################
########################################################################

# Solves the variational problem

variational_framework.hyperelasticity_displacementBased(
constitutive_model, traction_dictionary, maximum_loadingSteps, t_final, 
post_processes, mesh_fileName, solver_parameters, polynomial_degree=
polynomial_degree, t=t, dirichlet_boundaryConditions=bcs_dictionary, 
verbose=True)

# Saves a snapshot of the solution using the automatization of ParaView

new_figures = False 

if new_figures:

    frozen_snapshots(displacement_fileName, "Displacement", input_path=
    get_parent_path_of_file(), time=t_final, representation_type=
    "Surface With Edges", axes_color=[0.0, 0.0, 0.0], legend_bar_font="Tim"+
    "es", zoom_factor=1.0, component_to_plot="2", resolution_ratio=10, clip=
    False, clip_plane_origin=[0.15, 0.5, 0.1], clip_plane_normal_vector=
    [1.0, 1.0, 1.0], warp_by_vector=True, background_color="WhiteBackground",
    display_reference_configuration=False, transparent_background=True,
    legend_bar_font_color=[0.0, 0.0, 0.0], set_camera_interactively=False,
    camera_position=[-2.2202601190413063, 2.7845260002185928, 2.412838032797255],
    camera_focal_point=[0.500121318497246, 0.8894379799910188, 0.03488656665544461],
    camera_up_direction=[0.19171813331267484, 0.8626133794219494, -0.46812638784984917],
    camera_parallel_scale=0.872714431199938,
    camera_rotation=[0.0, 0.0, 0.0],
    legend_bar_position=[0.6048594377510041, 0.09555555555555562],
    legend_bar_length=0.7344444444444449,
    size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

    frozen_snapshots(displacement_fileName, "Displacement", input_path=
    get_parent_path_of_file(), time=t_final, representation_type=
    "Surface With Edges", axes_color=[0.0, 0.0, 0.0], legend_bar_font="Tim"+
    "es", zoom_factor=1.0, component_to_plot="2", resolution_ratio=10, clip=
    False, clip_plane_origin=[0.15, 0.5, 0.1], clip_plane_normal_vector=
    [1.0, 1.0, 1.0], warp_by_vector=True, background_color="WhiteBackground",
    display_reference_configuration=False, transparent_background=True,
    legend_bar_font_color=[0.0, 0.0, 0.0], set_camera_interactively=False, 
    output_imageFileName="plot2.png",
    camera_position=[-1.9564135985404367, 3.578848197233133, 1.3405449192875027],
    camera_focal_point=[0.2025655268960625, 0.7311641302620293, -0.6282088835248996],
    camera_up_direction=[-0.7902858082954154, -0.6124372353110827, 0.019208696243617818],
    camera_parallel_scale=0.872714431199938,
    camera_rotation=[0.0, 0.0, 0.0],
    legend_bar_position=[0.4838246202873453, 0.10398826979472141],
    legend_bar_length=0.6516788856304988,
    size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

# Creates a collage

create_box_collage("collage.png", input_path=get_parent_path_of_file(),
input_image_list=[{"file name": "plot.png", "position": [50.0, 200.0], 
"size": 50.0, "trim transparent background": True, "origin point":
"bottom-left"}, 
{"file name": "plot2.png", "position": [107.0, 200.0], "size": 50.0, 
"trim transparent background": True, "origin point": "bottom-left"}], 

input_text_list=[{"text": "Lateral", "position": [100.0, 195.0], "font size": 
10, "origin point": "bottom-right"}, 
{"text": "Upper", "position": [157.0, 195.0], "font size": 10,
"origin point": "bottom-right"},
{"text": "$\MaterialDivOf{\\boldsymbol{P}}=\\boldsymbol{0}$", 
"position": [50.0, 195.0], "font size": 10, "origin point": "bottom-left"}],

boxes_list=[{"contour color": "black", "fill color": "red 2", "contour"+
" thickness": 0.2, "position": [46.5, 193.0], "width": 114.0, "height": 58.0,
"contour style": "solid", "origin point": "bottom-left"},
{"contour color": "black", "fill color": "red 1", "contour"+
" thickness": 0.2, "position": [47.5, 194.0], "width": 55.0, "height": 56.0,
"contour style": "solid", "origin point": "bottom-left"},
{"contour color": "black", "fill color": "red 1", "contour"+
" thickness": 0.2, "position": [104.5, 194.0], "width": 55.0, "height": 56.0,
"contour style": "solid", "origin point": "bottom-left"}], 

arrows_and_lines_list=[{"start point": [99.0, 200.0], "end point": [109.0, 200.0],
"polygonal points": [[104., 202.0]], "thickness": 0.2, "arrow style": "inkscape angular arrow",
"line style": "dashed 7x3"},
{"start point": [99.0, 240.0], "end point": [99.0, 240.0], 
"spline points": [[102.3, 243.0], [105.6, 243.0], [109.0, 240.0], 
[105.6, 237.0], [102.3, 237.0]], "closed path": True, "thickness": 0.2,
"arrow style": "no arrow", "fill path with color": "red 3"}],

verbose=True, no_padding=True, add_overlaying_grid=True, dpi=500)