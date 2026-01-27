# Routine to mesh the intervertebral disc given points
# python3 -m source.Grante.CuboidGmsh.aa_tests.solids.intervertebral_disc

import numpy as np

from ...solids import cuboid_prisms as prisms

from ...tool_box import meshing_tools as tools

from ....PythonicUtilities.interpolation_tools import spline_3D_interpolation

# Defines a function to construct the intervertebral disc using prisms

def mesh_disc():

    height = 2.5

    n_points_spline = 15

    inner_radius = 3.0

    outer_radius = 5.0

    cube_radius = 0.5*inner_radius

    bias_axial = ["Bump", 0.5]

    transfinite_directions = [5, 7, 9]

    # Square curve

    points_array_inferior_square = [[cube_radius*np.cos(np.pi*(1/4)), 
    cube_radius*np.sin(np.pi*(1/4)), 0.0], [cube_radius*np.cos(np.pi*(3/
    4)), cube_radius*np.sin(np.pi*(3/4)), 0.0], [cube_radius*np.cos(
    np.pi*(5/4)), cube_radius*np.sin(np.pi*(5/4)), 0.0], [cube_radius*
    np.cos(np.pi*(7/4)), cube_radius*np.sin(np.pi*(7/4)), 0.0]]

    points_array_superior_square = [[cube_radius*np.cos(np.pi*(1/4)), 
    cube_radius*np.sin(np.pi*(1/4)), height], [cube_radius*np.cos(np.pi*
    (3/4)), cube_radius*np.sin(np.pi*(3/4)), height], [cube_radius*
    np.cos(np.pi*(5/4)), cube_radius*np.sin(np.pi*(5/4)), height], [
    cube_radius*np.cos(np.pi*(7/4)), cube_radius*np.sin(np.pi*(7/4)), 
    height]]

    inferior_curve_cube, initial_angle, final_angle = spline_3D_interpolation(points_array=
    points_array_inferior_square, add_initial_point_as_end_point=True,
    polar_angle_around_z=True, verbose=True, return_initial_and_final_polar_angles=True)

    print("Initial angle: "+str(initial_angle)+"; final angle: "+str(
    final_angle)+"\n")

    superior_curve_cube = spline_3D_interpolation(points_array=
    points_array_superior_square, add_initial_point_as_end_point=True,
    polar_angle_around_z=True, verbose=True)

    # Inner curve

    points_array_inferior_inner = [[inner_radius*np.cos(np.pi*(1/4)), 
    inner_radius*np.sin(np.pi*(1/4)), 0.0], [inner_radius*np.cos(np.pi*(3/
    4)), inner_radius*np.sin(np.pi*(3/4)), 0.0], [inner_radius*np.cos(
    np.pi*(5/4)), inner_radius*np.sin(np.pi*(5/4)), 0.0], [inner_radius*
    np.cos(np.pi*(7/4)), inner_radius*np.sin(np.pi*(7/4)), 0.0]]

    points_array_superior_inner = [[inner_radius*np.cos(np.pi*(1/4)), 
    inner_radius*np.sin(np.pi*(1/4)), height], [inner_radius*np.cos(np.pi*
    (3/4)), inner_radius*np.sin(np.pi*(3/4)), height], [inner_radius*
    np.cos(np.pi*(5/4)), inner_radius*np.sin(np.pi*(5/4)), height], [
    inner_radius*np.cos(np.pi*(7/4)), inner_radius*np.sin(np.pi*(7/4)), 
    height]]

    """inferior_curve_inner = lambda theta: [inner_radius*np.cos((2*np.pi*
    theta)+((7/4)*np.pi)), inner_radius*np.sin((2*np.pi*theta)+((7/4)*
    np.pi)), 0.0]

    superior_curve_inner = lambda theta: [inner_radius*np.cos((2*np.pi*
    theta)+((7/4)*np.pi)), inner_radius*np.sin((2*np.pi*theta)+((7/4)*
    np.pi)), height]"""

    inferior_curve_inner = spline_3D_interpolation(points_array=
    points_array_inferior_inner, add_initial_point_as_end_point=True,
    polar_angle_around_z=True, verbose=True)

    superior_curve_inner = spline_3D_interpolation(points_array=
    points_array_superior_inner, add_initial_point_as_end_point=True,
    polar_angle_around_z=True, verbose=True)

    # Outer curve

    points_array_inferior_outer = [[outer_radius*np.cos(np.pi*(1/4)), 
    outer_radius*np.sin(np.pi*(1/4)), 0.0], [outer_radius*np.cos(np.pi*(3/
    4)), outer_radius*np.sin(np.pi*(3/4)), 0.0], [outer_radius*np.cos(
    np.pi*(5/4)), outer_radius*np.sin(np.pi*(5/4)), 0.0], [outer_radius*
    np.cos(np.pi*(7/4)), outer_radius*np.sin(np.pi*(7/4)), 0.0]]

    points_array_superior_outer = [[outer_radius*np.cos(np.pi*(1/4)), 
    outer_radius*np.sin(np.pi*(1/4)), height], [outer_radius*np.cos(np.pi*
    (3/4)), outer_radius*np.sin(np.pi*(3/4)), height], [outer_radius*
    np.cos(np.pi*(5/4)), outer_radius*np.sin(np.pi*(5/4)), height], [
    outer_radius*np.cos(np.pi*(7/4)), outer_radius*np.sin(np.pi*(7/4)), 
    height]]

    """inferior_curve_outer = lambda theta: [outer_radius*np.cos((2*np.pi*
    theta)+((7/4)*np.pi)), outer_radius*np.sin((2*np.pi*theta)+((7/4)*
    np.pi)), 0.0]

    superior_curve_outer = lambda theta: [outer_radius*np.cos((2*np.pi*
    theta)+((7/4)*np.pi)), outer_radius*np.sin((2*np.pi*theta)+((7/4)*
    np.pi)), height]"""

    inferior_curve_outer = spline_3D_interpolation(points_array=
    points_array_inferior_outer, add_initial_point_as_end_point=True,
    polar_angle_around_z=True, verbose=True)

    superior_curve_outer = spline_3D_interpolation(points_array=
    points_array_superior_outer, add_initial_point_as_end_point=True,
    polar_angle_around_z=True, verbose=True)

    parametric_curves = {"inferior square": inferior_curve_cube, "supe"+
    "rior square": superior_curve_cube, "inferior inner": 
    inferior_curve_inner, "superior inner": superior_curve_inner, "inf"+
    "erior outer": inferior_curve_outer, "superior outer":
    superior_curve_outer}

    ####################################################################
    #                     Boundary surfaces setting                    #
    ####################################################################

    # Sets the names of the surface regions

    surface_regionsNames = ['lower', 'upper', 'outer side']

    ####################################################################
    #                    Volumetric regions setting                    #
    ####################################################################

    # Sets the names of the volume regions

    volume_regionsNames = ['nucleus', 'annulus']

    ####################################################################
    #                              Cuboids                             #
    ####################################################################

    # Initializes the geometric data

    geometric_data = tools.gmsh_initialization(surface_regionsNames=
    surface_regionsNames, volume_regionsNames=volume_regionsNames)

    # Center cube

    corner_points = [["inferior square", 45.0], ["inferior square", 135.0], 
    ["inferior square", 225.0], ["inferior square", 315.0], ["superior sq"+
    "uare", 45.0], ["superior square", 135.0], ["superior square", 225.0], [
    "superior square", 315.0]]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    # First nucleus flare

    theta_1 = 45.0

    theta_2 = 135.0

    corner_points = [["inferior inner", theta_1], ["inferior inner", 
    theta_2], ["inferior square", theta_2], ["inferior square", theta_1
    ], ["superior inner", theta_1], ["superior inner", theta_2], ["sup"+
    "erior square", theta_2], ["superior square", theta_1]]

    edge_points_1 = ["inferior inner", theta_1, theta_2, n_points_spline]

    edge_points_5 = ["superior inner", theta_1, theta_2, n_points_spline]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    # Second nucleus flare

    theta_1 = 135.0

    theta_2 = 225.0

    corner_points = [["inferior inner", theta_1], ["inferior inner", 
    theta_2], ["inferior square", theta_2], ["inferior square", theta_1
    ], ["superior inner", theta_1], ["superior inner", theta_2], ["sup"+
    "erior square", theta_2], ["superior square", theta_1]]

    edge_points_1 = ["inferior inner", theta_1, theta_2, n_points_spline]

    edge_points_5 = ["superior inner", theta_1, theta_2, n_points_spline]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    # Third nucleus flare

    theta_1 = 225.0

    theta_2 = 315.0

    corner_points = [["inferior inner", theta_1], ["inferior inner", 
    theta_2], ["inferior square", theta_2], ["inferior square", theta_1
    ], ["superior inner", theta_1], ["superior inner", theta_2], ["sup"+
    "erior square", theta_2], ["superior square", theta_1]]

    edge_points_1 = ["inferior inner", theta_1, theta_2, n_points_spline]

    edge_points_5 = ["superior inner", theta_1, theta_2, n_points_spline]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    # Fourth nucleus flare

    theta_1 = 315.0

    theta_2 = 405.0

    corner_points = [["inferior inner", theta_1], ["inferior inner", 
    theta_2], ["inferior square", theta_2], ["inferior square", theta_1
    ], ["superior inner", theta_1], ["superior inner", theta_2], ["sup"+
    "erior square", theta_2], ["superior square", theta_1]]

    edge_points_1 = ["inferior inner", theta_1, theta_2, n_points_spline]

    edge_points_5 = ["superior inner", theta_1, theta_2, n_points_spline]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"},
    edges_points={1: edge_points_1, 5: edge_points_5},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    """# First annulus flare

    theta_1 = 45.0

    theta_2 = 135.0

    corner_points = [["inferior outer", theta_1], ["inferior outer", 
    theta_2], ["inferior inner", theta_2], ["inferior inner", theta_1],
    ["superior outer", theta_1], ["superior outer", theta_2], ["superi"+
    "or inner", theta_2], ["superior inner", theta_1]]

    edge_points_1 = ["inferior outer", theta_1, theta_2, n_points_spline]

    edge_points_5 = ["superior outer", theta_1, theta_2, n_points_spline]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="annulus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper", 2:
    "outer side"}, edges_points={1: edge_points_1, 5: edge_points_5},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    # Second annulus flare

    theta_1 = 135.0

    theta_2 = 225.0

    corner_points = [["inferior outer", theta_1], ["inferior outer", 
    theta_2], ["inferior inner", theta_2], ["inferior inner", theta_1],
    ["superior outer", theta_1], ["superior outer", theta_2], ["superi"+
    "or inner", theta_2], ["superior inner", theta_1]]

    edge_points_1 = ["inferior outer", theta_1, theta_2, n_points_spline]

    edge_points_5 = ["superior outer", theta_1, theta_2, n_points_spline]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="annulus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper", 2:
    "outer side"}, edges_points={1: edge_points_1, 5: edge_points_5},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    # Third annulus flare

    theta_1 = 225.0

    theta_2 = 315.0

    corner_points = [["inferior outer", theta_1], ["inferior outer", 
    theta_2], ["inferior inner", theta_2], ["inferior inner", theta_1],
    ["superior outer", theta_1], ["superior outer", theta_2], ["superi"+
    "or inner", theta_2], ["superior inner", theta_1]]

    edge_points_1 = ["inferior outer", theta_1, theta_2, n_points_spline]

    edge_points_5 = ["superior outer", theta_1, theta_2, n_points_spline]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="annulus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper", 2:
    "outer side"}, edges_points={1: edge_points_1, 5: edge_points_5},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    # Fourth annulus flare

    theta_1 = 315.0

    theta_2 = 405.0

    corner_points = [["inferior outer", theta_1], ["inferior outer", 
    theta_2], ["inferior inner", theta_2], ["inferior inner", theta_1],
    ["superior outer", theta_1], ["superior outer", theta_2], ["superi"+
    "or inner", theta_2], ["superior inner", theta_1]]

    edge_points_1 = ["inferior outer", theta_1, theta_2, n_points_spline]

    edge_points_5 = ["superior outer", theta_1, theta_2, n_points_spline]

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, explicit_volume_physical_group_name="annulus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper", 2:
    "outer side"}, edges_points={1: edge_points_1, 5: edge_points_5},
    parametric_curves=parametric_curves, bias_directions={"z": 
    bias_axial})

    # Creates the geometry and meshes it"""

    tools.gmsh_finalize(geometric_data=geometric_data, file_name="inte"+
    "rvertebral_disc")

# Test block

if __name__=="__main__":

    mesh_disc()