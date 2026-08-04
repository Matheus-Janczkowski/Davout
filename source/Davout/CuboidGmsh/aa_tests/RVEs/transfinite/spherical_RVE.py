from ......Davout.CuboidGmsh.solids import cuboid_prisms as prisms
from ......Davout.CuboidGmsh.tool_box import meshing_tools as tools
from ......Davout.PythonicUtilities.path_tools import get_parent_path_of_file

import numpy as np

from copy import deepcopy

####################################################################
# Defines a code that generates the spherical geoemetry of the RVE #
####################################################################

# Defines a function that uses the Cuboid from Davout to 
# generates the sphere RVE. The arguments are the radius of the
# sphere and the TOTAL length of the inner cube's edge

def sphere_geometry_RVE(sphere_radius, inner_cube_half_edge_ratio, 
mesh_file_name, n_points_per_edge=100, transfinite_x=5, transfinite_y=5, 
transfinite_z=5, transfinite_radial=5, bias_radial=1.0, bias_x=1.0, 
bias_y=1.0, bias_z=1.0, p_exponent=2.0, mapping_matrix=np.eye(3)): 

    # Verifies if the inner cube is contained within the sphere. To as-
    # sert this, checks for the length of the outermost corner of the
    # cube to the center origin

    if inner_cube_half_edge_ratio>=(1.0/np.sqrt(3.0)):

        raise ValueError("The given 'inner_cube_half_edge_ratio' is "+
        str(inner_cube_half_edge_ratio)+", but it must be less than 1/"+
        "sqrt(3) or "+str(1.0/np.sqrt(3.0))+" for the inner cube to be"+
        " contained within the sphere")

    # Evaluates the inner cube half edge length

    inner_cube_half_length = inner_cube_half_edge_ratio*sphere_radius

    # Pre-computes some useful quantities

    half_pi = 0.5*np.pi 

    fourth_pi = 0.25*np.pi 

    fifth_pi = np.arctan(1.0/np.sqrt(2.0))

    pi = np.pi

    three_fourths_pi = (3/4)*pi
    
    five_fourths_pi = (5/4)*pi

    three_half_pi = (3/2)*pi

    seven_fourths = (7/4)*pi

    # Creates the matrix of points of the vertices of the cuboids in 
    # spherical coordinates. A numpy array of shape (n_points,3) is cre-
    # ated. The first column is the spherical radius, the second column 
    # is the azimuth angle (confined in XY plane), and the third column 
    # is the zenith angle (angle from XY plane to the Z axis)

    vertices_spherical_coordinates = np.array([[0.0, 0.0, 0.0],#00
    [inner_cube_half_length, 0.0, 0.0], #01
    [inner_cube_half_length, fourth_pi, 0.0], #02
    [inner_cube_half_length, half_pi, 0.0],#03
    [inner_cube_half_length, 0.0, half_pi], #04
    [inner_cube_half_length, 0.0, fourth_pi], #05
    [inner_cube_half_length, fourth_pi, fifth_pi],#06
    [inner_cube_half_length, half_pi, fourth_pi],#07
    [sphere_radius, 0.0, 0.0],#08
    [sphere_radius, fourth_pi, 0.0],#09
    [sphere_radius, half_pi, 0.0],#10
    [sphere_radius, 0.0, fourth_pi],#11
    [sphere_radius, fourth_pi, fifth_pi],#12
    [sphere_radius, half_pi, fourth_pi],#13
    [sphere_radius, 0.0, half_pi],#14
    #
    [inner_cube_half_length, three_fourths_pi, 0.0],#15
    [inner_cube_half_length, pi, 0.0],#16
    [inner_cube_half_length, three_fourths_pi, fifth_pi],#17
    [inner_cube_half_length, pi, fourth_pi],#18
    [sphere_radius, three_fourths_pi, 0.0],#19
    [sphere_radius, pi, 0.0],#20
    [sphere_radius, three_fourths_pi, fifth_pi],#21
    [sphere_radius, pi, fourth_pi],#22
    #
    [inner_cube_half_length, five_fourths_pi, 0.0],#23
    [inner_cube_half_length, three_half_pi, 0.0],#24
    [inner_cube_half_length, five_fourths_pi, fifth_pi],#25
    [inner_cube_half_length, three_half_pi, fourth_pi],#26
    [sphere_radius, five_fourths_pi, 0.0],#27
    [sphere_radius, three_half_pi, 0.0],#28
    [sphere_radius, five_fourths_pi, fifth_pi],#29
    [sphere_radius, three_half_pi, fourth_pi],#30
    #
    [inner_cube_half_length, seven_fourths, 0.0],#31
    [inner_cube_half_length, seven_fourths, fifth_pi],#32
    [sphere_radius, seven_fourths, 0.0],#33
    [sphere_radius, seven_fourths, fifth_pi],#34
    #
    # south hemisphere
    #
    [inner_cube_half_length, 0.0, -half_pi], #35
    [inner_cube_half_length, 0.0, -fourth_pi],#36
    [inner_cube_half_length, fourth_pi, -fifth_pi],#37
    [inner_cube_half_length, half_pi, -fourth_pi],#38
    [sphere_radius, 0.0, -fourth_pi],#39
    [sphere_radius, fourth_pi, -fifth_pi],#40
    [sphere_radius, half_pi, -fourth_pi],#41
    [sphere_radius, 0.0, -half_pi],#42
    #
    [inner_cube_half_length, three_fourths_pi, -fifth_pi],#43
    [inner_cube_half_length, pi, -fourth_pi],#44
    [sphere_radius, three_fourths_pi, -fifth_pi],#45
    [sphere_radius, pi, -fourth_pi],#46
    #
    [inner_cube_half_length, five_fourths_pi, -fifth_pi],#47
    [inner_cube_half_length, three_half_pi, -fourth_pi],#48
    [sphere_radius, five_fourths_pi, -fifth_pi],#49
    [sphere_radius, three_half_pi, -fourth_pi],#50
    #
    [inner_cube_half_length, seven_fourths, -fifth_pi],#51
    [sphere_radius, seven_fourths, -fifth_pi]])#52

    # Converts the matrix of points to cartesian coordinates

    vertices_cartesian_coordinates = spherical_to_retangular_coordinates(
    vertices_spherical_coordinates, p_exponent, mapping_matrix)

    # Sets the names of the physical regions

    surfaces_regions_names = ["bottom", "front", "left", "back",
    "right", "top"]

    volumes_regions_names = ["volume"]

    # Initializes the RVE mesh

    geometric_data = tools.gmsh_initialization(surface_regionsNames = 
    surfaces_regions_names, volume_regionsNames = volumes_regions_names)

    ####################################################################
    #                 Cuboid (00-01-02-03|04-05-06-07)                 #
    ####################################################################

    # Creates the inner cube

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    0, 1, 2, 3, 4, 5, 6, 7, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    bias_directions = {"x": -bias_y, "y": bias_x, "z": bias_z})

    ####################################################################
    #                 Cuboid (01-08-09-02|05-11-12-06)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_09_08 = linear_interpolation_in_spherical_coordinates(
    9, 8, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_08_11 = linear_interpolation_in_spherical_coordinates(
    8, 11, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_09_12 = linear_interpolation_in_spherical_coordinates(
    9, 12, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_12_11 = linear_interpolation_in_spherical_coordinates(
    12, 11, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(np.cos(theta)))

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    1, 8, 9, 2, 5, 11, 12, 6, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_09_08, 6: line_points_12_11,
    10: line_points_08_11, 11: line_points_09_12}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "front"},
    bias_directions = {"x": -bias_y, "y": bias_radial, "z": bias_z})

    ####################################################################
    #                 Cuboid (02-09-10-03|06-12-13-07)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_10_09 = linear_interpolation_in_spherical_coordinates(
    10, 9, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_10_13 = linear_interpolation_in_spherical_coordinates(
    10, 13, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -y+z=0 with the sphere

    line_points_13_12 = linear_interpolation_in_spherical_coordinates(
    13, 12, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(np.sin(theta)))

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    2, 9, 10, 3, 6, 12, 13, 7, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_x, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_10_09, 6: line_points_13_12,
    11: line_points_10_13}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "left"},
    bias_directions = {"x": bias_x, "y": bias_radial, "z": bias_z})

    ####################################################################
    #                 Cuboid (04-05-06-07|14-11-12-13)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_14_11 = linear_interpolation_in_spherical_coordinates(
    14, 11, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_13_14 = linear_interpolation_in_spherical_coordinates(
    13, 14, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix, fix_theta=half_pi)

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    4, 5, 6, 7, 14, 11, 12, 13, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x,
    transfinite_radial], geometric_data = geometric_data, 
    edges_points = {5: line_points_14_11, 8: line_points_13_14}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {6: "top"},
    bias_directions = {"x": -bias_y, "y": bias_x, "z": bias_radial})

    ####################################################################
    #                 Cuboid (00-03-15-16|04-07-17-18)                 #
    ####################################################################

    # Creates the second inner cube

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    0, 3, 15, 16, 4, 7, 17, 18, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_x, transfinite_y, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    bias_directions = {"x": -bias_x, "y": bias_y, "z": bias_z})

    ####################################################################
    #                 Cuboid (03-10-19-15|07-13-21-17)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_19_10 = linear_interpolation_in_spherical_coordinates(
    19, 10, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_10_13 = linear_interpolation_in_spherical_coordinates(
    10, 13, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_19_21 = linear_interpolation_in_spherical_coordinates(
    19, 21, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_21_13 = linear_interpolation_in_spherical_coordinates(
    13, 21, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(np.sin(theta)))

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    3, 10, 19, 15, 7, 13, 21, 17, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_x, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_19_10, 6: line_points_21_13,
    10: line_points_10_13, 11: line_points_19_21}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "left"},
    bias_directions = {"x": -bias_x, "y": bias_radial, "z": bias_z})

    ####################################################################
    #                 Cuboid (15-19-20-16|17-21-22-18)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_20_19 = linear_interpolation_in_spherical_coordinates(
    20, 19, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_20_22 = linear_interpolation_in_spherical_coordinates(
    20, 22, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -y+z=0 with the sphere

    line_points_22_21 = linear_interpolation_in_spherical_coordinates(
    22, 21, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(-np.cos(theta)))

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    15, 19, 20, 16, 17, 21, 22, 18, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_20_19, 6: line_points_22_21,
    11: line_points_20_22}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "back"},
    bias_directions = {"x": bias_y, "y": bias_radial, "z": bias_z})

    ####################################################################
    #                 Cuboid (04-07-17-18|14-13-21-22)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_22_14 = linear_interpolation_in_spherical_coordinates(
    22,14, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix, fix_theta=np.pi)

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    4, 7, 17, 18, 14, 13, 21, 22, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_x, transfinite_y,
    transfinite_radial], geometric_data = geometric_data, 
    edges_points = {8: line_points_22_14}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {6: "top"},
    bias_directions = {"x": -bias_x, "y": -bias_y, "z": bias_radial})

    ####################################################################
    #                 Cuboid (00-16-23-24|04-18-25-26)                 #
    ####################################################################

    # Creates the third inner cube

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    0, 16, 23, 24, 4, 18, 25, 26, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    bias_directions = {"x": -bias_y, "y": bias_x, "z": bias_z})

    ####################################################################
    #                 Cuboid (16-20-27-04|18-22-29-25)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_27_23 = linear_interpolation_in_spherical_coordinates(
    27, 23, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_29_27 = linear_interpolation_in_spherical_coordinates(
    29, 27, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_27_20 = linear_interpolation_in_spherical_coordinates(
    27, 20, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_29_22 = linear_interpolation_in_spherical_coordinates(
    29, 22, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(-np.cos(theta)))

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    16, 20, 27, 23, 18, 22, 29, 25, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_27_20, 6: line_points_29_22,
    3: line_points_27_23, 11: line_points_29_27}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "back"},
    bias_directions = {"x": -bias_y, "y": bias_radial, "z": bias_z})

    ####################################################################
    #                 Cuboid (23-27-28-24|25-29-30-26)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_30_26 = linear_interpolation_in_spherical_coordinates(
    30, 26, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_28_30 = linear_interpolation_in_spherical_coordinates(
    28, 30, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_27_28 = linear_interpolation_in_spherical_coordinates(
    27, 28, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_28_24 = linear_interpolation_in_spherical_coordinates(
    28, 24, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_30_29 = linear_interpolation_in_spherical_coordinates(
    30, 29, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(-np.sin(theta)))

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    23, 27, 28, 24, 25, 29, 30, 26, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_27_28, 6: line_points_30_29,
    11: line_points_28_30, 7: line_points_30_26, 3: line_points_28_24}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "right"},
    bias_directions = {"x": bias_x, "y": bias_radial, "z": bias_z})

    ####################################################################
    #                 Cuboid (04-18-25-26|14-22-29-30)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_30_14 = linear_interpolation_in_spherical_coordinates(
    30, 14, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix, fix_theta=three_half_pi)

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    4, 18, 25, 26, 14, 22, 29, 30, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {8: line_points_30_14}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {6: "top"},
    bias_directions = {"x": -bias_y, "y": bias_x, "z": bias_z})

    ####################################################################
    #                 Cuboid (00-24-31-01|04-26-32-05)                 #
    ####################################################################

    # Creates the fouth inner cube

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    0, 24, 31, 1, 4, 26, 32, 5, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_x, transfinite_y, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    bias_directions = {"x": -bias_x, "y": bias_y, "z": bias_z})

    ####################################################################
    #                 Cuboid (24-28-33-31|26-30-34-32)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_33_28 = linear_interpolation_in_spherical_coordinates(
    33, 28, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_33_31 = linear_interpolation_in_spherical_coordinates(
    33, 31, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_33_34 = linear_interpolation_in_spherical_coordinates(
    33, 34, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_32_34 = linear_interpolation_in_spherical_coordinates(
    32, 34, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_34_30 = linear_interpolation_in_spherical_coordinates(
    34, 30, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(-np.sin(theta)))

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    24, 28, 33, 31, 26, 30, 34, 32, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_33_28, 3: line_points_33_31, 11:
    line_points_33_34, 7: line_points_32_34, 6: line_points_34_30}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "right"},
    bias_directions = {"x": -bias_x, "y": bias_radial, "z": bias_z})

    ####################################################################
    #                  Cuboid (31-33-8-1|32-34-11-5)                   #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_33_8 = linear_interpolation_in_spherical_coordinates(
    33, 8, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix, fix_theta=[seven_fourths, 2*np.pi])

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_34_11 = linear_interpolation_in_spherical_coordinates(
    34, 11, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(np.cos(theta)), fix_theta=[seven_fourths, 
    2*np.pi])

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    31, 33, 8, 1, 32, 34, 11, 5, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_33_8, 6: line_points_34_11}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "front"},
    bias_directions = {"x": bias_y, "y": bias_radial, "z": bias_z})

    ####################################################################
    #                  Cuboid (4-26-32-5|14-30-34-11)                  #
    ####################################################################

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    4, 26, 32, 5, 14, 30, 34, 11, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {6: "top"},
    bias_directions = {"x": bias_x, "y": bias_y, "z": bias_radial})

    ####################################################################
    #                   Cuboid (35-36-37-38|0-1-2-3)                   #
    ####################################################################

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    35, 36, 37, 38, 0, 1, 2, 3, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    bias_directions = {"x": -bias_y, "y": bias_x, "z": -bias_z})

    ####################################################################
    #                   Cuboid (36-39-40-37|1-8-9-2)                   #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_36_39 = linear_interpolation_in_spherical_coordinates(
    36, 39, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_37_40 = linear_interpolation_in_spherical_coordinates(
    37, 40, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_8_39 = linear_interpolation_in_spherical_coordinates(
    8, 39, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_9_40 = linear_interpolation_in_spherical_coordinates(
    9, 40, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_39_40 = linear_interpolation_in_spherical_coordinates(
    39, 40, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(-np.cos(theta)))

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    36, 39, 40, 37, 1, 8, 9, 2, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {1: line_points_36_39, 2: line_points_39_40, 3:
    line_points_37_40, 10: line_points_8_39, 11: line_points_9_40}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "front"},
    bias_directions = {"x": -bias_y, "y": bias_radial, "z": -bias_z})

    ####################################################################
    #                   Cuboid (37-40-41-38|2-9-10-3)                  #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_41_38 = linear_interpolation_in_spherical_coordinates(
    41, 38, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_9_10 = linear_interpolation_in_spherical_coordinates(
    9, 10, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_10_41 = linear_interpolation_in_spherical_coordinates(
    10, 41, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_10_3 = linear_interpolation_in_spherical_coordinates(
    10, 3, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_40_41 = linear_interpolation_in_spherical_coordinates(
    40, 41, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(-np.sin(theta)))

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    37, 40, 41, 38, 2, 9, 10, 3, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, bias_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_40_41, 3: line_points_41_38, 6:
    line_points_9_10, 7: line_points_10_3, 11: line_points_10_41}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "left"},
    bias_directions = {"x": bias_x, "y": bias_radial, "z": -bias_z})

    ####################################################################
    #                 Cuboid (42-39-40-41|35-36-37-38)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_35_42 = linear_interpolation_in_spherical_coordinates(
    35, 42, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_42_39 = linear_interpolation_in_spherical_coordinates(
    42, 39, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_41_42 = linear_interpolation_in_spherical_coordinates(
    41, 42, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix, fix_theta=half_pi)

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    42, 39, 40, 41, 35, 36, 37, 38, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_radial], geometric_data = geometric_data, 
    edges_points = {1: line_points_42_39, 4: line_points_41_42, 9:
    line_points_35_42}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {1: "bottom"},
    bias_directions = {"x": -bias_y, "y": bias_x, "z": -bias_radial})

    ####################################################################
    #                  Cuboid (35-38-43-44|0-3-15-16)                  #
    ####################################################################

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    35, 38, 43, 44, 0, 3, 15, 16, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_x, transfinite_y, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    bias_directions = {"x": -bias_x, "y": bias_y, "z": -bias_z})

    ####################################################################
    #                 Cuboid (38-41-45-43|3-10-19-15)                  #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_45_43 = linear_interpolation_in_spherical_coordinates(
    45, 43, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_45_19 = linear_interpolation_in_spherical_coordinates(
    45, 19, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane y+z=0 with the sphere

    line_points_41_45 = linear_interpolation_in_spherical_coordinates(
    41, 45, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(-np.sin(theta)))

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    38, 41, 45, 43, 3, 10, 19, 15, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_x, transfinite_radial, 
    transfinite_radial], geometric_data = geometric_data, 
    edges_points = {2: line_points_41_45, 3: line_points_45_43, 11:
    line_points_45_19}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "left"},
    bias_directions = {"x": -bias_x, "y": bias_radial, "z": -bias_z})

    ####################################################################
    #                 Cuboid (43-45-46-44|15-19-20-16)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_46_44 = linear_interpolation_in_spherical_coordinates(
    46, 44, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_20_16 = linear_interpolation_in_spherical_coordinates(
    20, 16, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_20_46 = linear_interpolation_in_spherical_coordinates(
    20, 46, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_45_46 = linear_interpolation_in_spherical_coordinates(
    45, 46, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(np.cos(theta)))

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    43, 45, 46, 44, 15, 19, 20, 16, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_45_46, 3: line_points_46_44, 7:
    line_points_20_16, 11: line_points_20_46}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "back"},
    bias_directions = {"x": bias_y, "y": bias_radial, "z": -bias_z})

    ####################################################################
    #                 Cuboid (42-41-45-46|35-38-43-44)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_42_46 = linear_interpolation_in_spherical_coordinates(
    42, 46, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix, fix_theta=np.pi)

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    42, 41, 45, 46, 35, 38, 43, 44, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {4: line_points_42_46}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {1: "bottom"},
    bias_directions = {"x": -bias_x, "y": bias_y, "z": -bias_radial})

    ####################################################################
    #                  Cuboid (35-44-47-48|0-16-23-24)                 #
    ####################################################################

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    35, 44, 47, 48, 0, 16, 23, 24, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    bias_directions = {"x": -bias_y, "y": bias_x, "z": -bias_z})

    ####################################################################
    #                 Cuboid (44-46-49-47|16-20-27-23)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_49_47 = linear_interpolation_in_spherical_coordinates(
    49, 47, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_49_27 = linear_interpolation_in_spherical_coordinates(
    49, 27, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -x+z=0 with the sphere

    line_points_49_46 = linear_interpolation_in_spherical_coordinates(
    49, 46, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(np.cos(theta)))

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    44, 46, 49, 47, 16, 20, 27, 23, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_49_46, 3: line_points_49_47, 11:
    line_points_49_27}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "back"},
    bias_directions = {"x": -bias_y, "y": bias_radial, "z": -bias_z})

    ####################################################################
    #                 Cuboid (47-49-50-48|23-27-28-24)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_48_50 = linear_interpolation_in_spherical_coordinates(
    48, 50, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_50_28 = linear_interpolation_in_spherical_coordinates(
    50, 28, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -y+z=0 with the sphere

    line_points_49_50 = linear_interpolation_in_spherical_coordinates(
    49, 50, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(np.sin(theta)))

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    47, 49, 50, 48, 23, 27, 28, 24, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_49_50, 3: line_points_48_50, 11:
    line_points_50_28}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "right"},
    bias_directions = {"x": bias_x, "y": bias_radial, "z": -bias_z})

    ####################################################################
    #                 Cuboid (42-46-49-50|35-44-47-48)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_42_50 = linear_interpolation_in_spherical_coordinates(
    42, 50, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix, fix_theta=three_half_pi)

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    42, 46, 49, 50, 35, 44, 47, 48, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {4: line_points_42_50}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {1: "bottom"},
    bias_directions = {"x": -bias_y, "y": bias_x, "z": -bias_radial})

    ####################################################################
    #                  Cuboid (35-48-51-36|0-24-31-1)                  #
    ####################################################################

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    35, 48, 51, 36, 0, 24, 31, 1, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    bias_directions = {"x": -bias_x, "y": bias_y, "z": -bias_z})

    ####################################################################
    #                 Cuboid (48-50-52-51|24-28-33-31)                 #
    ####################################################################

    # Creates the points for the lines of this cuboid

    line_points_52_33 = linear_interpolation_in_spherical_coordinates(
    52, 33, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    line_points_52_51 = linear_interpolation_in_spherical_coordinates(
    52, 51, vertices_spherical_coordinates, n_points_per_edge, p_exponent,
    mapping_matrix)

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane -y+z=0 with the sphere

    line_points_52_50 = linear_interpolation_in_spherical_coordinates(
    52, 50, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(np.sin(theta)))

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    48, 50, 52, 51, 24, 28, 33, 31, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_52_50, 3: line_points_52_51, 11:
    line_points_52_33}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "right"},
    bias_directions = {"x": -bias_x, "y": bias_radial, "z": -bias_z})

    ####################################################################
    #                  Cuboid (51-52-39-36|31-33-8-1)                  #
    ####################################################################

    # For these lines, calculates phi as a function of theta to get the
    # intersection of the plane x+z=0 with the sphere

    line_points_52_39 = linear_interpolation_in_spherical_coordinates(
    52, 39, vertices_spherical_coordinates, n_points_per_edge, 
    p_exponent, mapping_matrix, function_to_get_phi_from_theta=
    lambda theta: np.arctan(-np.cos(theta)), fix_theta=[seven_fourths, 
    2*np.pi])

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    51, 52, 39, 36, 31, 33, 8, 1, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data = geometric_data, 
    edges_points = {2: line_points_52_39}, 
    explicit_volume_physical_group_name = "volume", 
    explicit_surface_physical_group_name = {3: "front"},
    bias_directions = {"x": bias_y, "y": bias_radial, "z": -bias_z})

    ####################################################################
    #                 Cuboid (42-50-52-39|35-48-51-36)                 #
    ####################################################################

    # Generates this Cuboid

    geometric_data = prisms.hexahedron_from_corners(get_corner_points(
    42, 50, 52, 39, 35, 48, 51, 36, vertices_cartesian_coordinates), 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data = geometric_data, 
    explicit_volume_physical_group_name = "volume",
    explicit_surface_physical_group_name = {1: "bottom"})

    ####################################################################
    #                          Mesh generation                         #
    ####################################################################
    
    tools.gmsh_finalize(geometric_data = geometric_data, file_name = 
    mesh_file_name, file_directory="")

########################################################################
#                              Utilities                               #
########################################################################

# Defines a class to scale the position vector to force this vector to
# reach the Lp-ball

class ScaleVectorToLpBall:
    
    def __init__(self, p_exponent, mapping_matrix):

        # Saves the exponent of the Lp norm

        self.p_exponent = p_exponent

        # And the matrix that deforms the vectors

        self.mapping_matrix = mapping_matrix

    # Defines a method to scale the vector

    def __call__(self, position_cartesian_coordinates, radial_component):
        
        # Gets the Lp norm of the position vector

        Lp_norm = np.maximum(1E-12, np.linalg.norm(
        position_cartesian_coordinates, ord=self.p_exponent, axis=1))

        # Gets the scale factor and multiplies it by the position vector
        
        scaling_factors = (radial_component/Lp_norm)[:,np.newaxis]

        return np.einsum('ij,kj->ki', self.mapping_matrix, 
        scaling_factors*position_cartesian_coordinates)

# Defines a function that transform the spherical coordinates in 
# retangular coordinates

def spherical_to_retangular_coordinates(spherical_coordinates, 
p_exponent, mapping_matrix):

    # Instantiates the class to scale the position vector by the Lp norm

    Lp_scaling_class = ScaleVectorToLpBall(p_exponent, mapping_matrix)

    # "r" is the radius, "theta" is the azimuth angle and "phi" is the 
    # polar or zenith angle. All angles in radians

    r = spherical_coordinates[:,0]
    
    theta = spherical_coordinates[:,1]
    
    phi = spherical_coordinates[:,2]

    x = r*np.cos(phi)*np.cos(theta)

    y = r*np.cos(phi)*np.sin(theta)

    z = r*np.sin(phi)

    # Stacks the cartesian coordinates and scales the corresponding posi-
    # tion vectors by the Lp norm 

    return Lp_scaling_class(np.column_stack((x, y, z)), r)

# Defines a function that generates a numpy array of shape (n_points,3)
# such that each column is a linear interpolation between two bounds.
# This function receives the start and end points in spherical coordinates
# and automatically interpolate them

def linear_interpolation_in_spherical_coordinates(start_index, end_index,
points_matrix, n_points, p_exponent, mapping_matrix, fix_r=None, 
fix_theta=None, fix_phi=None, function_to_get_phi_from_theta=None):

    # Recovers the start and end points

    start_point = points_matrix[start_index,:]

    end_point = points_matrix[end_index,:]

    # Gets the lower and upper bounds for each spherical coordinates

    lower_r = start_point[0]

    lower_theta = start_point[1]

    lower_phi = start_point[2]

    upper_r = end_point[0]

    upper_theta = end_point[1]

    upper_phi = end_point[2]

    # If fixed variables are given

    if isinstance(fix_r, list):

        lower_r = deepcopy(fix_r[0])

        upper_r = deepcopy(fix_r[1])

    elif fix_r is not None:

        lower_r = deepcopy(fix_r)

        upper_r = deepcopy(fix_r)

    if isinstance(fix_theta, list):

        lower_theta = deepcopy(fix_theta[0])

        upper_theta = deepcopy(fix_theta[1])
    
    elif fix_theta is not None:

        lower_theta = deepcopy(fix_theta)

        upper_theta = deepcopy(fix_theta)

    if isinstance(fix_phi, list):

        lower_phi = deepcopy(fix_phi[0])

        upper_phi = deepcopy(fix_phi[1])

    elif fix_phi is not None:

        lower_phi = deepcopy(fix_phi)

        upper_phi = deepcopy(fix_phi)

    # Generates the column for each spherical coordinates

    r = np.linspace(lower_r, upper_r, num=n_points)

    theta = np.linspace(lower_theta, upper_theta, num=n_points)

    phi = None

    # If phi is to be obtained as a function from theta

    if function_to_get_phi_from_theta is not None:

        # Applies the given function directly to the vector of theta

        phi = function_to_get_phi_from_theta(theta)

    else:

        phi = np.linspace(lower_phi, upper_phi, num=n_points)

    # Stacks the columns into a matrix. Converts the matrices from sphe-
    # rical to cartesian coordinates

    spherical_matrix = np.column_stack((r, theta, phi))

    return spherical_to_retangular_coordinates(spherical_matrix,
    p_exponent, mapping_matrix)

# Defines a function to get the corner points from the points matrix

def get_corner_points(vertex_1, vertex_2, vertex_3, vertex_4, vertex_5, 
vertex_6, vertex_7, vertex_8, points_matrix):

    # Gets the corners using the given indices

    corner_1 = points_matrix[vertex_1,:]

    corner_2 = points_matrix[vertex_2,:]

    corner_3 = points_matrix[vertex_3,:]

    corner_4 = points_matrix[vertex_4,:]

    corner_5 = points_matrix[vertex_5,:]

    corner_6 = points_matrix[vertex_6,:]

    corner_7 = points_matrix[vertex_7,:]

    corner_8 = points_matrix[vertex_8,:]

    # Returns a single numpy array (8,3)

    corners_matrix = np.row_stack((corner_1, corner_2, corner_3, corner_4, 
    corner_5, corner_6, corner_7, corner_8))

    return corners_matrix

# Testing block

if __name__=="__main__":

    # Defines the geometric properties of the spheric RVE

    sphere_radius = 1.0

    inner_cube_half_edge_ratio = 0.5

    mesh_file_name = get_parent_path_of_file()+"//spheric_RVE"

    # Defines the principal directions of the RVE's ellipsoid

    d_1 = np.array([1.0, 1.0, 0.0])

    d_1 = (1.0/np.linalg.norm(d_1))*d_1

    d_2 = np.array([-1.0, 1.0, 0.0])

    d_2 = (1.0/np.linalg.norm(d_2))*d_2

    d_3 = np.cross(d_1, d_2)

    # Defines the axis semi-length of each principal direction

    axis_semi_length_1 = 1.0

    axis_semi_length_2 = 1.5

    axis_semi_length_3 = 1.0

    mapping_matrix = (axis_semi_length_1*np.outer(d_1, d_1))+(
    axis_semi_length_2*np.outer(d_2, d_2))+(axis_semi_length_3*np.outer(
    d_3, d_3))

    sphere_geometry_RVE(sphere_radius, inner_cube_half_edge_ratio, 
    mesh_file_name, n_points_per_edge=100, transfinite_radial=4, 
    transfinite_x=5, transfinite_y=6, transfinite_z=7, bias_radial=3.0,
    bias_x=1.0, bias_y=1.7, bias_z=1.0, p_exponent=2.0, mapping_matrix=
    mapping_matrix)