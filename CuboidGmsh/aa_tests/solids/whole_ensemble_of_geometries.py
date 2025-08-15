# Routine to test the implementations of various geometries

import numpy as np

import os 

import CuboidGmsh.source.solids.cuboid_prisms as prisms

import CuboidGmsh.source.solids.cuboid_cylinders as cylinders

import CuboidGmsh.source.solids.cuboid_ellipsoids as ellipsoids

import CuboidGmsh.source.solids.cuboid_helicoids as helicoids

import CuboidGmsh.source.tool_box.meshing_tools as tools

########################################################################
#                           Battery of tests                           #
########################################################################

# Defines a function to test many geometries one after the other using 
# the below-listed test functions

def battery_ofTests():

    # Defines a list of tests to perform, or leave it empty to test all
    # of them

    test_list = [test_cylinderInBox]#, test_cylinder, 
    #test_rectangularPrismMicropolar, test_rectangularPrism,
    #test_sectorHollowCylinder, test_ellipsoidShellQuarter, 
    #test_ellipsoidPrismQuarter]

    # Iterates through the tests

    for test_function in test_list:

        test_function()

########################################################################
#                                Prisms                                #
########################################################################

def test_rectangularPrismMicropolar():

    print("###########################################################"+
    "#############\n#                 Rectangular prism mesh for micro"+
    "polar                #\n#########################################"+
    "###############################")

    gamma = 1.18E0

    mu = 26.12

    beta = 0.0

    ratio_Lb = 1.5E-1

    L = np.sqrt((beta+gamma)/(2*mu))

    b = L/ratio_Lb

    length_x = b*1.0

    length_y = b*1.0

    length_z = b*10.0#7.8

    axis_vector = [1.0, 0.0, 0.0]

    base_point = [0.0, 0.0, 0.0]

    transfinite_directions = [7,7,11]

    ####################################################################
    #                     Boundary surfaces setting                    #
    ####################################################################

    # Sets the finder for the boundary surfaces

    # XY plane at z = 0

    def back_expression(x, y, z):

        return z

    # XY plane at z = length_z

    def front_expression(x, y, z):

        return z-length_z

    # XZ plane at y = 0

    def lower_expression(x, y, z):

        return y

    # XZ plane at y = length_y

    def upper_expression(x, y, z):

        return y-length_y

    # YZ plane at x = 0

    def right_expression(x, y, z):

        return x

    # YZ plane at x = x_length

    def left_expression(x, y, z):

        return x-length_x

    # Sets a list of expressions to find the surfaces at the boundaries

    surface_regionsExpressions = [back_expression, front_expression, 
    lower_expression, upper_expression, right_expression, left_expression]

    # Sets the names of the surface regions

    surface_regionsNames = ['back', 'front', 'lower', 'upper', 'right',
    'left']

    geometric_data = tools.gmsh_initialization(surface_regionsExpressions
    =surface_regionsExpressions, surface_regionsNames=
    surface_regionsNames, tolerance_finders=1E-9)

    geometric_data = prisms.right_rectangularPrism(
    length_x, length_y, length_z, axis_vector, base_point, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data)

    tools.gmsh_finalize(geometric_data=geometric_data, file_directory=
    "tests//solids//results", file_name="micropolar_prism")

def test_rectangularPrism():

    print("###########################################################"+
    "#############\n#                        Rectangular prism test   "+
    "                     #\n#########################################"+
    "###############################")

    geometric_data = tools.gmsh_initialization()

    length_x = 10.0

    length_y = 1.0

    length_z = 2.0

    axis_vector = [1.0, 0.0, 0.0]

    base_point = [0.0, 0.0, 0.0]

    transfinite_directions = [20, 3, 4]

    bias_directions = dict()

    bias_directions["x"] = 1.1

    bias_directions["z"] = 1.0

    geometric_data = prisms.right_rectangularPrism(
    length_x, length_y, length_z, axis_vector, base_point, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, geometric_data=geometric_data)

    base_point = [0.0, 1.0, 2.0]

    transfinite_directions = [20, 3, 4]

    bias_directions["x"] = 1.3

    bias_directions["z"] = 1.0

    length_x = 10.0

    length_y = 2.0

    length_z = 1.0

    geometric_data = prisms.right_rectangularPrism(
    length_x, length_y, length_z, axis_vector, base_point, shape_spin=
    0.5*np.pi, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions, geometric_data=geometric_data)

    base_point = [0.0, 2.0, 2.0]

    geometric_data = prisms.right_rectangularPrism(
    length_x, length_y, length_z, axis_vector, base_point, shape_spin=
    0.5*np.pi, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions, geometric_data=geometric_data)

    base_point = [0.0, 2.0, 0.0]

    geometric_data = prisms.right_rectangularPrism(
    length_x, length_y, length_z, axis_vector, base_point, shape_spin=
    0.5*np.pi, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions, geometric_data=geometric_data)

    tools.gmsh_finalize(geometric_data=geometric_data, file_directory=
    "tests//solids//results", file_name="prism")

########################################################################
#                               Cylinders                              #
########################################################################

# Tests hollow cylinder

def test_sectorHollowCylinder():

    print("###########################################################"+
    "#############\n#                   Test of sector of hollow cylin"+
    "der                  #\n#########################################"+
    "###############################")

    geometric_data = tools.gmsh_initialization()

    inner_radius = 8

    outer_radius = 9

    length = 1

    axis_vector = [1.0, 0.0, 0.0]

    axis_vector1 = [-1.0, 0.0, 0.0]

    base_point = [10.0, 0.0, 0.0]

    base_point1 = [10.0, -0.5*(inner_radius+outer_radius), 0.5*(
    inner_radius+outer_radius)]

    polar_angle = (90/180)*np.pi

    transfinite_directions = [4, 3, 20]

    bias_directions = dict()

    bias_directions["x"] = -1.0

    bias_directions["z"] = -1.0

    geometric_data = cylinders.sector_hollowCylinder(inner_radius, 
    outer_radius, length, axis_vector, base_point, polar_angle, 
    shape_spin=0*np.pi, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions, geometric_data=geometric_data)

    geometric_data = cylinders.sector_hollowCylinder(inner_radius, 
    outer_radius, length, axis_vector1, base_point1, polar_angle, 
    shape_spin=0.5*np.pi, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions, geometric_data=geometric_data)

    tools.gmsh_finalize(geometric_data=geometric_data, file_directory=
    "tests//solids//results", file_name="hollow_cylinder")

# Tests cylinder inside a box

def test_cylinderInBox():

    print("###########################################################"+
    "#############\n#                  Test of whole cylinder inside a"+
    " box                 #\n#########################################"+
    "###############################")

    radius_x = 1.9

    radius_y = 1.0

    length = 5.0

    box_lengthX = 5.0

    box_lengthY = 2.5

    axis_vector = [0.0, 0.0, 1.0]

    bottom_normalVector = [0.0, 0.0, 1.0]#[1.0, 1.0, 1.0]

    top_normalVector = [0.0, 0.0, -1.0]#[-1.0, 1.0, 1.0]

    base_point = [0.0, 0.0, 0.0]

    transfinite_directions = [8, 10, 6, 5, 5]

    bias_directions = dict()

    bias_directions["x"] = 1.0

    bias_directions["y"] = 1.2

    bias_directions["z"] = 1.4

    bias_directions["cylinder radial"] = 1.5

    bias_directions["box radial"] = 1.2

    geometric_data = tools.gmsh_initialization()

    geometric_data = cylinders.cylinder_inBox(radius_x, radius_y, length, 
    box_lengthX, box_lengthY, axis_vector, bottom_normalVector, 
    top_normalVector, base_point, transfinite_directions=
    transfinite_directions, bias_directions=bias_directions, 
    geometric_data=geometric_data, shape_spin=0.0*np.pi)

    tools.gmsh_finalize(geometric_data=geometric_data, #file_directory=
    #os.getcwd()+"tests//solids//results", 
    file_name="whole_cylinder")

# Tests cylinder

def test_cylinder():

    print("###########################################################"+
    "#############\n#                        Test of whole cylinder   "+
    "                     #\n#########################################"+
    "###############################")

    radius_x = 1.9

    radius_y = 1.0

    length = 5.0

    axis_vector = [0.0, 0.0, 1.0]

    bottom_normalVector = [1.0, 1.0, 1.0]

    top_normalVector = [-1.0, 1.0, 1.0]

    base_point = [0.0, 0.0, 0.0]

    transfinite_directions = [8, 10, 6, 2]

    bias_directions = dict()

    bias_directions["x"] = 1.0

    bias_directions["y"] = 1.1

    bias_directions["z"] = 1.2

    bias_directions["radial"] = 1.3

    geometric_data = tools.gmsh_initialization()

    geometric_data = cylinders.cylinder(radius_x, radius_y, length, 
    axis_vector, bottom_normalVector, top_normalVector, base_point, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, geometric_data=geometric_data, shape_spin=0.0*np.pi)

    tools.gmsh_finalize(geometric_data=geometric_data, file_directory=
    "tests//solids//results", file_name="whole_cylinder")

########################################################################
#                              Ellipsoids                              #
########################################################################

# Tests a quarter of ellipsoid

def test_ellipsoidPrismQuarter():

    print("###########################################################"+
    "#############\n#                Test of quarter of ellipsoid with"+
    " prism               #\n#########################################"+
    "###############################")
    
    semi_ellipsoidXLength = 2.5
    
    semi_ellipsoidYLength = 1.5
    
    semi_ellipsoidZLength = 1.0 
    
    prism_lengthRatio = 0.5

    length_xBox = 4.0

    length_yBox = 2.5

    length_zBox = 2.0

    axis_vector = [1.0, -1.0, 1.0]

    base_point = [0.0, 0.0, 0.0]

    transfinite_directions = [9,8,7,6,5]

    bias_directions = dict()

    bias_directions["axial"] = 1.05

    bias_directions["circumferential"] = 1.2

    bias_directions["radial"] = 1.3

    bias_directions["radial ellipsoid"] = 1.2

    bias_directions["radial flare"] = 1.5

    geometric_data = tools.gmsh_initialization()

    geometric_data =ellipsoids.quarter_ellipsoidInsidePrism(length_xBox, 
    length_yBox, length_zBox, semi_ellipsoidXLength, 
    semi_ellipsoidYLength, semi_ellipsoidZLength, axis_vector, 
    base_point, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions, geometric_data=geometric_data,
    prism_lengthRatio=prism_lengthRatio)

    tools.gmsh_finalize(geometric_data=geometric_data, file_directory=
    "tests//solids//results", file_name="ellipsoid_inside_prism")

def test_ellipsoidShellQuarter():

    print("###########################################################"+
    "#############\n#                Test of quarter of ellipsoid with"+
    " shell               #\n#########################################"+
    "###############################")
    
    semi_ellipsoidXLength = 0.25
    
    semi_ellipsoidYLength = 0.15
    
    semi_ellipsoidZLength = 0.1 
    
    prism_lengthRatio = 0.5

    curvature = "x"

    if curvature=="y":

        inside_curvatureRadiusY = 5.0

        inside_curvatureRadiusX = 0.3

        angle_y = (10/180)*np.pi

        angle_x = (0/180)*np.pi

    if curvature=="x":

        inside_curvatureRadiusX = 5

        inside_curvatureRadiusY = 0.3

        angle_y = (0/180)*np.pi

        angle_x = (5/180)*np.pi

    length_zBox = 0.2

    axis_vector = [1.0, 0.0, 0.0]

    base_point = [0.0, 0.0, 0.0]

    transfinite_directions = [9,8,7,6,5]

    bias_directions = dict()

    bias_directions["axial"] = 1.05

    bias_directions["circumferential"] = 1.2

    bias_directions["radial"] = 1.3

    bias_directions["radial ellipsoid"] = 1.2

    bias_directions["radial flare"] = 1.5

    geometric_data = tools.gmsh_initialization()

    geometric_data = ellipsoids.quarter_ellipsoidInsideCylinder(angle_x, 
    angle_y, length_zBox, semi_ellipsoidXLength, semi_ellipsoidYLength, 
    semi_ellipsoidZLength, axis_vector, base_point, 
    inside_curvatureRadiusX, inside_curvatureRadiusY, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, geometric_data=geometric_data, shape_spin=np.pi, 
    prism_lengthRatio=prism_lengthRatio)

    base_point = [1.0, 0.0, 0.0]

    axis_vector = [0.0, 1.0, 0.0]

    geometric_data = ellipsoids.quarter_ellipsoidInsideCylinder(angle_x, 
    angle_y, length_zBox, semi_ellipsoidXLength, semi_ellipsoidYLength, 
    semi_ellipsoidZLength, axis_vector, base_point, 
    inside_curvatureRadiusX, inside_curvatureRadiusY, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, geometric_data=geometric_data, prism_lengthRatio=
    prism_lengthRatio)

    tools.gmsh_finalize(geometric_data=geometric_data, file_directory=
    "tests//solids//results", file_name="ellipsoid_inside_shell")

########################################################################
#                               Helicoids                              #
########################################################################

def test_cylindricalHelicoid():

    print("###########################################################"+
    "#############\n#                           Test of helicoid      "+
    "                     #\n#########################################"+
    "###############################")

    trajectory_length = 20.0

    radius_X = 0.25

    radius_Y = 0.25

    axis_vector = [0.0, 0.0, 1.0]

    base_pointAxis = [0.0, 0.8, 0.0]

    base_pointCrossSection = [0.0, 0.5, 0.0]

    transfinite_directions = [3,4,300]

    n_loops = 5.3

    bias_directions = dict()

    bias_directions["radial"] = 1

    bias_directions["circumferential"] = 1

    bias_directions["axial"] = 1.0

    geometric_data = tools.gmsh_initialization()

    geometric_data = helicoids.cylindrical_helicoidWithNormalFacet(
    radius_X, radius_Y, trajectory_length, n_loops, axis_vector, 
    base_pointCrossSection, base_pointAxis, geometric_data=
    geometric_data, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions)

    tools.gmsh_finalize(geometric_data=geometric_data, 
    mesh_topologicalDimension=3, file_directory="tests//solids//results",
    file_name="cylindrical_helicoid")

########################################################################
#                            Live-wire area!                           #
########################################################################

# Calls the battery of tests

battery_ofTests()