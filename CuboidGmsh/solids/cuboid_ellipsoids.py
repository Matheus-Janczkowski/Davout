# Routine to store some cuboid shapes

import numpy as np

import CuboidGmsh.source.solids.cuboid_generator as cuboid

import CuboidGmsh.source.tool_box.meshing_tools as tools

import CuboidGmsh.source.tool_box.geometric_tools as geo

# Defines a function to create an eigth of ellipsoid inside a sector of 
# cylinder. The angle_a defines the angle of curvature along the a axis,
# if the angle is zero, it means the cylinder is flat at the side normal
# to it. If the angle is zero, the curvature radius is used as length.
# Only one direction can be curved, for it is a cylinder

def quarter_ellipsoidInsideCylinder(angle_x, angle_y, 
semi_shellThickness, semi_lengthXEllipsoid, semi_lengthYEllipsoid, 
semi_lengthZEllipsoid, axis_vector, base_point, inside_curvatureRadiusX, 
inside_curvatureRadiusY, transfinite_directions=[], bias_directions=
dict(), shape_spin=0.0, geometric_data=[0, [[],[],[],[]], [[],[],[],[]], 
[[],[],[]], dict(), [], dict(), [], [], [], 0.5, False], 
prism_lengthRatio=0.5):
    
    # Verifies if just one angle is different than zero

    if angle_x!=0.0 and angle_y!=0.0:

        raise ValueError("Just one angle of curvature can be different"+
        " than zero. Choose one.\n")
    
    ####################################################################
    #                 Transfinite directions reordering                #
    ####################################################################

    # This solid has more transfinite directions, namely the x, y, z, 
    # and the radial direction inside the ellipsoid and the radial di-
    # rection inside the flared boxes. These directions must be split
    # into their classifications

    set_transfiniteVariables, set_biasVariables = tools.retrieve_transfiniteAndBiasData(
    transfinite_directions, ["axial", "circumferential", "radial",
    "radial ellipsoid", "radial flare"], bias_directions=bias_directions)

    (transfinite_axial, transfinite_circumferential, transfinite_radial,
    transfinite_radialEllipsoid, transfinite_radialFlare) = set_transfiniteVariables

    (bias_axial, bias_circumferential, bias_radial,bias_radialEllipsoid,
    bias_radialFlare) = set_biasVariables

    ####################################################################
    #                             Rotation                             #
    ####################################################################

    # Sets the native axis of this shape

    native_axis = [1.0, 0.0, 0.0]

    rotation_vector = geo.find_rotationToNewAxis(axis_vector, 
    native_axis, shape_spin)

    ####################################################################
    #                            Inner prism                           #
    ####################################################################

    ## Upper
    
    # Creates the 8 corners as if the long axis is the Z axis proper, 
    # and the prism starts at the XY plane towards positive Z

    prism_lengthX = prism_lengthRatio*semi_lengthXEllipsoid

    prism_lengthY = prism_lengthRatio*semi_lengthYEllipsoid

    prism_lengthZ = prism_lengthRatio*semi_lengthZEllipsoid

    corner_points = [[prism_lengthX, prism_lengthX, 0.0, 0.0, 
    prism_lengthX, prism_lengthX, 0.0, 0.0], [0.0, prism_lengthY, 
    prism_lengthY, 0.0, 0.0, prism_lengthY, prism_lengthY, 0.0], [0.0, 
    0.0, 0.0, 0.0, prism_lengthZ, prism_lengthZ, prism_lengthZ, 
    prism_lengthZ]]

    # Makes the shape

    transfinite_directions = [transfinite_axial, 
    transfinite_circumferential, transfinite_radial]
    
    bias_directions = dict()

    bias_directions["x"] = bias_axial

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, geometric_data=geometric_data)

    ## Lower
    
    # Creates the 8 corners as if the long axis is the Z axis proper, 
    # and the prism starts at the XY plane towards positive Z

    prism_lengthX = prism_lengthRatio*semi_lengthXEllipsoid

    prism_lengthY = prism_lengthRatio*semi_lengthYEllipsoid

    prism_lengthZ = prism_lengthRatio*semi_lengthZEllipsoid

    corner_points = [[prism_lengthX, prism_lengthX, 0.0, 0.0, 
    prism_lengthX, prism_lengthX, 0.0, 0.0], [0.0, prism_lengthY, 
    prism_lengthY, 0.0, 0.0, prism_lengthY, prism_lengthY, 0.0], [
    -prism_lengthZ, -prism_lengthZ, -prism_lengthZ, -prism_lengthZ, 0.0,
    0.0, 0.0, 0.0]]

    # Makes the shape

    bias_directions["z"] = -bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, geometric_data=geometric_data)

    ####################################################################
    #                    Lateral ellipsoid sections                    #
    ####################################################################

    # Calculates the ellipsoid lengths from the given semi lengths

    ellipsoid_xLength = 2*semi_lengthXEllipsoid

    ellipsoid_yLength = 2*semi_lengthYEllipsoid

    ellipsoid_zLength = 2*semi_lengthZEllipsoid

    # Sets the center points

    center_point = [0.0, 0.0, 0.0]

    # Asses the angle of the vertice of the prism with the origin on the
    # YZ plane, and on the XY plane

    angle_yz = np.arctan(ellipsoid_yLength/ellipsoid_zLength)

    angle_xy = np.arctan(ellipsoid_yLength/ellipsoid_xLength)

    angle_xz = np.arctan(ellipsoid_zLength/ellipsoid_xLength)

    # Calculates the radius of the plane that divides the ellipsoid a-
    # long the x axis

    r_x = ((0.5*ellipsoid_yLength*ellipsoid_zLength)/np.sqrt(((
    ellipsoid_yLength*np.cos(angle_yz))**2)+((ellipsoid_zLength*
    np.sin(angle_yz))**2)))

    # Calculates the radius of the plane that divides the ellipoid a-
    # long the y axis (pyramid flare)

    r_y = ((0.5*ellipsoid_xLength*ellipsoid_zLength)/np.sqrt(((
    ellipsoid_zLength*np.cos(angle_xz))**2)+((ellipsoid_xLength*np.sin(
    angle_xz))**2)))

    # Calculates the radius of the plane that divides the ellipoid a-
    # long the Z axis (pyramid flare)

    r_z = ((0.5*ellipsoid_yLength*ellipsoid_xLength)/np.sqrt(((
    ellipsoid_yLength*np.cos(angle_xy))**2)+((ellipsoid_xLength*
    np.sin(angle_xy))**2)))

    # Calculates the coordinates of the point on the edge of the ellip-
    # soid pyramid flare

    flare_edgeX = 0.5*(ellipsoid_xLength/np.sqrt(3))

    flare_edgeY = (ellipsoid_yLength/ellipsoid_xLength)*flare_edgeX

    flare_edgeZ = (ellipsoid_zLength/ellipsoid_xLength)*flare_edgeX

    ## Second section (lower)

    corner_points = [[prism_lengthX, r_z*np.cos(angle_xy), 0.0, 0.0, 
    prism_lengthX, flare_edgeX, 0.0, 0.0], [prism_lengthY, r_z*np.sin(
    angle_xy), semi_lengthYEllipsoid, prism_lengthY, prism_lengthY, 
    flare_edgeY, r_x*np.sin(angle_yz), prism_lengthY], [0.0, 0.0, 0.0, 
    0.0, prism_lengthZ, flare_edgeZ, r_x*np.cos(angle_yz), prism_lengthZ
    ]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 2

    lines_instructions[2] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], semi_lengthYEllipsoid], [center_point[2], 0.0]]]

    # Elliptic arc at line 6

    lines_instructions[6] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], r_x*np.cos(angle_yz)], [center_point[2], (r_x*
    np.sin(angle_yz))]]]

    # Elliptic arc at line 10

    lines_instructions[10] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Elliptic arc at line 11

    lines_instructions[11] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], semi_lengthYEllipsoid], [center_point[2], 0.0]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "surface filling", 
    "surface filling", "surface filling", "surface filling", 
    "surface filling"]

    # Makes the shape

    transfinite_directions = [transfinite_axial,  
    transfinite_radialEllipsoid, transfinite_radial]

    bias_directions["x"] = bias_axial

    bias_directions["y"] = bias_radialEllipsoid

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions,
    geometric_data=geometric_data)

    ## Lower second section (lower)

    corner_points = [[prism_lengthX, flare_edgeX, 0.0, 0.0, 
    prism_lengthX, r_z*np.cos(angle_xy), 0.0, 0.0], [prism_lengthY, 
    flare_edgeY, r_x*np.sin(angle_yz), prism_lengthY, prism_lengthY, 
    r_z*np.sin(angle_xy), semi_lengthYEllipsoid, prism_lengthY], [
    -prism_lengthZ, -flare_edgeZ, -r_x*np.cos(angle_yz), -prism_lengthZ,
    0.0, 0.0, 0.0, 0.0]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 2

    lines_instructions[2] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], r_x*np.cos(angle_yz)], [center_point[2], (-r_x*
    np.sin(angle_yz))]]]

    # Elliptic arc at line 6

    lines_instructions[6] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Elliptic arc at line 10

    lines_instructions[10] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Elliptic arc at line 11

    lines_instructions[11] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], semi_lengthYEllipsoid], [center_point[2], 0.0]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "surface filling", 
    "surface filling", "surface filling", "surface filling", 
    "surface filling"]

    # Makes the shape

    bias_directions["z"] = -bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions,
    geometric_data=geometric_data)

    ## Third section (pyramid flare)

    corner_points = [[prism_lengthX, prism_lengthX, 
    semi_lengthXEllipsoid, r_z*np.cos(angle_xy), prism_lengthX, 
    prism_lengthX, r_y*np.cos(angle_xz), flare_edgeX], [prism_lengthY,
    0.0, 0.0, r_z*np.sin(angle_xy), prism_lengthY, 0.0, 0.0, flare_edgeY
    ], [0.0, 0.0, 0.0, 0.0, prism_lengthZ, prism_lengthZ, r_y*np.sin(
    angle_xz), flare_edgeZ]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 3

    lines_instructions[3] = ["elliptic arc", [[center_point[0], 
    semi_lengthXEllipsoid], [center_point[1], 0.0], [center_point[2], 
    0.0]]]

    # Elliptic arc at line 7

    lines_instructions[7] = ["elliptic arc", [[center_point[0], (r_y*
    np.cos(angle_xz))], [center_point[1], 0.0], [center_point[2], (r_y*
    np.sin(angle_xz))]]]

    # Elliptic arc at line 11

    lines_instructions[11] = ["elliptic arc", [[center_point[0], 
    semi_lengthXEllipsoid], [center_point[1], 0.0], [center_point[2], 
    0.0]]]

    # Elliptic arc at line 12

    lines_instructions[12] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["plane surface", "surface filling", 
    "plane surface", "plane surface", "plane surface", "plane surface"]
    
    # Makes the shape

    bias_directions["x"] = -bias_radialEllipsoid

    bias_directions["y"] = -bias_circumferential

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions, 
    geometric_data=geometric_data)

    ## Lower third section (pyramid flare)

    corner_points = [[prism_lengthX, prism_lengthX, 
    semi_lengthXEllipsoid, r_z*np.cos(angle_xy), prism_lengthX, 
    prism_lengthX, r_y*np.cos(angle_xz), flare_edgeX], [prism_lengthY,
    0.0, 0.0, r_z*np.sin(angle_xy), prism_lengthY, 0.0, 0.0, flare_edgeY
    ], [0.0, 0.0, 0.0, 0.0, -prism_lengthZ, -prism_lengthZ, -r_y*np.sin(
    angle_xz), -flare_edgeZ]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 3

    lines_instructions[3] = ["elliptic arc", [[center_point[0], 
    semi_lengthXEllipsoid], [center_point[1], 0.0], [center_point[2], 
    0.0]]]

    # Elliptic arc at line 7

    lines_instructions[7] = ["elliptic arc", [[center_point[0], (r_y*
    np.cos(angle_xz))], [center_point[1], 0.0], [center_point[2], (-r_y*
    np.sin(angle_xz))]]]

    # Elliptic arc at line 11

    lines_instructions[11] = ["elliptic arc", [[center_point[0], 
    semi_lengthXEllipsoid], [center_point[1], 0.0], [center_point[2], 
    0.0]]]

    # Elliptic arc at line 12

    lines_instructions[12] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "surface filling", 
    "surface filling", "surface filling", "surface filling", 
    "surface filling"]

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions,
    geometric_data=geometric_data)

    ####################################################################
    #                     Lateral prismatic flares                     #
    ####################################################################

    # Adds the third flare, over the third ellipsoid partition

    length_xBox11 = inside_curvatureRadiusY*1.0

    length_xBox12 = inside_curvatureRadiusY*1.0

    length_yBox1 = inside_curvatureRadiusX*1.0

    length_zBox11 = 0.0

    length_zBox12 = 0.0

    length_xBox21 = inside_curvatureRadiusY*1.0

    length_xBox22 = inside_curvatureRadiusY*1.0

    length_yBox2 = inside_curvatureRadiusX*1.0

    length_zBox21 = semi_shellThickness*1.0

    length_zBox22 = semi_shellThickness*1.0

    lines_instructions = dict()

    # If there is a curvature in around the y axis

    if angle_y!=0:

        length_xBox11 = ((inside_curvatureRadiusY+semi_shellThickness)
        *np.sin(angle_y))

        length_xBox12 = length_xBox11*1.0

        length_xBox21 = ((inside_curvatureRadiusY+(2*semi_shellThickness
        ))*np.sin(angle_y))

        length_xBox22 = length_xBox21*1.0

        length_zBox11 = ((inside_curvatureRadiusY+semi_shellThickness)*(
        np.cos(angle_y)-1))

        length_zBox12 = length_zBox11*1.0

        length_zBox21 = (((inside_curvatureRadiusY+(2*
        semi_shellThickness))*np.cos(angle_y))-inside_curvatureRadiusY-
        semi_shellThickness)

        length_zBox22 = length_zBox21*1.0

    # If there is curvature around the x axis

    if angle_x!=0:

        length_yBox1 = ((inside_curvatureRadiusX+semi_shellThickness)
        *np.sin(angle_x))

        length_yBox2 = ((inside_curvatureRadiusX+(2*semi_shellThickness)
        )*np.sin(angle_x))

        length_zBox12 = ((inside_curvatureRadiusX+semi_shellThickness)*(
        np.cos(angle_x)-1))

        length_zBox22 = (((inside_curvatureRadiusX+(2*
        semi_shellThickness))*np.cos(angle_x))-inside_curvatureRadiusX-
        semi_shellThickness)

        lines_instructions[1] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusX-semi_shellThickness]]]

        lines_instructions[5] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusX-semi_shellThickness]]]

    corner_points = [[length_xBox11, length_xBox12, r_z*np.cos(angle_xy), 
    semi_lengthXEllipsoid, length_xBox21, length_xBox22, flare_edgeX, (
    r_y*np.cos(angle_xz))], [0.0, length_yBox1, r_z*np.sin(angle_xy), 
    0.0, 0.0, length_yBox2, flare_edgeY, 0.0], [length_zBox11, 
    length_zBox12, 0.0, 0.0, length_zBox21, length_zBox22, flare_edgeZ, 
    r_y*np.sin(angle_xz)]]

    # Makes the shape

    transfinite_directions = [transfinite_radialFlare, 
    transfinite_circumferential, transfinite_radial]

    bias_directions["x"] = bias_radialFlare

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, geometric_data=geometric_data)

    # Adds the lower third flare, over the third ellipsoid partition

    lines_instructions = dict()

    # If there is a curvature in around the y axis

    if angle_y!=0:

        length_xBox11 = (inside_curvatureRadiusY*np.sin(angle_y))

        length_xBox12 = length_xBox11*1.0

        length_xBox21 = ((inside_curvatureRadiusY+semi_shellThickness)*
        np.sin(angle_y))

        length_xBox22 = length_xBox21*1.0

        length_zBox11 = ((inside_curvatureRadiusY*(np.cos(angle_y)-1))-
        semi_shellThickness)

        length_zBox12 = length_zBox11*1.0

        length_zBox21 = ((inside_curvatureRadiusY+semi_shellThickness)*
        (np.cos(angle_y)-1))

        length_zBox22 = length_zBox21*1.0

    # If there is curvature around the x axis

    if angle_x!=0:

        length_yBox1 = (inside_curvatureRadiusX*np.sin(angle_x))

        length_yBox2 = ((inside_curvatureRadiusX+semi_shellThickness)
        *np.sin(angle_x))

        length_zBox12 = ((inside_curvatureRadiusX*(np.cos(angle_x)-1))-
        semi_shellThickness)

        length_zBox22 = (((inside_curvatureRadiusX+semi_shellThickness)*
        (np.cos(angle_x)-1)))

        length_zBox11 = -semi_shellThickness

        length_zBox21 = 0.0

        lines_instructions[1] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusX-semi_shellThickness]]]

        lines_instructions[5] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusX-semi_shellThickness]]]

    corner_points = [[length_xBox11, length_xBox12, flare_edgeX, (r_y*
    np.cos(angle_xz)), length_xBox21, length_xBox22, r_z*np.cos(angle_xy), 
    semi_lengthXEllipsoid], [0.0, length_yBox1, flare_edgeY, 0.0, 0.0, 
    length_yBox2, r_z*np.sin(angle_xy), 0.0], [length_zBox11, 
    length_zBox12, -flare_edgeZ, -r_y*np.sin(angle_xz), length_zBox21, 
    length_zBox22, 0.0, 0.0]]

    # Makes the shape

    bias_directions["z"] = -bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions, 
    geometric_data=geometric_data)

    # Adds the second flare, on the second ellipsoid section
    
    lines_instructions = dict()

    # If there is a curvature in around the y axis

    if angle_y!=0:

        length_xBox11 = ((inside_curvatureRadiusY+semi_shellThickness)*
        np.sin(angle_y))

        length_xBox21 = ((inside_curvatureRadiusY+(2*semi_shellThickness
        ))*np.sin(angle_y))

        length_zBox11 = (((inside_curvatureRadiusY+semi_shellThickness)*
        (np.cos(angle_y)-1)))

        length_zBox12 = 0.0

        length_zBox21 = ((inside_curvatureRadiusY+(2*semi_shellThickness
        ))*(np.cos(angle_y)-1)+semi_shellThickness)

        length_zBox22 = semi_shellThickness*1.0

        lines_instructions[3] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusY-semi_shellThickness]]]

        lines_instructions[7] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusY-semi_shellThickness]]]

    # If there is curvature around the x axis

    if angle_x!=0:

        length_yBox1 = ((inside_curvatureRadiusX+semi_shellThickness)
        *np.sin(angle_x))

        length_yBox2 = ((inside_curvatureRadiusX+(2*semi_shellThickness)
        )*np.sin(angle_x))

        length_zBox11 = (((inside_curvatureRadiusX+semi_shellThickness)*
        (np.cos(angle_x)-1)))

        length_zBox12 = length_zBox11*1.0

        length_zBox21 = ((inside_curvatureRadiusX+(2*semi_shellThickness
        ))*(np.cos(angle_x)-1)+semi_shellThickness)

        length_zBox22 = length_zBox21*1.0

        lines_instructions[4] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusX-semi_shellThickness]]]

    corner_points = [[0.0, r_z*np.cos(angle_xy), length_xBox11, 0.0, 0.0,
    flare_edgeX, length_xBox21, 0.0], [semi_lengthYEllipsoid, r_z*np.sin(
    angle_xy), length_yBox1, length_yBox1, r_x*np.sin(angle_yz), 
    flare_edgeY, length_yBox2, length_yBox2], [0.0, 0.0, length_zBox11, 
    length_zBox12, (r_x*np.cos(angle_yz)), flare_edgeZ, length_zBox21, 
    length_zBox22]]

    # Makes the shape

    bias_directions["x"] = -bias_radialFlare

    bias_directions["y"] = bias_axial

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, surfaces_instructions=
    surfaces_instructions, geometric_data=geometric_data, 
    lines_instructionsOriginal=lines_instructions)

    # Adds the lower second flare, on the second ellipsoid section

    lines_instructions = dict()

    # If there is a curvature in around the y axis

    if angle_y!=0:

        length_xBox11 = (inside_curvatureRadiusY*np.sin(angle_y))

        length_xBox21 = ((inside_curvatureRadiusY+semi_shellThickness)*
        np.sin(angle_y))

        length_zBox11 = ((inside_curvatureRadiusY*(np.cos(angle_y)-1))-
        semi_shellThickness)

        length_zBox12 = -semi_shellThickness

        length_zBox22 = 0.0

        length_zBox21 = ((inside_curvatureRadiusY+semi_shellThickness)*(
        np.cos(angle_y)-1))

        lines_instructions[3] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusY-semi_shellThickness]]]

        lines_instructions[7] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusY-semi_shellThickness]]]

    # If there is curvature around the x axis

    if angle_x!=0:

        length_yBox1 = (inside_curvatureRadiusX*np.sin(angle_x))

        length_yBox2 = ((inside_curvatureRadiusX+semi_shellThickness)*
        np.sin(angle_x))

        length_zBox11 = ((inside_curvatureRadiusX*(np.cos(angle_x)-1))-
        semi_shellThickness)

        length_zBox12 = length_zBox11*1.0

        length_zBox22 = ((inside_curvatureRadiusX+semi_shellThickness)*(
        np.cos(angle_x)-1))

        length_zBox21 = length_zBox22*1.0

    corner_points = [[0.0, flare_edgeX, length_xBox11, 0.0, 0.0, r_z*
    np.cos(angle_xy), length_xBox21, 0.0], [r_x*np.sin(angle_yz), 
    flare_edgeY, length_yBox1, length_yBox1, semi_lengthYEllipsoid, r_z*
    np.sin(angle_xy), length_yBox2, length_yBox2], [-r_x*np.cos(angle_yz
    ), -flare_edgeZ, length_zBox11, length_zBox12, 0.0, 0.0, 
    length_zBox21, length_zBox22]]

    # Makes the shape

    bias_directions["x"] = -bias_radialFlare

    bias_directions["y"] = bias_axial

    bias_directions["z"] = -bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, surfaces_instructions=
    surfaces_instructions, lines_instructionsOriginal=lines_instructions,
    geometric_data=geometric_data)

    ####################################################################
    #                Upper and lower ellipsoid sections                #
    ####################################################################

    ## First section (upper)

    corner_points = [[prism_lengthX, prism_lengthX, 0.0, 0.0,
    r_y*np.cos(angle_xz), flare_edgeX, 0.0, 0.0], [0.0, prism_lengthY, 
    prism_lengthY, 0.0, 0.0, flare_edgeY, r_x*np.sin(angle_yz), 0.0], [
    prism_lengthZ, prism_lengthZ, prism_lengthZ, prism_lengthZ, (r_y*
    np.sin(angle_xz)), flare_edgeZ, r_x*np.cos(angle_yz), 
    semi_lengthZEllipsoid]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 5

    lines_instructions[5] = ["elliptic arc", [[center_point[0], (r_y*
    np.cos(angle_xz))], [center_point[1], 0.0], [center_point[2], (r_y*
    np.sin(angle_xz))]]]

    # Elliptic arc at line 6

    lines_instructions[6] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], r_x*np.sin(angle_yz)], [center_point[2], (r_x*
    np.cos(angle_yz))]]]

    # Elliptic arc at line 7

    lines_instructions[7] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], 0.0], [center_point[2], semi_lengthZEllipsoid]]]

    # Elliptic arc at line 8

    lines_instructions[8] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], 0.0], [center_point[2], semi_lengthZEllipsoid]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["plane surface", "plane surface", 
    "plane surface", "plane surface", "plane surface", "surface filling"]

    # Makes the shape

    bias_directions["x"] = bias_axial

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_radialEllipsoid

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions, 
    geometric_data=geometric_data)

    ## Lower section
    
    corner_points = [[prism_lengthX, prism_lengthX, 0.0, 0.0, 
    flare_edgeX, r_y*np.cos(angle_xz), 0.0, 0.0], [prism_lengthY, 0.0, 
    0.0, prism_lengthY, flare_edgeY, 0.0, 0.0, r_x*np.sin(angle_yz)], [
    -prism_lengthZ, -prism_lengthZ, -prism_lengthZ, -prism_lengthZ, 
    -flare_edgeZ, -r_y*np.sin(angle_xz), -semi_lengthZEllipsoid, 
    -r_x*np.cos(angle_yz)]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 5

    lines_instructions[5] = ["elliptic arc", [[center_point[0], (r_y*
    np.cos(angle_xz))], [center_point[1], 0.0], [center_point[2], (-r_y*
    np.sin(angle_xz))]]]

    # Elliptic arc at line 6

    lines_instructions[6] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], 0.0], [center_point[2], -semi_lengthZEllipsoid]]]

    # Elliptic arc at line 7

    lines_instructions[7] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], 0.0], [center_point[2], -semi_lengthZEllipsoid]]]

    # Elliptic arc at line 8

    lines_instructions[8] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], r_x*np.sin(angle_yz)], [center_point[2], (-r_x*
    np.cos(angle_yz))]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "plane surface", 
    "plane surface", "plane surface", "plane surface", "plane surface"]
    
    # Makes the shape

    bias_directions["y"] = -bias_circumferential

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions,
    geometric_data=geometric_data)

    ####################################################################
    #                 Upper and lower prismatic flares                 #
    ####################################################################

    # Adds the upper flare

    lines_instructions = dict()

    # If there is a curvature in around the y axis

    if angle_y!=0:

        length_xBox11 = ((inside_curvatureRadiusY+(2*semi_shellThickness
        ))*np.sin(angle_y))

        length_xBox12 = length_xBox11*1.0

        length_zBox11 = (((inside_curvatureRadiusY+(2*semi_shellThickness
        ))*(np.cos(angle_y)-1))+semi_shellThickness)

        length_zBox12 = length_zBox11*1.0

        length_zBox22 = semi_shellThickness*1.0

        length_zBox21 = semi_shellThickness*1.0

        lines_instructions[8] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusY-semi_shellThickness]]]

    # If there is curvature around the x axis

    if angle_x!=0:

        length_yBox1 = ((inside_curvatureRadiusX+(2*semi_shellThickness)
        )*np.sin(angle_x))

        length_yBox2 = length_yBox1*1.0

        length_zBox11 = semi_shellThickness*1.0

        length_zBox12 = (((inside_curvatureRadiusX+(2*semi_shellThickness
        ))*(np.cos(angle_x)-1))+semi_shellThickness)

        length_zBox21 = length_zBox12*1.0

        length_zBox22 = semi_shellThickness*1.0

        lines_instructions[7] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusX-semi_shellThickness]]]
     
    corner_points = [[r_y*np.cos(angle_xz), flare_edgeX, 0.0, 0.0, 
    length_xBox11, length_xBox12, 0.0, 0.0], [0.0, flare_edgeY, r_x*
    np.sin(angle_yz), 0.0, 0.0, length_yBox1, length_yBox2, 0.0], [r_y*
    np.sin(angle_xz), flare_edgeZ, r_x*np.cos(angle_yz), 
    semi_lengthZEllipsoid, length_zBox11, length_zBox12, length_zBox21, 
    length_zBox22]]

    # Makes the shape

    bias_directions["x"] = bias_axial

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_radialFlare

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, geometric_data=geometric_data,
    lines_instructionsOriginal=lines_instructions)

    # Adds the lower flare

    lines_instructions = dict()

    # If there is a curvature in around the y axis

    if angle_y!=0:

        length_xBox11 = (inside_curvatureRadiusY*np.sin(angle_y))

        length_xBox12 = length_xBox11*1.0

        length_zBox11 = ((inside_curvatureRadiusY*(np.cos(angle_y)-1))-
        semi_shellThickness)

        length_zBox12 = length_zBox11*1.0

        length_zBox22 = semi_shellThickness*-1.0

        length_zBox21 = semi_shellThickness*-1.0

        lines_instructions[8] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusY-semi_shellThickness]]]

    # If there is curvature around the x axis

    if angle_x!=0:

        length_yBox1 = (inside_curvatureRadiusX*np.sin(angle_x))

        length_yBox2 = length_yBox1*1.0

        length_zBox11 = semi_shellThickness*-1.0

        length_zBox12 = ((inside_curvatureRadiusX*(np.cos(angle_x)-1))-
        semi_shellThickness)

        length_zBox21 = length_zBox12*1.0

        length_zBox22 = semi_shellThickness*-1.0

        lines_instructions[7] = ["circle arc", [[0.0], [0.0], [
        -inside_curvatureRadiusX-semi_shellThickness]]]
     
    corner_points = [[r_y*np.cos(angle_xz), flare_edgeX, 0.0, 0.0, 
    length_xBox11, length_xBox12, 0.0, 0.0], [0.0, flare_edgeY, r_x*
    np.sin(angle_yz), 0.0, 0.0, length_yBox1, length_yBox2, 0.0], [-r_y*
    np.sin(angle_xz), -flare_edgeZ, -r_x*np.cos(angle_yz), 
    -semi_lengthZEllipsoid, length_zBox11, length_zBox12, length_zBox21, 
    length_zBox22]]

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, surfaces_instructions=
    surfaces_instructions, geometric_data=geometric_data,
    lines_instructionsOriginal=lines_instructions)

    return geometric_data

# Defines a function to create an eigth of ellipsoid inside a rectangu-
# lar prism

def quarter_ellipsoidInsidePrism(length_xBox, length_yBox, length_zBox,
semi_lengthXEllipsoid, semi_lengthYEllipsoid, semi_lengthZEllipsoid, 
axis_vector, base_point, transfinite_directions=[], bias_directions=
dict(), shape_spin=0.0, geometric_data=[0, [[],[],[],[]], [[],[],[],[]], 
[[],[],[]], dict(), [], dict(), [], [], [], 0.5, False], 
prism_lengthRatio=0.5):
    
    ####################################################################
    #                 Transfinite directions reordering                #
    ####################################################################

    # This solid has more transfinite directions, namely the x, y, z, 
    # and the radial direction inside the ellipsoid and the radial di-
    # rection inside the flared boxes. These directions must be split
    # into their classifications

    transfinite_axial = 0

    transfinite_circumferential = 0

    transfinite_radial = 0

    transfinite_radialEllipsoid = 0

    transfinite_radialFlare = 0

    if len(transfinite_directions)>0:

        transfinite_axial = transfinite_directions[0]

        transfinite_circumferential = transfinite_directions[1]

        transfinite_radial = transfinite_directions[2]

        transfinite_radialEllipsoid = transfinite_directions[3]

        transfinite_radialFlare = transfinite_directions[4]

    # Recovers the biases

    bias_axial = 1.0

    bias_circumferential = 1.0

    bias_radial = 1.0

    bias_radialEllipsoid = 1.0

    bias_radialFlare = 1.0

    if "axial" in bias_directions.keys():

        bias_axial = bias_directions["axial"]

    if "circumferential" in bias_directions.keys():

        bias_circumferential = bias_directions["circumferential"]

    if "radial" in bias_directions.keys():

        bias_radial = bias_directions["radial"]

    if "radial ellipsoid" in bias_directions.keys():

        bias_radialEllipsoid = bias_directions["radial ellipsoid"]

    if "radial flare" in bias_directions.keys():

        bias_radialFlare = bias_directions["radial flare"]

    ####################################################################
    #                             Rotation                             #
    ####################################################################

    # Sets the native axis of this shape

    native_axis = [1.0, 0.0, 0.0]

    rotation_vector = geo.find_rotationToNewAxis(axis_vector, 
    native_axis, shape_spin)

    ####################################################################
    #                            Inner prism                           #
    ####################################################################

    ## Upper
    
    # Creates the 8 corners as if the long axis is the Z axis proper, 
    # and the prism starts at the XY plane towards positive Z

    prism_lengthX = prism_lengthRatio*semi_lengthXEllipsoid

    prism_lengthY = prism_lengthRatio*semi_lengthYEllipsoid

    prism_lengthZ = prism_lengthRatio*semi_lengthZEllipsoid

    corner_points = [[prism_lengthX, prism_lengthX, 0.0, 0.0, 
    prism_lengthX, prism_lengthX, 0.0, 0.0], [0.0, prism_lengthY, 
    prism_lengthY, 0.0, 0.0, prism_lengthY, prism_lengthY, 0.0], [0.0, 
    0.0, 0.0, 0.0, prism_lengthZ, prism_lengthZ, prism_lengthZ, 
    prism_lengthZ]]

    # Makes the shape

    transfinite_directions = [transfinite_axial, 
    transfinite_circumferential, transfinite_radial]
    
    bias_directions = dict()

    bias_directions["x"] = bias_axial

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, geometric_data=geometric_data)

    ## Lower
    
    # Creates the 8 corners as if the long axis is the Z axis proper, 
    # and the prism starts at the XY plane towards positive Z

    prism_lengthX = prism_lengthRatio*semi_lengthXEllipsoid

    prism_lengthY = prism_lengthRatio*semi_lengthYEllipsoid

    prism_lengthZ = prism_lengthRatio*semi_lengthZEllipsoid

    corner_points = [[prism_lengthX, prism_lengthX, 0.0, 0.0, 
    prism_lengthX, prism_lengthX, 0.0, 0.0], [0.0, prism_lengthY, 
    prism_lengthY, 0.0, 0.0, prism_lengthY, prism_lengthY, 0.0], [
    -prism_lengthZ, -prism_lengthZ, -prism_lengthZ, -prism_lengthZ, 0.0,
    0.0, 0.0, 0.0]]

    # Makes the shape

    bias_directions["z"] = -bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, geometric_data=geometric_data)

    ####################################################################
    #                    Lateral ellipsoid sections                    #
    ####################################################################

    # Calculates the ellipsoid lengths from the given semi lengths

    ellipsoid_xLength = 2*semi_lengthXEllipsoid

    ellipsoid_yLength = 2*semi_lengthYEllipsoid

    ellipsoid_zLength = 2*semi_lengthZEllipsoid

    # Sets the center points

    center_point = [0.0, 0.0, 0.0]

    # Asses the angle of the vertice of the prism with the origin on the
    # YZ plane, and on the XY plane

    angle_yz = np.arctan(ellipsoid_yLength/ellipsoid_zLength)

    angle_xy = np.arctan(ellipsoid_yLength/ellipsoid_xLength)

    angle_xz = np.arctan(ellipsoid_zLength/ellipsoid_xLength)

    # Calculates the radius of the plane that divides the ellipsoid a-
    # long the x axis

    r_x = ((0.5*ellipsoid_yLength*ellipsoid_zLength)/np.sqrt(((
    ellipsoid_yLength*np.cos(angle_yz))**2)+((ellipsoid_zLength*
    np.sin(angle_yz))**2)))

    # Calculates the radius of the plane that divides the ellipoid a-
    # long the y axis (pyramid flare)

    r_y = ((0.5*ellipsoid_xLength*ellipsoid_zLength)/np.sqrt(((
    ellipsoid_zLength*np.cos(angle_xz))**2)+((ellipsoid_xLength*np.sin(
    angle_xz))**2)))

    # Calculates the radius of the plane that divides the ellipoid a-
    # long the Z axis (pyramid flare)

    r_z = ((0.5*ellipsoid_yLength*ellipsoid_xLength)/np.sqrt(((
    ellipsoid_yLength*np.cos(angle_xy))**2)+((ellipsoid_xLength*
    np.sin(angle_xy))**2)))

    # Calculates the coordinates of the point on the edge of the ellip-
    # soid pyramid flare

    flare_edgeX = 0.5*(ellipsoid_xLength/np.sqrt(3))

    flare_edgeY = (ellipsoid_yLength/ellipsoid_xLength)*flare_edgeX

    flare_edgeZ = (ellipsoid_zLength/ellipsoid_xLength)*flare_edgeX

    ## Second section (lower)

    corner_points = [[prism_lengthX, r_z*np.cos(angle_xy), 0.0, 0.0, 
    prism_lengthX, flare_edgeX, 0.0, 0.0], [prism_lengthY, r_z*np.sin(
    angle_xy), semi_lengthYEllipsoid, prism_lengthY, prism_lengthY, 
    flare_edgeY, r_x*np.sin(angle_yz), prism_lengthY], [0.0, 0.0, 0.0, 
    0.0, prism_lengthZ, flare_edgeZ, r_x*np.cos(angle_yz), prism_lengthZ
    ]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 2

    lines_instructions[2] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], semi_lengthYEllipsoid], [center_point[2], 0.0]]]

    # Elliptic arc at line 6

    lines_instructions[6] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], r_x*np.cos(angle_yz)], [center_point[2], (r_x*
    np.sin(angle_yz))]]]

    # Elliptic arc at line 10

    lines_instructions[10] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Elliptic arc at line 11

    lines_instructions[11] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], semi_lengthYEllipsoid], [center_point[2], 0.0]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "surface filling", 
    "surface filling", "surface filling", "surface filling", 
    "surface filling"]

    # Makes the shape

    transfinite_directions = [transfinite_axial,  
    transfinite_radialEllipsoid, transfinite_radial]

    bias_directions["x"] = bias_axial

    bias_directions["y"] = bias_radialEllipsoid

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions,
    geometric_data=geometric_data)

    ## Lower second section (lower)

    corner_points = [[prism_lengthX, flare_edgeX, 0.0, 0.0, 
    prism_lengthX, r_z*np.cos(angle_xy), 0.0, 0.0], [prism_lengthY, 
    flare_edgeY, r_x*np.sin(angle_yz), prism_lengthY, prism_lengthY, 
    r_z*np.sin(angle_xy), semi_lengthYEllipsoid, prism_lengthY], [
    -prism_lengthZ, -flare_edgeZ, -r_x*np.cos(angle_yz), -prism_lengthZ,
    0.0, 0.0, 0.0, 0.0]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 2

    lines_instructions[2] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], r_x*np.cos(angle_yz)], [center_point[2], (-r_x*
    np.sin(angle_yz))]]]

    # Elliptic arc at line 6

    lines_instructions[6] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Elliptic arc at line 10

    lines_instructions[10] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Elliptic arc at line 11

    lines_instructions[11] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], semi_lengthYEllipsoid], [center_point[2], 0.0]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "surface filling", 
    "surface filling", "surface filling", "surface filling", 
    "surface filling"]

    # Makes the shape

    bias_directions["z"] = -bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions,
    geometric_data=geometric_data)

    ## Third section (pyramid flare)

    corner_points = [[prism_lengthX, prism_lengthX, 
    semi_lengthXEllipsoid, r_z*np.cos(angle_xy), prism_lengthX, 
    prism_lengthX, r_y*np.cos(angle_xz), flare_edgeX], [prism_lengthY,
    0.0, 0.0, r_z*np.sin(angle_xy), prism_lengthY, 0.0, 0.0, flare_edgeY
    ], [0.0, 0.0, 0.0, 0.0, prism_lengthZ, prism_lengthZ, r_y*np.sin(
    angle_xz), flare_edgeZ]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 3

    lines_instructions[3] = ["elliptic arc", [[center_point[0], 
    semi_lengthXEllipsoid], [center_point[1], 0.0], [center_point[2], 
    0.0]]]

    # Elliptic arc at line 7

    lines_instructions[7] = ["elliptic arc", [[center_point[0], (r_y*
    np.cos(angle_xz))], [center_point[1], 0.0], [center_point[2], (r_y*
    np.sin(angle_xz))]]]

    # Elliptic arc at line 11

    lines_instructions[11] = ["elliptic arc", [[center_point[0], 
    semi_lengthXEllipsoid], [center_point[1], 0.0], [center_point[2], 
    0.0]]]

    # Elliptic arc at line 12

    lines_instructions[12] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["plane surface", "surface filling", 
    "plane surface", "plane surface", "plane surface", "plane surface"]
    
    # Makes the shape

    bias_directions["x"] = -bias_radialEllipsoid

    bias_directions["y"] = -bias_circumferential

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions, 
    geometric_data=geometric_data)

    ## Lower third section (pyramid flare)

    corner_points = [[prism_lengthX, prism_lengthX, 
    semi_lengthXEllipsoid, r_z*np.cos(angle_xy), prism_lengthX, 
    prism_lengthX, r_y*np.cos(angle_xz), flare_edgeX], [prism_lengthY,
    0.0, 0.0, r_z*np.sin(angle_xy), prism_lengthY, 0.0, 0.0, flare_edgeY
    ], [0.0, 0.0, 0.0, 0.0, -prism_lengthZ, -prism_lengthZ, -r_y*np.sin(
    angle_xz), -flare_edgeZ]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 3

    lines_instructions[3] = ["elliptic arc", [[center_point[0], 
    semi_lengthXEllipsoid], [center_point[1], 0.0], [center_point[2], 
    0.0]]]

    # Elliptic arc at line 7

    lines_instructions[7] = ["elliptic arc", [[center_point[0], (r_y*
    np.cos(angle_xz))], [center_point[1], 0.0], [center_point[2], (-r_y*
    np.sin(angle_xz))]]]

    # Elliptic arc at line 11

    lines_instructions[11] = ["elliptic arc", [[center_point[0], 
    semi_lengthXEllipsoid], [center_point[1], 0.0], [center_point[2], 
    0.0]]]

    # Elliptic arc at line 12

    lines_instructions[12] = ["elliptic arc", [[center_point[0], (r_z*
    np.cos(angle_xy))], [center_point[1], r_z*np.sin(angle_xy)], [
    center_point[2], 0.0]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "surface filling", 
    "surface filling", "surface filling", "surface filling", 
    "surface filling"]

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions,
    geometric_data=geometric_data)

    ####################################################################
    #                     Lateral prismatic flares                     #
    ####################################################################

    # Adds the third flare, over the third ellipsoid partition

    corner_points = [[length_xBox, length_xBox, r_z*np.cos(angle_xy), 
    semi_lengthXEllipsoid, length_xBox, length_xBox, flare_edgeX, (r_y*
    np.cos(angle_xz))], [0.0, length_yBox, r_z*np.sin(angle_xy), 0.0, 
    0.0, length_yBox, flare_edgeY, 0.0], [0.0, 0.0, 0.0, 0.0, 
    length_zBox, length_zBox, flare_edgeZ, r_y*np.sin(angle_xz)]]

    # Makes the shape

    transfinite_directions = [transfinite_radialFlare, 
    transfinite_circumferential, transfinite_radial]

    bias_directions["x"] = bias_radialFlare

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, surfaces_instructions=
    surfaces_instructions, geometric_data=geometric_data)

    # Adds the lower third flare, over the third ellipsoid partition

    corner_points = [[length_xBox, length_xBox, flare_edgeX, r_y*np.cos(
    angle_xz), length_xBox, length_xBox, r_z*np.cos(angle_xy), 
    semi_lengthXEllipsoid], [0.0, length_yBox, flare_edgeY, 0.0, 0.0,
    length_yBox, r_z*np.sin(angle_xy), 0.0], [-length_zBox, -length_zBox, 
    -flare_edgeZ, -r_y*np.sin(angle_xz), 0.0, 0.0, 0.0, 0.0]]

    # Sets the line instructions

    lines_instructions = dict()

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "plane surface", 
    "plane surface", "surface filling", "plane surface", "plane surface"]

    # Makes the shape

    bias_directions["z"] = -bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions, 
    geometric_data=geometric_data)

    # Adds the second flare, on the second ellipsoid section

    corner_points = [[0.0, r_z*np.cos(angle_xy), length_xBox, 0.0, 0.0,
    flare_edgeX, length_xBox, 0.0], [semi_lengthYEllipsoid, r_z*np.sin(
    angle_xy), length_yBox, length_yBox, r_x*np.sin(angle_yz), 
    flare_edgeY, length_yBox, length_yBox], [0.0, 0.0, 0.0, 0.0, (r_x*
    np.cos(angle_yz)), flare_edgeZ, length_zBox, length_zBox]]

    # Sets the surfaces instructions

    surfaces_instructions = ["plane surface", "surface filling", 
    "plane surface", "plane surface", "plane surface", "surface filling"]

    # Makes the shape

    bias_directions["x"] = -bias_radialFlare

    bias_directions["y"] = bias_axial

    bias_directions["z"] = bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, surfaces_instructions=
    surfaces_instructions, geometric_data=geometric_data)

    # Adds the lower second flare, on the second ellipsoid section

    corner_points = [[0.0, flare_edgeX, length_xBox, 0.0, 0.0, r_z*
    np.cos(angle_xy), length_xBox, 0.0], [r_x*np.sin(angle_yz), 
    flare_edgeY, length_yBox, length_yBox, semi_lengthYEllipsoid, r_z*
    np.sin(angle_xy), length_yBox, length_yBox], [-r_x*np.cos(angle_yz),
    -flare_edgeZ, -length_zBox, -length_zBox, 0.0, 0.0, 0.0, 0.0]]

    # Sets the surfaces instructions

    surfaces_instructions = ["plane surface", "plane surface", 
    "surface filling", "plane surface", "plane surface", "plane surface"]

    # Makes the shape

    bias_directions["x"] = -bias_radialFlare

    bias_directions["y"] = bias_axial

    bias_directions["z"] = -bias_radial

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, surfaces_instructions=
    surfaces_instructions, geometric_data=geometric_data)

    ####################################################################
    #                Upper and lower ellipsoid sections                #
    ####################################################################

    ## First section (upper)

    corner_points = [[prism_lengthX, prism_lengthX, 0.0, 0.0,
    r_y*np.cos(angle_xz), flare_edgeX, 0.0, 0.0], [0.0, prism_lengthY, 
    prism_lengthY, 0.0, 0.0, flare_edgeY, r_x*np.sin(angle_yz), 0.0], [
    prism_lengthZ, prism_lengthZ, prism_lengthZ, prism_lengthZ, (r_y*
    np.sin(angle_xz)), flare_edgeZ, r_x*np.cos(angle_yz), 
    semi_lengthZEllipsoid]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 5

    lines_instructions[5] = ["elliptic arc", [[center_point[0], (r_y*
    np.cos(angle_xz))], [center_point[1], 0.0], [center_point[2], (r_y*
    np.sin(angle_xz))]]]

    # Elliptic arc at line 6

    lines_instructions[6] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], r_x*np.sin(angle_yz)], [center_point[2], (r_x*
    np.cos(angle_yz))]]]

    # Elliptic arc at line 7

    lines_instructions[7] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], 0.0], [center_point[2], semi_lengthZEllipsoid]]]

    # Elliptic arc at line 8

    lines_instructions[8] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], 0.0], [center_point[2], semi_lengthZEllipsoid]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["plane surface", "plane surface", 
    "plane surface", "plane surface", "plane surface", "surface filling"]

    # Makes the shape

    bias_directions["x"] = bias_axial

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_radialEllipsoid

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions, 
    geometric_data=geometric_data)

    ## Lower section
    
    corner_points = [[prism_lengthX, prism_lengthX, 0.0, 0.0, 
    flare_edgeX, r_y*np.cos(angle_xz), 0.0, 0.0], [prism_lengthY, 0.0, 
    0.0, prism_lengthY, flare_edgeY, 0.0, 0.0, r_x*np.sin(angle_yz)], [
    -prism_lengthZ, -prism_lengthZ, -prism_lengthZ, -prism_lengthZ, 
    -flare_edgeZ, -r_y*np.sin(angle_xz), -semi_lengthZEllipsoid, 
    -r_x*np.cos(angle_yz)]]

    # Sets the line instructions

    lines_instructions = dict()

    # Elliptic arc at line 5

    lines_instructions[5] = ["elliptic arc", [[center_point[0], (r_y*
    np.cos(angle_xz))], [center_point[1], 0.0], [center_point[2], (-r_y*
    np.sin(angle_xz))]]]

    # Elliptic arc at line 6

    lines_instructions[6] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], 0.0], [center_point[2], -semi_lengthZEllipsoid]]]

    # Elliptic arc at line 7

    lines_instructions[7] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], 0.0], [center_point[2], -semi_lengthZEllipsoid]]]

    # Elliptic arc at line 8

    lines_instructions[8] = ["elliptic arc", [[center_point[0], 0.0], [
    center_point[1], r_x*np.sin(angle_yz)], [center_point[2], (-r_x*
    np.cos(angle_yz))]]]

    # Sets the surfaces instructions

    surfaces_instructions = ["surface filling", "plane surface", 
    "plane surface", "plane surface", "plane surface", "plane surface"]
    
    # Makes the shape

    bias_directions["y"] = -bias_circumferential

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, lines_instructionsOriginal=
    lines_instructions, surfaces_instructions=surfaces_instructions,
    geometric_data=geometric_data)

    ####################################################################
    #                 Upper and lower prismatic flares                 #
    ####################################################################

    # Adds the upper flare
     
    corner_points = [[r_y*np.cos(angle_xz), flare_edgeX, 0.0, 0.0, 
    length_xBox, length_xBox, 0.0, 0.0], [0.0, flare_edgeY, r_x*np.sin(
    angle_yz), 0.0, 0.0, length_yBox, length_yBox, 0.0], [r_y*np.sin(
    angle_xz), flare_edgeZ, r_x*np.cos(angle_yz), semi_lengthZEllipsoid,
    length_zBox, length_zBox, length_zBox, length_zBox]]

    # Sets the surfaces instructions

    surfaces_instructions = ["plane surface", "plane surface", 
    "plane surface", "plane surface", "plane surface", 
    "surface filling"]

    # Makes the shape

    bias_directions["x"] = bias_axial

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_radialFlare

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, surfaces_instructions=
    surfaces_instructions, geometric_data=geometric_data)

    # Adds the lower flare
     
    corner_points = [[r_y*np.cos(angle_xz), flare_edgeX, 0.0, 0.0, 
    length_xBox, length_xBox, 0.0, 0.0], [0.0, flare_edgeY, r_x*np.sin(
    angle_yz), 0.0, 0.0, length_yBox, length_yBox, 0.0], [-r_y*np.sin(
    angle_xz), -flare_edgeZ, -r_x*np.cos(angle_yz), -semi_lengthZEllipsoid,
    -length_zBox, -length_zBox, -length_zBox, -length_zBox]]

    # Sets the surfaces instructions

    surfaces_instructions = ["plane surface", "plane surface", 
    "plane surface", "plane surface", "plane surface", 
    "surface filling"]

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, 
    translation_vector=base_point, surfaces_instructions=
    surfaces_instructions, geometric_data=geometric_data)

    return geometric_data