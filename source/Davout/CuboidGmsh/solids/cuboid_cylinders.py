# Routine to store cylindrical shapes

import numpy as np

from ..solids import cuboid_generator as cuboid

from ..tool_box import meshing_tools as tools

from ..tool_box import geometric_tools as geo

from ..tool_box import verification_tools as verification

# Defines a function to create a cylinder with no hole, but it can be 
# cut in the edges not perpendicular to its axis

def cylinder(radius_X, radius_Y, length, axis_vector, 
bottom_normalVector, top_normalVector, base_point, shape_spin=0.0,
transfinite_directions=[], bias_directions=dict(), geometric_data=[0, [[
],[],[],[]], [[],[],[],[]], [[],[],[]], dict(), [], dict(), [], [], [], 
0.5, False], prism_ratio=0.3, verbose=False):
    
    ####################################################################
    #                 Transfinite directions reordering                #
    ####################################################################

    # This solid has more transfinite directions, namely the x, y, z, 
    # and the radial direction inside the flared boxes. These directions
    # must be split into their classifications

    set_transfiniteVariables, set_biasVariables = tools.retrieve_transfiniteAndBiasData(
    transfinite_directions, ["x", "y", "z", "radial"], bias_directions=
    bias_directions)

    (transfinite_x, transfinite_y, transfinite_z, transfinite_radial
    ) = set_transfiniteVariables

    (bias_x, bias_y, bias_z, bias_radial) = set_biasVariables
    
    ####################################################################
    #                             Rotation                             #
    ####################################################################

    # Sets the native axis of the cylinder as along Z axis

    native_axisCylinder = [0.0, 0.0, 1.0]

    rotation_vector = geo.find_rotationToNewAxis(axis_vector, 
    native_axisCylinder, shape_spin)

    ####################################################################
    #                    Lower plane facets points                     #
    ####################################################################
    
    # Sets the points of the lower face of the cuboid

    semi_lengthX = radius_X*prism_ratio

    semi_lengthY = radius_Y*prism_ratio

    lower_squareCoordinates = [[semi_lengthX, semi_lengthX, 
    -semi_lengthX, -semi_lengthX], [-semi_lengthY, semi_lengthY, 
    semi_lengthY, -semi_lengthY], [0.0, 0.0, 0.0, 0.0]]

    # Sets the points of the lower face of the cylinder

    angle = np.arctan(radius_Y/radius_X)

    radius = ((radius_X*radius_Y)/np.sqrt(((radius_Y*np.cos(angle))**2)+
    ((radius_X*np.sin(angle))**2)))

    radial_x = radius*np.cos(angle)

    radial_y = radius*np.sin(angle)

    lower_cylinderCoordinates = [[radial_x, radial_x, -radial_x, 
    -radial_x], [-radial_y, radial_y, radial_y, -radial_y], [0.0, 0.0, 
    0.0, 0.0]]

    # Rotates these coordinates to the axis of the cylinder, but does 
    # not translate them

    lower_squareCoordinates = geo.rotate_translateList(
    lower_squareCoordinates, rotation_vector, [0.0, 0.0, 0.0])

    lower_cylinderCoordinates = geo.rotate_translateList(
    lower_cylinderCoordinates, rotation_vector, [0.0, 0.0, 0.0])

    # Projects these coordinates onto the plane of the lower facet

    lower_squareCoordinates = geo.project_shadowAndTranslate(
    lower_squareCoordinates, bottom_normalVector,axis_vector,base_point)

    lower_cylinderCoordinates = geo.project_shadowAndTranslate(
    lower_cylinderCoordinates, bottom_normalVector, axis_vector,
    base_point)

    ####################################################################
    #                    Upper plane facets points                     #
    ####################################################################
    
    # Sets the points of the upper face of the cuboid

    upper_squareCoordinates = [[semi_lengthX, semi_lengthX,
    -semi_lengthX, -semi_lengthX], [-semi_lengthY, semi_lengthY, 
    semi_lengthY, -semi_lengthY], [0.0, 0.0, 0.0, 0.0]]

    # Sets the points of the upper face of the cylinder

    upper_cylinderCoordinates = [[radial_x, radial_x, -radial_x, 
    -radial_x], [-radial_y, radial_y, radial_y, -radial_y], [0.0, 0.0, 
    0.0, 0.0]]

    # Rotates these coordinates to the axis of the cylinder, but does 
    # not translate them

    upper_squareCoordinates = geo.rotate_translateList(
    upper_squareCoordinates, rotation_vector, [0.0, 0.0, 0.0])

    upper_cylinderCoordinates = geo.rotate_translateList(
    upper_cylinderCoordinates, rotation_vector, [0.0, 0.0, 0.0])

    # Projects these coordinates onto the plane of the upper facet and
    # translate them to the upper facet

    upper_translation = geo.rotate_translateList([[0.0], [0.0], [length]
    ], rotation_vector, base_point)

    upper_translation = [upper_translation[0][0], upper_translation[1][0
    ], upper_translation[2][0]]

    upper_squareCoordinates = geo.project_shadowAndTranslate(
    upper_squareCoordinates, top_normalVector, axis_vector, 
    upper_translation)

    upper_cylinderCoordinates = geo.project_shadowAndTranslate(
    upper_cylinderCoordinates, top_normalVector, axis_vector,
    upper_translation)

    ####################################################################
    #                          Internal prisms                         #
    ####################################################################

    transfinite_directions = [transfinite_x, transfinite_y, 
    transfinite_z]

    bias_directions = dict()

    bias_directions["x"] = bias_x

    bias_directions["y"] = bias_y

    bias_directions["z"] = bias_z

    # Assembles the corner points of the prisms

    combinations = [[0,1,2,3]]

    for i in range(len(combinations)):

        prism_corners = [[lower_squareCoordinates[0][combinations[i][0]], 
        lower_squareCoordinates[0][combinations[i][1]], 
        lower_squareCoordinates[0][combinations[i][2]],
        lower_squareCoordinates[0][combinations[i][3]], 
        upper_squareCoordinates[0][combinations[i][0]],
        upper_squareCoordinates[0][combinations[i][1]], 
        upper_squareCoordinates[0][combinations[i][2]],
        upper_squareCoordinates[0][combinations[i][3]]], [
        lower_squareCoordinates[1][combinations[i][0]], 
        lower_squareCoordinates[1][combinations[i][1]], 
        lower_squareCoordinates[1][combinations[i][2]],
        lower_squareCoordinates[1][combinations[i][3]], 
        upper_squareCoordinates[1][combinations[i][0]],
        upper_squareCoordinates[1][combinations[i][1]], 
        upper_squareCoordinates[1][combinations[i][2]],
        upper_squareCoordinates[1][combinations[i][3]]], [
        lower_squareCoordinates[2][combinations[i][0]], 
        lower_squareCoordinates[2][combinations[i][1]], 
        lower_squareCoordinates[2][combinations[i][2]],
        lower_squareCoordinates[2][combinations[i][3]], 
        upper_squareCoordinates[2][combinations[i][0]],
        upper_squareCoordinates[2][combinations[i][1]], 
        upper_squareCoordinates[2][combinations[i][2]],
        upper_squareCoordinates[2][combinations[i][3]]]]

        # Makes the cuboid of the prism

        geometric_data = cuboid.make_cuboid(prism_corners, 
        transfinite_directions=transfinite_directions, bias_directions=
        bias_directions, geometric_data=geometric_data, verbose=verbose)

    ####################################################################
    #                           Curved flares                          #
    ####################################################################

    # Adds the flares

    combinations = [[0,1,1,0], [1,2,2,1], [2,3,3,2], [3,0,0,3]]

    ellipse_angles = [[-angle,angle], [angle,np.pi-angle], [np.pi-angle, 
    np.pi+angle], [np.pi+angle, (2*np.pi)-angle]]

    ellipctic_transfinite = [transfinite_y, transfinite_x, transfinite_y,
    transfinite_x]

    for i in range(4):

        flare_corners = [[lower_cylinderCoordinates[0][combinations[i][0
        ]], lower_cylinderCoordinates[0][combinations[i][1]],
        lower_squareCoordinates[0][combinations[i][2]],
        lower_squareCoordinates[0][combinations[i][3]],
        upper_cylinderCoordinates[0][combinations[i][0]],
        upper_cylinderCoordinates[0][combinations[i][1]],
        upper_squareCoordinates[0][combinations[i][2]],
        upper_squareCoordinates[0][combinations[i][3]]], [
        lower_cylinderCoordinates[1][combinations[i][0]], 
        lower_cylinderCoordinates[1][combinations[i][1]],
        lower_squareCoordinates[1][combinations[i][2]],
        lower_squareCoordinates[1][combinations[i][3]],
        upper_cylinderCoordinates[1][combinations[i][0]],
        upper_cylinderCoordinates[1][combinations[i][1]],
        upper_squareCoordinates[1][combinations[i][2]],
        upper_squareCoordinates[1][combinations[i][3]]], [
        lower_cylinderCoordinates[2][combinations[i][0]], 
        lower_cylinderCoordinates[2][combinations[i][1]],
        lower_squareCoordinates[2][combinations[i][2]],
        lower_squareCoordinates[2][combinations[i][3]],
        upper_cylinderCoordinates[2][combinations[i][0]],
        upper_cylinderCoordinates[2][combinations[i][1]],
        upper_squareCoordinates[2][combinations[i][2]],
        upper_squareCoordinates[2][combinations[i][3]]]]

        # Defines points of the elliptic line

        lower_ellipseLine = geo.ellipse_shadow(ellipse_angles[i][0], 
        ellipse_angles[i][1], radius_X, radius_Y, ellipctic_transfinite[
        i], False, axis_vector, rotation_vector, bottom_normalVector,
        base_point)

        upper_ellipseLine = geo.ellipse_shadow(ellipse_angles[i][0], 
        ellipse_angles[i][1], radius_X, radius_Y, ellipctic_transfinite[
        i], False, axis_vector, rotation_vector, top_normalVector,
        upper_translation)

        # Defines the lines instructions

        lines_instructions = dict()

        lines_instructions[1] = ["spline", lower_ellipseLine, 
        "do not rotate"]

        lines_instructions[5] = ["spline", upper_ellipseLine, 
        "do not rotate"]

        geometric_data = cuboid.make_cuboid(flare_corners, 
        transfinite_directions=transfinite_directions, bias_directions=
        bias_directions, lines_instructionsOriginal=lines_instructions,
        geometric_data=geometric_data, verbose=verbose)

    return geometric_data

# Defines a function to create a cylinder with no hole inside a bounding
# box, but both can be cut in the edges not perpendicular to its axis

def cylinder_inBox(radius_X, radius_Y, length, box_lengthX, box_lengthY,
axis_vector, bottom_normalVector, top_normalVector, base_point, 
shape_spin=0.0, transfinite_directions=[], bias_directions=dict(), 
geometric_data=[0, [[],[],[],[]], [[],[],[],[]], [[],[],[]], dict(), [],
dict(), [], [], [], 0.5, False], prism_ratio=0.3, verbose=False):
    
    # Verifies whether the bounding box is larger than the cylinder it-
    # self

    if box_lengthX<=(2*radius_X):

        raise ValueError("The bounding box must be larger than the inn"+
        "er cylinder diameter, the values of radius and box length in "+
        "X are, respectively: "+str(radius_X)+", "+str(box_lengthX))

    if box_lengthY<=(2*radius_Y):

        raise ValueError("The bounding box must be larger than the inn"+
        "er cylinder diameter, the values of radius and box length in "+
        "Y are, respectively: "+str(radius_Y)+", "+str(box_lengthY))
    
    ####################################################################
    #                 Transfinite directions reordering                #
    ####################################################################

    # This solid has more transfinite directions, namely the x, y, z, 
    # and the radial direction inside the flared boxes. These directions
    # must be split into their classifications

    set_transfiniteVariables, set_biasVariables = tools.retrieve_transfiniteAndBiasData(
    transfinite_directions, ["x", "y", "z", "cylinder radial", "box ra"+
    "dial"], bias_directions=bias_directions)

    (transfinite_x, transfinite_y, transfinite_z, 
    transfinite_cylinderRadial, transfinite_boxRadial) = set_transfiniteVariables

    (bias_x, bias_y, bias_z, bias_cylinderRadial, bias_boxRadial) = set_biasVariables
    
    ####################################################################
    #                             Rotation                             #
    ####################################################################

    # Sets the native axis of the cylinder as along Z axis

    native_axisCylinder = [0.0, 0.0, 1.0]

    rotation_vector = geo.find_rotationToNewAxis(axis_vector, 
    native_axisCylinder, shape_spin)

    ####################################################################
    #                    Lower plane facets points                     #
    ####################################################################
    
    # Sets the points of the lower face of the cuboid

    semi_lengthX = radius_X*prism_ratio

    semi_lengthY = radius_Y*prism_ratio

    lower_squareCoordinates = [[semi_lengthX, semi_lengthX, 
    -semi_lengthX, -semi_lengthX], [-semi_lengthY, semi_lengthY, 
    semi_lengthY, -semi_lengthY], [0.0, 0.0, 0.0, 0.0]]

    # Sets the points of the lower face of the cylinder

    angle = np.arctan(radius_Y/radius_X)

    radius = ((radius_X*radius_Y)/np.sqrt(((radius_Y*np.cos(angle))**2)+
    ((radius_X*np.sin(angle))**2)))

    radial_x = radius*np.cos(angle)

    radial_y = radius*np.sin(angle)

    lower_cylinderCoordinates = [[radial_x, radial_x, -radial_x, 
    -radial_x], [-radial_y, radial_y, radial_y, -radial_y], [0.0, 0.0, 
    0.0, 0.0]]

    # Sets the points of the lower face of the bounding box

    lower_boxCoordinates = [[box_lengthX*0.5, box_lengthX*0.5, (-0.5*
    box_lengthX), (-0.5*box_lengthX)], [-box_lengthY*0.5, (box_lengthY*
    0.5), box_lengthY*0.5, -box_lengthY*0.5], [0.0, 0.0, 0.0, 0.0]]

    # Rotates these coordinates to the axis of the cylinder, but does 
    # not translate them

    lower_squareCoordinates = geo.rotate_translateList(
    lower_squareCoordinates, rotation_vector, [0.0, 0.0, 0.0])

    lower_cylinderCoordinates = geo.rotate_translateList(
    lower_cylinderCoordinates, rotation_vector, [0.0, 0.0, 0.0])

    lower_boxCoordinates = geo.rotate_translateList(lower_boxCoordinates, 
    rotation_vector, [0.0, 0.0, 0.0])

    # Projects these coordinates onto the plane of the lower facet

    lower_squareCoordinates = geo.project_shadowAndTranslate(
    lower_squareCoordinates, bottom_normalVector,axis_vector,base_point)

    lower_cylinderCoordinates = geo.project_shadowAndTranslate(
    lower_cylinderCoordinates, bottom_normalVector, axis_vector,
    base_point)

    lower_boxCoordinates = geo.project_shadowAndTranslate(
    lower_boxCoordinates, bottom_normalVector, axis_vector, base_point)

    ####################################################################
    #                    Upper plane facets points                     #
    ####################################################################
    
    # Sets the points of the upper face of the cuboid

    upper_squareCoordinates = [[semi_lengthX, semi_lengthX,
    -semi_lengthX, -semi_lengthX], [-semi_lengthY, semi_lengthY, 
    semi_lengthY, -semi_lengthY], [0.0, 0.0, 0.0, 0.0]]

    # Sets the points of the upper face of the cylinder

    upper_cylinderCoordinates = [[radial_x, radial_x, -radial_x, 
    -radial_x], [-radial_y, radial_y, radial_y, -radial_y], [0.0, 0.0, 
    0.0, 0.0]]

    # Sets the points of the upper face of the bounding box

    upper_boxCoordinates = [[box_lengthX*0.5, box_lengthX*0.5, (-0.5*
    box_lengthX), (-0.5*box_lengthX)], [-box_lengthY*0.5, (box_lengthY*
    0.5), box_lengthY*0.5, -box_lengthY*0.5], [0.0, 0.0, 0.0, 0.0]]

    # Rotates these coordinates to the axis of the cylinder, but does 
    # not translate them

    upper_squareCoordinates = geo.rotate_translateList(
    upper_squareCoordinates, rotation_vector, [0.0, 0.0, 0.0])

    upper_cylinderCoordinates = geo.rotate_translateList(
    upper_cylinderCoordinates, rotation_vector, [0.0, 0.0, 0.0])

    upper_boxCoordinates = geo.rotate_translateList(upper_boxCoordinates, 
    rotation_vector, [0.0, 0.0, 0.0])

    # Projects these coordinates onto the plane of the upper facet and
    # translate them to the upper facet

    upper_translation = geo.rotate_translateList([[0.0], [0.0], [length]
    ], rotation_vector, base_point)

    upper_translation = [upper_translation[0][0], upper_translation[1][0
    ], upper_translation[2][0]]

    upper_squareCoordinates = geo.project_shadowAndTranslate(
    upper_squareCoordinates, top_normalVector, axis_vector, 
    upper_translation)

    upper_cylinderCoordinates = geo.project_shadowAndTranslate(
    upper_cylinderCoordinates, top_normalVector, axis_vector,
    upper_translation)

    upper_boxCoordinates = geo.project_shadowAndTranslate(
    upper_boxCoordinates, top_normalVector, axis_vector,
    upper_translation)

    #################################################################### 
    #                           Verification                           #
    ####################################################################

    # Verifies that the bounding box is not entangled

    flag_entanglement = verification.verify_cuboidEntanglement([
    lower_boxCoordinates[0]+upper_boxCoordinates[0], 
    lower_boxCoordinates[1]+upper_boxCoordinates[1],
    lower_boxCoordinates[2]+upper_boxCoordinates[2]])

    if flag_entanglement:

        raise ValueError("The bounding box of the cylinder is entangle"+
        "d, this means\nthat the upper facet is intersecting the lower"+
        " facet. Hence, the\nbounding box must be smaller or the facet"+
        "s have too a large angle\nbetween one another.")

    ####################################################################
    #                          Internal prisms                         #
    ####################################################################

    transfinite_directions = [transfinite_x, transfinite_y, 
    transfinite_z]

    bias_directions = dict()

    bias_directions["x"] = bias_x

    bias_directions["y"] = bias_y

    bias_directions["z"] = bias_z

    # Assembles the corner points of the prisms

    combinations = [[0,1,2,3]]

    for i in range(len(combinations)):

        prism_corners = [[lower_squareCoordinates[0][combinations[i][0]], 
        lower_squareCoordinates[0][combinations[i][1]], 
        lower_squareCoordinates[0][combinations[i][2]],
        lower_squareCoordinates[0][combinations[i][3]], 
        upper_squareCoordinates[0][combinations[i][0]],
        upper_squareCoordinates[0][combinations[i][1]], 
        upper_squareCoordinates[0][combinations[i][2]],
        upper_squareCoordinates[0][combinations[i][3]]], [
        lower_squareCoordinates[1][combinations[i][0]], 
        lower_squareCoordinates[1][combinations[i][1]], 
        lower_squareCoordinates[1][combinations[i][2]],
        lower_squareCoordinates[1][combinations[i][3]], 
        upper_squareCoordinates[1][combinations[i][0]],
        upper_squareCoordinates[1][combinations[i][1]], 
        upper_squareCoordinates[1][combinations[i][2]],
        upper_squareCoordinates[1][combinations[i][3]]], [
        lower_squareCoordinates[2][combinations[i][0]], 
        lower_squareCoordinates[2][combinations[i][1]], 
        lower_squareCoordinates[2][combinations[i][2]],
        lower_squareCoordinates[2][combinations[i][3]], 
        upper_squareCoordinates[2][combinations[i][0]],
        upper_squareCoordinates[2][combinations[i][1]], 
        upper_squareCoordinates[2][combinations[i][2]],
        upper_squareCoordinates[2][combinations[i][3]]]]

        # Makes the cuboid of the prism

        geometric_data = cuboid.make_cuboid(prism_corners, 
        transfinite_directions=transfinite_directions, bias_directions=
        bias_directions, geometric_data=geometric_data, verbose=verbose)

    ####################################################################
    #                           Curved flares                          #
    ####################################################################

    # Adds the flares

    transfinite_directions = [transfinite_cylinderRadial, transfinite_x, 
    transfinite_z]

    combinations = [[0,1,1,0], [1,2,2,1], [2,3,3,2], [3,0,0,3]]

    ellipse_angles = [[-angle,angle], [angle,np.pi-angle], [np.pi-angle, 
    np.pi+angle], [np.pi+angle, (2*np.pi)-angle]]

    ellipctic_transfinite = [transfinite_y, transfinite_x, transfinite_y,
    transfinite_x]

    bias_combinations = [[bias_cylinderRadial, bias_y, bias_z], [
    bias_cylinderRadial, bias_x, bias_z], [bias_cylinderRadial, -bias_y, 
    bias_z], [bias_cylinderRadial, bias_x, bias_z]]

    for i in range(4):

        flare_corners = [[lower_cylinderCoordinates[0][combinations[i][0
        ]], lower_cylinderCoordinates[0][combinations[i][1]],
        lower_squareCoordinates[0][combinations[i][2]],
        lower_squareCoordinates[0][combinations[i][3]],
        upper_cylinderCoordinates[0][combinations[i][0]],
        upper_cylinderCoordinates[0][combinations[i][1]],
        upper_squareCoordinates[0][combinations[i][2]],
        upper_squareCoordinates[0][combinations[i][3]]], [
        lower_cylinderCoordinates[1][combinations[i][0]], 
        lower_cylinderCoordinates[1][combinations[i][1]],
        lower_squareCoordinates[1][combinations[i][2]],
        lower_squareCoordinates[1][combinations[i][3]],
        upper_cylinderCoordinates[1][combinations[i][0]],
        upper_cylinderCoordinates[1][combinations[i][1]],
        upper_squareCoordinates[1][combinations[i][2]],
        upper_squareCoordinates[1][combinations[i][3]]], [
        lower_cylinderCoordinates[2][combinations[i][0]], 
        lower_cylinderCoordinates[2][combinations[i][1]],
        lower_squareCoordinates[2][combinations[i][2]],
        lower_squareCoordinates[2][combinations[i][3]],
        upper_cylinderCoordinates[2][combinations[i][0]],
        upper_cylinderCoordinates[2][combinations[i][1]],
        upper_squareCoordinates[2][combinations[i][2]],
        upper_squareCoordinates[2][combinations[i][3]]]]

        # Defines points of the elliptic line

        lower_ellipseLine = geo.ellipse_shadow(ellipse_angles[i][0], 
        ellipse_angles[i][1], radius_X, radius_Y, ellipctic_transfinite[
        i], False, axis_vector, rotation_vector, bottom_normalVector,
        base_point)

        upper_ellipseLine = geo.ellipse_shadow(ellipse_angles[i][0], 
        ellipse_angles[i][1], radius_X, radius_Y, ellipctic_transfinite[
        i], False, axis_vector, rotation_vector, top_normalVector,
        upper_translation)

        # Defines the lines instructions

        lines_instructions = dict()

        lines_instructions[1] = ["spline", lower_ellipseLine, 
        "do not rotate"]

        lines_instructions[5] = ["spline", upper_ellipseLine, 
        "do not rotate"]

        # Sets the biases

        bias_directions = dict()
        
        bias_directions["x"] = bias_combinations[i][0]
        
        bias_directions["y"] = bias_combinations[i][1]
        
        bias_directions["z"] = bias_combinations[i][2]

        geometric_data = cuboid.make_cuboid(flare_corners, 
        transfinite_directions=transfinite_directions, bias_directions=
        bias_directions, lines_instructionsOriginal=lines_instructions,
        geometric_data=geometric_data, verbose=verbose)

    ####################################################################
    #                        Bounding box prisms                       #
    #################################################################### 
    
    # Adds the bounding box prisms

    transfinite_directions = [transfinite_boxRadial, transfinite_x, 
    transfinite_z]

    bias_directions = dict()
    
    bias_directions["x"] = bias_boxRadial
    
    bias_directions["y"] = bias_x
    
    bias_directions["z"] = bias_z

    combinations = [[0,1,1,0], [1,2,2,1], [2,3,3,2], [3,0,0,3]]

    for i in range(4):

        flare_corners = [[lower_boxCoordinates[0][combinations[i][0]],
        lower_boxCoordinates[0][combinations[i][1]], 
        lower_cylinderCoordinates[0][combinations[i][2]],
        lower_cylinderCoordinates[0][combinations[i][3]],
        upper_boxCoordinates[0][combinations[i][0]],
        upper_boxCoordinates[0][combinations[i][1]],
        upper_cylinderCoordinates[0][combinations[i][2]],
        upper_cylinderCoordinates[0][combinations[i][3]]], 
        [lower_boxCoordinates[1][combinations[i][0]],
        lower_boxCoordinates[1][combinations[i][1]], 
        lower_cylinderCoordinates[1][combinations[i][2]],
        lower_cylinderCoordinates[1][combinations[i][3]],
        upper_boxCoordinates[1][combinations[i][0]],
        upper_boxCoordinates[1][combinations[i][1]],
        upper_cylinderCoordinates[1][combinations[i][2]],
        upper_cylinderCoordinates[1][combinations[i][3]]], 
        [lower_boxCoordinates[2][combinations[i][0]],
        lower_boxCoordinates[2][combinations[i][1]], 
        lower_cylinderCoordinates[2][combinations[i][2]],
        lower_cylinderCoordinates[2][combinations[i][3]],
        upper_boxCoordinates[2][combinations[i][0]],
        upper_boxCoordinates[2][combinations[i][1]],
        upper_cylinderCoordinates[2][combinations[i][2]],
        upper_cylinderCoordinates[2][combinations[i][3]]]]

        # Makes the cuboid

        geometric_data = cuboid.make_cuboid(flare_corners, 
        transfinite_directions=transfinite_directions, bias_directions=
        bias_directions, geometric_data=geometric_data, verbose=verbose)

    return geometric_data

# Defines a function to create a sector of a hollow cylinder

def sector_hollowCylinder(inner_radius, outer_radius, length, 
axis_vector, base_point, polar_angle, transfinite_directions=[],
bias_directions=dict(), shape_spin=0.0, geometric_data=[0, [[],[],[],[]], 
[[],[],[],[]], [[],[],[]], dict(), [], dict(), [], [], [], 0.5, False],
verbose=False):
    
    # The polar angle must be less than pi

    if polar_angle>=(np.pi-1E-6):

        raise ValueError("The polar angle for the sector of a hollow c"+
        "ylinder must be less than pi")

    # Creates the 8 corners as if the axis is the X axis proper, and the
    # cylinder starts at the XY plane towards positive Z

    p5_y = np.cos(polar_angle)*inner_radius

    p5_z = np.sin(polar_angle)*inner_radius

    p6_y = np.cos(polar_angle)*outer_radius

    p6_z = np.sin(polar_angle)*outer_radius

    y_shift = (inner_radius+outer_radius)*0.5

    corner_points = [[0.5*length, 0.5*length, -0.5*length, -0.5*length, 
    0.5*length, 0.5*length, -0.5*length, -0.5*length], [inner_radius-
    y_shift, outer_radius-y_shift, outer_radius-y_shift, inner_radius-
    y_shift, p5_y-y_shift, p6_y-y_shift, p6_y-y_shift, p5_y-y_shift], [
    0.0, 0.0, 0.0, 0.0, p5_z, p6_z, p6_z, p5_z]]

    # Creates the center points for the circular lines

    back_centerPoint = [[-0.5*length], [-y_shift], [0.0]]

    front_centerPoint = [[0.5*length], [-y_shift], [0.0]]

    # Creates the line instructions

    line_instructions = dict()

    line_instructions[9] = ["circle arc", front_centerPoint]

    line_instructions[12] = ["circle arc", back_centerPoint]

    line_instructions[10] = ["circle arc", front_centerPoint]

    line_instructions[11] = ["circle arc", back_centerPoint]

    ####################################################################
    #                             Rotation                             #
    ####################################################################

    # Sets the native axis of this shape

    native_axis = [1.0, 0.0, 0.0]

    rotation_vector = geo.find_rotationToNewAxis(axis_vector, 
    native_axis, shape_spin)

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, lines_instructionsOriginal=line_instructions, 
    rotation_vector=rotation_vector, translation_vector=base_point, 
    geometric_data=geometric_data, verbose=verbose)

    return geometric_data