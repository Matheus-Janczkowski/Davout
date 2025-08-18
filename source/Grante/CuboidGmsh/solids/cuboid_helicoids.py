# Routine to store methods for helicoids

import numpy as np

from ..solids import cuboid_generator as cuboid

from ..tool_box import meshing_tools as tools

from ..tool_box import geometric_tools as geo

# Defines a function to create a helicoid with circular cross section 
# and circular trajectory. Both ends have facets with the facet's normal
# collinear to original axis

def cylindrical_helicoidWithNormalFacet(radius_X, radius_Y, 
trajectory_length, n_loops, axis_vector, base_pointCrossSection, 
base_pointAxis, shape_spin=0.0, transfinite_directions=[], 
bias_directions=dict(), geometric_data=[0, [[],[],[],[]], [[],[],[],[]], 
[[],[],[]], dict(), [], dict(), [], [], [], 0.5, False], prism_ratio=0.5):
    
    ####################################################################
    #                 Transfinite directions reordering                #
    ####################################################################

    # This solid has more transfinite directions, namely the x, y, z, 
    # and the radial direction inside the flared boxes. These directions
    # must be split into their classifications

    set_transfiniteVariables, set_biasVariables = tools.retrieve_transfiniteAndBiasData(
    transfinite_directions, ["radial", "circumferential", "axial"], 
    bias_directions=bias_directions)

    (transfinite_radial, transfinite_circumferential, transfinite_axial
    ) = set_transfiniteVariables

    (bias_radial, bias_circumferential, bias_axial) = set_biasVariables
    
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
    
    # Makes sure the axial vector is unitary

    axial_norm = np.sqrt((axis_vector[0]**2)+(axis_vector[1]**2)+(
    axis_vector[2]**2))

    axis_vector = [axis_vector[0]/axial_norm, axis_vector[1]/axial_norm,
    axis_vector[2]/axial_norm]

    # Finds the point in the central axis that is the orthogonal projec-
    # tion of the initial point into the central axis

    central_to_initialPoint = [base_pointCrossSection[0]-base_pointAxis[
    0], base_pointCrossSection[1]-base_pointAxis[1], 
    base_pointCrossSection[2]-base_pointAxis[2]]

    # Projects this vector in the center axis direction

    projection = ((central_to_initialPoint[0]*axis_vector[0])+(
    central_to_initialPoint[1]*axis_vector[1])+(central_to_initialPoint[
    2]*axis_vector[2]))

    central_to_initialPoint = [projection*axis_vector[0], (projection*
    axis_vector[1]), (projection*axis_vector[2])]

    initial_centralPoint = [(base_pointAxis[0]+
    central_to_initialPoint[0]), (base_pointAxis[1]+
    central_to_initialPoint[1]), (base_pointAxis[2]+
    central_to_initialPoint[2])]

    print("Initial central point:", initial_centralPoint, "\n")

    # Evaluates the vector from the initial central point to the initial
    # point

    radial_vector = [base_pointCrossSection[0]-initial_centralPoint[0], 
    (base_pointCrossSection[1]-initial_centralPoint[1]), (
    base_pointCrossSection[2]-initial_centralPoint[2])]

    # Gets the norm of this vector

    norm_radial = np.sqrt((radial_vector[0]**2)+(radial_vector[1]**2)+(
    radial_vector[2]**2))

    # Divides the radial vector by the norm and multiplies by the ra-
    # dius, so much so that the radial vector has norm equal to the gi-
    # ven radius

    radial_vectorLowerUnitary = [1.0, 0.0, 0.0]

    if norm_radial>0.0:

        radial_vectorLowerUnitary = [(radial_vector[0]/norm_radial), (
        radial_vector[1]/norm_radial), radial_vector[2]/norm_radial]

    # Finds a normal vector to the radial vector

    ortho_radialVectorLower = geo.rotate_translateList([[
    radial_vectorLowerUnitary[0]], [radial_vectorLowerUnitary[1]], [
    radial_vectorLowerUnitary[2]]], [axis_vector[0]*0.5*np.pi, 
    axis_vector[1]*0.5*np.pi, axis_vector[2]*0.5*np.pi], [0.0, 0.0, 0.0])

    ortho_radialVectorLower = [ortho_radialVectorLower[0][0], 
    ortho_radialVectorLower[1][0], ortho_radialVectorLower[2][0]]

    # Creates the lower section points

    lower_squarePoints = [[base_pointCrossSection[0]+(prism_ratio*
    radius_X*ortho_radialVectorLower[0])-(prism_ratio*radius_Y*
    radial_vectorLowerUnitary[0]), base_pointCrossSection[0]-(
    prism_ratio*radius_X*ortho_radialVectorLower[0])-(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[0]), base_pointCrossSection[0]-(
    prism_ratio*radius_X*ortho_radialVectorLower[0])+(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[0]), base_pointCrossSection[0]+(
    prism_ratio*radius_X*ortho_radialVectorLower[0])+(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[0])], [base_pointCrossSection[1]+
    (prism_ratio*radius_X*ortho_radialVectorLower[1])-(prism_ratio*radius_Y*
    radial_vectorLowerUnitary[1]), base_pointCrossSection[1]-(
    prism_ratio*radius_X*ortho_radialVectorLower[1])-(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[1]), base_pointCrossSection[1]-(
    prism_ratio*radius_X*ortho_radialVectorLower[1])+(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[1]), base_pointCrossSection[1]+(
    prism_ratio*radius_X*ortho_radialVectorLower[1])+(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[1])], [base_pointCrossSection[2]+
    (prism_ratio*radius_X*ortho_radialVectorLower[2])-(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[2]), base_pointCrossSection[2]-(
    prism_ratio*radius_X*ortho_radialVectorLower[2])-(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[2]), base_pointCrossSection[2]-(
    prism_ratio*radius_X*ortho_radialVectorLower[2])+(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[2]), base_pointCrossSection[2]+(
    prism_ratio*radius_X*ortho_radialVectorLower[2])+(prism_ratio*
    radius_Y*radial_vectorLowerUnitary[2])]]

    lower_cylinderPoints = [[]]

    ####################################################################
    #                       Upper section points                       #
    ####################################################################

    # Updates the radial vector to the twisted configuration

    radial_vectorUpper = geo.rotate_translateList([[radial_vector[0
    ]], [radial_vector[1]], [radial_vector[2]]], [(
    axis_vector[0]*2*np.pi*n_loops), axis_vector[1]*2*np.pi*n_loops, 
    axis_vector[2]*2*np.pi*n_loops], [0.0, 0.0, 0.0])

    radial_vectorUpper = [radial_vectorUpper[0][0], radial_vectorUpper[1
    ][0], radial_vectorUpper[2][0]]

    radial_vectorUpperUnitary = geo.rotate_translateList([[
    radial_vectorLowerUnitary[0]], [radial_vectorLowerUnitary[1]], [
    radial_vectorLowerUnitary[2]]], [(axis_vector[0]*2*np.pi*n_loops), 
    axis_vector[1]*2*np.pi*n_loops, axis_vector[2]*2*np.pi*n_loops], [0.0, 
    0.0, 0.0])

    radial_vectorUpperUnitary = [radial_vectorUpperUnitary[0][0], 
    radial_vectorUpperUnitary[1][0], radial_vectorUpperUnitary[2][0]]

    # Rotates the vector orthogonal to the radial vector

    ortho_radialVectorUpper = geo.rotate_translateList([[
    ortho_radialVectorLower[0]], [ortho_radialVectorLower[1]], [
    ortho_radialVectorLower[2]]], [(axis_vector[0]*2*np.pi*n_loops), 
    axis_vector[1]*2*np.pi*n_loops, axis_vector[2]*2*np.pi*n_loops], [
    0.0, 0.0, 0.0])

    ortho_radialVectorUpper = [ortho_radialVectorUpper[0][0], 
    ortho_radialVectorUpper[1][0], ortho_radialVectorUpper[2][0]]

    # Evaluates the final point in the central axis

    axis_increment = [(axis_vector[0]*trajectory_length), (axis_vector[1
    ]*trajectory_length), (axis_vector[2]*trajectory_length)]

    final_centralPoint = [axis_increment[0]+initial_centralPoint[0], (
    axis_increment[1]+initial_centralPoint[1]), (axis_increment[2]+
    initial_centralPoint[2])]

    # Evaluates the upper points

    upper_squarePoints = [[final_centralPoint[0]+radial_vectorUpper[0]+(
    prism_ratio*radius_X*ortho_radialVectorUpper[0])-(prism_ratio*
    radius_Y*radial_vectorUpperUnitary[0]), final_centralPoint[0]+
    radial_vectorUpper[0]-(prism_ratio*radius_X*ortho_radialVectorUpper[
    0])-(prism_ratio*radius_Y*radial_vectorUpperUnitary[0]), 
    final_centralPoint[0]+radial_vectorUpper[0]-(prism_ratio*radius_X*
    ortho_radialVectorUpper[0])+(prism_ratio*radius_Y*
    radial_vectorUpperUnitary[0]), final_centralPoint[0]+
    radial_vectorUpper[0]+(prism_ratio*radius_X*ortho_radialVectorUpper[
    0])+(prism_ratio*radius_Y*radial_vectorUpperUnitary[0])], [
    final_centralPoint[1]+radial_vectorUpper[1]+(
    prism_ratio*radius_X*ortho_radialVectorUpper[1])-(prism_ratio*
    radius_Y*radial_vectorUpperUnitary[1]), final_centralPoint[1]+
    radial_vectorUpper[1]-(prism_ratio*radius_X*ortho_radialVectorUpper[
    1])-(prism_ratio*radius_Y*radial_vectorUpperUnitary[1]), 
    final_centralPoint[1]+radial_vectorUpper[1]-(prism_ratio*radius_X*
    ortho_radialVectorUpper[1])+(prism_ratio*radius_Y*
    radial_vectorUpperUnitary[1]), final_centralPoint[1]+
    radial_vectorUpper[1]+(prism_ratio*radius_X*ortho_radialVectorUpper[
    1])+(prism_ratio*radius_Y*radial_vectorUpperUnitary[1])], [
    final_centralPoint[2]+radial_vectorUpper[2]+(
    prism_ratio*radius_X*ortho_radialVectorUpper[2])-(prism_ratio*
    radius_Y*radial_vectorUpperUnitary[2]), final_centralPoint[2]+
    radial_vectorUpper[2]-(prism_ratio*radius_X*ortho_radialVectorUpper[
    2])-(prism_ratio*radius_Y*radial_vectorUpperUnitary[2]), 
    final_centralPoint[2]+radial_vectorUpper[2]-(prism_ratio*radius_X*
    ortho_radialVectorUpper[2])+(prism_ratio*radius_Y*
    radial_vectorUpperUnitary[2]), final_centralPoint[2]+
    radial_vectorUpper[2]+(prism_ratio*radius_X*ortho_radialVectorUpper[
    2])+(prism_ratio*radius_Y*radial_vectorUpperUnitary[2])]]

    upper_cylinderPoints = [[]]

    # Calculates the points for the splines of the lateral lines

    spline_points9Square = geo.hellicoid_splinePoints([
    lower_squarePoints[0][0], lower_squarePoints[1][0], 
    lower_squarePoints[2][0]], [upper_squarePoints[0][0], 
    upper_squarePoints[1][0], upper_squarePoints[2][0]], n_loops, 
    axis_increment, [[radial_vector[0]+(prism_ratio*radius_X*
    ortho_radialVectorLower[0])-(prism_ratio*radius_Y*
    radial_vectorLowerUnitary[0])], [radial_vector[1]+(prism_ratio*
    radius_X*ortho_radialVectorLower[1])-(prism_ratio*radius_Y*
    radial_vectorLowerUnitary[0])], [radial_vector[2]+(prism_ratio*
    radius*ortho_radialVectorLower[2])]], initial_centralPoint, 
    transfinite_axial-2, False, bias=bias_axial)

    spline_points10Square = geo.hellicoid_splinePoints([
    lower_squarePoints[0][1], lower_squarePoints[1][1], 
    lower_squarePoints[2][1]], [upper_squarePoints[0][1], 
    upper_squarePoints[1][1], upper_squarePoints[2][1]], n_loops, 
    axis_increment, [[radial_vector[0]-(prism_ratio*radius*
    radial_vectorLowerUnitary[0])], [radial_vector[1]-(prism_ratio*
    radius*radial_vectorLowerUnitary[1])], [radial_vector[2]-(
    prism_ratio*radius*radial_vectorLowerUnitary[2])]], 
    initial_centralPoint, transfinite_axial-2, False, bias=bias_axial)

    spline_points11Square = geo.hellicoid_splinePoints([
    lower_squarePoints[0][2], lower_squarePoints[1][2], 
    lower_squarePoints[2][2]], [upper_squarePoints[0][2], 
    upper_squarePoints[1][2], upper_squarePoints[2][2]], n_loops, 
    axis_increment, [[radial_vector[0]-(prism_ratio*radius*
    ortho_radialVectorLower[0])], [radial_vector[1]-(prism_ratio*radius*
    ortho_radialVectorLower[1])], [radial_vector[2]-(prism_ratio*
    radius*ortho_radialVectorLower[2])]], initial_centralPoint, 
    transfinite_axial-2, False, bias=bias_axial)

    spline_points12Square = geo.hellicoid_splinePoints([
    lower_squarePoints[0][3], lower_squarePoints[1][3], 
    lower_squarePoints[2][3]], [upper_squarePoints[0][3], 
    upper_squarePoints[1][3], upper_squarePoints[2][3]], n_loops, 
    axis_increment, [[radial_vector[0]+(prism_ratio*radius*
    radial_vectorLowerUnitary[0])], [radial_vector[1]+(prism_ratio*
    radius*radial_vectorLowerUnitary[1])], [radial_vector[2]+(
    prism_ratio*radius*radial_vectorLowerUnitary[2])]], 
    initial_centralPoint, transfinite_axial-2, False, bias=bias_axial)

    # Calculates the points for the splines of the lateral lines of the 
    # cylinders

    """

    spline_points9Cylinder = geo.hellicoid_splinePoints([
    lower_cylinderPoints[0][0], lower_cylinderPoints[1][0], 
    lower_cylinderPoints[2][0]], [upper_cylinderPoints[0][0], 
    upper_cylinderPoints[1][0], upper_cylinderPoints[2][0]], n_loops, 
    axis_increment, [[radial_vector[0]+(radius*ortho_radialVectorLower[0
    ])], [radial_vector[1]+(radius*ortho_radialVectorLower[1])], [
    radial_vector[2]+(radius*ortho_radialVectorLower[2])]], 
    initial_centralPoint, transfinite_axial-2, False, bias=bias_axial)

    spline_points10Cylinder = geo.hellicoid_splinePoints([
    lower_cylinderPoints[0][1], lower_cylinderPoints[1][1], 
    lower_cylinderPoints[2][1]], [upper_cylinderPoints[0][1], 
    upper_cylinderPoints[1][1], upper_cylinderPoints[2][1]], n_loops, 
    axis_increment, [[radial_vector[0]-(radius*
    radial_vectorLowerUnitary[0])], [radial_vector[1]-(radius*
    radial_vectorLowerUnitary[1])], [radial_vector[2]-(radius*
    radial_vectorLowerUnitary[2])]], initial_centralPoint, 
    transfinite_axial-2, False, bias=bias_axial)

    spline_points11Cylinder = geo.hellicoid_splinePoints([
    lower_cylinderPoints[0][2], lower_cylinderPoints[1][2], 
    lower_cylinderPoints[2][2]], [upper_cylinderPoints[0][2], 
    upper_cylinderPoints[1][2], upper_cylinderPoints[2][2]], n_loops, 
    axis_increment, [[radial_vector[0]-(radius*
    ortho_radialVectorLower[0])], [radial_vector[1]-(radius*
    ortho_radialVectorLower[1])], [radial_vector[2]-(radius*
    ortho_radialVectorLower[2])]], initial_centralPoint, 
    transfinite_axial-2, False, bias=bias_axial)

    spline_points12Cylinder = geo.hellicoid_splinePoints([
    lower_cylinderPoints[0][3], lower_cylinderPoints[1][3], 
    lower_cylinderPoints[2][3]], [upper_cylinderPoints[0][3], 
    upper_cylinderPoints[1][3], upper_cylinderPoints[2][3]], n_loops, 
    axis_increment, [[radial_vector[0]+(radius*
    radial_vectorLowerUnitary[0])], [radial_vector[1]+(radius*
    radial_vectorLowerUnitary[1])], [radial_vector[2]+(radius*
    radial_vectorLowerUnitary[2])]], initial_centralPoint, 
    transfinite_axial-2, False, bias=bias_axial)

    """

    # Sets the lines instructions

    line_instructions = dict()

    line_instructions[9] = ["spline", spline_points9Square, "do not rotate"]

    line_instructions[10] = ["spline", spline_points10Square, "do not rotate"]

    line_instructions[11] = ["spline", spline_points11Square, "do not rotate"]

    line_instructions[12] = ["spline", spline_points12Square, "do not rotate"]

    # Makes the volume

    square_corners = [[*lower_squarePoints[0], *upper_squarePoints[0]],
    [*lower_squarePoints[1], *upper_squarePoints[1]], [
    *lower_squarePoints[2], *upper_squarePoints[2]]]

    bias_directions = dict()

    bias_directions["x"] = bias_circumferential

    bias_directions["y"] = bias_circumferential

    bias_directions["z"] = bias_axial

    geometric_data = cuboid.make_cuboid(square_corners,
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, geometric_data=geometric_data, 
    lines_instructionsOriginal=line_instructions)

    ####################################################################
    #                        Cylindrical sides                         #
    ####################################################################

    #

    return geometric_data