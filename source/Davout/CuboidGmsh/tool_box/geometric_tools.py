# Routine to store tools for geometric operations with points and cur-
# ves; but not as gmsh objects with the same names

import numpy as np

import copy

########################################################################
########################################################################
##                      Basic curvilinear shapes                      ##
########################################################################
########################################################################

# Defines a function to generate points for a spline along a helicoid
# path given number of loops, axial vector, initial and final points

def hellicoid_splinePoints(initial_point, final_point, n_loops, 
axial_length, radial_vector, point_inCentralAxisOriginal, n_points, 
include_endPoints, bias=1.0):
    
    # Makes a copy of the point in the central axis

    point_inCentralAxis = copy.deepcopy(point_inCentralAxisOriginal)

    # Initializes the matrix of points

    points_coordinates = [[], [], []]

    # Normalizes the axial increment to obtain the rotation vector

    norm_increment = np.sqrt((axial_length[0]**2)+(axial_length[1
    ]**2)+(axial_length[2]**2))

    normalized_axialVector = [axial_length[0]/norm_increment, (
    axial_length[1]/norm_increment), (axial_length[2]/
    norm_increment)]

    # Calculates the pitch

    pitch = 0.0

    if include_endPoints:

        # If the number of points is less or equal to 2, makes sure at
        # least one mid point is added

        if n_points<=2:

            n_points = 3

        # Calculates the first pitch, for the base of the line

        pitch, bias = delta_geometricProgression(norm_increment, bias, 
        n_points-1)

        points_coordinates[0].append(initial_point[0])

        points_coordinates[0].append(initial_point[1])

        points_coordinates[0].append(initial_point[2])

    else:

        # If the number of points is less or equal to 0, makes sure at
        # least one mid point is added

        if n_points<=0:

            n_points = 1

        # Calculates the first pitch, for the base of the line

        pitch, bias = delta_geometricProgression(norm_increment, bias, 
        n_points+1)

    # Calculates the delta angle dividing the total angle of the hel-
    # licoid by the axial length

    delta_angle = (n_loops*2*np.pi)/norm_increment

    # Initializes a variable of accumulated axial length

    accumulated_length = 0.0

    # Iterates through the number of points

    for i in range(n_points):

        # Updates the accumulated axial length

        accumulated_length += pitch

        # Updates the central point

        point_inCentralAxis[0] += normalized_axialVector[0]*pitch

        point_inCentralAxis[1] += normalized_axialVector[1]*pitch

        point_inCentralAxis[2] += normalized_axialVector[2]*pitch

        # Rotates the radial vector to the new angle, but do not trans-
        # late it. Uses the axial vector as the axis of rotation

        rotated_radialVector = rotate_translateList(radial_vector, 
        [normalized_axialVector[0]*delta_angle*accumulated_length, 
        normalized_axialVector[1]*delta_angle*accumulated_length, (
        normalized_axialVector[2]*delta_angle*accumulated_length)], [
        0.0, 0.0, 0.0])

        # Adds the new point using this radial vector and translates u-
        # sing the central point

        points_coordinates[0].append(point_inCentralAxis[0]+
        rotated_radialVector[0][0])

        points_coordinates[1].append(point_inCentralAxis[1]+
        rotated_radialVector[1][0])

        points_coordinates[2].append(point_inCentralAxis[2]+
        rotated_radialVector[2][0])

        # Updates the pitch

        pitch = pitch*bias

    # Adds the final point if it is to be included

    if include_endPoints:

        points_coordinates[0].append(final_point[0])

        points_coordinates[1].append(final_point[1])

        points_coordinates[2].append(final_point[2])

    # Return the points

    return points_coordinates

# Defines a function to compute the initial delta of a geometric pro-
# gressed mesh

def delta_geometricProgression(total_length, factor, n_terms):

    # Takes care whether the factor is negative

    if factor<0:

        factor = -factor

        # Initializes the sum

        sum_value = 0.0

        # Sums the powers of the terms

        for i in range(n_terms):

            sum_value += (factor**i)

        # Evaluates the last delta, for the negative sign reverses 
        # the order

        first_step = (total_length/sum_value)*(factor**(n_terms-1))

        # Divides the total length by the sum to get the initial 
        # step length

        return first_step, 1/factor
        
    else:

        # Initializes the sum

        sum_value = 0.0

        # Sums the powers of the terms

        for i in range(n_terms):

            sum_value += (factor**i)

        # Evaluates the first delta

        first_step = (total_length/sum_value)

        # Divides the total length by the sum to get the initial 
        # step length

        return first_step, factor

########################################################################
########################################################################
##                         Basic plane shapes                         ##
########################################################################
########################################################################

# Defines a function to construct the shadow of an elliptic sector over
# a plane

def ellipse_shadow(initial_angle, final_angle, semi_length1, 
semi_length2, n_points, include_endPoints, axis_vector, rotation_vector,
normal_shadowVector, center_point):
    
    # Creates the matrix of points

    points_coordinates = ellipse_pointsXYPlane(initial_angle, 
    final_angle, semi_length1, semi_length2, n_points, include_endPoints)

    # Rotates these points to the plane normal to the axis, but keeps 
    # the center at the origin

    points_coordinates = rotate_translateList(points_coordinates, 
    rotation_vector, [0.0, 0.0, 0.0])

    # Projects the shadow of the points onto a plane

    points_coordinates = project_shadowAndTranslate(points_coordinates, 
    normal_shadowVector, axis_vector, center_point)

    # Returns the matrix of points

    return points_coordinates

# Defines a function to construct the points of a section of a quarter 
# of an ellipse on the plane XY

def ellipse_pointsXYPlane(initial_angle, final_angle, semi_lengthX, 
semi_lengthY, n_points, include_endPoints):
    
    # Initializes the matrix of points' coordinates

    points_coordinates = [[], [], []]

    # Evaluates the delta angle

    delta_angle = 0.0

    if include_endPoints:

        delta_angle = (final_angle-initial_angle)/(n_points-1)

    else:

        delta_angle = (final_angle-initial_angle)/(n_points+1)

        initial_angle += delta_angle

    # Iterates through the number of points

    for i in range(n_points):

        # Evaluates the radius

        cossine = np.cos(initial_angle)

        sine = np.sin(initial_angle)

        r = ((semi_lengthX*semi_lengthY)/np.sqrt(((semi_lengthY*cossine
        )**2)+((semi_lengthX*sine)**2)))

        # Calculates the coordinates

        points_coordinates[0].append(r*cossine)

        points_coordinates[1].append(r*sine)

        points_coordinates[2].append(0.0)

        # Updates the angle

        initial_angle += delta_angle

    # Returns the matrix of points

    return points_coordinates

########################################################################
########################################################################
##                       Mathematical operations                      ##
########################################################################
########################################################################

########################################################################
#                               Rotation                               #
########################################################################

# Defines a function to find the sequence of rotations from on plane to
# another one, and considering a rotation about the new axis

def find_rotationToNewAxis(new_axis, old_axis, rotation_aboutNewAxis): 

    # Evaluates the norm of both vectors

    norm_new = np.sqrt((new_axis[0]**2)+(new_axis[1]**2)+(new_axis[2]**2
    ))

    norm_old = np.sqrt((old_axis[0]**2)+(old_axis[1]**2)+(old_axis[2]**2
    ))

    # Calculates the rotation angle using the dot product

    dot_product = ((new_axis[0]*old_axis[0])+(new_axis[1]*old_axis[1])+(
    new_axis[2]*old_axis[2]))

    angle = np.arccos(dot_product/(norm_new*norm_old))

    # Guarantees the old and new axes are unitary

    new_unitary = [new_axis[0]/norm_new, new_axis[1]/norm_new, (new_axis[
    2]/norm_new)]

    old_unitary = [old_axis[0]/norm_old, old_axis[1]/norm_old, (old_axis[
    2]/norm_old)]

    # Initializes the rotation vector for the case of the new axis being 
    # oriented opposite to the old axis, but still the old axis being a
    # coordinate principal axis (X, Y, or Z)

    rotation_toAxis = [1.0, 1.0, 1.0]

    if (new_unitary[0]==1.0 or new_unitary[0]==-1.0) and (new_unitary[0]
    ==-old_unitary[0]):

        rotation_toAxis = [0.0, np.pi, 0.0]

    elif (new_unitary[1]==1.0 or new_unitary[1]==-1.0) and (new_unitary[
    1]==-old_unitary[1]):

        rotation_toAxis = [0.0, 0.0, np.pi]

    elif (new_unitary[2]==1.0 or new_unitary[2]==-1.0) and (new_unitary[
    2]==-old_unitary[2]):

        rotation_toAxis = [0.0, np.pi, 0.0]

    # If this is not the case, evalutes the rotation vector properly

    elif not ((new_unitary[0]==-old_unitary[0]) and (new_unitary[1]==(
    -old_unitary[1])) and (new_unitary[2]==-old_unitary[2])): 

        # Evaluates the normal between the new and old vectors using the
        # cross product

        rotation_toAxis = [((old_unitary[1]*new_unitary[2])-(old_unitary[
        2]*new_unitary[1])), ((old_unitary[2]*new_unitary[0])-(
        old_unitary[0]*new_unitary[2])), ((old_unitary[0]*new_unitary[1]
        )-(old_unitary[1]*new_unitary[0]))]

        # Makes the norm of the rotation vector equal to the rotation 
        # angle

        rotation_norm = (np.sqrt((rotation_toAxis[0]**2)+(
        rotation_toAxis[1]**2)+(rotation_toAxis[2]**2)))

        if rotation_norm!=0:

            rotation_norm = (angle/rotation_norm) 

        rotation_toAxis = [(rotation_toAxis[0]*rotation_norm), (
        rotation_toAxis[1]*rotation_norm), (rotation_toAxis[2]*
        rotation_norm)]

    else:
        
        for i in range(len(old_axis)):

            if old_axis[i]!=0.0:

                # Creates a vector orthogonal to the old axis

                summ = 0.0

                for j in range(3):

                    if j!=i:

                        summ += old_axis[j]

                rotation_toAxis[i] = (((-1)/old_axis[i])*summ)

        # Makes the vector unitary

        norm_vector = ((np.pi)/np.sqrt((rotation_toAxis[0]**2)+(
        rotation_toAxis[1]**2)+(rotation_toAxis[2]**2)))
        
        rotation_toAxis = [rotation_toAxis[0]*norm_vector, (rotation_toAxis[
        1]*norm_vector), rotation_toAxis[2]*norm_vector]

    # Calculates the vector of rotation about the new axis

    shape_spin = (rotation_aboutNewAxis/(np.sqrt((new_axis[0]**2)+(
    new_axis[1]**2)+(new_axis[2]**2))))

    shape_spin = [shape_spin*new_axis[0], shape_spin*new_axis[1], 
    shape_spin*new_axis[2]]

    # Concatenates the two rotation vectors

    return [rotation_toAxis, shape_spin]

# Defines a function to turn a list of points into a numpy array, rotate
# and translate it, then, transform back into a list

def rotate_translateList(list, rotation_vector, translation_vector):

    # Turns the list into a numpy array

    list_np = np.array(copy.deepcopy(list))

    # Rotate and translate it

    list_np = rotate_andTranslateEulerRodrigues(list_np, rotation_vector, 
    *translation_vector)

    # Transform back into a list and returns it

    return list_np.tolist()

# Defines the rotation matrix in a plane

def rotation_matrix(angle):

    c = np.cos(angle)

    s = np.sin(angle)

    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

# Defines the rotation matrix using a vector as the axis and its norm as
# the angle in radians

def rotation_matrixEulerRodrigues(rotation_vector):

    # Verifies if the rotation vector is indeed a list of vectors

    if isinstance(rotation_vector[0], list):

        # Initializes the rotation matrix

        R = np.identity(3)

        # Iterates through the rotation vectors

        for i in range(len(rotation_vector)):

            # Gets the components of the rotation vector

            phi_x = rotation_vector[i][0]

            phi_y = rotation_vector[i][1]

            phi_z = rotation_vector[i][2]

            # Gets the rotation angle

            angle = np.sqrt((phi_x*phi_x)+(phi_y*phi_y)+(phi_z*phi_z))

            # Evaluates the rotation matrix using the first part of the
            # Euler-Rodrigues formula

            cos_angle = np.cos(angle)

            sin_angle = 1.0

            coeff_tensor = 1.0

            if np.abs(angle)>1E-6:

                sin_angle = (np.sin(angle)/angle)

                coeff_tensor = ((1-cos_angle)/(angle*angle))

            W = np.array([[0.0, -phi_z, phi_y],[phi_z, 0.0, -phi_x], [
            -phi_y, phi_x, 0.0]])

            phi_tensor_phi = np.array([[phi_x*phi_x, phi_x*phi_y, (phi_x
            *phi_z)], [phi_y*phi_x, phi_y*phi_y, phi_y*phi_z], [phi_z*
            phi_x, phi_z*phi_y, phi_z*phi_z]])

            R = np.matmul((cos_angle*np.identity(3)+(sin_angle*W)+(
            coeff_tensor*phi_tensor_phi)), R)

        # Returns it

        return R

    else:

        # Gets the components of the rotation vector

        phi_x = rotation_vector[0]

        phi_y = rotation_vector[1]

        phi_z = rotation_vector[2]

        # Gets the rotation angle

        angle = np.sqrt((phi_x*phi_x)+(phi_y*phi_y)+(phi_z*phi_z))

        # Evaluates the rotation matrix using the first part of the Eu-
        # ler-Rodrigues formula

        cos_angle = np.cos(angle)

        sin_angle = 1.0

        coeff_tensor = 1.0

        if np.abs(angle)>1E-6:

            sin_angle = (np.sin(angle)/angle)

            coeff_tensor = ((1-cos_angle)/(angle*angle))

        W = np.array([[0.0, -phi_z, phi_y],[phi_z, 0.0, -phi_x], [-phi_y, 
        phi_x, 0.0]])

        phi_tensor_phi = np.array([[phi_x*phi_x, phi_x*phi_y, (phi_x
        *phi_z)], [phi_y*phi_x, phi_y*phi_y, phi_y*phi_z], [phi_z*phi_x,
        phi_z*phi_y, phi_z*phi_z]])

        R = (cos_angle*np.identity(3)+(sin_angle*W)+(coeff_tensor*
        phi_tensor_phi))

        # Returns it

        return R
    
# Defines a function to rotate, and, then, add a translation 

def rotate_andTranslate(matrix, polar_angle, x0, y0, z0):
    
    # Evaluates the rotation matrix and rotate the points

    R = rotation_matrix(polar_angle)

    matrix = np.matmul(R, matrix)

    # Adds the centroids back again

    for i in range(matrix.shape[1]):

        matrix[0,i] += x0

        matrix[1,i] += y0

        matrix[2,i] += z0

    return matrix
    
# Defines a function to rotate, and, then, add a translation. The rota-
# tion is performed using the Euler-Rodrigues formula

def rotate_andTranslateEulerRodrigues(matrix, rotation_vector, x0, y0, 
z0):
    
    # Evaluates the rotation matrix and rotate the points

    R = rotation_matrixEulerRodrigues(rotation_vector)

    matrix = np.matmul(R, matrix)

    # Adds the centroids back again

    for i in range(matrix.shape[1]):

        matrix[0,i] += x0

        matrix[1,i] += y0

        matrix[2,i] += z0

    return matrix

########################################################################
#                              Projection                              #
########################################################################

# Defines a function to calculate the shadows of a matrix of vectors on-
# to a plane when the light comes from a direction

def project_shadowAndTranslate(matrix_list, normal_projectionVector,
light_directionVector, translation_vector):
    
    # Evaluates the inner product of the light direction with the normal
    # vector of the shadowed plane

    inner_lightPlaneNormal = ((normal_projectionVector[0]*
    light_directionVector[0])+(normal_projectionVector[1]*
    light_directionVector[1])+(normal_projectionVector[2]*
    light_directionVector[2]))

    # If this inner product is zero, this means no shadow is possible

    if abs(inner_lightPlaneNormal)<1E-10:

        raise ValueError("The normal vector of the plane where upon th"+
        "e shadow of the vectors must be projected is orthogonal to th"+
        "e light source's direction, i.e., no shadow is possible\n")

    # Initializes the new shadow vectors

    shadow_list = [[], [], []]

    # Iterates through the points

    for i in range(len(matrix_list[0])):

        # The principal behind the following calculations lie on the
        # fact that the shadow is a linear combination of the original
        # projected vector with the light source direction. A length for
        # the source direction must be found so that the shadow is or-
        # thogonal to the plane where upon the shadow is projected

        direction_length = ((-1/inner_lightPlaneNormal)*((matrix_list[0
        ][i]*normal_projectionVector[0])+(matrix_list[1][i]*
        normal_projectionVector[1])+(matrix_list[2][i]*
        normal_projectionVector[2])))

        # Calculates the shadow vector and adds the translation

        for j in range(3):

            shadow_list[j].append(matrix_list[j][i]+(direction_length*
            light_directionVector[j])+translation_vector[j])

    return shadow_list
    
# Defines a function to project, and, then, add a translation to a set
# of points in a matrix

def project_andTranslate(matrix_list, normal_projectionVector, 
translation_vector):
    
    # Evaluates the projection matrix and project the points onto a pla-
    # ne centered at the origin

    P = projection_matrix(normal_projectionVector)

    matrix = np.array(copy.deepcopy(matrix_list))

    projected_matrix = np.matmul(P, matrix)

    # Adds the centroids back again

    for i in range(matrix.shape[1]):

        projected_matrix[0,i] += translation_vector[0]

        projected_matrix[1,i] += translation_vector[1]

        projected_matrix[2,i] += translation_vector[2]

    return projected_matrix.tolist()

# Defines a function to create a projection matrix onto a plane centered
# at the origin given its normal vector

def projection_matrix(normal_vector):

    # Initializes P (projection matrix) as the identity

    P = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    # Subtracts the tensor product of the normal vector with itself

    for i in range(3):

        for j in range(3):

            P[i][j] -= (normal_vector[i]*normal_vector[j])

    # Returns the matrix as a numpy array

    return np.array(P)
    
########################################################################
#                              Reflection                              #
########################################################################

# Defines the reflection matrix in a plane. The components of the 3D 
# vector that defines the reflection axis are n_x, n_y, and n_z. If no 
# reflection is intended, n_x, n_y, and n_z must both be zero and the 
# result will be the identity matrix

def reflection_matrix(reflection_vector):

    n_x = reflection_vector[0]

    n_y = reflection_vector[1]

    n_z = reflection_vector[2]

    if n_x==0 and n_y==0 and n_z==0.0:

        return np.identity(3)
    
    else:

        T = np.zeros((3,3))

        T[0,0] = (2*n_x*n_x)

        T[0,1] = 2*n_x*n_y

        T[0,2] = 2*n_x*n_z

        T[1,0] = 2*n_x*n_y

        T[1,1] = (2*n_y*n_y)

        T[1,2] = 2*n_y*n_z

        T[2,0] = 2*n_z*n_x

        T[2,1] = 2*n_z*n_y 

        T[2,2] = 2*n_z*n_z

        abs_n = 1/((n_x*n_x)+(n_y*n_y)+(n_z*n_z))

        return np.identity(3)-(abs_n*T)
    
########################################################################
#                            Linear algebra                            #
########################################################################

# Defines a function to evaluate the inner product between two vectors
# given as lists

def inner_productLists(list1, list2):

    value = 0.0

    for i in range(len(list1)):

        value += (list1[i]*list2[i])

    return value

# Defines a function to evaluate the cross product between two vectors
# given as lists

def cross_productFromLists(list1, list2):

    # Gets the 3D cross product

    list3 = [((list1[1]*list2[2])-(list1[2]*list2[1])), ((list1[2]*
    list2[0])-(list1[0]*list2[2])), ((list1[0]*list2[1])-(list1[1]*
    list2[0]))]

    return list3

# Defines a function to evaluate the norm of a vector given by a list

def norm_ofList(list1):

    norm_value = 0.0

    for element in list1:

        norm_value += (element**2)

    return np.sqrt(norm_value)

# Defines a function to normalize a vector given by a list

def normalize_list(list1):

    norm_value = norm_ofList(list1)

    norm_value = (1/norm_value)

    for i in range(len(list1)):

        list1[i] = norm_value*list1[i]

    return list1