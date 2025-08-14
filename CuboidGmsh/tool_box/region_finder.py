# Routine to store methods to identify regions in a mesh

import numpy as np

import CuboidGmsh.source.tool_box.geometric_tools as geo_tools

########################################################################
#                            Volumes finder                            #
########################################################################

# Defines a function to verify whether a set of points of a volume lies
# on a volumetric region given by a generic expression

def general_3DEnclosure(points_set, volume_expression):

    # Iterates through the points

    for point in points_set:

        # Evaluates the boundary expression, if it is true, the point is 
        # in the boundary

        if not volume_expression(*point):

            return False 
        
    # If no point was found out of the boundary so far, it means the 
    # surface is within the boundary

    return True

# Defines a function to identify if a point is located within a 3D re-
# gion enclosed in or out of a cylinder tappered by planes. The planes
# are given by normal vectors that point outwards. The base point is a
# point in the axial direction of the cylinder and gives the centroid of
# the lower plane facet of the cylinder. The flag_inside is True when 
# the point is to be searched inside the enclosure, and False otherwise

def cylindrical_enclosure(x, y, z, radius, length, axial_vector, 
planes_normals, base_point, flag_inside=True, tolerance=1E-4):
    
    # Guarantees the axial vector is unitary

    axial_vector = geo_tools.normalize_list(axial_vector)
    
    # Evaluates the biases of the planes

    planes_biases = []

    planes_biases.append(geo_tools.inner_productLists(base_point, 
    planes_normals[0]))

    planes_biases.append(geo_tools.inner_productLists([base_point[0]+
    (length*axial_vector[0]), base_point[1]+(length*axial_vector[1]), 
    base_point[2]+(length*axial_vector[2])], planes_normals[1]))  

    # Checks whether the point is in between these two planes

    if not plane_enclosure(x, y, z, planes_normals, planes_biases, 
    tolerance=tolerance):

        return False 

    # Shifts the position vector by taking out the base point

    shifted_position = [x-base_point[0], y-base_point[1], z-base_point[2
    ]]
    
    # If it is inside these planes, checks the cylindrical region. 
    # Projects the point into the axial vector

    axial_component = geo_tools.inner_productLists(shifted_position, 
    axial_vector)

    # Gets the component in the radial direction, i.e. orthogonal to the
    # axial direction

    radial_component = np.sqrt((geo_tools.norm_ofList(shifted_position)
    **2)-(axial_component*axial_component))

    # Compares the radial component with the radius

    if flag_inside:

        if radial_component<(radius+tolerance):

            return True 
        
    else:

        if radial_component>(radius-tolerance):

            return True 
        
    return False

# Defines a function to identify if a point is located within a 3D re-
# gion enclosed among a list of planes. The normals of the planes must 
# point OUT of the enclosure. The planes_normals is a list of lists, 
# where the inner lists are made of the components of the normal vectors
# of each plane; the planes_biases is a vector of the value of projec-
# tion of the point over the normal vector that must be met so the point
# is contained within the plane

def plane_enclosure(x, y, z, planes_normals, planes_biases, tolerance=
1E-4, flag_inside=True):

    # Iterates through the planes

    for i in range(len(planes_biases)):

        # Calculates the projection of the position vector over the nor-
        # mal vector of the plane. Discounts the bias

        projection = ((x*planes_normals[i][0])+(y*planes_normals[i][1])+
        (z*planes_normals[i][2])-planes_biases[i])

        # If the projection is positive, it means the point is out of 
        # the plane, thus, returns false

        if flag_inside:

            if projection>tolerance:

                return False
            
        else:

            if projection>-tolerance:

                return False

    return True

########################################################################
#                           Surfaces finder                            #
########################################################################

# Defines a function to verify if a general curved surface is within a
# two dimensional manifold

def general_2Dboundary(points_set, boundary_expression, tolerance=1E-4):

    # Iterates through the points

    for point in points_set:

        # Evaluates the boundary expression, if it is within a tolerance
        # from 0, the point is in the boundary

        if abs(boundary_expression(*point))>tolerance:

            return False 
        
    # If no point was found out of the boundary so far, it means the 
    # surface is within the boundary

    return True

# Defines a function to verify if a set of point is within a plane

def plane_boundary(points_set, plane_normal, plane_bias, 
limit_planeNormals=[], limit_planeBiases=[], tolerance=1E-4):
    
    # Iterates through the points

    for i in range(len(points_set)):

        # Gets the coordinates of the point

        x = points_set[i][0]

        y = points_set[i][1]

        z = points_set[i][2]

        # Verifies if the point is contained within the plane given a 
        # tolerance

        residue = ((x*plane_normal[0])+(y*plane_normal[1])+(z*
        plane_normal[2])-plane_bias)

        if abs(residue)<=tolerance:

            # Verifies if the point is within the bounding planes

            for i in range(len(limit_planeNormals)):

                # Calculates the projection of the position vector over
                # the normal vector of the plane. Discounts the bias

                projection = ((x*limit_planeNormals[i][0])+(y*
                limit_planeNormals[i][1])+(z*limit_planeNormals[i][2])-
                limit_planeBiases[i])

                # If the projection is positive, it means the point is 
                # out of the plane, thus, returns false

                if projection>0:

                    return False
                
            # If it hasn't returned False so far, it means that the 
            # point is within bounds AND within the surface plane. Thus,
            # just keep testing
        
        else:

            return False
        
    # As all points have been succesfully tested, returns true
        
    return True