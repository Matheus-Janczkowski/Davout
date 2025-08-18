# Routine to store verification methods

from ..tool_box import geometric_tools as geo_tools

# Defines a function to verify whether a cuboid is proper, i.e. if there
# are no overlaping or entangled facets. It uses the criterion that, 
# each one of the four height-direction lines must be inside the cone of 
# the two opposing normals. 
# The coordinate points are organized as in the common gmsh order

def verify_cuboidEntanglement(edges_coordinates):

    # Gets two adjacent lines in the upper and lower facets

    lower_line1 = [edges_coordinates[0][1]-edges_coordinates[0][0],
    edges_coordinates[1][1]-edges_coordinates[1][0], edges_coordinates[(
    2)][1]-edges_coordinates[2][0]]

    lower_line2 = [edges_coordinates[0][3]-edges_coordinates[0][0],
    edges_coordinates[1][3]-edges_coordinates[1][0], edges_coordinates[(
    2)][3]-edges_coordinates[2][0]]

    upper_line1 = [edges_coordinates[0][5]-edges_coordinates[0][4],
    edges_coordinates[1][5]-edges_coordinates[1][4], edges_coordinates[(
    2)][5]-edges_coordinates[2][4]]

    upper_line2 = [edges_coordinates[0][7]-edges_coordinates[0][4],
    edges_coordinates[1][7]-edges_coordinates[1][4], edges_coordinates[(
    2)][7]-edges_coordinates[2][4]]

    # Evaluates the outward-pointing normal of each facet and normalizes 
    # them

    upper_normal = geo_tools.cross_productFromLists(upper_line1, 
    upper_line2)

    upper_normal = geo_tools.normalize_list(upper_normal)

    lower_normal = geo_tools.cross_productFromLists(lower_line2, 
    lower_line1)

    lower_normal = geo_tools.normalize_list(lower_normal)

    # Initializes a flag of entanglement

    flag_entanglement = False

    # Iterates through each height-direction line

    for i in range(4):

        # Gets the line as a list

        height_line = [edges_coordinates[0][i+4]-edges_coordinates[0][i
        ], edges_coordinates[1][i+4]-edges_coordinates[1][i],
        edges_coordinates[2][i+4]-edges_coordinates[2][i]]

        # This line must have a positive inner product with the upper 
        # facet's normal

        if geo_tools.inner_productLists(height_line, upper_normal)<=0.0:

            flag_entanglement = True

            break

        # This line must have a negative inner product with the lower
        # facet's normal

        if geo_tools.inner_productLists(height_line, lower_normal)>=0.0:

            flag_entanglement = True

            break

    # Returns the flag

    return flag_entanglement