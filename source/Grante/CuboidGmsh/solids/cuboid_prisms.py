# Routine to store cylindrical shapes

import numpy as np

from ..solids import cuboid_generator as cuboid

from ..tool_box import geometric_tools as geo

# Defines a function to create a rectangular prism with right corners

def right_rectangularPrism(length_x, length_y, length_z, axis_vector, 
base_point, transfinite_directions=[], bias_directions=dict(), 
shape_spin=0.0, geometric_data=[0, [[],[],[],[]], [[],[],[],[]], [[],[],
[]], dict(), [], dict(), [], [], [], 0.5, False]):
    
    # Creates the 8 corners as if the long axis is the Z axis proper, and 
    # the prism starts at the XY plane towards positive Z

    corner_points = [[length_x, length_x, 0.0, 0.0, length_x, length_x, 
    0.0, 0.0], [0.0, length_y, length_y, 0.0, 0.0, length_y, length_y, 
    0.0], [0.0, 0.0, 0.0, 0.0, length_z, length_z, length_z, length_z]]

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
    bias_directions, rotation_vector=rotation_vector, translation_vector
    =base_point, geometric_data=geometric_data)

    return geometric_data

# Defines a function to create a generic 6-sided prism

def hexahedron_from_corners(corner_points, edges_points=None,
transfinite_directions=[], bias_directions=dict(), geometric_data=[0, [[
],[],[],[]], [[],[],[],[]], [[],[],[]], dict(), [], dict(), [], [], [], 
0.5, False]):

    ####################################################################
    #                       Arguments consistency                      #
    ####################################################################
    
    # Tests if the corner points is a numpy array

    if isinstance(corner_points, np.ndarray):

        # Tests if it has the right shape

        if corner_points.shape==(8,3):

            corner_points = corner_points.T

            # Sets the flag to inform if the rows represent points ins-
            # tead of coordinates

            rows_are_points = True

        elif corner_points.shape!=(3,8):

            raise ValueError("'corner_points' is a numpy array of shap"+
            "e "+str(corner_points.shape)+", but it should have shape "+
            "(3,8) to create a hexadron")
        
        # Transforms it to a list

        corner_points = corner_points.tolist()

    elif isinstance(corner_points, list):

        if len(corner_points)==8:

            corner_points = (np.array(corner_points).T).tolist()

            # Sets the flag to inform if the rows represent points ins-
            # tead of coordinates

            rows_are_points = True

        elif len(corner_points)!=3:

            raise IndexError("'corner_points' is a list, but it has le"+
            "ngth of "+str(len(corner_points))+", whereas it should be"+
            " 3 or 8 (transposed) to construct a hexadron")

        for point in corner_points:

            if len(point)!=8:

                raise IndexError("The sublist '"+str(point)+"' in "+
                "'corner_points' does not have length of 8. Thus, "+
                "it is not possible to create a hexadron")
                
    else:

        raise TypeError("'corner_points' should be a list of lists or "+
        "numpy array of shape (8,3) or (3,8) to create a hexadron")

    ####################################################################
    #                        Lines construction                        #
    ####################################################################   

    lines_instructions = dict()
    
    if edges_points is not None:

        # Verifies if it is a dictionary
        
        if isinstance(edges_points, dict):

            lines_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12"]

            # Iterates through the edges

            for edge_number, edge_coordinates in edges_points.items():

                # Verifies if the number is valid

                if not str(edge_number) in lines_numbers:

                    raise ValueError("The edge number '"+str(edge_number
                    )+"' is not a valid number to index lines in a hex"+
                    "adron. The keys must be 1, 2, ..., 12")
                
                # Verifies if the edge coordinates is a numpy array

                if isinstance(edge_coordinates, np.ndarray):

                    # Tests if it has the right shape

                    if not (edge_coordinates.shape[0]==3 or (
                    edge_coordinates.shape[1]==3)):

                        raise ValueError("'edge_coordinates' is a nump"+
                        "y array of shape "+str(edge_coordinates.shape)+
                        ", but it should have 3 rows or 3 columns to c"+
                        "reate a 3D spline as the edge of a hexadron")
                    
                    # If the rows represent points, the matrix must be
                    # transposed for the cuboid works with coordinates x
                    # points
                        
                    if rows_are_points:

                        edge_coordinates = edge_coordinates.T
                    
                    # Transforms it to a list

                    edge_coordinates = edge_coordinates.tolist()

                elif isinstance(edge_coordinates, list):

                    if not (len(edge_coordinates)==3 or len(
                    edge_coordinates[0])==3):

                        raise IndexError("'edge_coordinates' is a list"+
                        ", but it has length of "+str(len(
                        edge_coordinates))+", whereas it should be 3 t"+
                        "o construct a hexadron")
                        
                    # If the rows represent points, the matrix must be
                    # transposed for the cuboid works with coordinates x
                    # points
                        
                    if rows_are_points:

                        edge_coordinates = (np.array(edge_coordinates).T
                        ).tolist()
                            
                else:

                    raise TypeError("'edge_coordinates' should be a li"+
                    "st of lists or numpy array of 3 rows or 3 columns"+
                    " to create a hexadron")
                
                # After the verifications, adds the line

                lines_instructions[int(edge_number)] = ["spline", 
                edge_coordinates]

        else:

            raise TypeError("'egde_points' must be a dictionary with s"+
            "tring numbering from '1' to '12' as keys and a list or nu"+
            "mpy array as values. Each key-value is used to construct "+
            "the corresponding edge using splines for a hexadron")              

    ####################################################################
    #                        Geometry generation                       #
    ####################################################################

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, geometric_data=geometric_data, 
    lines_instructionsOriginal=lines_instructions)

    return geometric_data