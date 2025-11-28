# Routine to store cylindrical shapes

import numpy as np

from copy import deepcopy

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

def hexahedron_from_corners(corner_points, parametric_curves=None, 
edges_points=None, transfinite_directions=[], bias_directions=dict(), 
geometric_data=[0, [[],[],[],[]], [[],[],[],[]], [[],[],[]], dict(), [], 
dict(), [], [], [], 0.5, False], explicit_volume_physical_group_name=
None, explicit_surface_physical_group_name=None):

    ####################################################################
    #                       Arguments consistency                      #
    ####################################################################

    # Checks if parametric curves are given

    if parametric_curves is not None:

        # Tests it is not a dictionary

        if not isinstance(parametric_curves, dict):

            raise TypeError("'parametric_curves' must be a dictionary "+
            "in 'hexahedron_from_corners'. The keys must be strings; a"+
            "nd the values must be functions of a single variable")
        
        else:

            # Checks if the keys are strings

            for key in parametric_curves:

                if not isinstance(key, str):

                    raise TypeError("'parametric_curves' must be a dic"+
                    "tionary in 'hexahedron_from_corners'. The keys mu"+
                    "st be strings; and the values must be functions o"+
                    "f a single variable. The key '"+str(key)+"' is no"+
                    "t a string")
                
    else:

        parametric_curves = {}
    
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

            # Checks if the sublist has two elements only

            old_points = deepcopy(corner_points)

            corner_points = [[], [], []]

            for i in range(8):

                corner_points[0].append(old_points[i][0])

                corner_points[1].append(old_points[i][1])

                if len(old_points[i])==2:

                    corner_points[2].append(0.0)

                else:

                    corner_points[2].append(old_points[i][2])

            # Sets the flag to inform if the rows represent points ins-
            # tead of coordinates

            rows_are_points = True

        elif len(corner_points)!=3:

            raise IndexError("'corner_points' is a list, but it has le"+
            "ngth of "+str(len(corner_points))+", whereas it should be"+
            " 3 or 8 (transposed) to construct a hexadron")

        for dimension in corner_points:

            if len(dimension)!=8:

                raise IndexError("The sublist '"+str(dimension)+"' in "+
                "'corner_points' does not have length of 8. Thus, it i"+
                "s not possible to create a hexadron")
                
    else:

        raise TypeError("'corner_points' should be a list of lists or "+
        "numpy array of shape (8,3) or (3,8) to create a hexadron")
    
    # Checks if the corner points are given using the parametric curves

    for i in range(8):

        if isinstance(corner_points[0][i], str):

            # Gets the name of the parametric curve

            curve_name = corner_points[0][i]

            # Checks if this name is in the parametric curve dictionary

            if not (curve_name in parametric_curves):

                raise KeyError("The name '"+str(curve_name)+"' is not "+
                "a key of the 'parametric_curves' dictionary. Thus, it"+
                " is not possible to use it to get a corner in 'hexahe"+
                "dron_from_corners'. The 'parametric_curves' dictionar"+
                "y has the keys: "+str(parametric_curves.keys()))
            
            # Gets the parametric curve result

            parametric_result = parametric_curves[curve_name](
            corner_points[1][i])

            # Verifies if the result is a list and if it has three slots

            if not isinstance(parametric_result, list):

                raise TypeError("The result of the parametric curve '"+
                str(curve_name)+"' must be a list in 'hexahedron_from_"+
                "corners'")
            
            if len(parametric_result)!=3:

                raise IndexError("The result of the parametric curve '"+
                str(curve_name)+"' must be a list of length 3 in 'hexa"+
                "hedron_from_corners'. The current length is "+str(len(
                parametric_result)))
            
            # Puts the result in the x, y, and z order

            corner_points[0][i] = parametric_result[0]*1.0

            corner_points[1][i] = parametric_result[1]*1.0

            corner_points[2][i] = parametric_result[2]*1.0

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
                    edge_coordinates)==4 or len(edge_coordinates[0])==3):

                        raise IndexError("'edge_coordinates' is a list"+
                        ", but it has length of "+str(len(
                        edge_coordinates))+", whereas it should be 3 t"+
                        "o construct a hexadron or 4 to inform the par"+
                        "ametric curve name using a string, the initia"+
                        "l parameter, the final parameter, and the num"+
                        "ber of points in between")
                    
                    # Checks if a parametric curve name is asked

                    if isinstance(edge_coordinates[0], str):

                        if len(edge_coordinates)!=4:

                            raise IndexError("'"+str(edge_coordinates)+
                            "' has length of "+str(len(edge_coordinates)
                            )+", whereas it should have length of 4: s"+
                            "tring name for the parametric curve; init"+
                            "ial parameter value; final parameter valu"+
                            "e; number of points inside the edge")

                        # Checks if the first value is a string

                        curve_name = edge_coordinates[0]

                        # Checks if this name is in the dictionary 
                        # of parametric curves

                        if not (curve_name in parametric_curves):

                            raise KeyError("The name '"+str(
                            curve_name)+"' is not a key of the 'parame"+
                            "tric_curves' dictionary. Thus, it is not "+
                            "possible to use it to get an edge in 'hex"+
                            "ahedron_from_corners'. The 'parametric_cu"+
                            "rves' dictionary has the keys: "+str(
                            parametric_curves.keys()))
                        
                        # Calculates the delta of parameter

                        theta_initial = deepcopy(edge_coordinates[1])

                        delta_theta = ((edge_coordinates[2]-
                        edge_coordinates[1])/(edge_coordinates[3]+1))

                        n_points = deepcopy(edge_coordinates[3])

                        edge_coordinates = []

                        # Iterates through the points 

                        for i in range(n_points):

                            # Gets the result of the parametric curve

                            parametric_result = parametric_curves[
                            curve_name](theta_initial+((i+1)*delta_theta
                            ))

                            # Verifies if the result is a list and if it 
                            # has three slots

                            if not isinstance(parametric_result, list):

                                raise TypeError("The result of the par"+
                                "ametric curve '"+str(curve_name)+"' m"+
                                "ust be a list in 'hexahedron_from_cor"+
                                "ners'")
                            
                            if len(parametric_result)!=3:

                                raise IndexError("The result of the pa"+
                                "rametric curve '"+str(curve_name)+"' "+
                                "must be a list of length 3 in 'hexahe"+
                                "dron_from_corners'. The current lengt"+
                                "h is "+str(len(parametric_result)))
                            
                            # Adds the result as a point

                            edge_coordinates.append(parametric_result)
                        
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
    lines_instructionsOriginal=lines_instructions, 
    explicit_volume_physical_group_name=
    explicit_volume_physical_group_name,
    explicit_surface_physical_group_name=
    explicit_surface_physical_group_name)

    return geometric_data