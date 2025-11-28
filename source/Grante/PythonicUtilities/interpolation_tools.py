# Routine to store a set of interpolation routines

from scipy import interpolate

import numpy as np

########################################################################
#                               Splines                                #
########################################################################

# Defines a function to return a spline interpolation of a curve in a 3D
# space

def spline_3D_interpolation(x_points=None, y_points=None, z_points=None, 
points_array=None, add_initial_point_as_end_point=False):
    
    """
    Function for creating a cubic spline interpolation function of a
    parametric curve given a set of points. It returns a function of a
    single argument, which is the parameter variable that belongs to the
    interval [0,1]
    
    x_points: list or numpy array of x values of the points

    y_points: list or numpy array of y values of the points

    z_points: list or numpy array of z values of the points

    points_array: list of lists or numpy array of the values of the 
    points. It can be a list with 3 sublists (one for each dimension), or
    one sublist with 3 values for each point. Likewise for numpy arrays

    add_initial_point_as_end_point: Flag to add a copy of the initial 
    point as end point to close the curve
    """

    # Tests if the points array is not None

    if points_array is not None:

        # Tests if it is a numpy array

        if isinstance(points_array, np.ndarray):

            # Tests if has two shape informations

            if len(points_array.shape)!=2:

                raise IndexError("The 'points_array' in 'spline_3D_int"+
                "erpolation' has shape "+str(points_array.shape)+", wh"+
                "ereas it should be a matrix")

            # Tests which dimension has length of 3

            if points_array.shape[0]==3:

                # Gets the individual dimensions

                x_points = points_array[0,:]

                y_points = points_array[1,:]

                z_points = points_array[2,:]

            elif points_array.shape[1]==3:

                # Gets the individual dimensions

                x_points = points_array[:,0]

                y_points = points_array[:,1]

                z_points = points_array[:,2]

            else:

                raise IndexError("'points_array' in 'spline_3D_interpo"+
                "lation' has shape "+str(points_array.shape)+". It sho"+
                "uld have one dimension with length of 3")
            
        # Tests if it is a list

        elif isinstance(points_array, list):

            # Tests which dimension has length of 3

            if len(points_array)==3:

                # Gets the individual dimensions

                x_points = points_array[0]

                y_points = points_array[1]

                z_points = points_array[2]

            else:

                # Gets the individual dimensions

                x_points = []

                y_points = []

                z_points = []

                # Iterates though the points

                for point in points_array:

                    # Tests if this points has 3 dimensions

                    if len(point)!=3:

                        raise IndexError("'points_array' in 'spline_3D"+
                        "_interpolation' has the individual point "+str(
                        point)+"; it should have 3 elements: x, y, and"+
                        " z")
                    
                    x_points.append(point[0])

                    y_points.append(point[1])

                    z_points.append(point[2])
                    
        else:

            raise TypeError("'points_array' in 'spline_3D_interpolatio"+
            "n' must be a numpy array or a list. It currently is: "+str(
            points_array))
        
    else:

        # Verifies if all dimensions are not None

        if x_points is None:

            raise TypeError("'x_points' is None in 'spline_3D_interpol"+
            "ation', but so is 'points_array'")
        
        elif y_points is None:

            raise TypeError("'y_points' is None in 'spline_3D_interpol"+
            "ation', but so is 'points_array'")
        
        elif z_points is None:

            raise TypeError("'z_points' is None in 'spline_3D_interpol"+
            "ation', but so is 'points_array'")
        
    # Gets the number of points

    n_points = len(x_points)

    if n_points==0:

        raise IndexError("No points were given for 'spline_3D_interpol"+
        "ation', x_points="+str(x_points))

    if len(y_points)!=n_points:

        raise IndexError("There are "+str(n_points)+" points in the x "+
        "direction; whereas there are "+str(len(y_points))+" points in"+
        " the y direction. They must have the same size in 'spline_3D_"+
        "interpolation'")

    elif len(z_points)!=n_points:

        raise IndexError("There are "+str(n_points)+" points in the x "+
        "direction; whereas there are "+str(len(z_points))+" points in"+
        " the z direction. They must have the same size in 'spline_3D_"+
        "interpolation'")
    
    # If the initial point is meant to be copied as the end point

    if add_initial_point_as_end_point:

        x_points.append(x_points[0])

        y_points.append(y_points[0])

        z_points.append(z_points[0])

        n_points += 1

    # Gets the range of the parametric variable

    parametric_variable = np.linspace(0.0, 1.0, n_points)

    # Gets the spline interpolation for each dimension

    cubic_spline_x = interpolate.CubicSpline(parametric_variable, 
    x_points)

    cubic_spline_y = interpolate.CubicSpline(parametric_variable, 
    y_points)

    cubic_spline_z = interpolate.CubicSpline(parametric_variable, 
    z_points)

    # Gets a parametric curve as a function of the single parameter

    parametric_curve = lambda theta: [cubic_spline_x(theta), 
    cubic_spline_y(theta), cubic_spline_z(theta)]

    return parametric_curve