# Routine to store a set of interpolation routines

from scipy import interpolate

import numpy as np

########################################################################
#                               Splines                                #
########################################################################

# Defines a class to store spline results

class SplineInterpolation:

    def __init__(self, interpolation_function):

        self.interpolation_function = interpolation_function

    # Defines a function to call the class as a function

    def __call__(self, theta):
        
        return self.interpolation_function(theta)

# Defines a function to return a spline interpolation of a curve in a 3D
# space

def spline_3D_interpolation(x_points=None, y_points=None, z_points=None, 
points_array=None, add_initial_point_as_end_point=False, 
polar_angle_around_z=False, verbose=False, x_centroid=None, y_centroid=
None, z_centroid=None, return_initial_and_final_polar_angles=False, 
periodic_interpolation=False):
    
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

    polar_angle_around_z: the spline parametric variable becomes the
    polar angle around z in degrees
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
    
    # Calculates the centroid

    if x_centroid is None:

        x_centroid = np.mean(np.array(x_points))

    if y_centroid is None:

        y_centroid = np.mean(np.array(y_points))

    if z_centroid is None:

        z_centroid = np.mean(np.array(z_points))
    
    # If the initial point is meant to be copied as the end point

    if add_initial_point_as_end_point:

        x_points.append(x_points[0])

        y_points.append(y_points[0])

        z_points.append(z_points[0])

        n_points += 1

    # Gets the range of the parametric variable

    parametric_variable = None

    # If the original points are x and y values that have meaning in a
    # cylindrical coordinate system, the code makes sure that the para-
    # metric variable has the same interval as the polar angle

    if polar_angle_around_z:

        # Calculates the polar angle of the first point

        polar_angle = 0.5*np.pi

        # Verifies if the point lies on the first or fourth quadrants

        if (x_points[0]-x_centroid)>1E-9:

            polar_angle = np.arctan((y_points[0]-y_centroid)/(x_points[0
            ]-x_centroid))

            # If the point is in the fourth or third quadrants, adds 2 pi

            if (y_points[0]-y_centroid)<0.0:

                polar_angle += (2*np.pi)

        # Verifies if the point lies in the second or third quadrants

        elif (x_points[0]-x_centroid)<-1E-9:

            polar_angle = np.arctan((y_points[0]-y_centroid)/(x_points[0
            ]-x_centroid))

            # Adds pi to correct for the negative x
            
            polar_angle += np.pi

        elif (y_points[0]-y_centroid)<0.0:

            polar_angle = 1.5*np.pi 

        # Updates the list of parametric variable with this angle

        parametric_variable = [polar_angle*(180/np.pi)]

        # Iterates through the remaing points

        for i in range(1, n_points):

            # Calculates the angle between this point and the previous 
            # one using the cross product

            a_x = x_points[i-1]-x_centroid

            a_y = y_points[i-1]-y_centroid

            b_x = x_points[i]-x_centroid

            b_y = y_points[i]-y_centroid

            numerator = ((a_x*b_y)-(a_y*b_x))

            denominator = np.sqrt(((a_x*a_x)+(a_y*a_y))*((b_x*b_x)+(b_y*
            b_y)))

            # Takes care with values below -1.0 and above 1.0 

            delta_theta = np.arcsin(min(max(numerator/denominator, -1.0), 
            1.0))

            # If the angle is negative, adds 2 pi to get the complement

            if delta_theta<0.0:

                delta_theta += 2*np.pi

            # Adds the polar angle as parametric variable. Adds to the
            # last value to allow for values greater than 2*pi and in 
            # ascending order

            parametric_variable.append(parametric_variable[-1]+(
            delta_theta*(180/np.pi)))

        # Sorts the points in ascending order of 

        # Transforms the parametric variable list into a numpy array

        parametric_variable = np.array(parametric_variable)

    else:

        parametric_variable = np.linspace(0.0, 1.0, n_points)

    if verbose:

        print("The parametric variables and the coordinates for the sp"+
        "line interpolation\nof the curve whose centroid is at ("+str(
        x_centroid)+", "+str(y_centroid)+", "+str(z_centroid)+") are:")

        for i in range(len(x_points)):

            print("theta = "+str(parametric_variable[i])+"; x = "+str(
            x_points[i])+"; y = "+str(y_points[i])+"; z = "+str(z_points[
            i]))

        print("\n")

    # Gets the spline interpolation for each dimension

    cubic_spline_x = interpolate.CubicSpline(parametric_variable, 
    x_points)

    cubic_spline_y = interpolate.CubicSpline(parametric_variable, 
    y_points)

    cubic_spline_z = interpolate.CubicSpline(parametric_variable, 
    z_points)

    # Gets a parametric curve as a function of the single parameter

    parametric_curve = None 

    # If a periodic interpolation is to be created

    if periodic_interpolation:

        # Verifies if cylindrical coordinates were given

        if not polar_angle_around_z:

            raise ValueError("A periodic interpolation was asked for '"+
            "spline_3D_interpolation', but 'polar_angle_around_z' is n"+
            "ot True")
        
        # Gets the final and initial angles

        initial_angle = parametric_variable[0]

        final_angle = parametric_variable[-1]
        
        # Creates a function

        def parametric_curve_function(theta, initial_angle, final_angle):

            # If the angle is smaller than the initial angle

            if theta<initial_angle:

                # Gets the delta angle

                delta_theta = ((initial_angle-theta)%(final_angle-
                initial_angle))

                # Sums to the final angle

                theta = final_angle-delta_theta

            #  If the angle is larger than the final angle

            elif theta>final_angle:

                # Gets the remainder of the division of theta by the fi-
                # nal angle 

                delta_theta = theta%final_angle

                # Adds this to the initial angle

                theta = initial_angle+delta_theta

            # Gets the parametric curve result

            return [cubic_spline_x(theta), cubic_spline_y(theta), 
            cubic_spline_z(theta)]
        
        # Creates a driver

        parametric_curve = SplineInterpolation(lambda theta: (
        parametric_curve_function(theta,initial_angle, final_angle)))

    # Otherwise, creates a plain driver

    else:

        parametric_curve = SplineInterpolation(lambda theta: [
        cubic_spline_x(theta), cubic_spline_y(theta), cubic_spline_z(
        theta)])

    # Returns everything

    if polar_angle_around_z and return_initial_and_final_polar_angles:

        return (parametric_curve, parametric_variable[0], 
        parametric_variable[-1])

    return parametric_curve

# Defines a function to return a spline interpolation of a curve in 1D
# space

def spline_1D_interpolation(x_points=None, y_points=None, points_array=
None, add_initial_point_as_end_point=False):
    
    """
    Function for creating a cubic spline interpolation function of a
    parametric curve given a set of points. It returns a function of a
    single argument, which is the parameter variable that belongs to the
    interval [0,1]
    
    x_points: list or numpy array of x values of the points

    y_points: list or numpy array of y values of the points

    points_array: list of lists or numpy array of the values of the 
    points. It can be a list with 2 sublists (one for each dimension), or
    one sublist with 2 values for each point. Likewise for numpy arrays

    add_initial_point_as_end_point: Flag to add a copy of the initial 
    point as end point to close the curve
    """

    # Tests if the points array is not None

    if points_array is not None:

        # Tests if it is a numpy array

        if isinstance(points_array, np.ndarray):

            # Tests if has two shape informations

            if len(points_array.shape)!=2:

                raise IndexError("The 'points_array' in 'spline_1D_int"+
                "erpolation' has shape "+str(points_array.shape)+", wh"+
                "ereas it should be a matrix")

            # Tests which dimension has length of 3

            if points_array.shape[0]==2:

                # Gets the individual dimensions

                x_points = points_array[0,:]

                y_points = points_array[1,:]

            elif points_array.shape[1]==2:

                # Gets the individual dimensions

                x_points = points_array[:,0]

                y_points = points_array[:,1]

            else:

                raise IndexError("'points_array' in 'spline_1D_interpo"+
                "lation' has shape "+str(points_array.shape)+". It sho"+
                "uld have one dimension with length of 2")
            
        # Tests if it is a list

        elif isinstance(points_array, list):

            # Tests which dimension has length of 2

            if len(points_array)==2:

                # Gets the individual dimensions

                x_points = points_array[0]

                y_points = points_array[1]

            else:

                # Gets the individual dimensions

                x_points = []

                y_points = []

                # Iterates though the points

                for point in points_array:

                    # Tests if this points has 2 dimensions

                    if len(point)!=2:

                        raise IndexError("'points_array' in 'spline_1D"+
                        "_interpolation' has the individual point "+str(
                        point)+"; it should have 2 elements: x, y, and"+
                        " z")
                    
                    x_points.append(point[0])

                    y_points.append(point[1])
                    
        else:

            raise TypeError("'points_array' in 'spline_1D_interpolatio"+
            "n' must be a numpy array or a list. It currently is: "+str(
            points_array))
        
    else:

        # Verifies if all dimensions are not None

        if x_points is None:

            raise TypeError("'x_points' is None in 'spline_1D_interpol"+
            "ation', but so is 'points_array'")
        
        elif y_points is None:

            raise TypeError("'y_points' is None in 'spline_1D_interpol"+
            "ation', but so is 'points_array'")
        
    # Gets the number of points

    n_points = len(x_points)

    if n_points==0:

        raise IndexError("No points were given for 'spline_1D_interpol"+
        "ation', x_points="+str(x_points))

    if len(y_points)!=n_points:

        raise IndexError("There are "+str(n_points)+" points in the x "+
        "direction; whereas there are "+str(len(y_points))+" points in"+
        " the y direction. They must have the same size in 'spline_1D_"+
        "interpolation'")
    
    # If the initial point is meant to be copied as the end point

    if add_initial_point_as_end_point:

        x_points.append(x_points[0])

        y_points.append(y_points[0])

        n_points += 1

    # Gets the spline interpolation of the y points

    cubic_spline_y = SplineInterpolation(interpolate.CubicSpline(
    x_points, y_points))

    return cubic_spline_y