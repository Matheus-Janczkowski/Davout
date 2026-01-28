# Routine to store some stochastic methods

import numpy as np

########################################################################
#                     Stochastic geometric regions                     #
########################################################################

# Defines a function to get a point on a surface of a n-dimensional eli-
# pse

def get_random_point_on_elipsoid_surface(limits, return_as_list=True):

    """
    Get a random point on the surface of a n-dimensional elipsoid

    limits: list of lists with the upper and lower limits of each 
    dimension. Example [[x_inf, x_sup], [y_inf, y_sup], [z_inf, z_sup]]

    return_as_list: return the point coordinates as a list if True; or
    as a numpy array otherwise. The default value is True
    """

    # Verifies if limits is a list

    if not isinstance(limits, list):

        raise TypeError("'limits' in 'get_random_point_on_elipsoid_sur"+
        "face' must be a list of lists, such as [[x_inf, x_sup], [y_in"+
        "f, y_sup], [z_inf, z_sup]]. Currently, it is:\n"+str(limits))
    
    # Creates a list with the centroid coordinates

    centroid = []

    # Creates a lsit with the directive vector of a line to intercept 
    # the ellipse surface. Uses normal to avoid clustering in equatorial
    # regions (take the example of a 2D ellipse, there are infinite 
    # points (0, y) which point to only two equatorial direction; but 
    # there is a single point at the direction (1,1) for instance. Thus,
    # it is much more probable to fall onto equatorial regions than on
    # vertices of the hypercube)

    direction_vector = np.random.normal(size=len(limits))

    direction_vector = ((1.0/np.linalg.norm(direction_vector))*
    direction_vector)

    # Iterates through the dimensions

    for i in range(len(limits)):

        dimension_limits = limits[i]

        # Verifies if dimension limits is a list

        if (not isinstance(dimension_limits, list)) or (len(
        dimension_limits)!=2):

            raise TypeError("Each component in the list 'limits' in 'g"+
            "et_random_point_on_elipsoid_surface' must be another list"+
            " of length 2, such as [x_inf, x_sup]. Currently, it is:\n"+
            str(dimension_limits))
        
        # Evaluates the centroid and the semiaxis

        centroid.append(0.5*(dimension_limits[0]+dimension_limits[1]))

        semiaxis = abs(0.5*(dimension_limits[1]-dimension_limits[0]))

        # Updates the direction vector according to the semiaxis

        direction_vector[i] = direction_vector[i]*semiaxis

    # Updates the direction vector by the length and adds the centroid

    direction_vector += np.array(centroid)

    # Returns a list if it is asked as so

    if return_as_list:

        return direction_vector.tolist()
    
    else:

        return direction_vector
    
########################################################################
#                                Testing                               #
########################################################################

if __name__=="__main__":

    factor = 1E-3

    limits = [[-1.0*factor, 1.0*factor], [2.0*factor, 3.5*factor], [-3.0, -2.0]]

    random_point = get_random_point_on_elipsoid_surface(limits)

    value = ((((random_point[0]-(0.5*(limits[0][0]+limits[0][1])))/(0.5*(
    limits[0][1]-limits[0][0])))**2)+(((random_point[1]-(0.5*(limits[1][
    0]+limits[1][1])))/(0.5*(limits[1][1]-limits[1][0])))**2)+(((
    random_point[2]-(0.5*(limits[2][0]+limits[2][1])))/(0.5*(limits[2][1
    ]-limits[2][0])))**2))

    print("The limits are:\n"+str(limits)+"\nThe random point is:\n"+str(
    random_point)+"\nf(x,y,z)="+str(value))