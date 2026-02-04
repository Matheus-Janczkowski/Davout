# Routine to store some stochastic methods

import numpy as np

########################################################################
#                     Stochastic geometric regions                     #
########################################################################

# Defines a function to get a point on a surface of a n-dimensional eli-
# pse

def get_random_point_on_elipsoid_surface(limits, return_as_list=True,
p_norm_value=2):

    """
    Get a random point on the surface of a n-dimensional elipsoid

    limits: list of lists with the upper and lower limits of each 
    dimension. Example [[x_inf, x_sup], [y_inf, y_sup], [z_inf, z_sup]]

    return_as_list: return the point coordinates as a list if True; or
    as a numpy array otherwise. The default value is True

    p_norm_value: p value of the L_p norm. The default value is 2 
    (ellipse in the euclidean space)
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
    # vertices of the hypercube). 
    #
    # This approach, however, is modified for p norms different than 2,
    # as in Barthe et al. (2005), A probabilistic approach to the geo-
    # metry of the Lp-ball

    direction_vector = None

    if abs(p_norm_value-2)<1E-5:
        
        direction_vector = np.random.normal(size=len(limits))

    else:

        # Gets the amplitude 

        amplitude = np.random.gamma(shape=1.0/p_norm_value, scale=1.0,
        size=len(limits))

        # Gets signs

        signs = np.random.choice([-1, 1], size=len(limits))

        # Multiplies everything

        direction_vector = signs*(amplitude**(1.0/p_norm_value))

    direction_vector = ((1.0/np.linalg.norm(direction_vector, ord=
    p_norm_value))*direction_vector)

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

    p_norm_value = 0.8

    factor = 1E-3

    limits = [[-1.0*factor, 1.0*factor], [2.0*factor, 3.5*factor], [-3.0, -2.0]]

    random_point = get_random_point_on_elipsoid_surface(limits, 
    p_norm_value=p_norm_value)

    value = ((abs((random_point[0]-(0.5*(limits[0][0]+limits[0][1])))/(
    0.5*(limits[0][1]-limits[0][0])))**p_norm_value)+(abs((random_point[
    1]-(0.5*(limits[1][0]+limits[1][1])))/(0.5*(limits[1][1]-limits[1][0
    ])))**p_norm_value)+(abs((random_point[2]-(0.5*(limits[2][0]+limits[
    2][1])))/(0.5*(limits[2][1]-limits[2][0])))**p_norm_value))

    print("The limits are:\n"+str(limits)+"\nThe random point is:\n"+str(
    random_point)+"\nf(x,y,z)="+str(value))

    from ..PythonicUtilities.plotting_tools import plane_plot

    limits = [[-3.0, 1.0], [-0.5, 1.5]]

    n_ellipses = 7

    n_points = 300

    x_data = []

    y_data = []

    p_min = 0.25

    labels = ["p = "+str(p_min*(2**j)) for j in range(n_ellipses)]

    for j in range(n_ellipses):

        x_data.append([])

        y_data.append([])

        for i in range(n_points):

            point = get_random_point_on_elipsoid_surface(limits, 
            p_norm_value=float(labels[j][4:len(labels[j])]))

            x_data[-1].append(point[0])

            y_data[-1].append(point[1])

    plane_plot("2D_ellipse", x_data=x_data, y_data=y_data, plot_type="scatter",
    element_size=2.5, label=labels, title="$E^{p}\\left(\mathbf{D},\mathbf{x}_{c}\\right)$",
    color_map="coolwarm", aspect_ratio=1.0)