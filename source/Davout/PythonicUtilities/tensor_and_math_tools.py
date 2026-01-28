# Routine to store some functions for tensors

import numpy as np

from copy import deepcopy

########################################################################
#                       Useful and famous tensors                      #
########################################################################

# Defines a function to get the Kronecker's delta

def kroneckers_delta(i,j):

    if i==j:

        return 1.0
    
    else:

        return 0.0

# Defines a function to get the third order permutation tensor in compo-
# nents

def third_order_permutation_tensor_components(i,j,k):

    # Gets the set

    index_set = str(i)+str(j)+str(k)

    # Checks for positive component

    if index_set in ["123", "231", "312"]:

        return 1.0
    
    elif index_set in ["132", "213", "321"]:

        return -1.0
    
    else:

        return 0.0
    
# Defines a function to give the 3D rotation tensor as a function of a
# rotation pseudo-vector using Euler-Rodrigues formula

def tridimensional_rotation_tensor(pseudo_vector: np.ndarray):

    """
    pseudo_vector: a numpy vector 3 x 1. The direction of this vector is
    the rotation axle, whereas its magnitude is the rotation angle in 
    radians"""

    angle = np.linalg.norm(pseudo_vector)

    # Gets some trigonometric quantities to be used later

    c_1 = 1.0

    c_2 = 1.0

    c_3 = 0.5

    if angle>1E-5:

        c_1 = np.cos(angle)

        c_2 = np.sin(angle)/angle 

        c_3 = (1-c_1)/(angle**2)

    # Creates the rotation tensor with the identity bit and the dyadic
    # product

    R = (c_1*np.eye(3))+(c_3*np.outer(pseudo_vector, pseudo_vector))

    # Adds the skew bit

    R[0,1] -= c_2*pseudo_vector[2]

    R[1,0] = c_2*pseudo_vector[2]

    R[0,2] = c_2*pseudo_vector[1]

    R[2,0] -= c_2*pseudo_vector[1]

    R[1,2] -= c_2*pseudo_vector[0]

    R[2,1] = c_2*pseudo_vector[0]

    # Returns the rotation tensor

    return R

########################################################################
#                          Finite differences                          #
########################################################################

# Defines a function to evaluate scalar derivatives with respect to vec-
# tor-valued arguments using central finite differences

def central_finite_differences(function, vector_argument, h=1E-5):

    # Gets the number of variables

    n_variables = None

    if isinstance(vector_argument, list):

        n_variables = len(vector_argument)

    else:

        n_variables = vector_argument.shape[0]

    # Initializes the vector of derivatives

    derivative = np.zeros(n_variables)

    # Iterates through the variables

    for i in range(n_variables):

        # Gets the input ahead

        input_ahead = deepcopy(vector_argument)

        input_ahead[i] += h 

        # Gets the input abaft

        input_abaft = deepcopy(vector_argument)

        input_abaft[i] -= h

        # Gets the function evaluations

        function_ahead = function(input_ahead)

        function_abaft = function(input_abaft)

        # Gets the derivative's approximation

        derivative[i] = (function_ahead-function_abaft)/(2*h)

    return derivative