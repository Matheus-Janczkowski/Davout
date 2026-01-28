# Routine to store functions for numerical analysis and numerical work-
# arounds

from dolfin import *

import ufl_legacy as ufl

import numpy as np

import scipy as sp

from ...PythonicUtilities import programming_tools

########################################################################
#                     Parametric loading functions                     #
########################################################################

# Defines a class with the loading curves

class LoadingCurves:

    def __init__(self, verify_curveNameExistence):
    
        self.verify_curveNameExistence = verify_curveNameExistence

    # Defines a linear function

    def linear(self, additional_parameters):
        
        if self.verify_curveNameExistence:

            return True
        
        # Check out if additional parameters have been been given

        default_parameters = check_additionalParameters(
        additional_parameters, {"end_point": [1.0, 1.0], "starting_poi"+
        "nt": [0.0, 0.0]})

        a1 = ((default_parameters["end_point"][1]-default_parameters[
        "starting_point"][1])/(default_parameters["end_point"][0]-
        default_parameters["starting_point"][0]))

        a0 = (default_parameters["starting_point"][1]-(a1*
        default_parameters["starting_point"][1]))

        return lambda x: (a1*x)+a0
    
    # Defines a function for a load curve equal to the square root

    def square_root(self, additional_parameters):

        if self.verify_curveNameExistence:

            return True
        
        # Check out if additional parameters have been been given

        default_parameters = check_additionalParameters(
        additional_parameters, {"end_point": [1.0, 1.0]})

        a1 = ((default_parameters["end_point"][1]**2)/
        default_parameters["end_point"][0])

        return lambda x: ufl.sqrt(x)

# Defines a function to give different options of parametric loading 
# curves using ufl functions (so that they can be used in the variatio-
# nal ecosystem). These loading functions must spit out numbers between
# 0 and 1

@programming_tools.optional_argumentsInitializer({('additional_paramet'+
'ers'): lambda: dict()})

def generate_loadingParametricCurves(curve_name, additional_parameters=
None, verify_curveNameExistence=False):
    
    # The flag verify_curveNameExistence is True when the interest is 
    # just to point out if the curve name is in the scope of the imple-
    # mented functions 

    # Gets a dictionary of the methods of the class of loads

    loads_class = LoadingCurves(verify_curveNameExistence)

    loads_dictionary = programming_tools.get_attribute(loads_class, 
    None, None, dictionary_of_methods=True, delete_init_key=True)

    # Tests if the curve name is one of the available methods

    if curve_name in loads_dictionary:

        return loads_dictionary[curve_name](additional_parameters)

    else:

        if verify_curveNameExistence:

            return False
        
        else:

            raise NameError("The parametric load curve '"+str(curve_name
            )+"' has not yet been implemented. Check out the available"+
            " parametric load curves: "+str(list(loads_dictionary.keys())))
        
# Defines a function to check additional parameters to each loading cur-
# ve

def check_additionalParameters(additional_parameters, default_parameters):

    if additional_parameters is None:

        return default_parameters 
    
    elif not isinstance(additional_parameters, dict):

        raise TypeError("The additional_parameters must be a dictionar"+
        "y to get the simple generators of load curves. Whereas it cur"+
        "rently is: "+str(additional_parameters))
    
    else:

        # Iterates through the dictionary of default parameters

        for name in additional_parameters:

            if name in default_parameters:

                # Checks if they have the same type

                if not (type(default_parameters[name])==type(
                additional_parameters[name])):
                    
                    raise TypeError("The '"+str(name)+"' additional pa"+
                    "rameter to create a loading curve has not the sam"+
                    "e type as of the default one. Look at the default"+
                    " parameter: "+str(default_parameters[name])+"\nan"+
                    "d the given parameter: "+str(additional_parameters[
                    name]))

                default_parameters[name] = additional_parameters[name]

            else:

                raise KeyError("The dictionary of additional parameter"+
                "s to get a simple generator of load curves has the ke"+
                "y '"+str(name)+"', but this key is not a valid additi"+
                "onal information. Check out the valid ones and their "+
                "respective default values: "+str(default_parameters))
            
        return default_parameters

########################################################################
#  Safe numerical operations to avoid lack of differentiability or di- #
#                            vision by zero                            #
########################################################################

# Defines a "safe" square root function, in order to avoid division by 
# zero in the variational form

def safe_sqrt(a):

    return ufl.sqrt(a+1.0e-15)

# Defines a function to give the closest value of a list to a given 
# float number

def closest_numberInList(number, list_ofNumbers):

    return min(list_ofNumbers, key=lambda x: abs(x-number))

# Defines a function to convert degrees to radians

def degrees_to_radians(number):

    return (number/180.0)*np.pi

# Defines a function to construct the rotation matrix numerically (with-
# out any ufl function)

def rotation_tensorEulerRodrigues(phi, return_numpy_array=False):

    if not isinstance(phi, list):

        raise TypeError("The numerical rotation_tensorEulerRodrigues w"+
        "as chosen and the vector 'phi' is not a list")
    
    elif isinstance(phi, np.ndarray):

        phi = phi.tolist()

    # Evaluates the rotation angle

    rotation_angle = np.sqrt((phi[0]**2)+(phi[1]**2)+(phi[2]**2))

    # Evaluates the skew tensor and the tensor given by the tensor pro-
    # duct of the axial vector by itself

    W = [[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0
    ], 0.0]]

    axial_tensorAxial = [[phi[0]**2, phi[0]*phi[1], phi[0]*phi[2]], [
    phi[1]*phi[0], phi[1]**2, phi[1]*phi[2]], [phi[2]*phi[0], phi[2]*phi[
    1], phi[2]**2]]

    # Evaluates the coefficients

    c1 = np.cos(rotation_angle)

    c2 = 1.0

    c3 = 0.5

    if rotation_angle>DOLFIN_EPS:

        c2 = np.sin(rotation_angle)/rotation_angle

        c3 = (1-np.cos(rotation_angle))/(rotation_angle**2)

    # Evaluates the Euler-Rodrigues formula

    R_bar = [[c1, 0.0, 0.0], [0.0, c1, 0.0], [0.0, 0.0, c1]]

    for i in range(3):

        for j in range(3):

            R_bar[i][j] += (c2*W[i][j])+(c3*axial_tensorAxial[i][j])

    # If it is asked to return a numpy array

    if return_numpy_array:

        return np.array(R_bar)
    
    else:

        return R_bar

# Defines a function to get the axial vector corresponding to a rotation
# tensor 

def get_axle_from_rotation_tensor(R, return_numpy_array=False):

    # Verifies if a list

    if isinstance(R, list):

        # Converts to a numpy array

        R = np.array(R)

    # Verifies if it is not a numpy array

    elif not isinstance(R, np.ndarray):

        raise TypeError("The axle vector of a rotation tensor was aske"+
        "d, but it is not a list nor a numpy array. The given rotation"+
        " tensor is: "+str(R))

    # As the rotation tensor is the exponential of a skew-symmetric ten-
    # sor whose axial vector is the rotation axle, we take the logarithm

    W = sp.linalg.logm(R)

    # Gets the axial vector

    axial_vector = [W[2,1], W[0,2], W[1,0]]

    if return_numpy_array:

        return np.array(axial_vector)
    
    else:

        return axial_vector