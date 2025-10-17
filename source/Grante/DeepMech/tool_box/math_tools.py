# Routine to store some mathematical tools

import numpy as np

from ...PythonicUtilities.tensor_tools import kroneckers_delta as delta

from ...PythonicUtilities.tensor_tools import third_order_permutation_tensor_components as epsilon

########################################################################
#                 Identification of rigid body motion                  #
########################################################################

# Defines a function to get the derivative of the objective function 
# with respect to the rotation pseudo-vector

def derivative_optimization_rigid_body(translated_deformed_points: 
np.ndarray, reference_points: np.ndarray, rotation_pseudovector: 
np.ndarray):
    
    """
    translated_deformed_points: a numpy array n x 3, where n is the 
    number of points to be rotated. The rows of this matrix, hence, are
    the coordinates of the deformed point translated already to the re-
    ference configuration

    reference_point: a numpy array n x 3, with the coordinates of the 
    corresponding points in the reference configuration
    
    rotation_pseudovector: a numpy vector 3 x 1, which gives the axis 
    and magnitude of the rigid body rotation to be evaluated"""
    
    # Initializes the derivative vector

    derivative_vector = np.zeros(3)

    # Gets the current angle of rotation

    angle = np.linalg.norm(rotation_pseudovector)

    # Gets some trigonometric quantities to be used later

    c_1 = 1.0

    c_2 = -0.5

    c_3 = -0.125

    c_4 = 0.5

    if angle>1E-5:

        sin_angle = np.sin(angle)

        cos_angle = np.cos(angle)

        c_1 = sin_angle/angle 

        c_2 = (((cos_angle*angle)-sin_angle)/(angle**3))

        c_3 = (((sin_angle*angle)+(2.0*cos_angle)-2.0)/(angle**4))

        c_4 = ((1.0-cos_angle)/(angle**2))

    # Iterates through the elements of the derivative vector

    for l in range(3):

        # Iterates through the indices of the translated deformed points

        for j in range(3):

            # Iterates through the indices of the reference points

            for k in range(3):

                # Evaluates the derivative of the rotation tensor w.r.t.
                # the pseudo-vector l-th component

                dR_jk_dPhi_l = (-(c_1*((rotation_pseudovector[l]*delta(j,
                k))+epsilon(j,k,l)))+(c_3*rotation_pseudovector[l]*
                rotation_pseudovector[j]*rotation_pseudovector[k])+(c_4*
                ((rotation_pseudovector[k]*delta(j,l))+(delta(k,l)*
                rotation_pseudovector[j]))))

                # Adds te term corresponding to the skew tensor

                skew_term = 0.0

                for m in range(3):

                    skew_term += (rotation_pseudovector[l]*epsilon(j,k,m
                    )*rotation_pseudovector[m])

                dR_jk_dPhi_l -= c_2*skew_term

                # Iterates though the points

                for i in range(reference_points.shape[0]):

                    # Mutliplies the vectors by the derivative of the  
                    # rotation tensor

                    derivative_vector[l] -= (translated_deformed_points[
                    i,j]*dR_jk_dPhi_l*reference_points[i,k])

    # Multiplies the vector of derivatives by 2 and returns it

    return 2*derivative_vector