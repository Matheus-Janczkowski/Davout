# Routine to store some mathematical tools

import numpy as np

from scipy.optimize import minimize

from ...PythonicUtilities.tensor_and_math_tools import kroneckers_delta as delta

from ...PythonicUtilities.tensor_and_math_tools import third_order_permutation_tensor_components as epsilon

from ...PythonicUtilities.tensor_and_math_tools import tridimensional_rotation_tensor as R_tensor

########################################################################
#                 Identification of rigid body motion                  #
########################################################################

# Defines a function to get the rigid body motion (translation and rota-
# tion that minimizes the deformation of a set of points)

def get_rigid_body_motion(displacement_points: np.ndarray, 
reference_points: np.ndarray, optimization_method="CG", 
gradient_tolerance=1E-4, n_max_iterations=1000, verbose=False):
    
    """
    displacement_points: a numpy array n x 3, with the displacement 
    vector for each one of the n points

    reference_points: a numpy array n x 3, with the coordinates of the 
    corresponding points in the reference configuration"""

    # Gets an estimate of the rotation pseudo-vector as an average of 
    # the cross-product of the reference configuration point position to
    # the displacement vector

    rotation_pseudovector = np.zeros(3)

    # Evaluates the average displacement

    average_displacement = np.mean(displacement_points, axis=0)

    # Uses the average displacement as the translation, and translates 
    # the points of the deformed configuration back to a superposition 
    # onto the reference configuration

    translated_deformed_points = displacement_points+reference_points

    for i in range(displacement_points.shape[0]):

        translated_deformed_points[i,:] -= average_displacement

        local_rotation_axle = np.cross(reference_points[i,:],
        displacement_points[i,:]-average_displacement)

        local_rotation_axle = ((1/np.linalg.norm(local_rotation_axle))*
        local_rotation_axle)

        # Gets the estimation of the angle

        estimated_angle = np.arcsin(1.0/max(np.linalg.norm(
        displacement_points[i,:]-average_displacement)*np.linalg.norm(
        reference_points[i,:]), 1E-5))

        rotation_pseudovector += estimated_angle*local_rotation_axle

    # Divides the rotation estimation by the number of points to get the
    # average

    rotation_pseudovector = (1/displacement_points.shape[0]
    )*rotation_pseudovector

    rotation_pseudovector = np.array([0.0, 0.0, 1.0])

    if verbose:

        print("\nThe initial estimation for the rotation pseudo-vector"+
        " is: "+str(rotation_pseudovector)+"\n")

    # Sets the objective function as a function of the rotation pseudo-
    # vector

    objective_function = lambda phi: objective_optimization_rigid_body(
    translated_deformed_points, reference_points, phi)

    # Does the same for the gradient

    objective_gradient = lambda phi: derivative_optimization_rigid_body(
    translated_deformed_points, reference_points, phi)

    # Sets the minimization problem using the minimize class from scipy

    minimization_problem = minimize(objective_function, 
    rotation_pseudovector, method=optimization_method, tol=
    gradient_tolerance, options={"maxiter": 
    n_max_iterations, "disp": verbose})

    # Gets the optimized rotation pseudo-vector

    rotation_pseudovector = minimization_problem.x

    if verbose:

        print("\nThe norm of the difference between the rigid body mot"+
        "ion and the original\ncomplete motion is: "+str(
        objective_function(rotation_pseudovector))+"\nThe norm of the "+
        "gradient of the objective function at the end of the\noptimiz"+
        "ation problem is: "+str(np.linalg.norm(objective_gradient(
        rotation_pseudovector)))+"\n")

    # Returns the translation motion, the rotation tensor, and the rota-
    # tion tensor

    return (average_displacement, R_tensor(rotation_pseudovector),
    rotation_pseudovector)

# Defines a function to get the objective function of the identification
# of a rigid body kinematics approximation

def objective_optimization_rigid_body(translated_deformed_points: 
np.ndarray, reference_points: np.ndarray, rotation_pseudovector: 
np.ndarray):
    
    """
    translated_deformed_points: a numpy array n x 3, where n is the 
    number of points to be rotated. The rows of this matrix, hence, are
    the coordinates of the deformed point translated already to the re-
    ference configuration

    reference_points: a numpy array n x 3, with the coordinates of the 
    corresponding points in the reference configuration
    
    rotation_pseudovector: a numpy vector 3 x 1, which gives the axis 
    and magnitude of the rigid body rotation to be evaluated"""

    # Gets the rotation tensor

    R = R_tensor(rotation_pseudovector)

    # Initializes the objective function, which is the norm of the defor-
    # med motion

    deformed_motion_norm = 0.0

    # Iterates through the points' position

    for i in range(translated_deformed_points.shape[0]):

        # Evaluates the distance from the translated deformed vector to
        # the rotation of the reference vector

        distance = (translated_deformed_points[i,:]-np.matmul(R, 
        reference_points[i,:]))

        # Evaluates the norm of this vector and adds it to the objective
        # function
        
        deformed_motion_norm += np.linalg.norm(distance)

    return 0.5*deformed_motion_norm 

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

    return derivative_vector