# Routine to store some tests to evaluate convex input neural networks

import unittest

import numpy as np

from ...tool_box import math_tools

from ....PythonicUtilities.tensor_and_math_tools import central_finite_differences

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        # Defines the reference points

        self.reference_points = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 
        0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])

        translation = np.array([10.0, 15.0, -2.0])

        deformed_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [
        -1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

        randomization_factor = 1.0

        for i in range(deformed_points.shape[0]):

            # Adds the translation and a random movement

            deformed_points[i,:] += translation+(randomization_factor*
            np.random.randn(3))

        self.displacement_points = deformed_points-self.reference_points

    # Defines a function to test the derivative of the objective func-
    # tion used to get the rigid body motion

    def test_derivative_rigid_body_optimization_problem(self):

        print("\n#####################################################"+
        "###################\n#    Tests the derivative of the optimiz"+
        "ation of rigid body motion     #\n###########################"+
        "#############################################\n")

        # Gets the analytical derivative

        rotation_pseudovector = [1.0, 0.5, -0.5]

        average_displacement = np.mean(self.displacement_points, axis=0)

        translated_deformed_points = (self.reference_points+
        self.displacement_points)

        for i in range(translated_deformed_points.shape[0]):

            translated_deformed_points[i,:] -= average_displacement

        analytical_derivative = math_tools.derivative_optimization_rigid_body(
        translated_deformed_points, self.reference_points, 
        rotation_pseudovector)

        f = lambda x: math_tools.objective_optimization_rigid_body(
        translated_deformed_points, self.reference_points, x)

        numerical_derivative = central_finite_differences(f, 
        rotation_pseudovector)

        print("The analytical derivative is: "+str(analytical_derivative
        ))

        print("\nThe numerical derivative is: "+str(numerical_derivative
        ))

    # Defines a function to test the identifier of rigid body motion

    def test_rigid_body_motion_identifier(self):

        print("\n#####################################################"+
        "###################\n#               Tests the identifier of "+
        "rigid body motion              #\n###########################"+
        "#############################################\n")

        # Evaluates the translation, rotation, and the pseudo-vector

        translation, rotation, pseudovector = math_tools.get_rigid_body_motion(
        self.displacement_points, self.reference_points, verbose=True)

        print("The calculated translation is: "+str(translation))

        print("\nThe calculated rotation tensor is: "+str(rotation))

        print("\nThe calculated rotation pseudo-vector is: "+str(
        pseudovector))

        print("\nThe rotation angle in degrees is: "+str((180/np.pi)*
        np.linalg.norm(pseudovector)))

# Runs all tests

if __name__=="__main__":

    unittest.main()