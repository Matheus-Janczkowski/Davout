# Routine to test custom gradients

import tensorflow as tf

import numpy as np

from ....PythonicUtilities.testing_tools import run_class_of_tests

from .....Davout.DeepMech.tool_box import SVD_tools 

# Defines a function to test the inclusion of custom gradients into 
# tensorflow functions

class TestCustomGradients:

    def __init__(self):

        self.verbose = True

        self.parameters_dtype = "float32"

        # Creates three variables

        self.x_variable = tf.Variable(2.0, dtype=tf.as_dtype(
        self.parameters_dtype))

        self.y_variable = tf.Variable(1.0, dtype=tf.as_dtype(
        self.parameters_dtype))

        self.z_variable = tf.Variable(3.0, dtype=tf.as_dtype(
        self.parameters_dtype))

        # Sets the dimensionality and the rank of the left-orthogonal
        # matrix to be created using a chain of Householder reflectors
        # (CHR)

        self.dimensionality = 50

        self.rank = 10

        # Counts the number of degrees of freedom (DOFs) of the left-or-
        # thogonal matrix to be created using CHR

        self.n_dofs = int(0.5*(self.rank*((2*self.dimensionality)-
        self.rank-1)))

        # Creates a list of tuples of the indices to recreate the vectors
        # of DOFs of the Householder vectors

        self.householder_indices = []

        parameters_counter = 0

        for i in range(min(self.rank, self.dimensionality-1)):

            # Appends the index of the first parameter for the corres-
            # ponding Householder vector; the number of parameters ne-
            # cessary for this vector, and the number of leading zeros.
            # Starts with the last Householder reflector since it will
            # be the first to multiply any vector to the right of the 
            # corresponding orthogonal matrix

            self.householder_indices.append(tuple([parameters_counter,
            self.dimensionality-i-1, i]))

        # Creates a variable with the Householder DOFs

        self.householder_parameters = tf.Variable(np.random.randn((
        self.n_dofs)), dtype=tf.as_dtype(self.parameters_dtype))

        # Defines the Householder epsilon for the regularization of the
        # first non-zero component of each Householder vector

        self.householder_epsilon_squared = 1.0

    # Defines a function to test the evalaution of a function with two
    # variables with custom gradient defined only for one

    def test_partial_custom_gradient(self):

        print("\n#####################################################"+
        "###################\n#                    Tests partial custo"+
        "m gradient                     #\n###########################"+
        "#############################################\n")

        # Defines a function to compute the function f(x,y,z) = (x^2)*
        # exp(y)*z

        @tf.custom_gradient
        def test_function(x, y, z):

            # Evaluates the forward pass

            forward_result = x*x*tf.math.exp(y)*z

            # Evaluates the gradient with respect to x and y only

            def gradient(upstream_gradient):

                return (upstream_gradient*2*x*tf.math.exp(y)*z, 
                upstream_gradient*x*x*tf.math.exp(y)*z, None)

            # Returns the forward result and a function for the gradients 

            return forward_result, gradient

        # Tests the forward pass

        forward_pass_result = test_function(self.x_variable, 
        self.y_variable, self.z_variable)

        print("The forward pass for x="+str(self.x_variable.numpy())+
        ", y="+str(self.y_variable.numpy())+", and z="+str(
        self.z_variable.numpy())+" is: "+str(forward_pass_result.numpy()
        )+"\n")

        # Computes the gradient using automatic differentiation

        with tf.GradientTape() as tape:

            forward_pass_result = test_function(self.x_variable, 
            self.y_variable, self.z_variable)

        gradient_x, gradient_y, gradient_z = tape.gradient(
        forward_pass_result, [self.x_variable, self.y_variable, 
        self.z_variable])

        print("The gradient computed using automatic differentiation i"+
        "s:\ndf/dx = "+str(gradient_x.numpy())+"\ndf/dy = "+str(
        gradient_y.numpy())+"\ndf/dz = "+str(gradient_z)+"\n")

    # Defines a function to test the evalaution of a function with two
    # variables with custom gradient defined only for one

    def test_derivative_of_normalization_factor(self):

        print("\n#####################################################"+
        "###################\n# Tests derivative of normalization fact"+
        "or and first component, v_bar  #\n###########################"+
        "#############################################\n")

        # Defines the number of Householder vectors to check the deriva-
        # tives

        number_of_householder_vectors_to_check = 5

        number_of_householder_vectors_to_check = min(self.rank, 
        number_of_householder_vectors_to_check)

        # Iterates over the Householder vectors to check

        for i in range(number_of_householder_vectors_to_check):

            # Gets the Householder vector

            (householder_vector, v_bar, alpha, vector_of_dofs,
            unnormalized_first_component) = SVD_tools.get_householder_vector_from_parameters(
            self.householder_indices, self.householder_parameters, i,
            self.householder_epsilon_squared)

            # Gets the derivative of v_bar and of alpha

            derivative_v_bar, derivative_alpha = SVD_tools.evaluate_derivative_of_v_bar_and_alpha(
            v_bar, alpha, self.householder_indices[i][1], vector_of_dofs, 
            unnormalized_first_component, tf.as_dtype(
            self.parameters_dtype)) 

# Runs all tests

if __name__=="__main__":

    # Instantiates the class with the methods to be tested

    class_of_tests = TestCustomGradients()

    # Creates a list of methods (using their names) that are not to be
    # tested

    reserved_methods = []

    # Calls the function to run all the necessary tests

    run_class_of_tests(class_of_tests, reserved_methods=reserved_methods,
    sort_methods_alphabetically=False)