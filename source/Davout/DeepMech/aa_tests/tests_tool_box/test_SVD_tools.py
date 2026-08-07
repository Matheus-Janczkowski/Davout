# Routine to test custom gradients

import tensorflow as tf

import numpy as np

from .....Davout.PythonicUtilities.testing_tools import run_class_of_tests

from .....Davout.DeepMech.tool_box import SVD_tools 

from .....Davout.MultiMech.tool_box.numerical_tools import derivative_scalar_valued_function_of_vector_argument

# Defines a function to test the inclusion of custom gradients into 
# tensorflow functions

class TestCustomGradients:

    def __init__(self):

        self.verbose = True

        self.parameters_dtype = "float64"

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

        self.dimensionality = 10

        self.rank = 5

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

        # Defines the tensor of batched input vectors X [dimensionality,
        # n_samples] for the operation 
        # y = I(rank,dimensionality)*H_1*H_2*...*H_rank*X 
        #
        # such that y is a tensor [rank, n_samples]

        self.n_samples = 5

        self.X_input_vectors = tf.constant(np.random.randn((
        self.dimensionality, self.n_samples)), dtype=tf.as_dtype(
        self.parameters_dtype))

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

    # Defines a function to test the analytical derivatives of the nor-
    # malization factor and of the first non-zero component of the 
    # Householder vector

    def test_derivative_of_normalization_factor(self):

        print("\n#####################################################"+
        "###################\n# Tests derivative of normalization fact"+
        "or and first component, v_tilde  #\n#########################"+
        "###############################################\n")

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

            # Gets the derivative of v_tilde and of alpha

            derivative_v_tilde, derivative_alpha = SVD_tools.evaluate_derivative_of_v_tilde_and_alpha(
            v_bar, alpha, self.householder_indices[i][1], vector_of_dofs, 
            unnormalized_first_component, tf.as_dtype(
            self.parameters_dtype)) 

            # Gets the first and last indices of the Householder vector, 
            # then the vector that will become the Householder vector

            initial_index, length, number_of_leading_zeros = (
            self.householder_indices[i])

            slice_of_vector_of_dofs = tf.slice(
            self.householder_parameters, [initial_index], [length]
            ).numpy()

            # Evaluates the derivatives of v_tilde using central finite
            # differences

            def get_v_tilde(slice_of_vector_of_dofs):

                slice_of_dofs_tensor = tf.constant(
                slice_of_vector_of_dofs, dtype=tf.as_dtype(
                self.parameters_dtype))

                # Computes the average of the raw vector
                
                average_raw_vector = tf.reduce_mean(
                slice_of_dofs_tensor)
            
                # Calculates the non-free component of the Householder 
                # vector to the first position
            
                unnormalized_first_component = tf.sqrt((
                average_raw_vector*average_raw_vector)+
                self.householder_epsilon_squared)
            
                appended_raw_vector = tf.concat([
                unnormalized_first_component[None], 
                slice_of_dofs_tensor], axis=0)
            
                # Rescales the raw vector to have unit norm
            
                alpha = tf.math.rsqrt(tf.reduce_sum(tf.square(
                appended_raw_vector)))
            
                return alpha*unnormalized_first_component
            
            CFD_v_tilde = derivative_scalar_valued_function_of_vector_argument(
            get_v_tilde, slice_of_vector_of_dofs, epsilon=1E-5)

            # Evaluates the derivatives of alpha using central finite
            # differences

            def get_alpha(slice_of_vector_of_dofs):

                slice_of_dofs_tensor = tf.constant(
                slice_of_vector_of_dofs, dtype=tf.as_dtype(
                self.parameters_dtype))

                # Computes the average of the raw vector
                
                average_raw_vector = tf.reduce_mean(
                slice_of_dofs_tensor)
            
                # Concatenates the non-free component of the Householder 
                # vector to the first position
            
                unnormalized_first_component = tf.sqrt((
                average_raw_vector*average_raw_vector)+
                self.householder_epsilon_squared)
            
                appended_raw_vector = tf.concat([
                unnormalized_first_component[None], 
                slice_of_dofs_tensor], axis=0)
            
                # Rescales the raw vector to have unit norm
            
                return tf.math.rsqrt(tf.reduce_sum(tf.square(
                appended_raw_vector)))
            
            CFD_alpha = derivative_scalar_valued_function_of_vector_argument(
            get_alpha, slice_of_vector_of_dofs, epsilon=1E-5)

            # Compacts the analytical and the numerical gradients for 
            # each derivative

            comparison_derivative_v_tilde = np.vstack([
            derivative_v_tilde.numpy(), CFD_v_tilde])

            comparison_derivative_alpha = np.vstack([
            derivative_alpha.numpy(), CFD_alpha])

            print("\nThe derivative of v_tilde with respect to the DOF"+
            "s of the "+str(i+1)+"-th Householder vector is:\n"+str(
            comparison_derivative_v_tilde)+"\nThe maximum difference b"+
            "etween components is: "+str(tf.reduce_max(tf.abs(
            derivative_v_tilde-tf.constant(CFD_v_tilde, dtype=
            tf.as_dtype(self.parameters_dtype)))).numpy())+"\n")

            print("\nThe derivative of alpha with respect to the DOFs "+
            "of the "+str(i+1)+"-th Householder vector is:\n"+str(
            comparison_derivative_alpha)+"\nThe maximum difference bet"+
            "ween components is: "+str(tf.reduce_max(tf.abs(
            derivative_alpha-tf.constant(CFD_alpha, dtype=tf.as_dtype(
            self.parameters_dtype)))).numpy())+"\n")

    # Defines a function to test the evaluation of the analytical deri-
    # vative of the following operation
    # 
    # y = QX, such that (Q^T)*Q = I and Q = H1*H*...*H_rank
    #
    # with respect to the Householder vector v^i of the i-th Householder
    # reflector H_i

    def test_derivative_of_application_of_CHR(self):

        print("\n#####################################################"+
        "###################\n#  Tests derivative of application of ch"+
        "ain of Householder reflectors  #\n###########################"+
        "#############################################\n")

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