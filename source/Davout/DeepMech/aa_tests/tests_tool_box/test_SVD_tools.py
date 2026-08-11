# Routine to test custom gradients

import tensorflow as tf

import numpy as np

from .....Davout.PythonicUtilities.testing_tools import run_class_of_tests

from .....Davout.DeepMech.tool_box import SVD_tools 

from .....Davout.MultiMech.tool_box.numerical_tools import derivative_scalar_valued_function_of_vector_argument, derivative_tensor_valued_function_of_vector_argument

# Defines a function to test the inclusion of custom gradients into 
# tensorflow functions

class TestCustomGradients:

    def __init__(self):

        self.verbose = True

        self.parameters_dtype = "float64"

        self.integer_type = "int32"

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

        self.householder_first_index = []

        self.householder_length = []

        self.householder_number_of_leading_zeros = []

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

            # Appends to the simple flat tensors

            self.householder_first_index.append(parameters_counter)

            self.householder_length.append(self.dimensionality-i-1)

            self.householder_number_of_leading_zeros.append(i)

            # Updates the parameter counter

            parameters_counter += self.dimensionality-i-1

        # Converts the flat vectors to flat tensors

        self.householder_first_index = tf.constant(
        self.householder_first_index, dtype=tf.as_dtype(
        self.integer_type))

        self.householder_length = tf.constant(self.householder_length, 
        dtype=tf.as_dtype(self.integer_type))

        self.householder_number_of_leading_zeros = tf.constant(
        self.householder_number_of_leading_zeros, dtype=tf.as_dtype(
        self.integer_type))

        # Creates a variable with the Householder DOFs

        self.householder_parameters = tf.Variable(np.random.randn((
        self.n_dofs)), dtype=tf.as_dtype(self.parameters_dtype))

        # Defines the Householder epsilon for the regularization of the
        # first non-zero component of each Householder vector

        self.householder_epsilon_squared = tf.constant(1.0, dtype=
        tf.as_dtype(self.parameters_dtype))

        # Defines the number of Householder vectors to check the deriva-
        # tives

        self.number_of_householder_vectors_to_check = 5

        self.number_of_householder_vectors_to_check = min(self.rank, 
        self.number_of_householder_vectors_to_check, self.dimensionality
        -1)

        # Defines the tensor of batched input vectors X [dimensionality,
        # n_samples] for the operation 
        # y = I(rank,dimensionality)*H_1*H_2*...*H_rank*X 
        #
        # such that y is a tensor [n_samples, rank]

        self.n_samples = 5

        self.X_input_vectors = tf.constant(np.random.randn(
        self.n_samples, self.dimensionality), dtype=tf.as_dtype(
        self.parameters_dtype))

        # Saves the constant two

        self.constant_two = tf.constant(2.0, dtype=tf.as_dtype(
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

    # Defines a function to test getting and assembling a Householder 
    # vector from a flat vector of DOFs to construct a Householder chain

    def test_householder_vector_assembly(self):

        print("\n#####################################################"+
        "###################\n#            Tests assembling a set of H"+
        "ouseholder vectors             #\n###########################"+
        "#############################################\n")

        # Iterates over the Householder vectors to check

        for i in range(self.number_of_householder_vectors_to_check):

            # Converts the Householder index to a tensorflow integer

            householder_index = tf.constant(i, dtype=tf.as_dtype(
            self.integer_type))

            # Gets the Householder vector

            (householder_vector, v_bar, alpha, vector_of_dofs,
            unnormalized_first_component, local_n_dofs) = SVD_tools.get_householder_vector_from_parameters(
            self.householder_first_index, self.householder_length,
            self.householder_number_of_leading_zeros, 
            self.householder_parameters, householder_index, 
            self.householder_epsilon_squared)

            print("The assembled "+str(i)+"-th Householder vector is:"+
            "\n"+str(householder_vector.numpy())+"\n")

    # Defines a function to test multiplying an input tensor [n_samples,
    # dimensionality] by a Householder chain

    def test_householder_chain_multiplication(self):

        print("\n#####################################################"+
        "###################\n#       Tests multiplying a Householder "+
        "chain to an input tensor       #\n###########################"+
        "#############################################\n")

        y_tensor = SVD_tools.multiply_input_vector_by_householder_chain(
        self.X_input_vectors, self.householder_first_index,
        self.householder_length, 
        self.householder_number_of_leading_zeros, 
        self.householder_parameters, self.constant_two, 
        self.householder_epsilon_squared)

        print("The result of the input tensor multiplied by the Househ"+
        "older chain has shape "+str(y_tensor.shape)+" and is:\n"+str(
        y_tensor.numpy())+"\n")

        # Takes the y tensor and applied the Householder chain in rever-
        # se order to verify if the resulting tensor is the same as the
        # input

        reverse_tensor = SVD_tools.multiply_input_vector_by_householder_chain(
        y_tensor, tf.reverse(self.householder_first_index, axis=[0]),
        tf.reverse(self.householder_length, axis=[0]), 
        tf.reverse(self.householder_number_of_leading_zeros, axis=[0]), 
        self.householder_parameters, self.constant_two, 
        self.householder_epsilon_squared)

        print("Applies the Householder chain in reverse order to the f"+
        "ormer result to verify if it gets back to the original input "+
        "tensor. The maximum difference component-wise between the ori"+
        "ginal input tensor and the reconstructed one is: "+str(
        tf.reduce_max(tf.abs(reverse_tensor-self.X_input_vectors)))+"\n")

    # Defines a function to test the analytical derivatives of the nor-
    # malization factor and of the first non-zero component of the 
    # Householder vector

    def test_derivative_of_normalization_factor(self):

        print("\n#####################################################"+
        "###################\n# Tests derivative of normalization fact"+
        "or and first component, v_tilde  #\n#########################"+
        "###############################################\n")

        # Iterates over the Householder vectors to check

        for i in range(self.number_of_householder_vectors_to_check):

            # Converts the Householder index to a tensorflow integer

            householder_index = tf.constant(i, dtype=tf.as_dtype(
            self.integer_type))

            # GEts the number of DOFs of this Householder vectors and 
            # cast it into a float

            n_dofs = tf.cast(self.householder_length[householder_index],
            dtype=tf.as_dtype(self.parameters_dtype))

            # Gets the Householder vector

            (householder_vector, v_bar, alpha, vector_of_dofs,
            unnormalized_first_component, local_n_dofs) = SVD_tools.get_householder_vector_from_parameters(
            self.householder_first_index, self.householder_length,
            self.householder_number_of_leading_zeros, 
            self.householder_parameters, householder_index,
            self.householder_epsilon_squared)

            # Gets the derivative of v_tilde and of alpha

            derivative_v_tilde, derivative_alpha = SVD_tools.evaluate_derivative_of_v_tilde_and_alpha(
            v_bar, alpha, n_dofs, vector_of_dofs, 
            unnormalized_first_component) 

            # Gets the first and last indices of the Householder vector, 
            # then the vector that will become the Householder vector

            initial_index = self.householder_first_index[
            householder_index] 
            
            length = self.householder_length[householder_index]
            
            number_of_leading_zeros = self.householder_number_of_leading_zeros[
            householder_index]
            
            slice_of_vector_of_dofs = tf.slice(
            self.householder_parameters, begin=tf.expand_dims(
            initial_index, axis=0), size=tf.expand_dims(length, axis=0)
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
    # y = QX, such that (Q^T)*Q = I and Q = H_i
    #
    # with respect to the Householder vector v^i of the i-th Householder
    # reflector H_i

    def test_derivative_of_application_of_single_reflector(self):

        print("\n#####################################################"+
        "###################\n#    Tests derivative of application of "+
        "single Householder reflector   #\n###########################"+
        "#############################################\n")

        # Sets the left tensor to be multiplied as the identity

        left_tensor_to_be_multiplied = tf.eye(self.rank,
        self.dimensionality, dtype=tf.as_dtype(self.parameters_dtype))

        # Iterates over the Householder vectors to check
        
        for i in range(self.number_of_householder_vectors_to_check):

            # Converts the Householder index to a tensorflow integer

            householder_index = tf.constant(i, dtype=tf.as_dtype(
            self.integer_type))

            # GEts the number of DOFs of this Householder vectors and 
            # cast it into a float

            n_dofs = tf.cast(self.householder_length[householder_index],
            dtype=tf.as_dtype(self.integer_type))

            # Gets the Householder vector
            
            (householder_vector, v_bar, alpha, vector_of_dofs,
            unnormalized_first_component, local_n_dofs) = SVD_tools.get_householder_vector_from_parameters(
            self.householder_first_index, self.householder_length,
            self.householder_number_of_leading_zeros, 
            self.householder_parameters, householder_index,
            self.householder_epsilon_squared)

            # Gets the derivative of the operation using the analytical
            # derivative implemented in SVD tools

            dy_dv = SVD_tools.evaluate_derivative_of_householder_reflector_application(
            v_bar, householder_vector, alpha, n_dofs, vector_of_dofs, 
            householder_index, self.X_input_vectors, 
            left_tensor_to_be_multiplied, unnormalized_first_component, 
            self.constant_two, tf.as_dtype(self.parameters_dtype))

            # Gets the first and last indices of the Householder vector, 
            # then the vector that will become the Householder vector

            initial_index = self.householder_first_index[
            householder_index] 
            
            length = self.householder_length[householder_index]
            
            slice_of_vector_of_dofs = tf.slice(
            self.householder_parameters, begin=tf.expand_dims(
            initial_index, axis=0), size=tf.expand_dims(length, axis=0)
            ).numpy()

            # Evaluates the derivatives of the operation using central 
            # finite differences

            def get_y(slice_of_vector_of_dofs):

                slice_tensor = tf.constant(slice_of_vector_of_dofs, 
                dtype=tf.as_dtype(self.parameters_dtype))

                # Temporarily reconstructs the parameter vector with per-
                # turbed slice

                updated_householder_parameters = tf.tensor_scatter_nd_update(
                self.householder_parameters, tf.range(initial_index, 
                initial_index+length)[:, None], slice_tensor)

                # Gets the multiplication of the Householder reflector
                # by the input tensor. Also multiplies by the left matrix

                return tf.einsum('ij,kj->ki', left_tensor_to_be_multiplied, 
                SVD_tools.multiply_input_vector_by_householder_reflector(
                self.X_input_vectors, householder_index, 
                self.householder_first_index, self.householder_length, 
                self.householder_number_of_leading_zeros, 
                updated_householder_parameters, self.constant_two, 
                self.householder_epsilon_squared))
            
            CFD_y = derivative_tensor_valued_function_of_vector_argument(
            get_y, slice_of_vector_of_dofs, epsilon=1E-5)

            print("\nCompares the derivative of y=H_{"+str(i+1)+"}*X w"+
            "ith respect to the DOFs of the "+str(i+1)+"-th\nHousehold"+
            "er vector with central finite differences.\nThe shape of "+
            "the CFD derivative is "+str(CFD_y.shape)+"; while the sha"+
            "pe of the analytical derivative is "+str(dy_dv.shape)+"."+
            "\nThe maximum difference between components is: "+str(
            tf.reduce_max(tf.abs(dy_dv-tf.constant(CFD_y, dtype=
            tf.as_dtype(self.parameters_dtype)))).numpy())+"\n")

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

        # Defines the left orthogonal matrix 

        left_orthogonal_matrix = tf.eye(self.rank, self.dimensionality, 
        dtype=tf.as_dtype(self.parameters_dtype))

        # Evaluates the derivative of the operation y = QX analytically

        dy_dDofs = SVD_tools.evaluate_derivative_of_chain_of_householder_reflectors_application(
        self.X_input_vectors, self.householder_first_index, 
        self.householder_length, self.householder_number_of_leading_zeros,
        self.householder_parameters, self.householder_epsilon_squared,
        self.dimensionality, self.n_samples, tf.as_dtype(
        self.parameters_dtype), self.constant_two)

        # Defines a function to get the output as a function of a numpy
        # flat tensor of DOFs of the Householder chain

        def get_y(householder_parameters):

            updated_householder_parameters = tf.constant(
            householder_parameters, dtype=tf.as_dtype(
            self.parameters_dtype))

            # Gets the multiplication of the Householder reflector
            # by the input tensor. Also multiplies by the left matrix

            return tf.einsum('ij,kj->ki', left_orthogonal_matrix,
            SVD_tools.multiply_input_vector_by_householder_chain(
            self.X_input_vectors, self.householder_first_index, 
            self.householder_length, 
            self.householder_number_of_leading_zeros,
            updated_householder_parameters, self.constant_two,
            self.householder_epsilon_squared))

        # Evaluates the derivative using central finite differences

        CFD_y = derivative_tensor_valued_function_of_vector_argument(
        get_y, self.householder_parameters.numpy(), epsilon=1E-5)

        print("\nCompares the derivative of the Householder chain, y=Q"+
        "*X with respect to\nthe DOFs of the Householder vector chain "+
        "with central finite differences.\nThe shape of the CFD deriva"+
        "tive is "+str(CFD_y.shape)+"; while the shape of the\nanalyti"+
        "cal derivative is "+str(dy_dDofs.shape)+".\nThe maximum diffe"+
        "rence between components is: "+str(tf.reduce_max(tf.abs(
        dy_dDofs-tf.constant(CFD_y, dtype=tf.as_dtype(
        self.parameters_dtype)))).numpy())+"\n")

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