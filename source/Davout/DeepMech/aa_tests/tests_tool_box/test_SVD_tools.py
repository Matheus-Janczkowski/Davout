# Routine to test custom gradients

import tensorflow as tf

import numpy as np

import time 

import gc

import tracemalloc

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

        # Sets the input and the output dimensionality of the left-or-
        # thogonal matrix to be created using a chain of Householder re-
        # flectors (CHR)

        self.input_dimension = 10

        self.output_dimension = 5

        # Sets the epsilon to be used to calculate the first non-zero
        # component of each Householder vector

        householder_epsilon_squared = 1.0

        # Creates all tensors used to recreate the chain of Householder
        # reflectors and the underlying necessary information

        (self.householder_parameters, self.householder_first_index, 
        self.householder_length, self.householder_number_of_leading_zeros,
        self.householder_epsilon_squared, self.n_dofs, self.rank
        ) = SVD_tools.create_householder_tensors(self.input_dimension, 
        self.output_dimension, self.parameters_dtype, self.integer_type,
        householder_epsilon_squared)

        # Defines the number of Householder vectors to check the deriva-
        # tives

        self.number_of_householder_vectors_to_check = 5

        self.number_of_householder_vectors_to_check = min(self.rank, 
        self.number_of_householder_vectors_to_check, self.input_dimension
        -1)

        # Defines the tensor of batched input vectors X [dimensionality,
        # n_samples] for the operation 
        # y = I(rank,dimensionality)*H_1*H_2*...*H_rank*X 
        #
        # such that y is a tensor [n_samples, rank]

        self.n_samples = 5

        self.X_input_vectors = tf.constant(np.random.randn(
        self.n_samples, self.input_dimension), dtype=tf.as_dtype(
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
            self.householder_epsilon_squared, self.input_dimension)

            print("The assembled "+str(i)+"-th Householder vector is:"+
            "\n"+str(householder_vector.numpy())+"\n")

    # Defines a function to test multiplying an input tensor [n_samples,
    # dimensionality] by a Householder chain

    def test_householder_chain_multiplication(self):

        print("\n#####################################################"+
        "###################\n#       Tests multiplying a Householder "+
        "chain to an input tensor       #\n###########################"+
        "#############################################\n")

        # Sets the output dimension as the same as the input dimension
        # to be able to test the inversion of the orthogonal transforma-
        # tion

        output_dimension = self.input_dimension

        y_tensor = SVD_tools.multiply_input_vector_by_householder_chain(
        self.X_input_vectors, self.householder_first_index,
        self.householder_length, 
        self.householder_number_of_leading_zeros, 
        self.householder_parameters, self.constant_two, 
        self.householder_epsilon_squared, output_dimension, 
        self.input_dimension)

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
        self.householder_epsilon_squared, output_dimension,
        self.input_dimension)

        print("Reverse tensor has shape: "+str(reverse_tensor.shape)+
        "\n")

        print("Applies the Householder chain in reverse order to the f"+
        "ormer result to\nverify if it gets back to the original input"+
        "tensor. The maximum diffe-\nrence component-wise between the "+
        "original input tensor and the recons-\ntructed one is: "+str(
        tf.reduce_max(tf.abs(reverse_tensor-self.X_input_vectors)).numpy(
        ))+"\n")

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
            self.householder_epsilon_squared, self.input_dimension)

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
        self.input_dimension, dtype=tf.as_dtype(self.parameters_dtype))

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
            self.householder_epsilon_squared, self.input_dimension)

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
                self.householder_epsilon_squared, self.input_dimension))
            
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

        left_orthogonal_matrix = tf.eye(self.rank, self.rank, 
        dtype=tf.as_dtype(self.parameters_dtype))

        # Initializes the matrix Q as the product of all Householder re-
        # flectors

        Q = tf.eye(self.rank, self.input_dimension, dtype=tf.as_dtype(
        self.parameters_dtype))

        for i in tf.range(self.rank):

            # Get the Householder vector of the i-th reflector

            householder_vector, _, _, _, _, _ = SVD_tools.get_householder_vector_from_parameters(
            self.householder_first_index, self.householder_length, 
            self.householder_number_of_leading_zeros, 
            self.householder_parameters, i, 
            self.householder_epsilon_squared, self.input_dimension)

            # Multiplies this Householder reflector to the right of Q u-
            # sing the corresponding rank-1 update

            Q -= self.constant_two*tf.einsum('ik,k,j->ij', Q, 
            householder_vector, householder_vector)

        # Evaluates the derivative of the operation y = QX analytically

        dy_dDofs = SVD_tools.evaluate_derivative_of_chain_of_householder_reflectors_application(
        self.X_input_vectors, self.householder_first_index, 
        self.householder_length, self.householder_number_of_leading_zeros,
        self.householder_parameters, self.householder_epsilon_squared,
        self.n_samples, tf.as_dtype(self.parameters_dtype), 
        self.constant_two, Q)

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
            self.householder_epsilon_squared, self.rank, 
            self.input_dimension))

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

    # Defines a function to test the evaluation of the analytical deri-
    # vative of the following operation
    # 
    # y = QX, such that (Q^T)*Q = I and Q = H1*H*...*H_rank
    #
    # with respect to the Householder vector v^i of the i-th Householder
    # reflector H_i and with respect to the input tensor using the ap-
    # propriate method that was decorated with tf.custom_gradient

    def test_derivative_of_application_of_CHR_using_custom_grad(self):

        print("\n#####################################################"+
        "###################\n#      Tests chain of Householder reflec"+
        "tors with custom gradient      #\n###########################"+
        "#############################################\n")

        # Sets the input tensor for performance test

        n_samples_performance_test = 100

        self.X_input_performance_test = tf.constant(np.random.randn(
        n_samples_performance_test, self.input_dimension), dtype=
        tf.as_dtype(self.parameters_dtype))

        # Sets the output dimension as the rank

        output_dimension = self.rank

        # Tests the derivative of a toy loss function with the custom 
        # gradient. GradientTape from tensorflow will automatically the
        # derived custom gradient

        with tf.GradientTape() as custom_tape:

            # Sets the tape to watch the input tensor

            custom_tape.watch(self.X_input_performance_test)

            y_custom = SVD_tools.multiply_input_vector_by_householder_chain_with_custom_gradient(
            self.X_input_performance_test, self.householder_first_index, 
            self.householder_length, 
            self.householder_number_of_leading_zeros,
            self.householder_parameters, self.constant_two,
            self.householder_epsilon_squared, output_dimension,
            self.input_dimension)

            # Defines a scalar loss L = 0.5 * sum(y^2)

            custom_loss = 0.5*tf.reduce_sum(tf.square(y_custom))

        # Evaluates the custom gradient

        custom_gradient_parameters, custom_gradient_input_tensor = custom_tape.gradient(
        custom_loss, [self.householder_parameters, 
        self.X_input_performance_test])

        # Tests the gradient with pure automatic differentiation (AD)

        with tf.GradientTape() as AD_tape:

            # Sets the tape to watch both the Householder parameters and
            # the input tensor

            AD_tape.watch(self.X_input_performance_test)

            AD_tape.watch(self.householder_parameters)

            y_AD = SVD_tools.multiply_input_vector_by_householder_chain(
            self.X_input_performance_test, self.householder_first_index, 
            self.householder_length, 
            self.householder_number_of_leading_zeros,
            self.householder_parameters, self.constant_two,
            self.householder_epsilon_squared, output_dimension,
            self.input_dimension)

            AD_loss = 0.5*tf.reduce_sum(tf.square(y_AD))

        # Evaluates the gradient with AD

        AD_gradient_parameters, AD_gradient_input = AD_tape.gradient(
        AD_loss, [self.householder_parameters, 
        self.X_input_performance_test])

        print("\nCompares the derivative of the Householder chain, y=Q"+
        "*X with respect to\nthe DOFs of the Householder vector chain "+
        "with automatic differentiation (AD).\nThe shape of the AD der"+
        "ivative is "+str(AD_gradient_parameters.shape)+"; while the s"+
        "hape of the\nanalytical derivative is "+str(
        custom_gradient_parameters.shape)+".\nThe maximum difference b"+
        "etween components is: "+str(
        tf.reduce_max(tf.abs(custom_gradient_parameters-
        AD_gradient_parameters)).numpy())+"\n")

        print("Compares the derivative of the Householder chain, y=Q*X"+
        " with respect to\nthe input tensor with automatic differentia"+
        "tion (AD).\nThe shape of the AD derivative is "+str(
        AD_gradient_input.shape)+"; while the shape of the\nanalytical"+
        " derivative is "+str(custom_gradient_input_tensor.shape)+".\n"+
        "The maximum difference between components is: "+str(
        tf.reduce_max(tf.abs(custom_gradient_input_tensor-
        AD_gradient_input)).numpy())+"\n")

    # Defines a function to compare runtime and memory cost using the a-
    # nalytical derivative against automatic differentiation

    def test_runtime_and_memory_cost(self):
        
        print("\n#####################################################"+
        "###################\n#  Tests runtime and memory cost of anal"+
        "ytical derivatives against AD  #\n###########################"+
        "#############################################\n")

        # Sets the number of runs to warm-up both functions to force 
        # graph compilation. Also sets the number of runs to evaluate
        # computation time

        n_warm_up_runs = 10

        n_execution_runs = 100

        # Sets the input and the output dimensionality of the left-or-
        # thogonal matrix to be created using a chain of Householder re-
        # flectors (CHR)

        input_dimension_performance = 50

        output_dimension_performance = 1000

        # Sets the type of the parameters

        parameters_type_performance = "float32"

        # Sets the epsilon to be used to calculate the first non-zero
        # component of each Householder vector

        householder_epsilon_squared = 1.0

        # Creates all tensors used to recreate the chain of Householder
        # reflectors and the underlying necessary information

        (householder_parameters_performance, 
        householder_first_index_performance, 
        householder_length_performance,
        householder_number_of_leading_zeros_performance,
        householder_epsilon_squared_performance, n_dofs_performance, 
        rank_performance) = SVD_tools.create_householder_tensors(
        input_dimension_performance, output_dimension_performance, 
        parameters_type_performance, self.integer_type,
        householder_epsilon_squared)

        # Defines the tensor of batched input vectors X [dimensionality,
        # n_samples] for the operation 
        # y = I(rank,dimensionality)*H_1*H_2*...*H_rank*X 
        #
        # such that y is a tensor [n_samples, rank]

        n_samples_performance = 1000

        X_input_vectors_performance = tf.constant(np.random.randn(
        n_samples_performance, input_dimension_performance), dtype=
        tf.as_dtype(parameters_type_performance))

        # Saves the constant two

        constant_two_performance = tf.constant(2.0, dtype=tf.as_dtype(
        parameters_type_performance))

        @tf.function#(jit_compile=True)
        def optimized_AD_chain_multiplication(X, first_index, length, 
        leading_zeros, params, constant_two, epsilon_sq, output_dim, 
        input_dimension):
            
            return SVD_tools.multiply_input_vector_by_householder_chain(
            X, first_index, length, leading_zeros, params, constant_two, 
            epsilon_sq, output_dim, input_dimension)

        """@tf.function(jit_compile=True)
        def optimized_custom_chain_multiplication(X, first_index, 
        length, leading_zeros, params, constant_two, epsilon_sq, 
        output_dim, input_dimension):

            return SVD_tools.multiply_input_vector_by_householder_chain_with_custom_gradient(
            X, first_index, length, leading_zeros, params, constant_two, 
            epsilon_sq, output_dim, input_dimension)"""

        """# Defines a function to compute the gradient of the toy loss 
        # function using the custom gradient

        def run_custom_implementation():

            with tf.GradientTape() as custom_tape:
            
                # Sets the tape to watch the input tensor
    
                custom_tape.watch(X_input_vectors_performance)
    
                y_custom = optimized_custom_chain_multiplication(
                X_input_vectors_performance, householder_first_index_performance, 
                householder_length_performance, 
                householder_number_of_leading_zeros_performance,
                householder_parameters_performance, 
                constant_two_performance,
                householder_epsilon_squared_performance, 
                output_dimension_performance)
    
                # Defines a scalar loss L = 0.5 * sum(y^2)
    
                custom_loss = 0.5*tf.reduce_sum(tf.square(y_custom))
    
            # Evaluates the custom gradient
    
            return custom_tape.gradient(custom_loss, [
            householder_parameters_performance, 
            X_input_vectors_performance])"""

        # Defines a function to compute the gradient of the toy loss 
        # function using automatic differentiation

        def run_AD_implementation():

            with tf.GradientTape() as AD_tape:
            
                # Sets the tape to watch both the Householder parameters 
                # and the input tensor
    
                AD_tape.watch(X_input_vectors_performance)
    
                AD_tape.watch(householder_parameters_performance)
    
                y_AD = optimized_AD_chain_multiplication(
                X_input_vectors_performance, 
                householder_first_index_performance, 
                householder_length_performance, 
                householder_number_of_leading_zeros_performance,
                householder_parameters_performance, 
                constant_two_performance,
                householder_epsilon_squared_performance, 
                output_dimension_performance, 
                input_dimension_performance)
    
                AD_loss = 0.5*tf.reduce_sum(tf.square(y_AD))
    
            # Evaluates the gradient with AD
    
            return AD_tape.gradient(AD_loss, [
            householder_parameters_performance, X_input_vectors_performance])

        # Warms up to force graph compilation

        for _ in range(n_warm_up_runs):

            # Runs both functions

            #_ = run_custom_implementation()

            _ = run_AD_implementation()

        # Collects all garbage to avoid AD sneaking through left-over 
        # data

        gc.collect()

        # Evaluates the AD running time

        start_time = time.perf_counter()

        for _ in range(n_execution_runs):

            _ = run_AD_implementation()

        # Gets the average time. Multiply by 1000 to get in miliseconds

        whole_time = time.perf_counter()-start_time

        AD_average_time = ((whole_time/n_execution_runs)*1000.0)

        print("Finalizes executing the AD implementation in a total of"+
        " "+str(whole_time)+" seconds\n")

        # Collects all garbage to avoid AD sneaking through left-over 
        # data

        gc.collect()

        """# Evaluates the custom gradient running time

        start_time = time.perf_counter()

        for _ in range(n_execution_runs):

            _ = run_custom_implementation()

        # Gets the average time. Multiply by 1000 to get in miliseconds

        whole_time = time.perf_counter()-start_time

        custom_average_time = ((whole_time/n_execution_runs)*1000.0)

        print("Finalizes executing the custom implementation in a tota"+
        "l of "+str(whole_time)+" seconds\n")"""

        custom_average_time = 0.0

        # Defines a function to measure peak memory

        def measure_peak_memory(target_function):

            # Collects all garbage

            gc.collect()

            # Starts tracing memory footprint

            tracemalloc.start()

            # Executes the function

            _ = target_function()

            # Retrieves current and peak memory allocations in bytes

            current_bytes, peak_bytes = tracemalloc.get_traced_memory()

            # Stops tracing to reset counters

            tracemalloc.stop()
            
            # Returns memory delta in Megabytes (MB)

            return peak_bytes/(1024**2)

        # Measures the memory peak for both implementations

        custom_memory_peak = 0.0

        #custom_memory_peak = measure_peak_memory(
        #run_custom_implementation)

        AD_memory_peak = measure_peak_memory(run_AD_implementation)

        # Prints the results

        print("The testing set-up is:\n1. input dimension ---------- "+
        str(input_dimension_performance)+"\n2. output dimension ------"+
        "--- "+str(output_dimension_performance)+"\n3. number of sampl"+
        "es -------- "+str(n_samples_performance)+"\n4. number of exec"+
        "ution runs - "+str(n_execution_runs)+"\n")

        print("The average time per execution for the AD implementatio"+
        "n in miliseconds is: "+str(AD_average_time)+"\nThe average ti"+
        "me per execution for the custom implementation in miliseconds"+
        " is: "+str(custom_average_time)+"\n")

        print("The memory peak of the AD implementation in MB is: "+str(
        AD_memory_peak)+"\nThe memory peak of the custom implementatio"+
        "n in MB is: "+str(custom_memory_peak))

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