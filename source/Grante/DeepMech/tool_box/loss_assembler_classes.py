# Routine to store classes that assemble loss functions given mutable or
# immutable externaly-defined parameters

import tensorflow as tf

import scipy as sp

from ..tool_box import loss_tools

from ..tool_box import numerical_tools

########################################################################
#                             Linear loss                              #
########################################################################

# Defines a class to assemble the linear loss

@tf.keras.utils.register_keras_serializable(package="custom_losses")
class LinearLossAssembler(tf.keras.losses.Loss):

    def __init__(self, coefficient_matrix, trainable_coefficient_matrix=
    False, dtype=tf.float32, name="linear_loss", reduction=
    tf.keras.losses.Reduction.SUM, check_tensors=False):

        super().__init__(name=name, reduction=reduction)

        # Stores some parameters

        self.trainable_coefficient_matrix = trainable_coefficient_matrix

        self.tensorflow_type = dtype

        self.check_tensors = check_tensors

        # Stores the coefficient matrix as a TensorFlow constant
        
        self.coefficient_matrix = tf.Variable(coefficient_matrix, dtype=
        dtype, trainable=trainable_coefficient_matrix)

        # Stores the expected and true size of the coefficient matrix

        self.n_rows_coefficient_matrix = 0

        self.n_columns_coefficient_matrix = 0

    # Redefines the call method

    def call(self, expected_response, model_response):

        # If the flag for checking parameters is on

        if self.check_tensors:

            self.check_arguments_consistency(model_response)

        # Evaluates the loss and returns it
        
        return loss_tools.linear_loss(model_response, 
        self.coefficient_matrix)
    
    # Defines a method to update the coefficient matrix

    def update_arguments(self, coefficient_matrix):

        """if self.check_tensors:

            self.check_arguments_consistency(None)"""

        # Updates the coefficient matrix as a TensorFlow constant

        self.coefficient_matrix.assign(coefficient_matrix)
        
        """self.coefficient_matrix = tf.Variable(coefficient_matrix, dtype=
        self.tensorflow_type, trainable=self.trainable_coefficient_matrix)"""

    # Defines a method for checking arguments consistency

    def check_arguments_consistency(self, model_response):

        if model_response is None:

            # Checks the number of rows and columns of the coefficient
            # matrix

            tf.debugging.assert_equal(tf.shape(self.coefficient_matrix)[
            0], self.n_rows_coefficient_matrix, message="The number of"+
            " samples being evaluated by the model is different than t"+
            "he number of rows of the coefficient matrix")

            tf.debugging.assert_equal(tf.shape(self.coefficient_matrix)[
            1], self.n_columns_coefficient_matrix, message="The number"+
            " of output neurons of the model is different than the num"+
            "ber of columns of the coefficient matrix")

        else:

            # Uses the debuggin interface of tensorflow to ease computa-
            # tional cost and optimize the use of graph mode. Checks the
            # number of samples

            tf.debugging.assert_equal(tf.shape(self.coefficient_matrix)[
            0], tf.shape(model_response)[0], message="The number of sa"+
            "mples being evaluated by the model is different than the "+
            "number of rows of the coefficient matrix")

            # Checks the number of output neurons

            tf.debugging.assert_equal(tf.shape(self.coefficient_matrix)[
            1], tf.shape(model_response)[-1], message="The number of o"+
            "utput neurons of the model is different than the number o"+
            "f columns of the coefficient matrix")

            # Updates the number of rows and columns of the coefficient
            # matrix

            self.n_rows_coefficient_matrix = tf.shape(model_response)[0]

            self.n_columns_coefficient_matrix = tf.shape(model_response
            )[-1]

    # Redefines configurations for model saving

    def get_config(self):

        config = super().get_config()

        config.update({"coefficient_matrix": 
        self.coefficient_matrix.numpy().tolist(), "dtype":
        self.coefficient_matrix.dtype.name})

        return config

########################################################################
#                            Quadratic loss                            #
########################################################################

# Defines a class to assemble the quadratic loss over a linear residual,
# i.e. L = 0.5*(R.T*D*R); R=Ax-b

@tf.keras.utils.register_keras_serializable(package="custom_losses")
class QuadraticLossOverLinearResidualAssembler(tf.keras.losses.Loss):

    def __init__(self, A_matrix, b_vector, conditioning_matrix="identi"+
    "ty", trainable_A_matrix=False, trainable_b_vector=False, dtype=
    tf.float32, name="quadratic_loss_over_linear_residual", reduction=
    tf.keras.losses.Reduction.SUM, check_tensors=False, 
    block_multiplication=False):

        super().__init__(name=name, reduction=reduction)

        # Stores some parameters

        self.trainable_A_matrix = trainable_A_matrix

        self.trainable_b_matrix = trainable_b_vector

        self.tensorflow_type = dtype

        self.check_tensors = check_tensors

        # There are two options for evaluating this loss function:
        #
        # 1. Block multiplication. Each one of the A matrices (one for
        # each sample) is allocated in a large block-diagonal matrix. 
        # The b vectors and the model outputs will also be stacked for 
        # multiplication.
        #
        # 2. Loop multiplication. Each set of A, b, and model response 
        # (one for each sample) is multiplied independently inside a for 
        # loop.
        #
        # Option 1 is supposed to be faster than option 2, since it ta-
        # kes advantage of the batch treatment native to TensorFlow. 
        # However, it can lead to memory issues. Thus, option 2 is de-
        # fault, as a conservative option

        self.block_multiplication = block_multiplication

        # Gets the number of output neurons

        self.n_outputs = b_vector.shape[0]

        # Gets the number of samples

        self.n_samples = b_vector.shape[1]

        if not isinstance(A_matrix, list):

            raise TypeError("The coefficient matrix is not a list in '"+
            "QuadraticLossOverLinearResidualAssembler' even though the"+
            " block multiplication flag is on")

        # Verifies if the the coefficient matrix has the same number of
        # samples

        if len(A_matrix)!=self.n_samples:

            raise IndexError("The coefficient matrix has size "+str(len(
            A_matrix))+", which is different than the number of sample"+
            "s retrieved from the coefficient vector, "+str(
            self.n_samples)+". Thus, the quadratic loss function for a"+
            "linear residual cannot be performed")
        
        # If the conditioning matrix is equal to the identity

        if conditioning_matrix=="identity":

            conditioning_matrix = sp.sparse.identity(self.n_outputs, 
            dtype=float, format='coo')

        # If block multiplication is selected

        if self.block_multiplication:

            # Builds the block diagonal matrix

            self.A_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            A_matrix, block_multiplication=self.block_multiplication)

            # Creates a long vector out of the b vector, each sample af-
            # ter the previous one

            self.b_vector = tf.reshape(b_vector, (self.n_samples*
            self.n_outputs, 1))

            # Creates a block diagonal matrix for the conditioning matrix

            self.conditioning_matrix = tf.linalg.LinearOperatorBlockDiag([
            conditioning_matrix]*self.n_samples)

        # If batch multiplication is selected

        else:

            # Builds the sparse tensor as tensor with 3 indices, where 
            # the first one tells the batch

            self.A_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            A_matrix, block_multiplication=self.block_multiplication)

            # Creates a variable for the b vector

            self.b_vector = tf.Variable(b_vector, dtype=
            self.tensorflow_type)

            # Creates a sparse tensor for the conditioning matrix

            self.conditioning_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            conditioning_matrix, block_multiplication=False)

    # Redefines the call method

    def call(self, expected_response, model_response):

        # If block multiplication is selected

        if self.block_multiplication:

            # Reshapes the model response into a long vector of samples

            flat_model_response = tf.reshape(model_response, (
            self.n_samples*self.n_outputs, 1))

            # Evaluates the residual vector and reshapes it

            R = tf.reshape(tf.sparse.sparse_dense_matmul(self.A_matrix, 
            flat_model_response)-self.b_vector, (self.n_samples*
            self.n_outputs,))

            # Evaluates the quadratic loss function and returns it
            
            return 0.5*tf.tensordot(R, self.conditioning_matrix.matvec(R
            ), axes=1)
        
        # Tackles batch multiplication
        
        else:

            # Initializes a list of residual vectors, one for each sam-
            # ple

            R_list = []

            # Iterates through the samples

            for i in range(self.n_samples):

                # Gets the residual vector

                R_sample = tf.sparse.sparse_dense_matmul(self.A_matrix[i
                ], model_response[:,i:(i+1)])-self.b_vector[:,i:(i+1)]

                # Flattens the residual vector and appends to the list

                R_list.append(tf.reshape(R_sample, [-1]))

            # Concatenates the residual list into a single vector

            R_list = tf.concat(R_list, axis=0)

            # Evaluates the loss and returns it

            return 0.5*tf.tensordot(R_list, 
            tf.sparse.sparse_dense_matmul(self.conditioning_matrix, 
            R_list), axes=1)
    
    # Defines a method to update the coefficient matrix

    def update_arguments(self, A_matrix, b_vector):

        # Gets the number of output neurons

        self.n_outputs = b_vector.shape[0]

        # Gets the number of samples

        self.n_samples = b_vector.shape[1]

        if not isinstance(A_matrix, list):

            raise TypeError("The coefficient matrix is not a list in '"+
            "QuadraticLossOverLinearResidualAssembler' even though the"+
            " block multiplication flag is on")

        # Verifies if the the coefficient matrix has the same number of
        # samples

        if len(A_matrix)!=self.n_samples:

            raise IndexError("The coefficient matrix has size "+str(len(
            A_matrix))+", which is different than the number of sample"+
            "s retrieved from the coefficient vector, "+str(
            self.n_samples)+". Thus, the quadratic loss function for a"+
            "linear residual cannot be performed")

        # If block multiplication is selected

        if self.block_multiplication:

            # Builds the block diagonal matrix

            self.A_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            A_matrix, block_multiplication=self.block_multiplication)

            # Creates a long vector out of the b vector, each sample af-
            # ter the previous one

            self.b_vector = tf.reshape(b_vector, (self.n_samples*
            self.n_outputs, 1))

        # If batch multiplication is selected

        else:

            # Builds the sparse tensor as tensor with 3 indices, where 
            # the first one tells the batch

            self.A_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            A_matrix, block_multiplication=self.block_multiplication)

            # Creates a variable for the b vector

            self.b_vector.assign(tf.transpose(b_vector))

    # Redefines configurations for model saving

    def get_config(self):

        config = super().get_config()

        config.update({"coefficient_matrix": 
        self.A_matrix.numpy().tolist(), "dtype":
        self.A_matrix.dtype.name})

        return config