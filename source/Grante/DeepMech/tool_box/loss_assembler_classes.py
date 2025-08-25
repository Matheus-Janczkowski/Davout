# Routine to store classes that assemble loss functions given mutable or
# immutable externaly-defined parameters

import tensorflow as tf

from ..tool_box import loss_tools

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

    def __init__(self, A_matrix, b_vector, trainable_A_matrix=False,
    trainable_b_vector=False, dtype=tf.float32, name="quadratic_loss_o"+
    "ver_linear_residual", reduction=tf.keras.losses.Reduction.SUM, 
    check_tensors=False, block_multiplication=False):

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

        # If block multiplication is selected

        if self.block_multiplication:

            # Gets the number of output neurons

            self.n_outputs = b_vector.shape[0]

            # Gets the number of samples

            self.n_samples = b_vector.shape[1]

            # Verifies if the the coefficient matrix has the same number
            # of samples

            if len(A_matrix)!=self.n_samples:

                raise IndexError("The coefficient matrix has size "+str(
                len(A_matrix))+", which is different than the number o"+
                "f samples retrieved from the coefficient vector, "+str(
                self.n_samples)+". Thus, the quadratic loss function f"+
                "or a linear residual cannot be performed")

            # Builds the block diagonal matrix. Thus, initializes a list
            # of indices and of the values

            all_indices = []

            all_values = []

            tf.sparse.SparseTensor()

            for i, Ai in enumerate(A_matrix):

                indices = Ai.indices + tf.constant([[i*self.n_outputs, i*self.n_outputs]], dtype=tf.int64) - tf.constant([[0,0]])

                all_indices.append(indices)

                all_values.append(Ai.values)

            all_indices = tf.concat(all_indices, axis=0)

            all_values = tf.concat(all_values, axis=0)

            A_block = tf.sparse.SparseTensor(indices=all_indices, values=
            all_values, dense_shape=[self.n_samples*self.n_outputs, self.n_samples*self.n_outputs])

            # Compute R

            y_flat = tf.reshape(y, (self.n_samples*self.n_outputs, 1))  # (Nm, 1)
            b_flat = tf.reshape(b, (self.n_samples*self.n_outputs, 1))
            R = tf.sparse.sparse_dense_matmul(A_block, y_flat) - b_flat  # (Nm,1)
            R = tf.reshape(R, (self.n_samples*self.n_outputs,))  # flat vector

            # Compute D-block
            D_block = tf.linalg.LinearOperatorBlockDiag([D]*self.n_samples)  # (Nm,Nm) dense

            return 0.5 * tf.tensordot(R, D_block.matvec(R), axes=1)

    # Redefines the call method

    def call(self, expected_response, model_response):

        # Multiplies the coefficient matrix by the model output, and 
        # subtracts by the coefficient vector to get the residual

        R = (self.A_matrix*model_response)-self.b_vector

        # Evaluates the quadratic loss function and returns it
        
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