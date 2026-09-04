# Routine to store classes that assemble loss functions given mutable or
# immutable externaly-defined parameters

import tensorflow as tf

import scipy as sp

from ..tool_box import loss_tools

from ..tool_box import numerical_tools

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
# i.e. L = 0.5*(R.T*D*R); R=Ax-b. No custom gradient is provided. Block
# multiplication is the only available option

@tf.keras.utils.register_keras_serializable(package="custom_losses")
class QuadraticLossOverLinearResidualAssembler(tf.keras.losses.Loss):

    def __init__(self, A_matrix, b_vector, conditioning_matrix="identi"+
    "ty", trainable_A_matrix=False, trainable_b_vector=False, dtype=
    tf.float32, name="quadratic_loss_over_linear_residual", reduction=
    tf.keras.losses.Reduction.SUM, check_tensors=False):

        super().__init__(name=name, reduction=reduction)

        # Stores some parameters

        self.trainable_A_matrix = trainable_A_matrix

        self.trainable_b_matrix = trainable_b_vector

        self.tensorflow_type = dtype

        self.check_tensors = check_tensors

        # Gets the number of output neurons

        self.n_outputs = b_vector.shape[1]

        # Gets the number of samples

        self.n_samples = b_vector.shape[0]

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
        
        # If the conditioning matrix is the identity matrix

        if conditioning_matrix=="identity":

            conditioning_matrix = sp.sparse.identity(self.n_outputs, 
            dtype=float, format="coo")

        # The loss function is evaluated using block multiplication. 
        # Each one of the A matrices (one for each sample) is allocated 
        # in a large block-diagonal matrix. The b vectors and the model 
        # outputs will also be stacked for multiplication.

        # Builds the block diagonal matrix

        self.A_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
        A_matrix, block_multiplication=True)

        # Creates a long vector out of the b vector, each sample af-
        # ter the previous one

        self.b_vector = tf.Variable(tf.cast(tf.reshape(b_vector, (
        self.n_samples*self.n_outputs, 1)), self.tensorflow_type), 
        dtype=self.tensorflow_type)

        # Creates a block diagonal matrix for the conditioning matrix

        self.conditioning_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
        conditioning_matrix, block_multiplication=True, n_samples=
        self.n_samples)

    # Redefines the call method

    def call(self, expected_response, model_response):

        # Reshapes the model response into a long vector of samples

        flat_model_response = tf.reshape(model_response, (
        self.n_samples*self.n_outputs, 1))

        # Evaluates the residual vector and reshapes it

        R = tf.reshape(tf.sparse.sparse_dense_matmul(self.A_matrix, 
        flat_model_response)-self.b_vector, (self.n_samples*
        self.n_outputs,))

        # Evaluates the quadratic loss function and returns it
        
        return 0.5*tf.tensordot(R, tf.reshape(
        tf.sparse.sparse_dense_matmul(self.conditioning_matrix, 
        tf.expand_dims(R, -1)), [-1]), axes=1)
    
    # Defines a method to update the coefficient matrix

    def update_arguments(self, A_matrix, b_vector):

        # Gets the number of output neurons

        self.n_outputs = b_vector.shape[1]

        # Gets the number of samples

        self.n_samples = b_vector.shape[0]

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

        # Builds the coefficient matrix

        self.A_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
        A_matrix, block_multiplication=True)

        # Creates a long vector out of the b vector, each sample af-
        # ter the previous one

        self.b_vector.assign(tf.cast(tf.reshape(b_vector, (
        self.n_samples*self.n_outputs, 1)), self.tensorflow_type))

    # Redefines configurations for model saving

    def get_config(self):

        config = super().get_config()

        config.update({"coefficient_matrix": 
        self.A_matrix.numpy().tolist(), "dtype":
        self.A_matrix.dtype.name})

        return config

# Defines a class to assemble the quadratic loss over a linear residual,
# i.e. L = 0.5*(R.T*D*R); R=Ax-b. A custom gradient is also defined

@tf.keras.utils.register_keras_serializable(package="custom_losses")
class QuadraticLossOverLinearResidualAssemblerWithCustomGrad(
tf.keras.losses.Loss):

    def __init__(self, A_matrix, b_vector, conditioning_matrix="identi"+
    "ty", trainable_A_matrix=False, trainable_b_vector=False, dtype=
    tf.float32, name="quadratic_loss_over_linear_residual", reduction=
    tf.keras.losses.Reduction.SUM, check_tensors=False, 
    block_multiplication=True, custom_gradient_usage=False):

        super().__init__(name=name, reduction=reduction)

        # Stores some parameters

        self.trainable_A_matrix = trainable_A_matrix

        self.trainable_b_matrix = trainable_b_vector

        self.tensorflow_type = dtype

        self.check_tensors = check_tensors

        # Gets the number of output neurons

        self.n_outputs = b_vector.shape[1]

        # Gets the number of samples

        self.n_samples = b_vector.shape[0]

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
        
        # If the conditioning matrix is the identity matrix

        if conditioning_matrix=="identity":

            conditioning_matrix = sp.sparse.identity(self.n_outputs, 
            dtype=float, format="coo")

        # Some loss classes can have their own implementation of the 
        # gradient w.r.t. the model trainable parameters, due to compu-
        # tational cost concerns. This way, there is a flag to inform if
        # this custom gradient is to be used or not

        self.custom_gradient_usage = custom_gradient_usage

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
        # However, it can lead to memory issues.

        self.block_multiplication = block_multiplication

        # If block multiplication is selected

        if self.block_multiplication:

            # Builds the block diagonal matrix

            self.A_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            A_matrix, block_multiplication=self.block_multiplication)

            # Creates a long vector out of the b vector, each sample af-
            # ter the previous one

            self.b_vector = tf.Variable(tf.cast(tf.reshape(b_vector, (
            self.n_samples*self.n_outputs, 1)), self.tensorflow_type), 
            dtype=self.tensorflow_type)

            # Creates a block diagonal matrix for the conditioning matrix

            self.conditioning_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            conditioning_matrix, block_multiplication=True, n_samples=
            self.n_samples)

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
            
            return 0.5*tf.tensordot(R, tf.reshape(
            tf.sparse.sparse_dense_matmul(self.conditioning_matrix, 
            tf.expand_dims(R, -1)), [-1]), axes=1)
        
        # Tackles batch multiplication
        
        else:

            # Initializes the value of the loss value

            loss_value = tf.constant(0.0, dtype=self.tensorflow_type)

            # Iterates through the samples

            for i in range(self.n_samples):

                # Evaluates the residual of the linear form and flattens
                # it

                R_sample = tf.reshape(tf.sparse.sparse_dense_matmul(
                self.A_matrix[i], tf.expand_dims(model_response[i,:], 
                axis=-1))-tf.expand_dims(self.b_vector[i, :], axis=-1), 
                [-1])

                # Evaluates the residual functional and adds it to the
                # loss

                loss_value += tf.tensordot(R_sample, tf.reshape(
                tf.sparse.sparse_dense_matmul(self.conditioning_matrix, 
                tf.expand_dims(R_sample, -1)), [-1]), axes=1)

            # Returns the loss multiplied by 0.5

            return 0.5*loss_value
        
    # Defines a function to build a custom implementation of the gradient
    # w.r.t. the trainable parameters. This code leverages the computa-
    # tional cost of the evaluation of a linear loss function

    def custom_gradient(self, expected_response, model_response,
    trainable_parameters):

        # If block multiplication is selected

        if self.block_multiplication:

            # Reshapes the model response into a long vector of samples

            flat_model_response = tf.reshape(model_response, (
            self.n_samples*self.n_outputs, 1))

            # Evaluates the residual

            R = tf.sparse.sparse_dense_matmul(self.A_matrix, 
            flat_model_response)-self.b_vector
            
            # Multiplies it by the conditional matrix

            R = tf.sparse.sparse_dense_matmul(self.conditioning_matrix,
            R)

            # Multiplies it by the coefficient matrix transpose, then,
            # reshapes it to the format (n_samples, n_outputs)

            R = tf.reshape(tf.sparse.sparse_dense_matmul(
            tf.sparse.transpose(self.A_matrix), R), (self.n_samples, 
            self.n_outputs))

            # Updates the coefficient matrix of the helper linear loss 
            # inside the gradient function

            self.helper_gradient.update_function(R)

            # Returns the gradient

            return self.helper_gradient(trainable_parameters)
        
        # Tackles batch multiplication
        
        else:

            # Initializes a list of samples for the coefficient matrix
            # of the linear helper

            helper_coefficient_list = []

            # Iterates through the samples

            for i in range(self.n_samples):

                # Evaluates the residual

                R_sample = tf.sparse.sparse_dense_matmul(self.A_matrix[i
                ], tf.expand_dims(model_response[i,:], axis=-1)
                )-tf.expand_dims(self.b_vector[i, :], axis=-1)
                
                # Multiplies it by the conditional matrix

                R_sample = tf.sparse.sparse_dense_matmul(
                self.conditioning_matrix, R_sample)

                # Multiplies it by the coefficient matrix transpose, 
                # then, reshapes it to a vector and appends to the list 
                # of samples

                helper_coefficient_list.append(tf.reshape(
                tf.sparse.sparse_dense_matmul(tf.sparse.transpose(
                self.A_matrix[i]), R_sample), [-1]))

            # Updates the coefficient matrix of the helper linear loss 
            # inside the gradient function

            self.helper_gradient.update_function(tf.stack(
            helper_coefficient_list, axis=0))

            # Returns the gradient

            return self.helper_gradient(trainable_parameters)
    
    # Defines a method to update the coefficient matrix

    def update_arguments(self, A_matrix, b_vector):

        # Gets the number of output neurons

        self.n_outputs = b_vector.shape[1]

        # Gets the number of samples

        self.n_samples = b_vector.shape[0]

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

        # Builds the coefficient matrix

        self.A_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
        A_matrix, block_multiplication=self.block_multiplication)

        # If block multiplication is selected

        if self.block_multiplication:

            # Creates a long vector out of the b vector, each sample af-
            # ter the previous one

            self.b_vector.assign(tf.cast(tf.reshape(b_vector, (
            self.n_samples*self.n_outputs, 1)), self.tensorflow_type))

        # If batch multiplication is selected

        else:

            # Creates a variable for the b vector

            self.b_vector.assign(b_vector)

    # Defines a function to create the helper loss function

    def prepare_custom_gradient(self, model, input_tensor):

        # If the custom implementation is to be performed, makes a set 
        # of precalculations

        # Creates the coefficient matrix for the helper loss, which is a
        # linear loss fucntion whose gradient w.r.t. the trainable para-
        # meters is the same as the given quadratic loss. Then, instan-
        # tiates the linear loss class

        helper_loss = LinearLossAssembler(tf.random.normal((
        self.n_samples, self.n_outputs)))

        # Uses this loss to create a gradient helper

        self.helper_gradient, _ = loss_tools.build_loss_gradient_varying_model_parameters(
        model, helper_loss, input_tensor)

    # Redefines configurations for model saving

    def get_config(self):

        config = super().get_config()

        config.update({"coefficient_matrix": 
        self.A_matrix.numpy().tolist(), "dtype":
        self.A_matrix.dtype.name})

        return config

# Defines a class to assemble the quadratic loss over any residual, i.e.
# L = 0.5*(R.T*D*R). But the derivative of the residual vector w.r.t. 
# the model response must be provided

@tf.keras.utils.register_keras_serializable(package="custom_losses")
class QuadraticLossOverAnyResidualAssembler(tf.keras.losses.Loss):

    def __init__(self, dR_du, R_vector, conditioning_matrix="identi"+
    "ty", trainable_dR_du=False, trainable_R_vector=False, dtype=
    tf.float32, name="quadratic_loss_over_any_residual", reduction=
    tf.keras.losses.Reduction.SUM, check_tensors=False, 
    block_multiplication=True, custom_gradient_usage=True):

        super().__init__(name=name, reduction=reduction)

        # Stores some parameters

        self.trainable_dR_du = trainable_dR_du

        self.trainable_b_matrix = trainable_R_vector

        self.tensorflow_type = dtype

        self.check_tensors = check_tensors

        # Gets the number of output neurons

        self.n_outputs = R_vector.shape[1]

        # Gets the number of samples

        self.n_samples = R_vector.shape[0]

        if not isinstance(dR_du, list):

            raise TypeError("The coefficient matrix is not a list in '"+
            "QuadraticLossOverLinearResidualAssembler' even though the"+
            " block multiplication flag is on")

        # Verifies if the the coefficient matrix has the same number of
        # samples

        if len(dR_du)!=self.n_samples:

            raise IndexError("The coefficient matrix has size "+str(len(
            dR_du))+", which is different than the number of samples r"+
            "etrieved from the coefficient vector, "+str(self.n_samples
            )+". Thus, the quadratic loss function for alinear residua"+
            "l cannot be performed")
        
        # If the conditioning matrix is the identity matrix

        if conditioning_matrix=="identity":

            conditioning_matrix = sp.sparse.identity(self.n_outputs, 
            dtype=float, format="coo")

        # Some loss classes can have their own implementation of the 
        # gradient w.r.t. the model trainable parameters, due to compu-
        # tational cost concerns. This way, there is a flag to inform if
        # this custom gradient is to be used or not

        self.custom_gradient_usage = custom_gradient_usage

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
        # However, it can lead to memory issues.

        self.block_multiplication = block_multiplication

        # If block multiplication is selected

        if self.block_multiplication:

            # Builds the block diagonal matrix

            self.dR_du = numerical_tools.scipy_sparse_to_tensor_sparse(
            dR_du, block_multiplication=self.block_multiplication)

            # Creates a long vector out of the b vector, each sample af-
            # ter the previous one

            self.R_vector = tf.Variable(tf.cast(tf.reshape(R_vector, (
            self.n_samples*self.n_outputs, 1)), self.tensorflow_type), 
            dtype=self.tensorflow_type)

            # Creates a block diagonal matrix for the conditioning matrix

            self.conditioning_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            conditioning_matrix, block_multiplication=True, n_samples=
            self.n_samples)

        # If batch multiplication is selected

        else:

            # Builds the sparse tensor as tensor with 3 indices, where 
            # the first one tells the batch

            self.dR_du = numerical_tools.scipy_sparse_to_tensor_sparse(
            dR_du, block_multiplication=self.block_multiplication)

            # Creates a variable for the b vector

            self.R_vector = tf.Variable(R_vector, dtype=
            self.tensorflow_type)

            # Creates a sparse tensor for the conditioning matrix

            self.conditioning_matrix = numerical_tools.scipy_sparse_to_tensor_sparse(
            conditioning_matrix, block_multiplication=False)

    # Redefines the call method

    def call(self, expected_response, model_response):

        # If block multiplication is selected

        if self.block_multiplication:

            # Evaluates the quadratic loss function and returns it
            
            return 0.5*tf.tensordot(tf.reshape(self.R_vector, [-1]), 
            tf.reshape(tf.sparse.sparse_dense_matmul(
            self.conditioning_matrix, self.R_vector), [-1]), axes=1)
        
        # Tackles batch multiplication
        
        else:

            # Initializes the value of the loss value

            loss_value = tf.constant(0.0, dtype=self.tensorflow_type)

            # Iterates through the samples

            for i in range(self.n_samples):

                # Evaluates the residual functional and adds it to the
                # loss

                loss_value += tf.tensordot(self.R_vector[i,:], tf.reshape(
                tf.sparse.sparse_dense_matmul(self.conditioning_matrix, 
                tf.expand_dims(self.R_vector[i,:], -1)), [-1]), axes=1)

            # Returns the loss multiplied by 0.5

            return 0.5*loss_value
        
    # Defines a function to build a custom implementation of the gradient
    # w.r.t. the trainable parameters. This code leverages the computa-
    # tional cost of the evaluation of a linear loss function

    def custom_gradient(self, expected_response, model_response,
    trainable_parameters):

        # If block multiplication is selected

        if self.block_multiplication:
            
            # Multiplies it by the conditional matrix transposed

            R = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(
            self.conditioning_matrix), self.R_vector)

            # Multiplies it by the coefficient matrix transposed

            R = tf.reshape(tf.sparse.sparse_dense_matmul(
            tf.sparse.transpose(self.dR_du), R), (self.n_samples, 
            self.n_outputs))

            # Updates the coefficient matrix of the helper linear loss 
            # inside the gradient function

            self.helper_gradient.update_function(R)

            # Returns the gradient

            return self.helper_gradient(trainable_parameters)
        
        # Tackles batch multiplication
        
        else:

            # Initializes a list of samples for the coefficient matrix
            # of the linear helper

            helper_coefficient_list = []

            # Iterates through the samples

            for i in range(self.n_samples):
                
                # Multiplies it by the conditional matrix transposed

                R_sample = tf.sparse.sparse_dense_matmul(
                tf.sparse.transpose(self.conditioning_matrix), 
                tf.expand_dims(self.R_vector[i,:], axis=-1))

                # Multiplies it by the coefficient matrix transposed, 
                # then, reshapes it to a vector and appends to the list 
                # of samples

                helper_coefficient_list.append(tf.reshape(
                tf.sparse.sparse_dense_matmul(tf.sparse.transpose(
                self.dR_du[i]), R_sample), [-1]))

            # Updates the coefficient matrix of the helper linear loss 
            # inside the gradient function

            self.helper_gradient.update_function(tf.stack(
            helper_coefficient_list, axis=0))

            # Returns the gradient

            return self.helper_gradient(trainable_parameters)
    
    # Defines a method to update the coefficient matrix

    def update_arguments(self, dR_du, R_vector):

        # Gets the number of output neurons

        self.n_outputs = R_vector.shape[1]

        # Gets the number of samples

        self.n_samples = R_vector.shape[0]

        if not isinstance(dR_du, list):

            raise TypeError("The coefficient matrix is not a list in '"+
            "QuadraticLossOverLinearResidualAssembler' even though the"+
            " block multiplication flag is on")

        # Verifies if the the coefficient matrix has the same number of
        # samples

        if len(dR_du)!=self.n_samples:

            raise IndexError("The coefficient matrix has size "+str(len(
            dR_du))+", which is different than the number of sample"+
            "s retrieved from the coefficient vector, "+str(
            self.n_samples)+". Thus, the quadratic loss function for a"+
            "linear residual cannot be performed")

        # Builds the coefficient matrix

        self.dR_du = numerical_tools.scipy_sparse_to_tensor_sparse(
        dR_du, block_multiplication=self.block_multiplication)

        # If block multiplication is selected

        if self.block_multiplication:

            # Creates a long vector out of the b vector, each sample af-
            # ter the previous one

            self.R_vector.assign(tf.cast(tf.reshape(R_vector, (
            self.n_samples*self.n_outputs, 1)), self.tensorflow_type))

        # If batch multiplication is selected

        else:

            # Creates a variable for the b vector

            self.R_vector.assign(R_vector)

    # Defines a function to create the helper loss function

    def prepare_custom_gradient(self, model, input_tensor):

        # If the custom implementation is to be performed, makes a set 
        # of precalculations

        # Creates the coefficient matrix for the helper loss, which is a
        # linear loss fucntion whose gradient w.r.t. the trainable para-
        # meters is the same as the given quadratic loss. Then, instan-
        # tiates the linear loss class

        helper_loss = LinearLossAssembler(tf.random.normal((
        self.n_samples, self.n_outputs)))

        # Uses this loss to create a gradient helper

        self.helper_gradient, _ = loss_tools.build_loss_gradient_varying_model_parameters(
        model, helper_loss, input_tensor)

    # Redefines configurations for model saving

    def get_config(self):

        config = super().get_config()

        config.update({"coefficient_matrix": 
        self.dR_du.numpy().tolist(), "dtype":
        self.dR_du.dtype.name})

        return config
    
########################################################################
#                          Conventional losses                         #
########################################################################

# Defines a class for the maximum absolute error

class MaximumAbsoluteError(tf.keras.losses.Loss):

    # Defines just the call method to return the maximum absolute value

    def call(self, y_true, y_pred):

        return tf.reduce_max(tf.abs(y_true - y_pred))
    
    # Defines a function to get the minimum absolute value

    def minimum_absolute_error(self, y_true, y_pred):

        return tf.reduce_min(tf.abs(y_true - y_pred))

# Defines a class for the Lp norm of the absolute error

class LpNormError(tf.keras.losses.Loss):

    """
    Lp norm of the error. This loss evaluates the following expression,

    L = (E_ij^p)^(1/p),

    where E_ij is the error measured at the j-th component of the i-th 
    sample. 
    
    Parameters
    ----------
    p : float, default=2.0
        p exponent of the Lp norm.
        
    name : str, default="lp_norm_loss"
        Name of the loss function instance.
    """

    def __init__(self, p=2.0, epsilon=1E-7, dtype=tf.float32, name="lp"+
    "_norm_loss", use_stable_implementation=False, **kwargs):

        super().__init__(name=name, dtype=tf.as_dtype(dtype), **kwargs)

        # Saves the original rawly given objects

        self.raw_p = float(p)

        self.raw_epsilon = float(epsilon)

        self.raw_dtype = tf.as_dtype(dtype).name

        self.use_stable_implementation = use_stable_implementation

        # Saves the live object

        self.epsilon = tf.cast(epsilon, dtype=self.dtype)

        # Verifies if the stable implementation is to be used

        if self.use_stable_implementation:

            self.p = tf.cast(self.raw_p, dtype=self.dtype)
            
            # Selects the call method that scales and rescales the loss
            # function to avoid numerical instabilities

            self.call_method = self.scale_evaluate_and_rescale_method

        # Verifies if the exponent p is an even positive integer

        elif self.raw_p%2==0 and self.raw_p>1.0:

            self.p = tf.cast(self.raw_p, dtype=self.dtype)

            # Selects the call method to avoid evaluating the absolute
            # value

            self.call_method = self.even_exponent_loss

        # Otherwise, changes p to half of p to compensate for the smooth
        # absolute value

        elif self.raw_p>=1.0:

            self.p = tf.cast(self.raw_p*0.5, dtype=self.dtype)

            # Selects the call method to ensure evaluating the absolute
            # value before evaluating the Lp norm

            self.call_method = self.non_even_exponent_loss

        else:

            raise ValueError("The 'p' exponent in 'LpNormError' loss c"+
            "lass must be greater than or equal to 1. Currently, it is"+
            ": "+str(p))

        # Sets the reciprocal of the original p exponent

        self.p_reciprocal = tf.cast(1.0/self.raw_p, dtype=self.dtype)

    # Defines the generic call method to be differentiated using AD

    def call(self, y_true, y_model):

        return self.call_method(y_true, y_model)

    # Defines the call method to evaluate the loss function when the ex-
    # ponent is an even integer

    def even_exponent_loss(self, y_true, y_model):

        # Computes the Lp norm of the error tensor. Adds a constant epsi-
        # lon to the computation of the p-th root to avoid exploding gra-
        # dients when the p-th root tends to zero

        return tf.pow(tf.reduce_sum(tf.pow(y_true-y_model, self.p), 
        axis=-1)+self.epsilon, self.p_reciprocal)

    # Defines the call method to evaluate the loss function when the ex-
    # ponent is not an integer or when it is an odd integer. Thus, the 
    # absolute value is taken first

    def non_even_exponent_loss(self, y_true, y_model):

        # Evaluates the smooth absolute value of the error tensor first

        absolute_error = tf.square(y_true-y_model)+self.epsilon

        # Computes the Lp norm of the error tensor

        return tf.pow(tf.reduce_sum(tf.pow(absolute_error, self.p),
        axis=-1), self.p_reciprocal)

    # Defines a safe and numerically stable evaluation of this loss 
    # function common to both even and odd exponents. With the caveat
    # that the absolute value is calculated using tf.abs

    def scale_evaluate_and_rescale_method(self, y_true, y_model):

        # Evaluates the absolute error tensor

        absolute_error_tensor = tf.abs(y_true-y_model)

        # Gets the maximum error per sample as a tensor

        maximum_per_sample = tf.reduce_max(absolute_error_tensor, axis=
        -1, keepdims=True)

        # Divides the error tensor by the maximum absolute error. But 
        # care is taken to avoid division by zero

        absolute_error_tensor = tf.math.divide_no_nan(
        absolute_error_tensor, maximum_per_sample)

        # Evaluates the Lp norm by computing the powers with the scaled
        # components then muliplying by the scale factor

        return tf.squeeze(maximum_per_sample, axis=-1)*tf.pow(
        tf.reduce_sum(tf.pow(absolute_error_tensor, self.p), axis=-1),
        self.p_reciprocal)

    # Defines the get-config method to ensure serialization when saving
    # and loading models

    def get_config(self):

        config = super().get_config()

        config.update({"p": self.raw_p, "dtype": self.raw_dtype, "epsi"+
        "lon": self.raw_epsilon, "use_stable_implementation":
        self.use_stable_implementation})

        return config