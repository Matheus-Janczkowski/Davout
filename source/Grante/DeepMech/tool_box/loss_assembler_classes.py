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
    tf.keras.losses.Reduction.SUM):

        super().__init__(name=name, reduction=reduction)

        # Stores some parameters

        self.trainable_coefficient_matrix = trainable_coefficient_matrix

        self.tensorflow_type = dtype

        # Stores the coefficient matrix as a TensorFlow constant
        
        self.coefficient_matrix = tf.Variable(coefficient_matrix, dtype=
        dtype, trainable=trainable_coefficient_matrix)

    # Redefines the call method

    def call(self, expected_response, model_response):
        
        return loss_tools.linear_loss(model_response, 
        self.coefficient_matrix)
    
    # Defines a method to update the coefficient matrix

    def update_arguments(self, coefficient_matrix):

        # Updates the coefficient matrix as a TensorFlow constant
        
        self.coefficient_matrix = tf.Variable(coefficient_matrix, dtype=
        self.tensorflow_type, trainable=self.trainable_coefficient_matrix)
    
    # Redefines configurations for model saving

    def get_config(self):

        config = super().get_config()

        config.update({"coefficient_matrix": 
        self.coefficient_matrix.numpy().tolist(), "dtype":
        self.coefficient_matrix.dtype.name})

        return config