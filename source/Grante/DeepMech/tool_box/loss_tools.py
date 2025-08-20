# Routine to store loss functions

import tensorflow as tf

from ..tool_box import ANN_tools

from ..tool_box import differentiation_tools as diff_tools

########################################################################
#                            Loss functions                            #
########################################################################

# Defines a function to give the loss function as the product of the mo-
# del outputs with a matrix of coeficients

def linear_loss(x, model, coefficient_matrix):

    # Gets the response of the model and multiplies by the coefficient
    # matrix, then, sums everything together

    y = model(x)

    return tf.reduce_sum(coefficient_matrix*y)

########################################################################
#             Loss functions updating the model parameters             #
########################################################################

# Defines a function to build a function to give the loss function and 
# its gradient as a function of the model trainable parameters

def build_loss_varying_model_parameters(model, loss, input_tensor):
    
    def parameterizable_loss(model_parameters_numpy, model=model, loss=
    loss, input_tensor=input_tensor):

        # Reassigns the same model parameters

        model = ANN_tools.update_model_parameters(model, 
        model_parameters_numpy)

        # Gets the loss and the gradient

        loss_value, gradient = diff_tools.scalar_gradient_wrt_trainable_params(
        loss, model, input_tensor)

        # Converts both to numpy

        return (loss_value.numpy(), 
        diff_tools.convert_scalar_gradient_to_numpy(gradient))
    
    return parameterizable_loss