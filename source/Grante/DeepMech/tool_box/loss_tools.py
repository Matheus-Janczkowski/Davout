# Routine to store loss functions

import tensorflow as tf

from ..tool_box import differentiation_tools as diff_tools

from ..tool_box import parameters_tools

########################################################################
#                            Loss functions                            #
########################################################################

# Defines a function to give the loss function as the product of the mo-
# del outputs with a matrix of coeficients

def linear_loss(model_output, coefficient_matrix):

    # Gets the response of the model and multiplies by the coefficient
    # matrix, then, sums everything together

    return tf.reduce_sum(coefficient_matrix*model_output)

########################################################################
#             Loss functions updating the model parameters             #
########################################################################

# Defines a function to build a function to give the gradient of the 
# loss function w.r.t. the model trainable parameters as a function of 
# the model trainable parameters

def build_loss_varying_model_parameters(model, loss, input_tensor, 
trainable_variables_type="tensorflow"):
    
    # Defines a function to give the gradient of a scalar loss function
    # w.r.t. the trainable parameters of the model given as a numpy ar-
    # ray
    
    if trainable_variables_type=="numpy":

        # Gets the model parameters as a list

        model_parameters = parameters_tools.model_parameters_to_numpy(
        model)
    
        def parameterizable_loss(model_parameters_numpy, model=model, 
        loss=loss, input_tensor=input_tensor):

            # Reassigns the same model parameters

            model = parameters_tools.update_model_parameters(model, 
            model_parameters_numpy)

            # Gets the gradient

            gradient = diff_tools.scalar_gradient_wrt_trainable_params(
            loss, model, input_tensor)

            # Converts both to numpy

            return diff_tools.convert_scalar_gradient_to_numpy(gradient)
        
        return parameterizable_loss, model_parameters
    
    # Defines a function to give the gradient of a scalar loss function
    # w.r.t. the trainable parameters of the model given as a tensorflow
    # array
    
    elif trainable_variables_type=="tensorflow":

        # Gets the 1D tensor of model trainable parameters and their 
        # tensors' shapes

        model_parameters, parameters_shapes = parameters_tools.model_parameters_to_flat_tensor_and_shapes(
        model)
    
        def parameterizable_loss(model_parameters_tensorflow, model=
        model, loss=loss, input_tensor=input_tensor, parameters_shapes=
        parameters_shapes):

            # Gets the gradient

            gradient = diff_tools.scalar_gradient_wrt_trainable_params_given_parameters(
            loss, model, input_tensor, model_parameters_tensorflow,
            parameters_shapes)

            # Returns it

            return gradient
        
        return parameterizable_loss, model_parameters
    
    else:

        raise NameError("The flag 'trainable_variables_type' must be '"+
        "numpy' or 'tensorflow'")