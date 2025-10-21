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
#        Loss functions gradients updating the model parameters        #
########################################################################

# Defines a function to build a function to give the gradient of the 
# loss function w.r.t. the model trainable parameters as a function of 
# the model trainable parameters

def build_loss_gradient_varying_model_parameters(model, loss, 
input_tensor, trainable_variables_type="tensorflow", model_true_values=
None, convex_input_model=None, regularizing_function="smooth absolute "+
"value", verbose=False):
    
    """
    Function to build the gradient of the loss function and the vector 
    of trainable parameters of a neural network model. The evaluated 
    gradient of the model is with respect to the trainabale parameters.
    
    The inputs are the model itself (Keras or custom layer), the loss 
    class or function, the input tensor as a tensorflow tensor"""
    
    # Defines a function to give the gradient of a scalar loss function
    # w.r.t. the trainable parameters of the model given as a numpy ar-
    # ray
    
    if trainable_variables_type=="numpy":

        # Gets the model parameters as a list

        model_parameters = parameters_tools.model_parameters_to_numpy(
        model)

        # Assembles the class to evaluate the gradient

        gradient_class = diff_tools.ScalarGradientWrtTrainableParams(
        loss, input_tensor)
    
        def parameterizable_loss(model_parameters_numpy, model=model):

            # Reassigns the same model parameters

            model = parameters_tools.update_model_parameters(model, 
            model_parameters_numpy)

            # Gets the gradient, and converts it to numpy

            return diff_tools.convert_scalar_gradient_to_numpy(
            gradient_class(model))
        
        return parameterizable_loss, model_parameters
    
    # Defines a function to give the gradient of a scalar loss function
    # w.r.t. the trainable parameters of the model given as a tensorflow
    # array when the model is convex input by construction
    
    elif convex_input_model:

        # Gets the 1D tensor of model trainable parameters and their 
        # tensors' shapes

        model_parameters, parameters_shapes = parameters_tools.model_parameters_to_flat_tensor_and_shapes(
        model)

        if verbose:

            print("\nThe shape of the model parameters flat tensor: "+
            str(model_parameters.shape)+".\n\nThere are "+str(len(
            parameters_shapes))+" tensors of trainable parameters\n")

        # Gets the class instance to evaluate the gradient and returns 
        # it alongside the 1D tensor of model parameters

        gradient_class = diff_tools.ScalarGradientWrtTrainableParamsGivenParametersConvexModel(
        loss, model, input_tensor, parameters_shapes, 
        regularizing_function=regularizing_function, model_true_values=
        model_true_values, parameters_type=model_parameters.dtype)
        
        return gradient_class, model_parameters
    
    # Defines a function to give the gradient of a scalar loss function
    # w.r.t. the trainable parameters of the model given as a tensorflow
    # array
    
    elif trainable_variables_type=="tensorflow":

        # Gets the 1D tensor of model trainable parameters and their 
        # tensors' shapes

        model_parameters, parameters_shapes = parameters_tools.model_parameters_to_flat_tensor_and_shapes(
        model)

        # Gets the class instance to evaluate the gradient and returns 
        # it alongside the 1D tensor of model parameters

        gradient_class = diff_tools.ScalarGradientWrtTrainableParamsGivenParameters(
        loss, model, input_tensor, parameters_shapes, model_true_values=
        model_true_values)
        
        return gradient_class, model_parameters
    
    else:

        raise NameError("The flag 'trainable_variables_type' must be '"+
        "numpy' or 'tensorflow'")