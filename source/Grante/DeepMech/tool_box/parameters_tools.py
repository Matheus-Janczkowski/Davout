# Routine to store functionalities to evaluate NN models and manage 
# their parameters

import numpy as np

import tensorflow as tf
    
########################################################################
#                     Assembly of model parameters                     #
########################################################################

# Defines a function to get the model parameters, flatten them, and con-
# vert to a numpy array

def model_parameters_to_numpy(model, as_numpy=True):

    flat_parameters = tf.concat([tf.reshape(var, [-1]) for var in (
    model.trainable_variables)], axis=0)

    return flat_parameters.numpy() if as_numpy else flat_parameters

# Defines a function to update the trainable parameters of a NN model 
# given a list of them

def update_model_parameters(model, new_parameters):

    # Converts the parameters to tensorflow constant

    new_parameters = tf.constant(new_parameters, dtype=
    model.trainable_variables[0].dtype)

    # Iterates over the layers

    offset = 0

    for layer in model.trainable_variables:

        size = tf.size(layer)

        # Gets a slice of the parameters

        parameters_slice = tf.reshape(new_parameters[offset:(offset+size
        )], layer.shape)

        # Assigns the values

        layer.assign(parameters_slice)

        # Updates the offset 

        offset += size

    # Returns the updated model

    return model

# Defines a function to get the model parameters, flatten them, and keep
# them as a tensor

def model_parameters_to_flat_tensor_and_shapes(model):

    # Initializes the flat list of parameters and the list of shapes of
    # the parameters tensors in each layer
    
    flat_parameters = []

    shapes = []

    # Iterates over the layers

    for layer in model.trainable_variables:

        # Adds the shape of the layer tensor of parameters

        shapes.append(layer.shape)

        # Adds the parameters as a vector tensor

        flat_parameters.append(tf.reshape(layer, [-1]))

    # Concatenates and returns everything

    return tf.concat(flat_parameters, axis=0), shapes

# Defines a function to get the flat tensor of parameters back to the
# tensors of parameters

def unflatten_parameters(flat_parameters, shapes):

    # Initializes the tensors list and the index
    
    tensors = []

    parameter_index = 0

    # Iterates over the list of shapes of the tensors

    for shape in shapes:

        # Gets the number of elements for this tensor

        size = np.prod(shape)

        # Gets the parameters for this tensor, and appends to the ten-
        # sors list

        tensors.append(tf.reshape(flat_parameters[parameter_index:(
        parameter_index+size)], shape))

        # Updates the index counter

        parameter_index += size

    return tensors

# Defines a function to compute the output of a NN model given the para-
# meters (weights and biases) as input

def model_output_given_trainable_parameters(input_variables, model,
model_parameters, parameters_shapes):
    
    # Gets the parameters from a 1D tensor to the conventional tensor 
    # format for building models
    
    parameters = unflatten_parameters(model_parameters, 
    parameters_shapes)

    # Initializes the index of the parameters to be read in the new ten-
    # sor format
    
    parameter_index = 0

    # Iterates through the layers

    for layer in model.layers:
        
        # Verifies if the layer has the call with parameters attribute,
        # which signals it as an instance of the MixedActivationLayer 
        # class

        if hasattr(layer, "call_with_parameters"):

            # Gets the number of parameters in this layer

            n_parameters = len(layer.trainable_variables)

            # Gets the output of this layer from the method call with 
            # parameters
            
            input_variables = layer.call_with_parameters(input_variables, 
            parameters[parameter_index:(parameter_index+n_parameters)])

            # Updates the index of the parameter tensors

            parameter_index += n_parameters

        # Verifies if it is not an input layer, throws an error, because
        # the input layer does not do anything really

        elif layer.__class__.__name__!="InputLayer":

            raise TypeError("A layer is not an instance of 'MixedActiv"+
            "ationLayer' nor of 'InputLayer'")
        
    # Returns the input variables as the output of the NN model, because
    # it has been passed through the NN model
        
    return input_variables

def linear_loss(theta, model, x, coeffs, shapes):
    y = model_output_given_trainable_parameters(x, model, theta, shapes)
    return tf.reduce_sum(coeffs * y)

@tf.function
def loss_and_grad(theta, model, x, coeffs, shapes):
    with tf.GradientTape() as tape:
        tape.watch(theta)
        L = linear_loss(theta, model, x, coeffs, shapes)
    g = tape.gradient(L, theta)
    return L, g