# Routine to store functionalities to evaluate NN models and manage 
# their parameters

import numpy as np

import tensorflow as tf

########################################################################
#                       Parameters initialization                      #
########################################################################

# Defines a function to reinitialize a model parameters using its own i-
# initializers

def reinitialize_model_parameters(model):

    # Iterates through the layers of parameters

    for i in range(len(model.layers)):

        # Gets the layer

        layer = model.layers[i]

        # Treats the standard keras layer case

        if isinstance(layer, tf.keras.layers.Dense):

            # Gets the initializer for the weights

            initializer = type(layer.kernel_initializer)() 

            # Reinitializes the weights

            model.layers[i].kernel.assign(initializer(shape=
            layer.kernel.shape, dtype=layer.kernel.dtype))

            # Gets the initializer for the biases

            initializer = type(layer.bias_initializer)() 
            
            # Reinitializes the biases

            model.layers[i].bias.assign(initializer(shape=
            layer.bias.shape, dtype=layer.bias.dtype))

        # Treats the case of a layer as instance of MixedActivationLayer

        elif hasattr(layer, "functions_dict"):

            # Iterates through the attributes of the class to look for
            # dense objects

            for attribute_value in vars(layer).values():

                # Verifies if it is a weight matrix

                if (hasattr(attribute_value, "kernel") and hasattr(
                attribute_value, "kernel_initializer")):
                    
                    if attribute_value.kernel is not None:

                        # Gets the initializer for the weights

                        initializer = type(attribute_value.kernel_initializer)() 

                        # Reinitializes the weights

                        attribute_value.kernel.assign(initializer(shape=
                        attribute_value.kernel.shape, dtype=
                        attribute_value.kernel.dtype))

                # Verifies if it is a bias vector

                elif  (hasattr(attribute_value, "bias") and hasattr(
                attribute_value, "bias_initializer")):
                    
                    if attribute_value.bias is not None:

                        # Gets the initializer for the biases

                        initializer = type(attribute_value.bias_initializer)() 
                        
                        # Reinitializes the biases

                        attribute_value.bias.assign(initializer(shape=
                        attribute_value.bias.shape, dtype=
                        attribute_value.bias.dtype))

        # Some layers don't have parameters to be reinitialized. Raises
        # an error only if this layer is not one of such layer types

        elif not (isinstance(layer, tf.keras.layers.InputLayer)):

            raise TypeError("The parameters of the model cannot be rei"+
            "nitialized because it can either handle a standard keras "+
            "layer or the MixedActivationLayer. The current layer is: "+
            str(layer))
        
    return model
    
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

def update_model_parameters(model, new_parameters, regularizing_function=
None):

    # Converts the parameters to tensorflow constant

    new_parameters = tf.constant(new_parameters, dtype=
    model.trainable_variables[0].dtype)

    # Iterates over the layers

    offset = 0

    for i in range(len(model.trainable_variables)):

        size = tf.size(model.trainable_variables[i])

        # Gets a slice of the parameters

        parameters_slice = tf.reshape(new_parameters[offset:(offset+size
        )], model.trainable_variables[i].shape)

        # Assigns the values and regularizes only if the layer is of 
        # weights

        if (regularizing_function is None) or (not hasattr(
        model.trainable_variables[i], "regularizable")):

            model.trainable_variables[i].assign(parameters_slice)

        else:

            model.trainable_variables[i].assign(regularizing_function(
            parameters_slice))

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

    # Iterates over the tensors of weights and biases

    for layer in model.trainable_variables:

        # Adds the shape of the tensor of parameters, and adds if it has
        # the regularizable attribute

        shapes.append((layer.shape, getattr(layer, "name")))

        # Adds the parameters as a vector tensor

        flat_parameters.append(tf.reshape(layer, [-1]))

    # Concatenates and returns everything

    return tf.concat(flat_parameters, axis=0), shapes

# Defines a function to get the split the flat tensor of parameters into
# the different parameters tensors

@tf.function
def unflatten_parameters(flat_parameters, n_parameters_per_tensor):

    return tf.split(flat_parameters, n_parameters_per_tensor, axis=-1)

# Defines a function to get the flat tensor of parameters back to the
# tensors of parameters and regularizes each weight using the regulari-
# zation function

@tf.function
def unflatten_regularize_parameters(flat_parameters, shapes, 
regularization_function):

    # Initializes the tensors list and the index
    
    tensors = []

    parameter_index = 0

    # Iterates over the list of shapes of the tensors

    for shape in shapes:

        # Gets the number of elements for this tensor

        size = np.prod(shape[0])

        # Gets the the flag to tell if the tensor is to be regularized
        # or not

        regularizable_tensor = shape[1]

        # Gets the parameters for this tensor, and appends to the ten-
        # sors list. Regularizes only if the tensor is indeed to be re-
        # gularized

        if regularizable_tensor:

            tensors.append(regularization_function(tf.reshape(
            flat_parameters[parameter_index:(parameter_index+size)], 
            shape[0])))

        else:

            tensors.append(tf.reshape(flat_parameters[parameter_index:(
            parameter_index+size)], shape[0]))

        # Updates the index counter

        parameter_index += size

    return tensors

########################################################################
#             Call with parameters method for custom layers            #
########################################################################

# Defines a class to compute the output of a NN model given the parame-
# ters (weights and biases) as input. The regularizing function modula-
# tes the weights and biases, one example is with convex-input neural
# networks, where the weights must be positive

class ModelOutputGivenTrainableParameters:

    def __init__(self, model, parameters_shapes):

        # Stores the model

        self.model = model

        self.parameters_shapes = parameters_shapes

        # Initializes lists of initial and final indices for the list of
        # parameters

        self.initial_index = []

        self.final_index = []
        
        parameter_index = 0

        # Initializes a list of layer objects

        self.layers_list = []

        # Validates the consistency of the layers of the model

        for layer in self.model.layers:
            
            # Verifies if the layer has the call with parameters attri-
            # bute, which signals it as an instance of the MixedActiva-
            # tionLayer class

            if hasattr(layer, "call_with_parameters"):
                
                self.layers_list.append(layer)

                # Gets the number of parameters in this layer

                n_parameters = len(layer.trainable_variables)
                
                # Gets the initial and final index in the list of traina-
                # ble parameters

                self.initial_index.append(parameter_index)
                
                self.final_index.append(parameter_index+n_parameters)

                # Updates the index of the parameter tensors

                parameter_index += n_parameters

            # Verifies if it is not an input layer, throws an error, be-
            # cause the input layer does not do anything really
                
            elif layer.__class__.__name__!="InputLayer":

                raise TypeError("Layer '"+str(layer.__class__.__name__)+
                "' is not an instance of 'MixedActivationLayer' nor of"+
                " 'InputLayer'")
            
        # Gets the number of parameters per variable tensor

        self.n_parameters_per_tensor = [np.prod(shape[0]) for shape in (
        self.parameters_shapes)]

        # Converts the lists of indices to tuples

        self.initial_index = tuple(self.initial_index)

        self.final_index = tuple(self.final_index)

        self.n_layers = len(self.layers_list)

        self.layers_list = tuple(self.layers_list)

        self.n_parameters_per_tensor = tuple(self.n_parameters_per_tensor)

    # Defines a method to evaluate the model

    @tf.function
    def __call__(self, input_variables, model_parameters):
        
        # Gets the parameters from a 1D tensor to the conventional ten-
        # sor format for building models

        parameters = unflatten_parameters(model_parameters, 
        self.n_parameters_per_tensor)

        # Iterates through the layers

        for i in range(self.n_layers):

            # Gets the layer

            layer = self.layers_list[i]

            # Gets the output of this layer from the method call with 
            # parameters
            
            input_variables = layer.call_with_parameters(input_variables, 
            parameters[self.initial_index[i]:self.final_index[i]])
            
        # Returns the input variables as the output of the NN model, be-
        # cause it has been passed through the NN model
            
        return input_variables

    # Defines a method to update the model parameters

    @tf.function
    def update_model_parameters(self, model, model_parameters):
        
        # Gets the parameters from a 1D tensor to the conventional ten-
        # sor format for building models

        parameters = unflatten_parameters(model_parameters, 
        self.n_parameters_per_tensor)

        # Iterates through the layers

        for i, layer in enumerate(model.layers):

            # Updates the model parameters in place given the flat vec-
            # tor of parameters

            layer.apply_parameters_to_layer(parameters[
            self.initial_index[i]:self.final_index[i]])

########################################################################
#           Call with parameters method for non-custom layers          #
########################################################################

# Defines a function to work as the method call_with_parameters in a Ke-
# ras Dense layer

def keras_dense_call_with_parameters(self, inputs, parameters):

    # Gets the weights and biases

    weights, biases = parameters

    # Multiplies the weights by the inputs, adds the biases and applies
    # the activation function

    return self.activation(tf.matmul(inputs, weights)+biases)

# Defines a class to call Keras dense layers given parameters. This class
# is meant to store the parameters shape as well

class KerasCallWithParameters:

    def __init__(self, trainable_variables_shapes, activation):
        
        # Saves the shapes of the trainable variables and the activation
        # function

        self.trainable_variables_shapes = trainable_variables_shapes

        self.activation = activation

    # Defines a method to get a vector of split flat vectors, then, calls
    # the activation function

    def __call__(self, inputs, parameters):

        # Gets the weights and biases, then, reshapes them according to 
        # the parameters shapes

        weights = tf.reshape(parameters[0], 
        self.trainable_variables_shapes[0])

        biases = tf.reshape(parameters[1], 
        self.trainable_variables_shapes[1])

        # Multiplies the weights by the inputs, adds the biases and ap-
        # plies the activation function

        return self.activation(tf.matmul(inputs, weights)+biases)