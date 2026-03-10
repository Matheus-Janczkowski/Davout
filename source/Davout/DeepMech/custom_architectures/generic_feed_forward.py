# Routine to store classes for special and custom neural network archi-
# tectural

import tensorflow as tf

from ...PythonicUtilities.dictionary_tools import verify_obligatory_and_optional_keys

########################################################################
#                        Generic feed-forward NN                       #
########################################################################

# Defines a class for a generic feed forward neural network

class GenericFeedForwardNNs:

    def __init__(self, layer_self, activation_functionDict, 
    custom_activations_class, architecture_info_dict):
        
        # Verifies if the necessary information of the architecture pro-
        # vided by the user has been actually supplied

        architecture_info_dict = verify_obligatory_and_optional_keys(
        architecture_info_dict, ["name"], {}, "custom_architecture", 
        "GenericFeedForwardNNs")

        # Saves the objects saved into the MixedActivationLayer class 
        # instance

        self.layer_self = layer_self

        # Gets all the live activation functions into a tuple

        self.activation_list = tuple([
        layer_self.live_activationFunctions[name] for name in (
        activation_functionDict.keys())])

        # Defines the method that will be used to call the layer's 
        # response when the trainable parameters are fixed

        self.layer_self.call_from_input_method = self.layer_call_from_input

        # Defines the method that will be used to call the layer's 
        # response when the trainable parameters are given

        self.layer_self.call_given_parameters = self.call_with_parameters

        # Selects the method to update the model parameters in place

        self.layer_self.apply_parameters_to_layer = self.apply_parameters_to_layer

        # Selects the method for reshaping the model parameters from a
        # flat vector

        self.update_layer_parameters = self.layer_update_parameters
    
    # Defines a method for getting the layer value given the input when
    # an accessory layer is NOT necessary

    def layer_call_from_input(self, input):

        # Initializes the input as dense layer and split it into the 
        # different families of activation functions. This keeps the in-
        # put as a tensor

        x_splits = tf.split(self.layer_self.dense(input), 
        self.layer_self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations = [activation_function(split) for (
        activation_function), split in zip(self.activation_list, 
        x_splits)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row

        return tf.concat(output_activations, axis=-1)
    
    # Defines a function to get the output of a layer that constitutes a
    # fully input convex neural network. The first layer might have ne-
    # gative weights

    def call_with_parameters(self, layer_input, parameters):
        
        # Gets the weights and biases, then, reshapes them according to 
        # the parameters shapes

        weights, biases = self.update_layer_parameters(parameters)

        # Multiplies the weights by the inputs and adds the biases, then
        # splits by activation function family

        x_splits = tf.split(tf.matmul(layer_input, weights)+biases, 
        self.layer_self.neurons_per_activation, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations = [activation_function(split) for (
        activation_function), split in zip(self.activation_list, 
        x_splits)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row

        return tf.concat(output_activations, axis=-1)
    
    # Defines a function to build this layer during serialization

    def mixed_layer_builder(self, input_shape):

        # Constructs a dense layer with identity activation functions

        self.layer_self.dense = tf.keras.layers.Dense(
        self.layer_self.total_neurons)

        # Constructs the layer

        super(type(self.layer_self), self.layer_self).build(input_shape)

    # Defines a function to get the flat and sliced parameters, then, 
    # converts it to tensors given the shapes of the trainables varia-
    # bles

    def layer_update_parameters(self, flat_parameters):

        # Gets the weights and biases, then, reshapes them according to 
        # the parameters shapes

        weights = tf.reshape(flat_parameters[0], 
        self.layer_self.trainable_variables_shapes[0])

        biases = tf.reshape(flat_parameters[1], 
        self.layer_self.trainable_variables_shapes[1])

        return weights, biases
    
    # Defines a function to apply the weights and biases in the model

    def apply_parameters_to_layer(self, flat_parameters):

        # Gets the weights and biases from the flat vector

        weights, biases = self.update_layer_parameters(flat_parameters)

        # Updates the weights and biases matrices

        self.layer_self.dense.kernel.assign(weights)

        self.layer_self.dense.bias.assign(biases)