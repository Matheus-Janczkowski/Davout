# Routine to store classes for special and custom neural network archi-
# tectural

import tensorflow as tf

from ...PythonicUtilities.dictionary_tools import verify_obligatory_and_optional_keys

########################################################################
#                        Gated quotient space NN                       #
########################################################################

# Defines a class for a neural network that operates on a quotient space
# defined by a gate function. This function nullifies a subset of input
# neurons once the other set is null. All layers are biasless

class GatedQuotientSpace:

    def __init__(self, layer_self, activation_functionDict, 
    custom_activations_class, architecture_info_dict):
        
        # Verifies if the necessary information of the architecture pro-
        # vided by the user has been actually supplied

        architecture_info_dict = verify_obligatory_and_optional_keys(
        architecture_info_dict, {"name": {}, "quotient space dimension": 
        {"type": int, "description": "Integer with the number of neuro"+
        "ns attached to inputs from the quotient space. The quotient s"+
        "pace has the input vector which makes the whole neural networ"+
        "k output the null when the former is also null."}}, 
        {"damping factor": {"type": float, "description": "Float facto"+
        "r used in the exponential of the gate function", "default": 1E3
        }}, "custom_architecture", "GatedQuotientSpace")

        # Saves the objects saved into the MixedActivationLayer class 
        # instance

        self.layer_self = layer_self

        self.layer_number = self.layer_self.code_given_info_class.layer

        self.quotient_space_dimensionality = architecture_info_dict["q"+
        "uotient space dimension"]

        self.damping_factor = tf.constant(architecture_info_dict["damp"+
        "ing factor"], dtype=
        self.layer_self.code_given_info_class.float_dtype)

        # Gets all the live activation functions into a tuple

        self.activation_list = tuple([
        layer_self.live_activationFunctions[name] for name in (
        activation_functionDict.keys())])

        # Defines the method to call this layer and get the forward res-
        # ponse. If this layer is the first layer

        if self.layer_number==0:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.layer_self.call_from_input_method = self.first_layer_call_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.layer_self.call_given_parameters = self.first_layer_call_with_parameters

        # If it is not the first layer

        else:

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
    # a gate function is necessary to nullify a section of the input 
    # vector

    def first_layer_call_from_input(self, input):

        # If it's the first layer, the input tensor must be sliced: one
        # bit for the quotient space, the rest for the remaining input 
        # space

        quotient_space_input = input[..., 
        :self.quotient_space_dimensionality]

        # Evaluates the norm of the section of the input tensor corres-
        # ponding to the quotient space

        quotient_input_norm = tf.norm(quotient_space_input, ord='eucli'+
        'dean', axis=1, keepdims=True)

        # Calculates the gate function and returns a tensor [n_samples,1]

        gate_function = (1.0-tf.exp(-self.damping_factor*
        quotient_input_norm))

        # Gets the input tensor by multiplying the gate function to the
        # section corresponding to the remaining input space

        gated_input = tf.concat([input[..., 
        :self.quotient_space_dimensionality], gate_function*input[..., 
        self.quotient_space_dimensionality:]], axis=-1)

        # Multiplies the gated input by the weight matrix

        x_splits = tf.split(self.layer_self.dense(gated_input), 
        self.layer_self.neurons_per_activation, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations = [activation_function(split) for (
        activation_function), split in zip(self.activation_list, 
        x_splits)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row

        return tf.concat(output_activations, axis=-1)
    
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
    # a gated architecture from the input

    def first_layer_call_with_parameters(self, layer_input, parameters):
        
        # If it's the first layer, the input tensor must be sliced: one
        # bit for the quotient space and another for the remaining space

        quotient_space_input = layer_input[..., 
        :self.quotient_space_dimensionality]

        # Evaluates the norm of the section of the input tensor corres-
        # ponding to the quotient space

        quotient_input_norm = tf.norm(quotient_space_input, ord='eucli'+
        'dean', axis=1, keepdims=True)

        # Calculates the gate function and returns a tensor [n_samples,1]

        gate_function = (1.0-tf.exp(-self.damping_factor*
        quotient_input_norm))

        # Gets the input tensor by multiplying the gate function to the
        # section corresponding to the remaining input space

        gated_input = tf.concat([layer_input[..., 
        :self.quotient_space_dimensionality], gate_function*layer_input[
        ..., self.quotient_space_dimensionality:]], axis=-1)

        # Gets the weights, then, reshapes them according to the parame-
        # ters shapes

        weights = self.update_layer_parameters(parameters)

        # Multiplies the weights by the inputs, thennsplits by activa-
        # tion function family

        x_splits = tf.split(tf.matmul(gated_input, weights), 
        self.layer_self.neurons_per_activation, axis=-1)

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
        
        # Gets the weights, then, reshapes them according to the parame-
        # ters shapes

        weights = self.update_layer_parameters(parameters)

        # Multiplies the weights by the inputs, then splits by activation 
        # function family

        x_splits = tf.split(tf.matmul(layer_input, weights), 
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
        self.layer_self.total_neurons, use_bias=False)

        # Constructs the layer

        super(type(self.layer_self), self.layer_self).build(input_shape)

    # Defines a function to get the flat and sliced parameters, then, 
    # converts it to tensors given the shapes of the trainables varia-
    # bles

    def layer_update_parameters(self, flat_parameters):

        # Gets the weights, then, reshapes them according to the parame-
        # ters shapes

        weights = tf.reshape(flat_parameters, 
        self.layer_self.trainable_variables_shapes[0])

        return weights
    
    # Defines a function to apply the weights and biases in the model

    def apply_parameters_to_layer(self, flat_parameters):

        # Gets the weights from the flat vector

        weights = self.update_layer_parameters(flat_parameters)

        # Updates the weights and biases matrices

        self.layer_self.dense.kernel.assign(weights)