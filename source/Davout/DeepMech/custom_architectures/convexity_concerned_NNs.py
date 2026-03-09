# Routine to store classes for special and custom neural network archi-
# tectural

import tensorflow as tf

from ..tool_box.activation_function_utilities import verify_activationDict

from ..tool_box.numerical_tools import build_tensorflow_math_expressions

from ..tool_box.custom_activation_functions import CustomActivationFunctions

########################################################################
#                      Fully convex neural network                     #
########################################################################

# Defines a class for fully input-convex architecture using Amos et al

class FullyConvexNNs:

    def __init__(self, layer_self, activation_functionDict, 
    custom_activations_class, activations_accessory_layer_dict, 
    input_size_main_network, layer, regularization_function, dtype):

        # Gets all the live activation functions into a tuple

        self.activation_list = tuple([
        layer_self.live_activationFunctions[name] for name in (
        activation_functionDict.keys())])

        # Defines the method that will be used to call the layer's 
        # response when the trainable parameters are fixed

        self.call_from_input_method = self.layer_call_from_input

        # Defines the method that will be used to call the layer's 
        # response when the trainable parameters are given

        self.call_given_parameters = self.call_with_parameters

        # Selects the method for reshaping the model parameters from a
        # flat vector

        if layer==0:

            self.update_layer_parameters = self.first_layer_update_parameters

        # Otherwise, if it is any of the intermediate layers

        else:

            self.update_layer_parameters = self.middle_layer_update_parameters

        # Gets the regularization function

        self.regularization_function = build_tensorflow_math_expressions(
        regularization_function, dtype)

        # Saves the objects saved into the MixedActivationLayer class 
        # instance

        self.layer_self = layer_self
    
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

        # If the model is to be fully input convex, the weight tensors
        # must be strictly positive, except for the first layer

        if self.layer_self.layer!=0:

            # Builds the layer's parameters

            self.layer_self.dense.build(input_shape)

    # Defines a function to get the flat and sliced parameters, then, 
    # converts it to tensors given the shapes of the trainables varia-
    # bles

    def first_layer_update_parameters(self, flat_parameters):

        # Gets the weights and biases, then, reshapes them according to 
        # the parameters shapes. The weights do not have to be regulari-
        # zed in the first layer

        weights = tf.reshape(flat_parameters[0], 
        self.layer_self.trainable_variables_shapes[0])

        biases = tf.reshape(flat_parameters[1], 
        self.layer_self.trainable_variables_shapes[1])

        return weights, biases

    # Defines a function to get the flat and sliced parameters, then, 
    # converts it to tensors given the shapes of the trainables varia-
    # bles

    def middle_layer_update_parameters(self, flat_parameters):

        # Gets the weights and biases, then, reshapes them according to 
        # the parameters shapes. The weights have to be regularized in 
        # the middle and output layers

        weights = self.regularization_function(tf.reshape(
        flat_parameters[0], self.layer_self.trainable_variables_shapes[0
        ]))

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

########################################################################
#                    Partially convex neural network                   #
########################################################################

# Defines a class for partially input-convex architecture using Amos et
# al

class PartiallyConvexNNs:

    def __init__(self, layer_self, activation_functionDict, 
    custom_activations_class, activations_accessory_layer_dict, 
    input_size_main_network, layer, regularization_function_name, dtype):
        
        # Stores variables that will be used in the get_config for seri-
        # alization and class rebuilding

        self.layer_self = layer_self

        self.input_size_main_network = input_size_main_network
        
        # Verifies if there are activations in the dictionary of activa-
        # tions for the accessory layer

        if not activations_accessory_layer_dict:

            raise ValueError("'activations_accessory_layer_dict' must "+
            "have at least one activation function for setting up 'Par"+
            "tiallyConvexNNs', since this class constructs an accessor"+
            "ry neural network")

        # Adds the dictionary of live-wired activation functions. But 
        # checks if it is given as argument

        if (layer_self.live_activationFunctions is None) or (
        layer_self.live_activationFunctions=={}): 

            # Concatenates the two dictionaries, but overrides the val-
            # ues of the accessory dictionary with the values of the con-
            # ventional one

            self.live_activationFunctions, *_ = verify_activationDict(
            activations_accessory_layer_dict | activation_functionDict, 
            layer, {}, True, custom_activations_class)

        else:

            self.live_activationFunctions = layer_self.live_activationFunctions

        # Gets all the live activation functions into a tuple

        self.activation_list = tuple([self.live_activationFunctions[name
        ] for name in activation_functionDict.keys()])

        # Gets all the live activation functions into a tuple

        self.activation_list_acessory_network = tuple([
        self.live_activationFunctions[name] for name in (
        activations_accessory_layer_dict.keys())])

        # Defines the method to call this layer and get the forward res-
        # ponse. If this layer is the first layer

        if layer==0:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.call_from_input_method = self.first_layer_call_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.call_given_parameters = self.first_layer_call_partially_convex_with_parameters

            # Selects the method for reshaping the model parameters from 
            # a flat vector

            self.update_layer_parameters = self.first_layer_update_parameters

            # Selects the method to update the model parameters in place

            self.apply_parameters_to_layer = self.apply_parameters_to_first_layer

        # If this layer is the output layer

        elif layer==-1:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.call_from_input_method = self.output_layer_call_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.call_given_parameters = self.output_layer_call_partially_convex_with_parameters

            # Selects the method for reshaping the model parameters from 
            # a flat vector

            self.update_layer_parameters = self.output_layer_update_parameters

            # Selects the method to update the model parameters in place

            self.apply_parameters_to_layer = self.apply_parameters_to_output_layer

        # Otherwise, if it is any of the intermediate layers

        else:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.call_from_input_method = self.middle_layer_call_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.call_given_parameters = self.middle_layer_call_partially_convex_with_parameters

            # Selects the method for reshaping the model parameters from 
            # a flat vector

            self.update_layer_parameters = self.middle_layer_update_parameters

            # Selects the method to update the model parameters in place

            self.apply_parameters_to_layer = self.apply_parameters_to_middle_layer

        # Gets the regularization function

        self.regularization_function = build_tensorflow_math_expressions(
        regularization_function_name, dtype)
    
    # Defines a method for getting the layer value given the input when
    # an accessory layer is necessary. In this case, the input must be a
    # tuple. It follows the rationale from Amos et al, Input convex neu-
    # ral networks. This is the case of the first layer

    def first_layer_call_from_input(self, input):

        # The first element in the input tuple is the main layer. The 
        # second element is due to the accessory layer. The third element
        # is the initial convex input

        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        segmented_input = (input[..., :self.input_size_main_network], 
        input[..., self.input_size_main_network:])

        # Gets the multiplication of the parcel of the accessory 
        # layer by its corresponding matrix

        parcel_1 = self.layer_self.dense_Wu(segmented_input[1])

        # Gets the parcel of the convex input multiplied by the bit made
        # of the accessory layer using the Hadamard product

        parcel_2 = self.layer_self.dense_Wy(tf.multiply(segmented_input[0], 
        self.layer_self.dense_Wyu(segmented_input[1])))

        # Initializes the input as dense layer and split it into the dif-
        # ferent families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(parcel_1+parcel_2, 
        self.layer_self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Does the same for the accessory layer

        x_splits_accessory_layer = tf.split(
        self.layer_self.dense_accessory_layer(segmented_input[1]), 
        self.layer_self.neurons_per_activation_acessory_layer,  axis=-1)

        output_activations_accessory_layer = [activation_function(
        split) for activation_function, split in zip(
        self.activation_list_acessory_network, 
        x_splits_accessory_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer 
        # first, then the accessory layer, then the initial convex input

        return (tf.concat(output_activations_main_layer, axis=-1), 
        tf.concat(output_activations_accessory_layer, axis=-1), 
        segmented_input[0])
    
    # Defines a method for getting the layer value given the input when
    # an accessory layer is necessary. In this case, the input must be a
    # tuple. It follows the rationale from Amos et al, Input convex neu-
    # ral networks. This is for a middle layer

    def middle_layer_call_from_input(self, input):

        # The first element in the input tuple is the main layer. The 
        # second element is due to the accessory layer. The third element
        # is the initial convex input

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix

        parcel_1 = self.layer_self.dense_Wu(input[1])

        # Gets the parcel of the convex input multiplied by the bit made
        # of the accessory layer using the Hadamard product

        parcel_2 = self.layer_self.dense_Wy(tf.multiply(input[2], 
        self.layer_self.dense_Wyu(input[1])))

        # Gets the parcel of the input of the main network multiplied by
        # the absolute value of the bit given by accessory previous 
        # layer, using the Hadamard product. Then, multiplies by the cor-
        # responding weight matrix

        parcel_3 = self.layer_self.dense(tf.multiply(input[0], tf.abs(
        self.layer_self.dense_Wzu(input[1]))))

        # Initializes the input as dense layer and split it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(parcel_1+parcel_2+parcel_3, 
        self.layer_self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Does the same for the accessory layer

        x_splits_accessory_layer = tf.split(
        self.layer_self.dense_accessory_layer(input[1]), 
        self.layer_self.neurons_per_activation_acessory_layer,  axis=-1)

        output_activations_accessory_layer = [activation_function(split
        ) for activation_function, split in zip(
        self.activation_list_acessory_network, x_splits_accessory_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer
        # first, then the accessory layer

        return (tf.concat(output_activations_main_layer, axis=-1), 
        tf.concat(output_activations_accessory_layer, axis=-1), input[2])
    
    # Defines a method for getting the layer value given the input when
    # an accessory layer is necessary. In this case, the input must be a
    # tuple. It follows the rationale from Amos et al, Input convex neu-
    # ral networks. This is for the output layer

    def output_layer_call_from_input(self, input):

        # The first element in the input tuple is the main layer. The 
        # second element is due to the accessory layer. The third element
        # is the initial convex input

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix

        parcel_1 = self.layer_self.dense_Wu(input[1])

        # Gets the parcel of the convex input multiplied by the bit made
        # of the accessory layer using the Hadamard product

        parcel_2 = self.layer_self.dense_Wy(tf.multiply(input[2], 
        self.layer_self.dense_Wyu(input[1])))

        # Gets the parcel of the input of the main network multiplied by
        # the absolute value of the bit given by accessory previous lay-
        # er, using the Hadamard product. Then, multiplies by the corres-
        # ponding weight matrix

        parcel_3 = self.layer_self.dense(tf.multiply(input[0], tf.abs(
        self.layer_self.dense_Wzu(input[1]))))

        # Initializes the input as dense layer and split it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(parcel_1+parcel_2+parcel_3, 
        self.layer_self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. As this layer is the last 
        # one, returns just the main layer

        return tf.concat(output_activations_main_layer, axis=-1)
    
    # Defines a function to get the output of a layer that constitutes a
    # partially input-convex neural network (AMOS ET AL, Input Convex 
    # Neural Networks). This function is for the first layer

    def first_layer_call_partially_convex_with_parameters(self, 
    layer_input, parameters):
        
        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        (W_tilde, b_tilde, W_yu, b_y, W_u, b_layer, W_y
        ) = self.first_layer_update_parameters(parameters)

        segmented_layer_input = (layer_input[..., :(
        self.input_size_main_network)], layer_input[..., 
        self.input_size_main_network:])

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix

        parcel_1 = tf.matmul(segmented_layer_input[1], W_u)+b_layer

        # Gets the parcel of the convex input multiplied by the bit ma-
        # de of the accessory layer using the Hadamard product

        parcel_2 = tf.matmul(tf.multiply(segmented_layer_input[0], 
        tf.matmul(segmented_layer_input[1], W_yu)+b_y), W_y)

        # Initializes the input as dense layer and split it into the
        # different families of activation functions for the main layer.
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(parcel_1+parcel_2, 
        self.layer_self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-)
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Does the same for the accessory layer

        x_splits_accessory_layer = tf.split(tf.matmul(
        segmented_layer_input[1], W_tilde)+b_tilde, 
        self.layer_self.neurons_per_activation_acessory_layer, axis=-1)

        output_activations_accessory_layer = [activation_function(
        split) for activation_function, split in zip(
        self.activation_list_acessory_network, 
        x_splits_accessory_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer
        # layer first, then the accessory layer, then the initial convex 
        # input

        return (tf.concat(output_activations_main_layer, axis=-1), 
        tf.concat(output_activations_accessory_layer, axis=-1), 
        segmented_layer_input[0])
    
    # Defines a function to get the output of a layer that constitutes a
    # partially input-convex neural network (AMOS ET AL, Input Convex 
    # Neural Networks). This function is for the middle layers

    def middle_layer_call_partially_convex_with_parameters(self, 
    layer_input, parameters):

        ################################################################
        #                    Accessory layer update                    #
        ################################################################
        
        # Gets the weights and biases

        (W_z, W_tilde, b_tilde, W_zu, b_z, W_yu, b_y, W_u, b_layer, W_y
        ) = self.middle_layer_update_parameters(parameters)

        # Multiplies the weights of the acessory network (u) by the in-
        # puts of the acessory layer and, then, adds the biases. Finally,
        # splits by activation function family

        x_splits_u = tf.split(tf.matmul(layer_input[1], W_tilde)+b_tilde, 
        self.layer_self.neurons_per_activation_acessory_layer, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions) for the accessory layer

        output_activations_u = [activation_function(split) for (
        activation_function), split in zip(
        self.activation_list_acessory_network, x_splits_u)]

        # Concatenates the response and saves it into u_(i+1). Uses flag 
        # axis=-1 to concatenate next to the last row

        output_u = tf.concat(output_activations_u, axis=-1)

        ################################################################
        #                      Main layer update                       #
        ################################################################

        # Gets the multiplication of the matrix W_u by the output of the
        # accessory previous layer, and adds this layer bias

        parcel_1 = tf.matmul(layer_input[1], W_u)+b_layer

        # Gets the parcel of the convex input multiplied by the bit made
        # of the accessory layer using the Hadamard product. Then, mul-
        # tiplies the whole by the corresponding weight matrix

        parcel_2 = tf.matmul(tf.multiply(layer_input[2], tf.matmul(
        layer_input[1], W_yu)+b_y), W_y)

        # Gets the parcel of the input of the main network multiplied by
        # the absolute value of the bit given by accessory previous 
        # layer, using the Hadamard product. Then, multiplies by the cor-
        # responding weight matrix

        parcel_3 = tf.matmul(tf.multiply(layer_input[0], tf.abs(
        tf.matmul(layer_input[1], W_zu)+b_z)), W_z)

        # Sums the parcels and splits it according to the families of 
        # activation functions

        x_splits_z = tf.split(parcel_1+parcel_2+parcel_3,
        self.layer_self.neurons_per_activation, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions) for the main layer

        output_activations_z = [activation_function(split) for (
        activation_function), split in zip(self.activation_list, 
        x_splits_z)]

        # Concatenates the response and saves it into z_(i+1). Uses flag 
        # axis=-1 to concatenate next to the last row

        output_z = tf.concat(output_activations_z, axis=-1)

        # Returns both outputs if this is not the last layer

        return output_z, output_u, layer_input[2]
    
    # Defines a function to get the output of a layer that constitutes a
    # partially input-convex neural network (AMOS ET AL, Input Convex 
    # Neural Networks). This function is for the output layer

    def output_layer_call_partially_convex_with_parameters(self, 
    layer_input, parameters):

        ################################################################
        #                    Accessory layer update                    #
        ################################################################
        
        # Gets the weights and biases

        (W_z, W_zu, b_z, W_yu, b_y, W_u, b_layer, W_y
        ) = self.output_layer_update_parameters(parameters)

        ################################################################
        #                      Main layer update                       #
        ################################################################

        # Gets the multiplication of the matrix W_u by the output of the
        # accessory previous layer, and adds this layer bias

        parcel_1 = tf.matmul(layer_input[1], W_u)+b_layer

        # Gets the parcel of the convex input multiplied by the bit made
        # of the accessory layer using the Hadamard product. Then, mul-
        # tiplies the whole by the corresponding weight matrix

        parcel_2 = tf.matmul(tf.multiply(layer_input[2], tf.matmul(
        layer_input[1], W_yu)+b_y), W_y)

        # Gets the parcel of the input of the main network multiplied by
        # the absolute value of the bit given by accessory previous 
        # layer, using the Hadamard product. Then, multiplies by the cor-
        # responding weight matrix

        parcel_3 = tf.matmul(tf.multiply(layer_input[0], tf.abs(
        tf.matmul(layer_input[1], W_zu)+b_z)), W_z)

        # Sums the parcels and splits it according to the families of 
        # activation functions

        x_splits_z = tf.split(parcel_1+parcel_2+parcel_3,
        self.layer_self.neurons_per_activation, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions) for the main layer

        output_activations_z = [activation_function(split) for (
        activation_function), split in zip(self.activation_list, 
        x_splits_z)]

        # Concatenates the response and saves it into z_(i+1). Uses flag 
        # axis=-1 to concatenate next to the last row

        output_z = tf.concat(output_activations_z, axis=-1)

        # Returns just the main layer

        return output_z
    
    # Defines a function to build this layer during serialization

    def mixed_layer_builder(self, input_shape):

        # Constructs a dense layer with identity activation functions
        # with no biases only if this is not the first layer. Because,
        # at the first hidden layer, the whole input of the model is 
        # simply fed into this first hidden layer

        if self.layer_self.layer!=0:

            self.layer_self.dense = tf.keras.layers.Dense(
            self.layer_self.total_neurons, use_bias=False, name="Wz")

        # Gets a list with the numbers of neurons per activation functi-
        # on for the accessory network (u)

        self.layer_self.neurons_per_activation_acessory_layer = [value[
        "number of neurons"] if isinstance(value, dict) else value for (
        value) in self.layer_self.functions_dict_acessory_network.values(
        )]

        # Creates a dense layer for the accessory network

        self.layer_self.dense_accessory_layer = tf.keras.layers.Dense(
        sum(self.layer_self.neurons_per_activation_acessory_layer), 
        name="Wtilde")

        ################################################################
        #           Amos et at, Input convex neural networks           #
        ################################################################

        # Creates a dense layer for the bit of the accessory layer's re-
        # sult that multiplies the main layer response using the Hada-
        # mard product

        self.layer_self.dense_Wzu = tf.keras.layers.Dense(
        self.layer_self.input_size_main_layer, name="Wzu")

        # Creates a dense layer for the bit of the accessory layer's re-
        # sult that multiplies the initial convex input using the Hada-
        # mard product

        self.layer_self.dense_Wyu = tf.keras.layers.Dense(
        self.input_size_main_network, name="Wyu")

        # Creates a dense layer without biases for the multiplication of 
        # the accessory layer's result by a weight matrix, which, in 
        # turn, is used as a complement to the bias of the main layer. 
        # The bias of this operation will be the bias to the main net-
        # work, too

        self.layer_self.dense_Wu = tf.keras.layers.Dense(
        self.layer_self.total_neurons, name="Wu")

        # Creates a dense layer without biases for the multiplica-
        # tion of the result of the Hadamard product between the o-
        # riginal convex input and the product of the acessory layer
        # result

        self.layer_self.dense_Wy = tf.keras.layers.Dense(
        self.layer_self.total_neurons, use_bias=False, name="Wy")

        # Constructs the layer

        super(type(self.layer_self), self.layer_self).build(input_shape)

        # Verifies if the dense layer is present

        if hasattr(self.layer_self, "dense"):

            self.layer_self.dense.build(input_shape[0])

    # Defines a function to get the flat and sliced parameters, then, 
    # converts it to tensors given the shapes of the trainables varia-
    # bles

    def first_layer_update_parameters(self, flat_parameters):

        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        W_tilde = tf.reshape(flat_parameters[0], 
        self.layer_self.trainable_variables_shapes[0])
        
        b_tilde = tf.reshape(flat_parameters[1], 
        self.layer_self.trainable_variables_shapes[1])
        
        W_yu = tf.reshape(flat_parameters[2], 
        self.layer_self.trainable_variables_shapes[2])
        
        b_y = tf.reshape(flat_parameters[3], 
        self.layer_self.trainable_variables_shapes[3])
        
        W_u = tf.reshape(flat_parameters[4], 
        self.layer_self.trainable_variables_shapes[4])
        
        b_layer = tf.reshape(flat_parameters[5], 
        self.layer_self.trainable_variables_shapes[5])
        
        W_y = tf.reshape(flat_parameters[6], 
        self.layer_self.trainable_variables_shapes[6])

        return (W_tilde, b_tilde, W_yu, b_y, W_u, b_layer, W_y)

    # Defines a function to get the flat and sliced parameters, then, 
    # converts it to tensors given the shapes of the trainables varia-
    # bles

    def middle_layer_update_parameters(self, flat_parameters):

        W_z = self.regularization_function(tf.reshape(flat_parameters[0], 
        self.layer_self.trainable_variables_shapes[0]))

        W_tilde = tf.reshape(flat_parameters[1], 
        self.layer_self.trainable_variables_shapes[1])

        b_tilde = tf.reshape(flat_parameters[2], 
        self.layer_self.trainable_variables_shapes[2])

        W_zu = tf.reshape(flat_parameters[3], 
        self.layer_self.trainable_variables_shapes[3])

        b_z = tf.reshape(flat_parameters[4], 
        self.layer_self.trainable_variables_shapes[4])

        W_yu = tf.reshape(flat_parameters[5], 
        self.layer_self.trainable_variables_shapes[5])

        b_y = tf.reshape(flat_parameters[6], 
        self.layer_self.trainable_variables_shapes[6])

        W_u = tf.reshape(flat_parameters[7], 
        self.layer_self.trainable_variables_shapes[7])

        b_layer = tf.reshape(flat_parameters[8], 
        self.layer_self.trainable_variables_shapes[8])
        
        W_y = tf.reshape(flat_parameters[9], 
        self.layer_self.trainable_variables_shapes[9])

        return (W_z, W_tilde, b_tilde, W_zu, b_z, W_yu, b_y, W_u, 
        b_layer, W_y)

    # Defines a function to get the flat and sliced parameters, then, 
    # converts it to tensors given the shapes of the trainables varia-
    # bles

    def output_layer_update_parameters(self, flat_parameters):

        W_z = self.regularization_function(tf.reshape(flat_parameters[0], 
        self.layer_self.trainable_variables_shapes[0]))

        W_zu = tf.reshape(flat_parameters[1], 
        self.layer_self.trainable_variables_shapes[1])

        b_z = tf.reshape(flat_parameters[2], 
        self.layer_self.trainable_variables_shapes[2])

        W_yu = tf.reshape(flat_parameters[3], 
        self.layer_self.trainable_variables_shapes[3])

        b_y = tf.reshape(flat_parameters[4], 
        self.layer_self.trainable_variables_shapes[4])

        W_u = tf.reshape(flat_parameters[5], 
        self.layer_self.trainable_variables_shapes[5])

        b_layer = tf.reshape(flat_parameters[6], 
        self.layer_self.trainable_variables_shapes[6])
        
        W_y = tf.reshape(flat_parameters[7], 
        self.layer_self.trainable_variables_shapes[7])

        return (W_z, W_zu, b_z, W_yu, b_y, W_u, b_layer, W_y)
    
    # Defines a function to apply the weights and biases to the first 
    # layer

    def apply_parameters_to_first_layer(self, flat_parameters):

        # Gets the weights and biases from the flat vector

        (W_tilde, b_tilde, W_yu, b_y, W_u, b_layer, W_y
        ) = self.update_layer_parameters(flat_parameters)

        # Updates the weights and biases matrices

        self.layer_self.dense_Wu.kernel.assign(W_u)

        self.layer_self.dense_Wu.bias.assign(b_layer)

        self.layer_self.dense_accessory_layer.kernel.assign(W_tilde)

        self.layer_self.dense_accessory_layer.bias.assign(b_tilde)

        self.layer_self.dense_Wy.kernel.assign(W_y)

        self.layer_self.dense_Wyu.kernel.assign(W_yu)

        self.layer_self.dense_Wyu.bias.assign(b_y)
    
    # Defines a function to apply the weights and biases to a middle 
    # layer

    def apply_parameters_to_middle_layer(self, flat_parameters):

        # Gets the weights and biases from the flat vector

        (W_z, W_tilde, b_tilde, W_zu, b_z, W_yu, b_y, W_u, b_layer, W_y
        ) = self.update_layer_parameters(flat_parameters)

        # Updates the weights and biases matrices

        self.layer_self.dense.kernel.assign(W_z)

        self.layer_self.dense_Wzu.kernel.assign(W_zu)

        self.layer_self.dense_Wzu.bias.assign(b_z)

        self.layer_self.dense_Wu.kernel.assign(W_u)

        self.layer_self.dense_Wu.bias.assign(b_layer)

        self.layer_self.dense_accessory_layer.kernel.assign(W_tilde)

        self.layer_self.dense_accessory_layer.bias.assign(b_tilde)

        self.layer_self.dense_Wy.kernel.assign(W_y)

        self.layer_self.dense_Wyu.kernel.assign(W_yu)

        self.layer_self.dense_Wyu.bias.assign(b_y)
    
    # Defines a function to apply the weights and biases to a middle 
    # layer

    def apply_parameters_to_output_layer(self, flat_parameters):

        # Gets the weights and biases from the flat vector

        (W_z, W_zu, b_z, W_yu, b_y, W_u, b_layer, W_y
        ) = self.update_layer_parameters(flat_parameters)

        # Updates the weights and biases matrices

        self.layer_self.dense.kernel.assign(W_z)

        self.layer_self.dense_Wzu.kernel.assign(W_zu)

        self.layer_self.dense_Wzu.bias.assign(b_z)

        self.layer_self.dense_Wu.kernel.assign(W_u)

        self.layer_self.dense_Wu.bias.assign(b_layer)

        self.layer_self.dense_Wy.kernel.assign(W_y)

        self.layer_self.dense_Wyu.kernel.assign(W_yu)

        self.layer_self.dense_Wyu.bias.assign(b_y)