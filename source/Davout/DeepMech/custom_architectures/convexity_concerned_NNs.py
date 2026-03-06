# Routine to store classes for special and custom neural network archi-
# tectural

import tensorflow as tf

from ..tool_box.activation_function_utilities import verify_activationDict

########################################################################
#                    Partially convex neural network                   #
########################################################################

class PartiallyConvexNNs:

    def __init__(self, activation_functionDict, custom_activations_class,
    trainable_variables_shapes, live_activationsDict=dict(), 
    activations_accessory_layer_dict=dict(), input_size_main_network=
    None, layer=0):
        
        # Saves the size of the tensors of trainable parameters

        self.trainable_variables_shapes = trainable_variables_shapes

        # Saves the custom activations_class

        self.custom_activations_class = custom_activations_class

        # Adds the dictionary of live-wired activation functions. But 
        # checks if it is given as argument

        if (live_activationsDict is None) or live_activationsDict=={}: 

            # Checks for the activation functions of the accessory net-
            # work, too

            if activations_accessory_layer_dict:

                # Concatenates the two dictionaries, but overrides the 
                # values of the accessory dictionary with the values of
                # the conventional one

                self.live_activationFunctions, *_ = verify_activationDict(
                activations_accessory_layer_dict | activation_functionDict, 
                layer, {}, True, self.custom_activations_class)

        else:

            self.live_activationFunctions = live_activationsDict

        # Gets the dictionary of functions into the class
        
        self.functions_dict = activation_functionDict

        self.layer = layer

        # Gets all the live activation functions into a tuple

        self.activation_list = tuple([self.live_activationFunctions[name
        ] for name in self.functions_dict.keys()])

        # Gets the dictionary of functions of the accessory network in 
        # the case of partially input-convex networks

        self.functions_dict_acessory_network = activations_accessory_layer_dict

        # Gets all the live activation functions into a tuple

        self.activation_list_acessory_network = tuple([
        self.live_activationFunctions[name] for name in (
        self.functions_dict_acessory_network.keys())])

        # Selects the method that will give the ouput of the layer based
        # on the existence or not of an accessory layer

        if self.functions_dict_acessory_network:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.call_from_input_method = self.call_from_input_with_accessory_layer

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.call_given_parameters = self.call_partially_convex_layer_with_parameters

            # Saves the number of input neurons for the main network

            self.input_size_main_network = input_size_main_network
    
    # Defines a method for getting the layer value given the input when
    # an accessory layer is necessary. In this case, the input must be a
    # tuple. It follows the rationale from Amos et al, Input convex neu-
    # ral networks

    def call_from_input_with_accessory_layer(self, input):

        # The first element in the input tuple is the main layer. The 
        # second element is due to the accessory layer. The third element
        # is the initial convex input

        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        if self.layer==0:

            segmented_input = (input[..., :self.input_size_main_network], 
            input[..., self.input_size_main_network:])

            # Gets the multiplication of the parcel of the accessory 
            # layer by its corresponding matrix

            parcel_1 = self.dense_Wu(segmented_input[1])

            # Gets the parcel of the convex input multiplied by the bit 
            # made of the accessory layer using the Hadamard product

            parcel_2 = self.dense_Wy(tf.multiply(segmented_input[0], 
            self.dense_Wyu(segmented_input[1])))

            # Initializes the input as dense layer and split it into the 
            # different families of activation functions for the main 
            # layer. This keeps the input as a tensor

            x_splits_main_layer = tf.split(parcel_1+parcel_2, 
            self.neurons_per_activation,  axis=-1)

            # Initializes a list of outputs for each family of neurons 
            # (organized by their activation functions)

            output_activations_main_layer = [activation_function(split
            ) for activation_function, split in zip(self.activation_list, 
            x_splits_main_layer)]

            # Does the same for the accessory layer

            x_splits_accessory_layer = tf.split(
            self.dense_accessory_layer(segmented_input[1]), 
            self.neurons_per_activation_acessory_layer,  axis=-1)

            output_activations_accessory_layer = [activation_function(
            split) for activation_function, split in zip(
            self.activation_list_acessory_network, 
            x_splits_accessory_layer)]

            # Concatenates the response and returns it. Uses flag axis=-1 
            # to concatenate next to the last row. Returns always the 
            # main layer first, then the accessory layer, then the ini-
            # convex input

            return (tf.concat(output_activations_main_layer, axis=-1), 
            tf.concat(output_activations_accessory_layer, axis=-1), 
            segmented_input[0])

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix

        parcel_1 = self.dense_Wu(input[1])

        # Gets the parcel of the convex input multiplied by the bit made
        # of the accessory layer using the Hadamard product

        parcel_2 = self.dense_Wy(tf.multiply(input[2], self.dense_Wyu(
        input[1])))

        # Gets the parcel of the input of the main network multiplied by
        # the absolute value of the bit given by accessory previous 
        # layer, using the Hadamard product. Then, multiplies by the cor-
        # responding weight matrix

        parcel_3 = self.dense(tf.multiply(input[0], tf.abs(
        self.dense_Wzu(input[1]))))

        # Initializes the input as dense layer and split it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(parcel_1+parcel_2+parcel_3, 
        self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Does the same for the accessory layer

        x_splits_accessory_layer = tf.split(self.dense_accessory_layer(
        input[1]), self.neurons_per_activation_acessory_layer,  axis=-1)

        output_activations_accessory_layer = [activation_function(split
        ) for activation_function, split in zip(
        self.activation_list_acessory_network, x_splits_accessory_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer
        # first, then the accessory layer. If this layer is the last one,
        # returns just the main layer

        if self.layer==-1:

            return tf.concat(output_activations_main_layer, axis=-1)
        
        else:

            return (tf.concat(output_activations_main_layer, axis=-1), 
            tf.concat(output_activations_accessory_layer, axis=-1), 
            input[2])
    
    # Defines a function to get the output of a layer that constitutes a
    # partially input-convex neural network (AMOS ET AL, Input Convex 
    # Neural Networks)

    def call_partially_convex_layer_with_parameters(self, layer_input, 
    parameters):
        
        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        if self.layer==0:

            #W_tilde, b_tilde, W_yu, b_y, W_u, b_layer, W_y = (
            #parameters)

            W_tilde = tf.reshape(parameters[0], 
            self.trainable_variables_shapes[0])
            
            b_tilde = tf.reshape(parameters[1], 
            self.trainable_variables_shapes[1])
            
            W_yu = tf.reshape(parameters[2], 
            self.trainable_variables_shapes[2])
            
            b_y = tf.reshape(parameters[3], 
            self.trainable_variables_shapes[3])
            
            W_u = tf.reshape(parameters[4], 
            self.trainable_variables_shapes[4])
            
            b_layer = tf.reshape(parameters[5], 
            self.trainable_variables_shapes[5])
            
            W_y = tf.reshape(parameters[6], 
            self.trainable_variables_shapes[6])

            segmented_layer_input = (layer_input[..., :(
            self.input_size_main_network)], layer_input[..., 
            self.input_size_main_network:])

            # Gets the multiplication of the parcel of the accessory 
            # layer by its corresponding matrix

            parcel_1 = tf.matmul(segmented_layer_input[1], W_u)+b_layer

            # Gets the parcel of the convex input multiplied by the bit 
            # made of the accessory layer using the Hadamard product

            parcel_2 = tf.matmul(tf.multiply(segmented_layer_input[0], 
            tf.matmul(segmented_layer_input[1], W_yu)+b_y), W_y)

            # Initializes the input as dense layer and split it into the 
            # different families of activation functions for the main 
            # layer. This keeps the input as a tensor

            x_splits_main_layer = tf.split(parcel_1+parcel_2, 
            self.neurons_per_activation,  axis=-1)

            # Initializes a list of outputs for each family of neurons 
            # (organized by their activation functions)

            output_activations_main_layer = [activation_function(split
            ) for activation_function, split in zip(self.activation_list, 
            x_splits_main_layer)]

            # Does the same for the accessory layer

            x_splits_accessory_layer = tf.split(tf.matmul(
            segmented_layer_input[1], W_tilde)+b_tilde, 
            self.neurons_per_activation_acessory_layer, axis=-1)

            output_activations_accessory_layer = [activation_function(
            split) for activation_function, split in zip(
            self.activation_list_acessory_network, 
            x_splits_accessory_layer)]

            # Concatenates the response and returns it. Uses flag axis=-1 
            # to concatenate next to the last row. Returns always the 
            # main layer first, then the accessory layer, then the ini-
            # convex input

            return (tf.concat(output_activations_main_layer, axis=-1), 
            tf.concat(output_activations_accessory_layer, axis=-1), 
            segmented_layer_input[0])

        ################################################################
        #                    Accessory layer update                    #
        ################################################################
        
        # Gets the weights and biases

        W_z = tf.reshape(parameters[0], self.trainable_variables_shapes[
        0])

        W_tilde = tf.reshape(parameters[1], 
        self.trainable_variables_shapes[1])

        b_tilde = tf.reshape(parameters[2], 
        self.trainable_variables_shapes[2])

        W_zu = tf.reshape(parameters[3], 
        self.trainable_variables_shapes[3])

        b_z = tf.reshape(parameters[4], 
        self.trainable_variables_shapes[4])

        W_yu = tf.reshape(parameters[5], 
        self.trainable_variables_shapes[5])

        b_y = tf.reshape(parameters[6], 
        self.trainable_variables_shapes[6])

        W_u = tf.reshape(parameters[7], 
        self.trainable_variables_shapes[7])

        b_layer = tf.reshape(parameters[8], 
        self.trainable_variables_shapes[8])
        
        W_y = tf.reshape(parameters[9], 
        self.trainable_variables_shapes[9])

        # Multiplies the weights of the acessory network (u) by the in-
        # puts of the acessory layer and, then, adds the biases. Finally,
        # splits by activation function family

        x_splits_u = tf.split(tf.matmul(layer_input[1], W_tilde)+b_tilde, 
        self.neurons_per_activation_acessory_layer, axis=-1)

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
        self.neurons_per_activation, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions) for the main layer

        output_activations_z = [activation_function(split) for (
        activation_function), split in zip(self.activation_list, 
        x_splits_z)]

        # Concatenates the response and saves it into z_(i+1). Uses flag 
        # axis=-1 to concatenate next to the last row

        output_z = tf.concat(output_activations_z, axis=-1)

        # Returns both outputs if this is not the last layer. Otherwise,
        # returns just the main layer

        if self.layer==-1:

            return output_z

        return output_z, output_u, layer_input[2]