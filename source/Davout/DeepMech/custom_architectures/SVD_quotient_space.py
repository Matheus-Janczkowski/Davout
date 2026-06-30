# Routine to store the class for the SVD-based architecture

import tensorflow as tf

from ..tool_box.activation_function_utilities import verify_activationDict

from ..tool_box.numerical_tools import build_tensorflow_math_expressions

from ...PythonicUtilities.dictionary_tools import verify_obligatory_and_optional_keys

########################################################################
#                      SVD-based quotient space NN                     #
########################################################################

# Defines a class for NN architecture based on SVD decomposition

class SVDQuotientSpace:

    def __init__(self, layer_self, activation_functionDict, 
    custom_activations_class, architecture_info_dict):
        
        # Verifies if the necessary information of the architecture pro-
        # vided by the user has been actually supplied

        architecture_info_dict = verify_obligatory_and_optional_keys(
        architecture_info_dict, {"name": {}, "quotient space dimension": 
        {"type": int, "description": "Integer with the number of neuro"+
        "ns attached to inputs from the quotient space. The quotient s"+
        "pace has the input vector which makes the whole neural networ"+
        "k output the null vector when the former is also null."}}, 
        {"weights modulating function": {"type": str, "description": 
        "String with the name of the function that modulates the weigh"+
        "ts matrix", "default": "identity"}, "Householder epsilon": {
        "type": float, "description": "Float number with the tolerance"+
        " to the calculation of the non-free component of each Househo"+
        "lder vector"}}, "custom_architecture", "SVDQuotientSpace")
        
        # Stores variables that will be used in the get_config for seri-
        # alization and class rebuilding

        self.layer_self = layer_self

        self.layer_number = self.layer_self.code_given_info_class.layer

        self.input_size_main_network = (
        self.layer_self.code_given_info_class.input_size_main_network)

        self.input_size_main_layer = (
        self.layer_self.code_given_info_class.input_size_main_layer)

        # Verifies if the size of the main network is not None

        if self.input_size_main_network is None:

            raise ValueError(" The key 'name' in 'custom_architecture'"+
            " was selected as 'SVDQuotientSpace', but the argument 'in"+
            "put_size_main_network' in 'MultiLayerModel' is None.\nYou"+
            " have to insert the number of neurons of the main layer t"+
            "o 'input_size_main_network', this quantity is equal to th"+
            "e number of variables of input to which the NN model must"+
            " be convex to or variables that construct the quotient sp"+
            "ace")

        # Gets the dictionary of activation functions for the accessory
        # layer

        self.activations_accessory_layer_dict = architecture_info_dict[
        "activations accessory layer list"][self.layer_number]
        
        # Verifies if there are activations in the dictionary of activa-
        # tions for the accessory layer

        if not self.activations_accessory_layer_dict:

            raise ValueError("'activations_accessory_layer_dict' must "+
            "have at least one activation function for setting up 'SVD"+
            "QuotientSpace', since this class constructs an accessorry"+
            " neural network")

        # Concatenates the two dictionaries, but overrides the val-
        # ues of the accessory dictionary with the values of the con-
        # ventional one

        self.live_activationFunctions, *_ = verify_activationDict(
        self.activations_accessory_layer_dict | activation_functionDict, 
        self.layer_number, {}, True, custom_activations_class)

        # Gets all the live activation functions into a tuple

        self.activation_list = tuple([self.live_activationFunctions[name
        ] for name in activation_functionDict.keys()])

        # Gets all the live activation functions into a tuple

        self.activation_list_acessory_network = tuple([
        self.live_activationFunctions[name] for name in (
        self.activations_accessory_layer_dict.keys())])

        # Verifies if the modulating function is the identity

        if (architecture_info_dict["weights modulating function"]=="id"+
        "entity"):

            # Sets the evaluator of the generic layer from input to the
            # method that multiplies the input vector by the chain of
            # Householder reflector instead of the method that builds 
            # the orthogonal matrix first and, then, computes the weight
            # matrix

            self.generic_layers_from_input = self.identity_modulation_from_input

        # Otherwise, gets the modulating function

        else:

            self.modulating_function = build_tensorflow_math_expressions(
            architecture_info_dict["weights modulating function"], 
            self.layer_self.code_given_info_class.float_dtype)

            # Sets the evaluator of the generic layer from input to the
            # method that assembles the SVD first and, then, applies the
            # modulation mapping

            self.generic_layers_from_input = self.non_identity_modulation_from_input

        # Defines the method to call this layer and get the forward res-
        # ponse. If this layer is the first layer

        if self.layer_number==0:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.layer_self.call_from_input_method = self.first_layer_call_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.layer_self.call_given_parameters = self.first_layer_call_SVD_architecture_with_parameters

            # Selects the method for reshaping the model parameters from 
            # a flat vector

            self.update_layer_parameters = self.first_layer_update_parameters

            # Selects the method to update the model parameters in place

            self.layer_self.apply_parameters_to_layer = self.apply_parameters_to_first_layer

        # If this layer is the output layer

        elif self.layer_number==-1:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.layer_self.call_from_input_method = self.output_layer_call_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.layer_self.call_given_parameters = self.output_layer_call_SVD_architecture_with_parameters

            # Selects the method for reshaping the model parameters from 
            # a flat vector

            self.update_layer_parameters = self.output_layer_update_parameters

            # Selects the method to update the model parameters in place

            self.layer_self.apply_parameters_to_layer = self.apply_parameters_to_output_layer

        # Otherwise, if it is any of the intermediate layers

        else:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.layer_self.call_from_input_method = self.generic_layers_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.layer_self.call_given_parameters = self.middle_layer_call_SVD_architecture_with_parameters

            # Selects the method for reshaping the model parameters from 
            # a flat vector

            self.update_layer_parameters = self.middle_layer_update_parameters

            # Selects the method to update the model parameters in place

            self.layer_self.apply_parameters_to_layer = self.apply_parameters_to_middle_layer

        # Gets the epsilon constant for the evaluation of the non-free
        # component of the Householder vector

        self.householder_epsilon_squared = tf.constant(
        architecture_info_dict["Householder epsilon"]**2, dtype=
        self.layer_self.code_given_info_class.float_dtype)

        # Stores the constant 2.0 with the correct float type

        self.two = tf.constant(2.0, dtype=
        self.layer_self.code_given_info_class.float_dtype)
    
    # Defines a method for evaluating the output of a layer in the main
    # network and in the accessory network. These layers must not be in-
    # put or output layers. This method assembles the weights matrix and
    # then applies the modulation mapping

    def non_identity_modulation_from_input(self, input):

        # The first element in the input tuple is the main layer. The 
        # second element is due to the accessory layer

        # Gets the output of the accessory layer and splits into the 
        # different families of activation functions

        x_splits_accessory_layer = tf.split(
        self.layer_self.dense_W_accessory(input[1]), 
        self.layer_self.neurons_per_activation_acessory_layer,  axis=-1)

        # Applies the split multiplication to each activation function
        # and, then, concatenates them back into a single vector

        output_activations_accessory_layer = tf.concat(
        [activation_function(split) for activation_function, split in (
        zip(self.activation_list_acessory_network, 
        x_splits_accessory_layer))], axis=-1)

        # Initializes the weight matrix as the identity

        weight_matrix = tf.eye(self.n_neurons_current_main_layer,
        self.n_neurons_last_main_layer)

        weight_matrix = tf.foldl(
        self.update_B_matrix_with_householder_reflector, tf.range(
        self.weights_rank), initializer=weight_matrix)

        # Multiplies this result by the singular values coming from the
        # accessory layer. The singular values are a tensor [n_samples,
        # rank]; the reconstructed weight matrix so far is a tensor [
        # rank, p_i]. The final result must be [n_samples, rank, p_i]

        weight_matrix = output_activations_accessory_layer*weight_matrix
        
        # Multiplies by the A matrix of the SVD to reconstruct the SVD.
        # Then, uses the modulating function to get the weights matrix

        weight_matrix = self.modulating_function(tf.foldl(
        self.update_A_matrix_with_householder_reflector, tf.range(
        self.weights_rank), initializer=(weight_matrix)))

        # Gets the output of the main network and splits it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(tf.einsum('sij,sj->si', 
        weight_matrix, input[0]), self.layer_self.neurons_per_activation,  
        axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer 
        # first, then the accessory layer

        return (tf.concat(output_activations_main_layer, axis=-1), 
        output_activations_accessory_layer)
    
    # Defines a method for evaluating the output of a layer in the main
    # network and in the accessory network. These layers must not be in-
    # put or output layers. This method must be used when the mapping 
    # method is the identity, hence the weight matrix does not have to
    # be assembled

    def identity_modulation_from_input(self, input):

        # The first element in the input tuple is the main layer. The 
        # second element is due to the accessory layer

        # Gets the output of the accessory layer and splits into the 
        # different families of activation functions

        x_splits_accessory_layer = tf.split(
        self.layer_self.dense_W_accessory(input[1]), 
        self.layer_self.neurons_per_activation_acessory_layer,  axis=-1)

        # Applies the split multiplication to each activation function
        # and, then, concatenates them back into a single vector

        output_activations_accessory_layer = tf.concat(
        [activation_function(split) for activation_function, split in (
        zip(self.activation_list_acessory_network, 
        x_splits_accessory_layer))], axis=-1)

        # Multiplies the incoming batched input vector by the transposed
        # B matrix of the SVD

        output_B = tf.foldl(self.multiply_B_matrix_householder_reflector,
        tf.range(self.weights_rank), initializer=input[0])

        # Multiplies this result by the singular values coming from the
        # accessory layer and, then, multiplies by the A matrix of the 
        # SVD

        output_A = tf.foldl(self.multiply_A_matrix_householder_reflector,
        tf.range(self.weights_rank), initializer=(
        output_activations_accessory_layer*output_B))

        # Gets the output of the main network and splits it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(output_A, 
        self.layer_self.neurons_per_activation, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer 
        # first, then the accessory layer

        return (tf.concat(output_activations_main_layer, axis=-1), 
        output_activations_accessory_layer)
    
    # Defines a method for getting the input vector of the NN and 
    # breaking it down into the main and accessory networks. This method
    # can be used only once the model has been trained, since this me-
    # thod does not ensure orthogonality of the orthogonal matrices of
    # the SVD decomposition

    def first_layer_call_from_input(self, input):

        # The first element in the input tuple is the main layer. The 
        # second element is due to the accessory layer

        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        segmented_input = (input[..., :self.input_size_main_network], 
        input[..., self.input_size_main_network:])

        # Evaluates the response of the two layers using the method for 
        # a generic layer

        return self.generic_layers_from_input(segmented_input)
    
    # Defines a method to get the output value of SVD-based architecture.
    # This method cannot be used during training, since the orthogonality
    # of the SVD matrices is not enforced

    def output_layer_call_from_input(self, input):

        # Evaluates the response of the two layers using the method for 
        # a generic layer, but returns only the main layer's response

        return self.generic_layers_from_input(input)[0]
    
    # Defines a method for getting the input vector of the NN and 
    # breaking it down into the main and accessory networks. This method
    # must be used during training to enforce orthogonality to the SVD
    # matrices

    def first_layer_call_SVD_architecture_with_parameters(self, 
    layer_input, parameters):
        
        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        segmented_layer_input = (layer_input[..., :(
        self.input_size_main_network)], layer_input[..., 
        self.input_size_main_network:])

        # Gets the trainable parameters

        (W_accessory, b_accessory, left_orthogonal_matrix, 
        right_orthogonal_matrix) = self.first_layer_update_parameters(
        parameters)

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix. Then, splits the result into the
        # different families of activation functions and evaluates the
        # output of the accessory layer

        x_splits_accessory_layer = tf.split(tf.matmul(
        segmented_layer_input[1], W_accessory)+b_accessory, 
        self.layer_self.neurons_per_activation_acessory_layer, axis=-1)

        # Applies the split multiplication to each activation function
        # and, then, concatenates them back into a single vector

        output_activations_accessory_layer = tf.concat(
        [activation_function(split) for activation_function, split in (
        zip(self.activation_list_acessory_network, 
        x_splits_accessory_layer))], axis=-1)

        # Reconstructs the weights matrix by the SVD decomposition and
        # modulates it using the modulating function

        weight_matrix = self.modulating_function(tf.einsum('ir,sr,jr->'+
        'sij', left_orthogonal_matrix, 
        output_activations_accessory_layer, right_orthogonal_matrix))

        # Gets the output of the main network and splits it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(tf.einsum('sij,sj->si', 
        weight_matrix, segmented_layer_input[0]), 
        self.layer_self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer 
        # first, then the accessory layer

        return (tf.concat(output_activations_main_layer, axis=-1), 
        output_activations_accessory_layer)
    
    # Defines a method to get the output of a middle layer given the in-
    # puts of the main and accessory networks, respectively. This method
    # must be used for training, since it enforeces the orthogonality
    # constraints to the SVD matrices

    def middle_layer_call_SVD_architecture_with_parameters(self, 
    layer_input, parameters):
        
        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        (W_accessory, b_accessory, left_orthogonal_matrix, 
        right_orthogonal_matrix) = self.middle_layer_update_parameters(
        parameters)

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix. Then, splits the result into the
        # different families of activation functions and evaluates the
        # output of the accessory layer

        x_splits_accessory_layer = tf.split(tf.matmul(
        layer_input[1], W_accessory)+b_accessory, 
        self.layer_self.neurons_per_activation_acessory_layer, axis=-1)

        # Applies the split multiplication to each activation function
        # and, then, concatenates them back into a single vector

        output_activations_accessory_layer = tf.concat(
        [activation_function(split) for activation_function, split in (
        zip(self.activation_list_acessory_network, 
        x_splits_accessory_layer))], axis=-1)

        # Reconstructs the weights matrix by the SVD decomposition and
        # modulates it using the modulating function

        weight_matrix = self.modulating_function(tf.einsum('ir,sr,jr->'+
        'sij', left_orthogonal_matrix, 
        output_activations_accessory_layer, right_orthogonal_matrix))

        # Gets the output of the main network and splits it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(tf.einsum('sij,sj->si', 
        weight_matrix, layer_input[0]), 
        self.layer_self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer 
        # first, then the accessory layer

        return (tf.concat(output_activations_main_layer, axis=-1), 
        output_activations_accessory_layer)
    
    # Defines a function to get the output of a layer that constitutes a
    # partially input-convex neural network (AMOS ET AL, Input Convex 
    # Neural Networks). This function is for the output layer

    def output_layer_call_SVD_architecture_with_parameters(self, 
    layer_input, parameters):
        
        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        (W_accessory, b_accessory, left_orthogonal_matrix, 
        right_orthogonal_matrix) = self.output_layer_update_parameters(
        parameters)

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix. Then, splits the result into the
        # different families of activation functions and evaluates the
        # output of the accessory layer

        x_splits_accessory_layer = tf.split(tf.matmul(
        layer_input[1], W_accessory)+b_accessory, 
        self.layer_self.neurons_per_activation_acessory_layer, axis=-1)

        # Applies the split multiplication to each activation function
        # and, then, concatenates them back into a single vector

        output_activations_accessory_layer = tf.concat(
        [activation_function(split) for activation_function, split in (
        zip(self.activation_list_acessory_network, 
        x_splits_accessory_layer))], axis=-1)

        # Reconstructs the weights matrix by the SVD decomposition and
        # modulates it using the modulating function

        weight_matrix = self.modulating_function(tf.einsum('ir,sr,jr->'+
        'sij', left_orthogonal_matrix, 
        output_activations_accessory_layer, right_orthogonal_matrix))

        # Gets the output of the main network and splits it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(tf.einsum('sij,sj->si', 
        weight_matrix, layer_input[0]), 
        self.layer_self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations_main_layer = [activation_function(split
        ) for activation_function, split in zip(self.activation_list, 
        x_splits_main_layer)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns the main layer only

        return tf.concat(output_activations_main_layer, axis=-1)
    
    # Defines a function to build this layer during serialization

    def mixed_layer_builder(self, input_shape):

        # Constructs a dense layer with identity activation functions
        # with no biases only if this is not the first layer. Because,
        # at the first hidden layer, the whole input of the model is 
        # simply fed into this first hidden layer

        if self.layer_number!=0:

            self.layer_self.dense = tf.keras.layers.Dense(
            self.layer_self.total_neurons, use_bias=False, name="Wz")

        # Gets a list with the numbers of neurons per activation functi-
        # on for the accessory network (u)

        self.layer_self.neurons_per_activation_acessory_layer = [value[
        "number of neurons"] if isinstance(value, dict) else value for (
        value) in self.activations_accessory_layer_dict.values(
        )]

        # Creates a dense layer for the accessory network

        self.layer_self.dense_W_accessory = tf.keras.layers.Dense(
        sum(self.layer_self.neurons_per_activation_acessory_layer), 
        name="W_accessory")

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

        self.layer_self.dense_W_accessory.kernel.assign(W_u)

        self.layer_self.dense_W_accessory.bias.assign(b_layer)

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

        self.layer_self.dense_W_accessory.kernel.assign(W_u)

        self.layer_self.dense_W_accessory.bias.assign(b_layer)

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

        self.layer_self.dense_W_accessory.kernel.assign(W_u)

        self.layer_self.dense_W_accessory.bias.assign(b_layer)

        self.layer_self.dense_Wy.kernel.assign(W_y)

        self.layer_self.dense_Wyu.kernel.assign(W_yu)

        self.layer_self.dense_Wyu.bias.assign(b_y)

    ####################################################################
    #                      Householder reflectors                      #
    ####################################################################

    # Defines a function to initialize the trainable parameters of the
    # Householder chain of each orthogonal matrix of the SVD of this 
    # layer. This function also creates the tuples of indices

    def initialize_householder_parameters(self):

        # Gets the number of neurons of the incoming layer and the num-
        # ber of neurons of this layer

        self.n_neurons_last_main_layer = self.layer_self.code_given_info_class.number_neurons_per_main_layer[
        self.layer_number-1]

        self.n_neurons_current_main_layer = self.layer_self.code_given_info_class.number_neurons_per_main_layer[
        self.layer_number]

        # Computes the rank of the weights matrix

        self.weights_rank = min(self.n_neurons_last_main_layer, 
        self.n_neurons_current_main_layer)

        # Sets the initializer of the Householder parameters using Glo-
        # rot initialization with the scaling of the corresponding 
        # weights matrix and zero mean

        standard_deviation = tf.sqrt(self.two/tf.cast(
        self.n_neurons_current_main_layer+self.n_neurons_last_main_layer, 
        self.layer_self.code_given_info_class.float_dtype))

        initializer = tf.keras.initializers.RandomNormal(mean=0.0,
        stddev=standard_deviation)

        # Counts the number of degrees of freedom to compute each ortho-
        # gonal matrix

        n_parameters_A = int(0.5*(self.weights_rank*((2*
        self.n_neurons_current_main_layer)-self.weights_rank-1)))

        n_parameters_B = int(0.5*(self.weights_rank*((2*
        self.n_neurons_last_main_layer)-self.weights_rank-1)))

        # Initializes the vector of Householder parameters of the ortho-
        # gonal matrices of the SVD of the weights matrix W, W=A*diag(
        # sigma)*transpose(B)

        self.householder_parameters_B_matrix = self.layer_self.add_weight(
        name="householder_parameters_B_matrix", shape=(n_parameters_B,), 
        initializer=initializer, dtype=
        self.layer_self.code_given_info_class.float_dtype, trainable=
        True)

        self.householder_parameters_A_matrix = self.layer_self.add_weight(
        name="householder_parameters_A_matrix", shape=(n_parameters_A,), 
        initializer=initializer, dtype=
        self.layer_self.code_given_info_class.float_dtype, trainable=
        True)

        # Assembles the tuples of first index, number of parameters, and
        # number of zeros necessary to construct each Householder vector
        # from the corresponding slice of the vector of trainable param-
        # eters

        self.householder_indices_A_matrix = []

        self.householder_indices_B_matrix = []

        counter_parameters_A = 0

        counter_parameters_B = 0

        for reflector_index in range(self.weights_rank):

            # Verifies if there is any component left to build a House-
            # holder vector for the A matrix

            if (self.n_neurons_current_main_layer-1-reflector_index)>0:

                # Appends the index of the first parameter for the cor-
                # responding Householder vector; the number of parame-
                # ters necessary for this vector, and the number of 
                # trailing zeros. Commences with the last Householder 
                # reflector since the multiplication by a vector is from
                # the right side

                n_dofs_householder_vector = reflector_index+1

                self.householder_indices_A_matrix.append(tuple([
                counter_parameters_A, n_dofs_householder_vector, 
                self.n_neurons_current_main_layer-
                n_dofs_householder_vector]))

                # Updates the counter of parameters for the A matrix 
                # that have already been mapped

                counter_parameters_A += n_dofs_householder_vector

            # Verifies if there is any component left to build a House-
            # holder vector for the B matrix

            if (self.n_neurons_last_main_layer-1-reflector_index)>0:

                # Appends the index of the first parameter for the cor-
                # responding Householder vector; the number of parame-
                # ters necessary for this vector, and the number of 
                # trailing zeros. As the B matrix is used in its trans-
                # posed version for calculation, the Householder reflec-
                # tors will be stored in the reversed order since it is
                # the exact same thing as transposing the final orthogo-
                # nal matrix. Commences with the last Householder re-
                # flector (which is the first in the transposed case) 
                # since the multiplication by a vector is from the right 
                # side

                n_dofs_householder_vector = (
                self.n_neurons_last_main_layer-reflector_index-1)

                self.householder_indices_B_matrix.append(tuple([
                counter_parameters_B, n_dofs_householder_vector, 
                self.n_neurons_last_main_layer-n_dofs_householder_vector
                ]))

                # Updates the counter of parameters for the B matrix 
                # that have already been mapped

                counter_parameters_B += n_dofs_householder_vector

        # Transforms the lists of indices for slicing into tuples to en-
        # sure performance

        self.householder_indices_A_matrix = tuple(
        self.householder_indices_A_matrix)

        self.householder_indices_B_matrix = tuple(
        self.householder_indices_B_matrix)
    
    # Defines a function to parse a single Householder vector from a
    # vector of all Householder vectors to construct an orthogonal ma-
    # trix. The incoming vector is a tensor [0.5*(m_rank*((2*n_rows)-
    # m_rank-1))], where m_rank is the number of columns of the final 
    # orthogonal matrix, and n_rows is the number of rows. Receives the
    # index of the Householder reflector in the Householder chain to
    # collect the corresponding Householder vector

    def get_householder_vector_from_parameters(self, householder_indices,
    householder_parameters, householder_reflector_index):
        
        # Gets the first and last indices of the Householder vector, 
        # then the vector that will become the Householder vector

        initial_index, length, number_of_trailing_zeros = (
        householder_indices[householder_reflector_index])

        raw_vector = tf.slice(householder_parameters, [initial_index], [
        length])

        # Computes the average of the raw vector

        average_raw_vector = tf.reduce_mean(raw_vector)

        # Concatenates the non-free component of the Householder vector
        # to the first position

        raw_vector = tf.concat([tf.sqrt((average_raw_vector*
        average_raw_vector)+self.householder_epsilon_squared)[None], 
        raw_vector], axis=0)

        # Rescales the raw vector to have unit norm, then adds the trai-
        # ling zeros and returns it

        return tf.pad(tf.math.l2_normalize(raw_vector), [[
        number_of_trailing_zeros, 0]])
    
    # Defines a function to evaluate the multiplication of one Househol-
    # der reflector of the Householder chain of the transposed B matrix 
    # of the SVD decomposition (A*diag(sigma)*transpose(B)) by the input 
    # vector of the corresponding layer. The input vector is a tensor
    # [n_samples, p_i] where p_i is the number of neurons of the i-th
    # layer

    def multiply_B_matrix_householder_reflector(self, input_vector, 
    householder_reflector_index):
        
        # Gets the Householder vector from the Householder parameters of
        # the B matrix. Keep in mind that the order of the Householder 
        # chain of the B matrix is versed with respect to the A matrix,
        # since B is transposed in the SVD

        householder_vector = self.get_householder_vector_from_parameters(
        self.householder_indices_B_matrix, 
        self.householder_parameters_B_matrix, householder_reflector_index)

        # Multiplies the input vector by the Householder reflector (the
        # operation is already broken down into the rank-1 calculation)

        return input_vector-(self.two*tf.reduce_sum(input_vector*
        householder_vector, axis=-1, keepdims=True)*householder_vector)
    
    # Defines a function to evaluate the multiplication of one Househol-
    # der reflector of the Householder chain of the A matrix of the SVD 
    # decomposition (A*diag(sigma)*transpose(B)) by the input vector of 
    # the corresponding layer. The input vector is a tensor [n_samples,
    # m_rank] where m_rank is the rank of the weights matrix correspon-
    # ding to this SVD

    def multiply_A_matrix_householder_reflector(self, input_vector, 
    householder_reflector_index):
        
        # Gets the Householder vector from the Householder parameters of
        # the A matrix

        householder_vector = self.get_householder_vector_from_parameters(
        self.householder_indices_A_matrix, 
        self.householder_parameters_A_matrix, householder_reflector_index)

        # Multiplies the input vector by the Householder reflector (the
        # operation is already broken down into the rank-1 calculation)

        return input_vector-(self.two*tf.reduce_sum(input_vector*
        householder_vector, axis=-1, keepdims=True)*householder_vector)
    
    # Defines a function to evaluate the multiplication of one Househol-
    # der reflector of the Householder chain of the transposed B matrix 
    # of the SVD decomposition (A*diag(sigma)*transpose(B)) by the re-
    # constructed B matrix so far. The partial_B_matrix is a tensor
    # [p_i, rank] where p_i is the number of neurons of the i-th layer 
    # and rank is the rank of the weight matrix

    def update_B_matrix_with_householder_reflector(self, 
    partial_B_matrix, householder_reflector_index):
        
        # Gets the Householder vector from the Householder parameters of
        # the B matrix. Keep in mind that the order of the Householder 
        # chain of the B matrix is reversed with respect to the A matrix,
        # since B is transposed in the SVD

        householder_vector = self.get_householder_vector_from_parameters(
        self.householder_indices_B_matrix, 
        self.householder_parameters_B_matrix, householder_reflector_index)

        # Multiplies the partially reconstructed B matrix by the House-
        # holder reflector to the left. However, the structure of the 
        # rank-1 projection is taken advantage of

        return partial_B_matrix-(self.two*householder_vector[:,None]*
        tf.matmul(householder_vector[None, :], partial_B_matrix))
    
    # Defines a function to evaluate the multiplication of one Househol-
    # der reflector of the Householder chain of the A matrix of the SVD 
    # decomposition (A*diag(sigma)*transpose(B)) by the reconstructed A
    # matrix so far. The partial_A_matrix is a tensor [p_(i+1), rank] 
    # where p_(i+1) is the number of neurons of the (i+1)-th layer and
    # rank is the rank of the weight matrix

    def update_A_matrix_with_householder_reflector(self, 
    partial_A_matrix, householder_reflector_index):
        
        # Gets the Householder vector from the Householder parameters of
        # the A matrix

        householder_vector = self.get_householder_vector_from_parameters(
        self.householder_indices_A_matrix, 
        self.householder_parameters_A_matrix, householder_reflector_index)

        # Multiplies the partially reconstructed A matrix by the House-
        # holder reflector to the left. However, the structure of the 
        # rank-1 projection is taken advantage of

        return partial_A_matrix-(self.two*householder_vector[:,None]*
        tf.matmul(householder_vector[None, :], partial_A_matrix))