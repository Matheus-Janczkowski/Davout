# Routine to store the class for the SVD-based architecture

import tensorflow as tf

from ..tool_box.activation_function_utilities import verify_activationDict

from ..tool_box.numerical_tools import BuildTensorflowMathExpressions

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
        architecture_info_dict, {"name": {}, "activations accessory la"+
        "yer list": {"type": list, "description": "List with dictionar"+
        "ies with names of activation functions as keys and number of "+
        "the corresponding neurons as values. These activations functi"+
        "ons are for the accessory layer. Each dictionary corresponds "+
        "to a layer"}}, 
        {"weights modulating function": {"type": str, "description": 
        "String with the name of the function that modulates the weigh"+
        "ts matrix", "default": "identity"}, "Householder epsilon": {
        "type": float, "description": "Float number with the tolerance"+
        " to the calculation of the non-free component of each Househo"+
        "lder vector", "default": 1.0}, "householder vector builder me"+
        "thod": {"type": str, "description": "String with the name of "+
        "the function that constructs each Householder vector from the"+
        " flat tensor of DOFs of the Householder chain for each orthog"+
        "onal matrix", "default": "get_householder_vector_from_paramet"+
        "ers"}}, "custom_architecture", "SVDQuotientSpace")

        # Selects the method for computing the Householder vectors from
        # the flat tensor of parameters

        self.method_for_householder_vector_from_parameters = getattr(
        self, architecture_info_dict["householder vector builder metho"+
        "d"])
        
        # Stores variables that will be used in the get_config for seri-
        # alization and class rebuilding

        self.layer_self = layer_self

        self.layer_number = self.layer_self.code_given_info_class.layer

        self.input_size_main_network = (
        self.layer_self.code_given_info_class.input_size_main_network)

        self.input_size_accessory_network = (
        self.layer_self.code_given_info_class.input_size_acessory_network)

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

        if self.layer_number<len(architecture_info_dict["activations a"+
        "ccessory layer list"]):

            self.activations_accessory_layer_dict = architecture_info_dict[
            "activations accessory layer list"][self.layer_number]

            # If this is the first layer, the input size of the accessory
            # layer is the input size of the accessory network. Multi-
            # plies by one to create a copy

            if self.layer_number==0:

                self.input_size_accessory_layer = (
                self.input_size_accessory_network*1)

            # Otherwise, gets the input size of the accessory layer by
            # subtracting one

            else:

                self.input_size_accessory_layer = sum([value["number o"+
                "f neurons"] if isinstance(value, dict) else value for (
                value) in architecture_info_dict["activations accessor"+
                "y layer list"][self.layer_number-1].values()])

        # If the number of the layer is not a index of the list of acti-
        # vations of the accessory network, returns an error

        else:

            raise IndexError("The "+str(self.layer_number)+"-th layer "+
            "of the main network does not have a corresponding layer i"+
            "n the accessory network. The provided list of dictionarie"+
            "s of activations for the accessory network within the var"+
            "iable 'activations accessory layer list' is:\n"+str(
            architecture_info_dict["activations accessory layer list"])+
            "\nwhich has a length of "+str(len(architecture_info_dict[
            "activations accessory layer list"])))
        
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
            
            self.modulation_option = "identity"

            # Sets the evaluator of the generic layer from input to the
            # method that multiplies the input vector by the chain of
            # Householder reflector instead of the method that builds 
            # the orthogonal matrix first and, then, computes the weight
            # matrix

            self.generic_layers_from_input = self.identity_modulation_from_input

            self.generic_layers_from_parameters = self.identity_modulation_call_with_parameters

        # Otherwise, gets the modulating function

        else:

            self.modulation_option = "not identity"

            # Initializes the class to build tensorflow expressions

            build_tensorflow_math_expressions = BuildTensorflowMathExpressions(
            dtype=self.layer_self.code_given_info_class.float_dtype)

            self.modulating_function = build_tensorflow_math_expressions(
            architecture_info_dict["weights modulating function"])

            # Sets the evaluator of the generic layer from input to the
            # method that assembles the SVD first and, then, applies the
            # modulation mapping

            self.generic_layers_from_input = self.non_identity_modulation_from_input

            self.generic_layers_from_parameters = self.non_identity_modulation_call_with_parameters

        # Defines the method to call this layer and get the forward res-
        # ponse. If this layer is the first layer

        if self.layer_number==0:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.layer_self.call_from_input_method = self.first_layer_call_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.layer_self.call_given_parameters = self.first_layer_call_with_parameters

            # Defines the method to compute the output shape for Keras
            # initialization

            self.layer_self.compute_output_shape = self.first_layer_output_shape

        # If this layer is the output layer

        elif self.layer_number==-1:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.layer_self.call_from_input_method = self.output_layer_call_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.layer_self.call_given_parameters = self.output_layer_call_with_parameters

            # Defines the method to compute the output shape for Keras
            # initialization

            self.layer_self.compute_output_shape = self.output_layer_output_shape

        # Otherwise, if it is any of the intermediate layers

        else:

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are fixed

            self.layer_self.call_from_input_method = self.generic_layers_from_input

            # Defines the method that will be used to call the layer's 
            # response when the trainable parameters are given

            self.layer_self.call_given_parameters = self.generic_layers_from_parameters

            # Defines the method to compute the output shape for Keras
            # initialization

            self.layer_self.compute_output_shape = self.hidden_layer_output_shape

        # Selects the method for reshaping the model parameters from a
        # flat vector

        self.update_layer_parameters = self.layer_update_parameters

        # Selects the method to update the model parameters in place

        self.layer_self.apply_parameters_to_layer = self.apply_parameters_layer

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

        # Initializes the weight matrix as the identity. The number of
        # rows of the initial matrix is the rank of the layer's weights
        # matrix since the B matrix is transposed

        weight_matrix = self.update_B_matrix_with_householder_chain(
        self.initial_weight_matrix, 
        self.householder_reflectors_indices_B, 
        self.householder_first_index_B, self.householder_length_B,
        self.householder_number_of_leading_zeros_B, 
        self.layer_self.householder_parameters_B_matrix,
        self.n_neurons_last_main_layer)

        # Multiplies this result by the singular values coming from the
        # accessory layer. The singular values are a tensor [n_samples,
        # rank]; the reconstructed weight matrix so far is a tensor [
        # rank, p_i]. The final result must be [n_samples, p_i+1, p_i]

        weight_matrix = self.singular_values_by_matrix(
        output_activations_accessory_layer, weight_matrix)
        
        # Multiplies by the A matrix of the SVD to reconstruct the SVD.
        # Then, uses the modulating function to get the weights matrix

        weight_matrix = self.modulating_function(
        self.update_A_matrix_with_householder_chain(weight_matrix, 
        self.householder_reflectors_indices_A, 
        self.householder_first_index_A, self.householder_length_A,
        self.householder_number_of_leading_zeros_A,
        self.layer_self.householder_parameters_A_matrix,
        self.n_neurons_current_main_layer))

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
        # B matrix of the SVD. Then multiplies it by the identity [rank,
        # number of neurons of the last layer]

        output_B = self.multiply_input_vector_by_householder_chain(input[
        0], self.householder_reflectors_indices_B, 
        self.householder_first_index_B, self.householder_length_B,
        self.householder_number_of_leading_zeros_B, 
        self.layer_self.householder_parameters_B_matrix,
        self.n_neurons_last_main_layer)

        # Multiplies this result by the singular values coming from the
        # accessory layer and, then, multiplies by the A matrix of the 
        # SVD. Note that only the components of the output of B linked
        # to the rank of the layer are used

        singular_values_output = self.singular_values_multiplier(
        output_activations_accessory_layer, output_B)

        output_A = self.multiply_input_vector_by_householder_chain(
        singular_values_output, self.householder_reflectors_indices_A, 
        self.householder_first_index_A, self.householder_length_A,
        self.householder_number_of_leading_zeros_A, 
        self.layer_self.householder_parameters_A_matrix,
        self.n_neurons_current_main_layer)

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
    
    # Defines a method to evaluate the accessory and the main layers of
    # a generic layer of the SVD-based architecture given the trainable
    # parameters as a flat tensor. This method is dedicated to the case
    # of a modulation mapping different than the identity

    def non_identity_modulation_call_with_parameters(self, layer_input, 
    parameters):

        # Gets the trainable parameters

        (W_accessory, b_accessory, householder_parameters_A_matrix, 
        householder_parameters_B_matrix) = self.layer_update_parameters(
        parameters)

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix. Then, splits the result into the
        # different families of activation functions and evaluates the
        # output of the accessory layer

        x_splits_accessory_layer = tf.split(tf.matmul(layer_input[1], 
        W_accessory)+b_accessory, 
        self.layer_self.neurons_per_activation_acessory_layer, axis=-1)

        # Applies the split multiplication to each activation function
        # and, then, concatenates them back into a single vector

        output_activations_accessory_layer = tf.concat(
        [activation_function(split) for activation_function, split in (
        zip(self.activation_list_acessory_network, 
        x_splits_accessory_layer))], axis=-1)

        # Initializes the weight matrix as the identity. The number of
        # rows of the initial matrix is the rank of the layer's weights
        # matrix since the B matrix is transposed

        weight_matrix = self.update_B_matrix_with_householder_chain(
        self.initial_weight_matrix, 
        self.householder_reflectors_indices_B, 
        self.householder_first_index_B, self.householder_length_B,
        self.householder_number_of_leading_zeros_B, 
        householder_parameters_B_matrix, self.n_neurons_last_main_layer)

        # Multiplies this result by the singular values coming from the
        # accessory layer. The singular values are a tensor [n_samples,
        # rank]; the reconstructed weight matrix so far is a tensor [
        # rank, p_i]. The final result must be [n_samples, p_i+1, p_i]

        weight_matrix = self.singular_values_by_matrix(
        output_activations_accessory_layer, weight_matrix)
        
        # Multiplies by the A matrix of the SVD to reconstruct the SVD.
        # Then, uses the modulating function to get the weights matrix

        weight_matrix = self.modulating_function(
        self.update_A_matrix_with_householder_chain(weight_matrix, 
        self.householder_reflectors_indices_A, 
        self.householder_first_index_A, self.householder_length_A,
        self.householder_number_of_leading_zeros_A, 
        householder_parameters_A_matrix, 
        self.n_neurons_current_main_layer))

        # Gets the output of the main network and splits it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(tf.einsum('sij,sj->si', 
        weight_matrix, layer_input[0]), 
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

    # Defines a method to evaluate the accessory and the main layers of
    # a generic layer of the SVD-based architecture given the trainable
    # parameters as a flat tensor. This method is dedicated to the case
    # of a modulation mapping equal to the identity

    def identity_modulation_call_with_parameters(self, layer_input, 
    parameters):

        # Gets the trainable parameters

        (W_accessory, b_accessory, householder_parameters_A_matrix, 
        householder_parameters_B_matrix) = self.layer_update_parameters(
        parameters)

        # Gets the multiplication of the parcel of the accessory layer
        # by its corresponding matrix. Then, splits the result into the
        # different families of activation functions and evaluates the
        # output of the accessory layer

        x_splits_accessory_layer = tf.split(tf.matmul(layer_input[1], 
        W_accessory)+b_accessory, 
        self.layer_self.neurons_per_activation_acessory_layer, axis=-1)

        # Applies the split multiplication to each activation function
        # and, then, concatenates them back into a single vector

        output_activations_accessory_layer = tf.concat(
        [activation_function(split) for activation_function, split in (
        zip(self.activation_list_acessory_network, 
        x_splits_accessory_layer))], axis=-1)

        # Multiplies the incoming batched input vector by the transposed
        # B matrix of the SVD. Then multiplies it by the identity [rank,
        # number of neurons of the last layer]

        output_B = self.multiply_input_vector_by_householder_chain(
        layer_input[0], self.householder_reflectors_indices_B, 
        self.householder_first_index_B, self.householder_length_B,
        self.householder_number_of_leading_zeros_B, 
        householder_parameters_B_matrix, self.n_neurons_last_main_layer)

        # Multiplies this result by the singular values coming from the
        # accessory layer and, then, multiplies by the A matrix of the 
        # SVD. Note that only the components of the output of B linked
        # to the rank of the layer are used

        singular_values_output = self.singular_values_multiplier(
        output_activations_accessory_layer, output_B)

        output_A = self.multiply_input_vector_by_householder_chain(
        singular_values_output, self.householder_reflectors_indices_A, 
        self.householder_first_index_A, self.householder_length_A,
        self.householder_number_of_leading_zeros_A, 
        householder_parameters_A_matrix, 
        self.n_neurons_current_main_layer)

        # Gets the output of the main network and splits it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(output_A, 
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
    
    # Defines a method for getting the input vector of the NN and 
    # breaking it down into the main and accessory networks. This method
    # must be used during training to enforce orthogonality to the SVD
    # matrices, since the parameters are given in a flat tensor

    def first_layer_call_with_parameters(self, layer_input, parameters):
        
        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        segmented_layer_input = (layer_input[..., :(
        self.input_size_main_network)], layer_input[..., 
        self.input_size_main_network:])

        # Evaluates the layer using the parameters

        return self.generic_layers_from_parameters(segmented_layer_input,
        parameters)
    
    # Defines a function to get the output of a layer. Only the result 
    # of the main layer is returned

    def output_layer_call_with_parameters(self, layer_input, parameters):
        
        # Evaluates the layer using the parameters

        return self.generic_layers_from_parameters(layer_input,
        parameters)[0]
    
    # Defines a function to build this layer during serialization

    def mixed_layer_builder(self, input_shape):

        # Gets a list with the numbers of neurons per activation functi-
        # on for the accessory network (u)

        self.layer_self.neurons_per_activation_acessory_layer = [value[
        "number of neurons"] if isinstance(value, dict) else value for (
        value) in self.activations_accessory_layer_dict.values(
        )]

        # Counts the number of neurons in the accessory layer that spits
        # the singular values to the SVD of the weights matrix of the 
        # main layer

        self.number_of_neurons_accessory_layer = sum(
        self.layer_self.neurons_per_activation_acessory_layer)

        # Creates a dense layer with biases for the accessory network

        self.layer_self.dense_W_accessory = tf.keras.layers.Dense(
        self.number_of_neurons_accessory_layer, name="W_accessory", 
        use_bias=True)

        # Effectively builds the accessory layer

        self.layer_self.dense_W_accessory.build((None, 
        self.input_size_accessory_layer))

        # Initializes all trainable parameters and indices for the
        # Householder chains

        self.initialize_householder_parameters()

        # Constructs the layer

        super(type(self.layer_self), self.layer_self).build(input_shape)

        # Recovers the indices of the trainables variables

        self.index_W_accessory = self.search_trainable_variable_index(
        self.layer_self.dense_W_accessory.kernel)

        self.index_bias_accessory = self.search_trainable_variable_index(
        self.layer_self.dense_W_accessory.bias)

        self.index_householder_A = self.search_trainable_variable_index(
        self.layer_self.householder_parameters_A_matrix)

        self.index_householder_B = self.search_trainable_variable_index(
        self.layer_self.householder_parameters_B_matrix)

    # Defines a function to tell the shape of the output of the layer in
    # case it is the first layer

    def first_layer_output_shape(self, input_shape):

        # Gets the first axis of the input shape, and returns the number 
        # of neurons neurons of each network alongside it. The output of
        # the main layer has the shape equal to the number of neurons of
        # the current main layer. The output of the auxiliary layer has
        # the shape equal to the rank of the corresponding weights matrix

        batch_size = input_shape[0]

        return ((batch_size, self.n_neurons_current_main_layer), (
        batch_size, self.weights_rank))

    # Defines a function to tell the shape of the output of the layer in
    # case it is any hidden layer

    def hidden_layer_output_shape(self, input_shape):

        # Gets the first axis of the input shape, and returns the number 
        # of neurons neurons of each network alongside it. The output of
        # the main layer has the shape equal to the number of neurons of
        # the current main layer. The output of the auxiliary layer has
        # the shape equal to the rank of the corresponding weights matrix

        batch_size = input_shape[0][0]

        return ((batch_size, self.n_neurons_current_main_layer), (
        batch_size, self.weights_rank))

    # Defines a function to tell the shape of the output of the layer in
    # case it is the output layer

    def output_layer_output_shape(self, input_shape):

        # Gets the first axis of the input shape, and returns the number 
        # of neurons neurons of the main network alongside. The output 
        # of the main layer has the shape equal to the number of neurons
        # of the current main layer. The output of the auxiliary layer 
        # is ignored

        batch_size = input_shape[0][0]

        return (batch_size, self.n_neurons_current_main_layer)

    # Defines a function to search for the index of a trainable variable
    # in the flat tensor

    def search_trainable_variable_index(self, trainable_variable):

        # Iterates through the trainable variables

        for index, variable in enumerate(
        self.layer_self.trainable_variables):
            
            # Verifies if the trainable variable is equal to this varia-
            # ble. In positive case, returns the index

            if trainable_variable is variable:

                return index
            
        # If no index has been returned, throw an error

        variables_names = ""

        for name in self.layer_self.trainable_variables:

            variables_names += "\n"+str(name)

        raise ValueError("The trainable variable '"+str(
        trainable_variable)+"' was not found in the list of trainable "+
        "variables of architecture '"+str(self.__class__.__name__)+"':"+
        variables_names)

    # Defines a function to get the flat and sliced parameters, then, 
    # converts it to tensors given the shapes of the trainables varia-
    # bles

    def layer_update_parameters(self, flat_parameters):

        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network, the rest for the accessory network

        W_accessory = tf.reshape(flat_parameters[self.index_W_accessory], 
        self.layer_self.trainable_variables_shapes[
        self.index_W_accessory])
        
        b_accessory = tf.reshape(flat_parameters[
        self.index_bias_accessory], 
        self.layer_self.trainable_variables_shapes[
        self.index_bias_accessory])
        
        householder_parameters_A_matrix = tf.reshape(flat_parameters[
        self.index_householder_A], 
        self.layer_self.trainable_variables_shapes[
        self.index_householder_A])
        
        householder_parameters_B_matrix = tf.reshape(flat_parameters[
        self.index_householder_B], 
        self.layer_self.trainable_variables_shapes[
        self.index_householder_B])

        return (W_accessory, b_accessory, 
        householder_parameters_A_matrix, householder_parameters_B_matrix)
    
    # Defines a function to apply the weights and biases to the first 
    # layer

    def apply_parameters_layer(self, flat_parameters):

        # Gets the weights and biases from the flat vector

        (W_accessory, b_accessory, householder_parameters_A_matrix, 
        householder_parameters_B_matrix) = self.update_layer_parameters(
        flat_parameters)

        # Updates the weights and biases matrices

        self.layer_self.dense_W_accessory.kernel.assign(W_accessory)

        self.layer_self.dense_W_accessory.bias.assign(b_accessory)

        self.layer_self.householder_parameters_A_matrix.assign(
        householder_parameters_A_matrix)

        self.layer_self.householder_parameters_B_matrix.assign(
        householder_parameters_B_matrix)

    ####################################################################
    #                      Householder reflectors                      #
    ####################################################################

    # Defines a function to initialize the trainable parameters of the
    # Householder chain of each orthogonal matrix of the SVD of this 
    # layer. This function also creates the tuples of indices

    def initialize_householder_parameters(self):

        # Gets the number of neurons of the incoming layer and the num-
        # ber of neurons of this layer. Sums 1 to the layer number since
        # the list of numbers of neurons in each layer include the input
        # layer

        # Verifies if this is the output layer, then corrects for the 
        # given layer number

        if self.layer_number==-1:

            self.n_neurons_last_main_layer = self.layer_self.code_given_info_class.number_neurons_per_main_layer[
            self.layer_number-1]

            self.n_neurons_current_main_layer = self.layer_self.code_given_info_class.number_neurons_per_main_layer[
            self.layer_number]

        # Otherwise, just add the number 1 to correct for the input layer

        else:

            self.n_neurons_last_main_layer = self.layer_self.code_given_info_class.number_neurons_per_main_layer[
            self.layer_number]

            self.n_neurons_current_main_layer = self.layer_self.code_given_info_class.number_neurons_per_main_layer[
            self.layer_number+1]

        # Computes the rank of the weights matrix

        self.weights_rank = min(self.n_neurons_last_main_layer, 
        self.n_neurons_current_main_layer)

        # Verifies if the rank of the weights matrix is equal to the 
        # number of neurons of the accessory layer (since the accessory
        # layer spits the singular values of the SVD of the weights ma-
        # trix of the main layer)

        if self.number_of_neurons_accessory_layer!=self.weights_rank:

            raise ValueError("The "+str(self.layer_number)+"-th main l"+
            "ayer has a rank of "+str(self.weights_rank)+", since the "+
            "previous main layer has "+str(
            self.n_neurons_last_main_layer)+" neurons and the current "+
            "layer has "+str(self.n_neurons_current_main_layer)+" neur"+
            "ons. The corresponding accessory layer has "+str(
            self.number_of_neurons_accessory_layer)+" neurons, but it "+
            "must have the same number of neurons as the rank of the c"+
            "orresponding main layer.\nThe main network has the follow"+
            "ing list of numbers of neurons per layer:\n"+str(
            self.layer_self.code_given_info_class.number_neurons_per_main_layer))

        # In case the modulating function is not the identity, creates
        # the initial weight matrix as the identity

        if self.modulation_option!="identity":

            self.initial_weight_matrix = tf.eye(self.weights_rank, 
            self.n_neurons_last_main_layer, dtype=
            self.layer_self.code_given_info_class.float_dtype)

        # Initializes the tuple with the indices of the Householder re-
        # flectors of the Householder chain for each orthogonal matrix
        # individually. This happens because, for full-rank orthogonal
        # matrices, the last Householder reflector is empty of DOFs

        if self.n_neurons_last_main_layer<self.n_neurons_current_main_layer:

            # In case the last layer has less neurons than the current
            # layer, the rank is dominated by the last layer

            self.householder_reflectors_indices_A = tf.range(
            self.weights_rank, dtype=
            self.layer_self.code_given_info_class.int_dtype)

            self.householder_reflectors_indices_B = tf.range(
            self.weights_rank-1, dtype=
            self.layer_self.code_given_info_class.int_dtype)

            # Sets the function to multiply the output of the operation
            # B.T*input by the tensor of singular values

            self.singular_values_multiplier = self.multiply_input_vector_by_singular_values_expanding_layer

            # Sets the function to multiply the B matrix by the tensor 
            # of singular values

            self.singular_values_by_matrix = self.multiply_B_matrix_by_singular_values_expanding_layer

        # If the current layer has less neurons than the last layer, the
        # rank of the weight matrix is dominated by the current layer 
        # and the orthogonal matrix A will be full-rank

        elif self.n_neurons_current_main_layer<self.n_neurons_last_main_layer:

            # In case the last layer has less neurons than the current
            # layer, the rank is dominated by the last layer

            self.householder_reflectors_indices_A = tf.range(
            self.weights_rank-1, dtype=
            self.layer_self.code_given_info_class.int_dtype)

            self.householder_reflectors_indices_B = tf.range(
            self.weights_rank, dtype=
            self.layer_self.code_given_info_class.int_dtype)

            # Sets the function to multiply the output of the operation
            # B.T*input by the tensor of singular values

            self.singular_values_multiplier = self.multiply_input_vector_by_singular_values_shirinking_layer

            # Sets the function to multiply the B matrix by the tensor 
            # of singular values

            self.singular_values_by_matrix = self.multiply_B_matrix_by_singular_values_shrinking_layer

        # If the current and the last layers have the same number of 
        # neurons, both orthogonal matrices A and B have full rank

        else:

            self.householder_reflectors_indices_A = tf.range(
            self.weights_rank-1, dtype=
            self.layer_self.code_given_info_class.int_dtype)
            
            self.householder_reflectors_indices_B = tf.range(
            self.weights_rank-1, dtype=
            self.layer_self.code_given_info_class.int_dtype)

            # Sets the function to multiply the output of the operation
            # B.T*input by the tensor of singular values

            self.singular_values_multiplier = self.multiply_input_vector_by_singular_values_shirinking_layer

            # Sets the function to multiply the B matrix by the tensor 
            # of singular values

            self.singular_values_by_matrix = self.multiply_B_matrix_by_singular_values_shrinking_layer

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

        self.layer_self.householder_parameters_A_matrix = self.layer_self.add_weight(
        name="householder_parameters_A_matrix", shape=(n_parameters_A,), 
        initializer=initializer, dtype=
        self.layer_self.code_given_info_class.float_dtype, trainable=
        True)

        self.layer_self.householder_parameters_B_matrix = self.layer_self.add_weight(
        name="householder_parameters_B_matrix", shape=(n_parameters_B,), 
        initializer=initializer, dtype=
        self.layer_self.code_given_info_class.float_dtype, trainable=
        True)

        # Assembles the tuples of first index, number of parameters, and
        # number of zeros necessary to construct each Householder vector
        # from the corresponding slice of the vector of trainable param-
        # eters

        self.householder_first_index_A = []

        self.householder_first_index_B = []

        self.householder_length_A = []

        self.householder_length_B = []

        self.householder_number_of_leading_zeros_A = []

        self.householder_number_of_leading_zeros_B = []

        counter_parameters_A = 0

        counter_parameters_B = 0

        for reflector_index in range(self.weights_rank):

            # Verifies if there is any component left to build a House-
            # holder vector for the A matrix

            if (self.n_neurons_current_main_layer-1-reflector_index)>0:

                # Appends the index of the first parameter for the cor-
                # responding Householder vector; the number of parame-
                # ters necessary for this vector, and the number of 
                # leading zeros. Commences with the last Householder re-
                # flector since the multiplication by a vector is from
                # the right side

                n_dofs_householder_vector = (reflector_index+
                self.n_neurons_current_main_layer-self.weights_rank)

                self.householder_first_index_A.append(
                counter_parameters_A)

                self.householder_length_A.append(
                n_dofs_householder_vector)

                self.householder_number_of_leading_zeros_A.append(
                self.n_neurons_current_main_layer-
                n_dofs_householder_vector-1)

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

                self.householder_first_index_B.append(
                counter_parameters_B)

                self.householder_length_B.append(
                n_dofs_householder_vector)

                self.householder_number_of_leading_zeros_B.append(
                self.n_neurons_last_main_layer-n_dofs_householder_vector
                -1)

                # Updates the counter of parameters for the B matrix 
                # that have already been mapped

                counter_parameters_B += n_dofs_householder_vector

        # Transforms the lists of indices for slicing into tuples to en-
        # sure performance

        self.householder_first_index_A = tf.constant(
        self.householder_first_index_A, dtype=
        self.layer_self.code_given_info_class.int_dtype)
        
        self.householder_first_index_B = tf.constant(
        self.householder_first_index_B, dtype=
        self.layer_self.code_given_info_class.int_dtype)

        self.householder_length_A = tf.constant(
        self.householder_length_A, dtype=
        self.layer_self.code_given_info_class.int_dtype)

        self.householder_length_B = tf.constant(
        self.householder_length_B, dtype=
        self.layer_self.code_given_info_class.int_dtype)

        self.householder_number_of_leading_zeros_A = tf.constant(
        self.householder_number_of_leading_zeros_A, dtype=
        self.layer_self.code_given_info_class.int_dtype)

        self.householder_number_of_leading_zeros_B = tf.constant(
        self.householder_number_of_leading_zeros_B, dtype=
        self.layer_self.code_given_info_class.int_dtype)

    # TEST-TensorArray
    # Defines a function to get a contiguous flat tensor and split it 
    # into a set of vectors of different sizes. Each vector comprises
    # the degrees of freedom of the corresponding Householder reflector

    def split_flat_tensor_into_householder_dofs(self,
    householder_parameters, householder_lengths):

        # Splits the flat tensor of Householder DOFs of the whole chain

        split_householder_raw_vectors = tf.split(householder_parameters,
        householder_lengths)

        householder_dofs_tensor_array = tf.TensorArray(dtype=
        self.layer_self.code_given_info_class.float_dtype,
        size=len(householder_lengths), infer_shape=False,
        clear_after_read=False)

        # Iterates over the python list generated by tf.split to write
        # the individual components into the TensorArray that stores the
        # raw vector of DOFs of each Householder reflector

        for i, split_householder_raw_vector in enumerate(
        split_householder_raw_vectors):

            householder_dofs_tensor_array = householder_dofs_tensor_array.write(
            i, split_householder_raw_vector)

        # Returns the TensorArray only

        return householder_dofs_tensor_array

    # Defines a function to multiply the result of the multiplication of
    # the input vector by the B matrix then by the tensor of singular 
    # values. This function is for the case of identity modulation func-
    # tion alongside the setting of more neurons in the current layer
    # than in the previous layer

    def multiply_input_vector_by_singular_values_expanding_layer(self,
    singular_values, output_B):
        
        # Gets the multiplication of the output of the multiplication of
        # the input tensor by B matrix. Recovers only the columns associ-
        # ated with the rank of the weights matrix

        output_rank = singular_values*output_B[:, :self.weights_rank]

        # If there are more neurons in the current layer than in the 
        # previous layer, columns of zeros must be added to the right to
        # compensate for the difference in number of columns in the up-
        # coming operations with the Householder reflectors of matrix A

        return tf.pad(output_rank, [[0, 0], [0, 
        self.n_neurons_current_main_layer-self.weights_rank]])
    
    # Defines a function to multiply the result of the multiplication of
    # the input vector by the B matrix then by the tensor of singular 
    # values. This function is for the case of identity modulation func-
    # tion alongside the setting of less neurons in the current layer
    # than in the previous layer

    def multiply_input_vector_by_singular_values_shirinking_layer(self,
    singular_values, output_B):
        
        # Gets the multiplication of the output of the multiplication of
        # the input tensor by B matrix. Recovers only the columns associ-
        # ated with the rank of the weights matrix

        return singular_values*output_B[:, :self.weights_rank]
    
    # Defines a function to multiply the reconstructed B matrix by the
    # singular values in case of an expanding layer. This function is u-
    # sed to pad the multiplication of the B matrix by the singular va-
    # lues with extra null columns, since the current layer has more 
    # neurons than the previous layer

    def multiply_B_matrix_by_singular_values_expanding_layer(self,
    singular_values, B_matrix):
        
        # Multiplies the B matrix by the singular values coming from the
        # accessory layer. The singular values are a tensor [n_samples,
        # rank]; the reconstructed weight matrix so far is a tensor [
        # rank, p_i]. The final result must be [n_samples, rank, p_i]

        weight_matrix = (singular_values[:,:,None]*B_matrix[None,:,:])

        # Pads the late result with zero rows below to make the partial 
        # weight matrix [n_samples, p_(i+1), p_i]

        return tf.pad(weight_matrix, paddings=[[0, 0], [0, 
        self.n_neurons_current_main_layer-self.weights_rank], [0, 0]])
    
    # Defines a function to multiply the reconstructed B matrix by the
    # singular values in case of a shrinking layer. This function is u-
    # sed when the current layer has less or an equal number of neurons 
    # than the previous layer

    def multiply_B_matrix_by_singular_values_shrinking_layer(self,
    singular_values, B_matrix):
        
        # Multiplies the B matrix by the singular values coming from the
        # accessory layer. The singular values are a tensor [n_samples,
        # rank]; the reconstructed weight matrix so far is a tensor [
        # rank, p_i]. The final result must be [n_samples, rank, p_i]

        return (singular_values[:,:,None]*B_matrix[None,:,:])

    # Defines the same function as get_householder_vector_from_parameters
    # but XLA-compliant

    def get_householder_vector_from_parameters_XLA_compliant(self, 
    householder_first_index, householder_length,
    householder_number_of_leading_zeros, householder_parameters, 
    householder_reflector_index, input_dimensionality):

        # Gets the first and last indices of the Householder vector
        
        initial_index = householder_first_index[
        householder_reflector_index] 
    
        length = householder_length[householder_reflector_index]
    
        number_of_leading_zeros = householder_number_of_leading_zeros[
        householder_reflector_index]
    
        # Gets the number of DOFs of the whole chain of Householder re-
        # flectors, then creates a vector of the indices of those DOFs
    
        n_dofs_of_chain = tf.shape(householder_parameters)[0]
    
        chain_dofs_indices = tf.range(n_dofs_of_chain, dtype=
        initial_index.dtype)
    
        # Creates a mask to take out the DOFs of this Householder vector
    
        current_householder_mask = (chain_dofs_indices>=initial_index
        ) & (chain_dofs_indices<(initial_index+length))
    
        # Gets the vector of DOFs of this Householder vector but in a 
        # tensor [n_dofs_of_chain], such that all components that are 
        # not related to this Householder vector are zero
    
        raw_vector = tf.where(current_householder_mask, 
        householder_parameters, 0.0)
    
        # Computes the average of the DOFs of this Householder vector
    
        average_raw_vector = tf.reduce_sum(raw_vector)/tf.cast(length,
        householder_parameters.dtype)
    
        # Calculates the first non-zero component of the Householder 
        # vector
    
        unnormalized_first_component = tf.sqrt(tf.square(
        average_raw_vector)+self.householder_epsilon_squared)
    
        # Calculates the normalizing factor to make the Householder vec-
        # tor unitary
    
        alpha = tf.math.rsqrt(tf.reduce_sum(tf.square(raw_vector))+
        tf.square(unnormalized_first_component))
    
        # Creates a range with all indices of the Householder vector
    
        all_indices_householder_vector = tf.range(input_dimensionality, 
        dtype=initial_index.dtype)
    
        # Creates a mask for the first non-zero component of the House-
        # holder vector
    
        first_non_zero_component_mask = ((all_indices_householder_vector
        )==number_of_leading_zeros)
    
        # Creates a mask for the indexing of the raw vector
    
        raw_vector_mask = (all_indices_householder_vector>(
        number_of_leading_zeros)) & (all_indices_householder_vector<=(
        number_of_leading_zeros+length))
    
        # Creates a range of indices for the Householder parameters
    
        householder_indices = tf.clip_by_value(initial_index+(
        all_indices_householder_vector-(number_of_leading_zeros+1)), 0, 
        n_dofs_of_chain-1)
    
        # Gets the Householder DOFs mapped to the right position in the 
        # final Householder vector, i.e. considering the leading zeros 
        # and the first non-zero component
    
        raw_vector_DOFs = tf.gather(householder_parameters, 
        householder_indices)
    
        # Assembles the Householder vector and multiplies by the norma-
        # lization factor
    
        householder_vector = alpha*tf.where(
        first_non_zero_component_mask, unnormalized_first_component, 
        tf.where(raw_vector_mask, raw_vector_DOFs, 0.0))

        return householder_vector
    
    # Defines a function to parse a single Householder vector from a
    # vector of all Householder vectors to construct an orthogonal ma-
    # trix. The incoming vector is a tensor [0.5*(m_rank*((2*n_rows)-
    # m_rank-1))], where m_rank is the number of columns of the final 
    # orthogonal matrix, and n_rows is the number of rows. Receives the
    # index of the Householder reflector in the Householder chain to
    # collect the corresponding Householder vector
    
    def get_householder_vector_from_parameters(self, 
    householder_first_index, householder_length, 
    householder_number_of_leading_zeros, 
    #split_householder_DOFs,
    #householder_parameters, 
    householder_dofs_tensor_array,
    householder_reflector_index, input_dimensionality):
        
        # Gets the first and last indices of the Householder vector, 
        # then the vector that will become the Householder vector

        initial_index = householder_first_index[
        householder_reflector_index]

        length = householder_length[householder_reflector_index]

        number_of_leading_zeros = householder_number_of_leading_zeros[
        householder_reflector_index]

        """
        raw_vector = tf.slice(householder_parameters, [initial_index], [
        length])#"""

        #raw_vector = split_householder_DOFs[householder_reflector_index]

        # TensorArray
        #"""
        raw_vector = householder_dofs_tensor_array.read(
        householder_reflector_index) #"""

        # Computes the average of the raw vector

        average_raw_vector = tf.reduce_mean(raw_vector)

        # Concatenates the non-free component of the Householder vector
        # to the first position

        raw_vector = tf.math.l2_normalize(tf.concat([tf.sqrt(
        tf.math.square(average_raw_vector)+
        self.householder_epsilon_squared)[None], raw_vector], axis=0))

        # Rescales the raw vector to have unit norm, then adds the trai-
        # ling zeros and returns it

        return raw_vector, number_of_leading_zeros
    
    # Defines a function to evaluate the multiplication of one Househol-
    # der reflector of the Householder chain of one of the two orthogo-
    # nal matrices of the SVD decomposition (A*diag(sigma)*transpose(B))
    # by the input vector of the corresponding layer. The input vector 
    # is a tensor [n_samples, p_i] where p_i is the number of neurons of
    # the i-th layer

    def multiply_input_vector_by_householder_reflector(self, 
    input_vector, householder_reflector_index, householder_first_index, 
    householder_length, householder_number_of_leading_zeros, 
    #split_householder_DOFs,
    #householder_parameters_orthogonal_matrix, 
    householder_dofs_tensor_array,
    input_dimensionality):
        
        # Gets the Householder vector from the Householder parameters of
        # the B matrix. Keep in mind that the order of the Householder 
        # chain of the B matrix is versed with respect to the A matrix,
        # since B is transposed in the SVD

        (householder_vector_core, number_of_leading_zeros
        ) = self.method_for_householder_vector_from_parameters(
        householder_first_index, householder_length, 
        householder_number_of_leading_zeros, 
        #split_householder_DOFs,
        #householder_parameters_orthogonal_matrix, 
        householder_dofs_tensor_array,
        householder_reflector_index, input_dimensionality)

        # Multiplies the input vector by the Householder reflector (the
        # operation is already broken down into the rank-1 calculation).
        # Note that the multiplication by the Householder vector is done
        # only for the non-zero components and that the result is padded
        # to be added to the original input vector

        return input_vector+tf.pad(-(self.two*tf.einsum('bi,i->b', 
        input_vector[:,number_of_leading_zeros:], 
        householder_vector_core)[:,None]*householder_vector_core[None,:]
        ), [[0,0], [number_of_leading_zeros,0]])
    
    # Defines a function to create a wrapper for the method that multi-
    # plies the input vector by an orthogonal matrix and, then, calls
    # foldl to evaluate the chain of Householder reflectors operating on
    # the input vector

    def multiply_input_vector_by_householder_chain(self, input_vector, 
    householder_reflector_indices, householder_first_index, 
    householder_length, householder_number_of_leading_zeros, 
    householder_parameters_orthogonal_matrix, input_dimensionality):

        # Splits the Householder DOFs of each Householder vector from 
        # the flat tensor of DOFs of the chain

        #split_householder_DOFs = tf.split(
        #householder_parameters_orthogonal_matrix, householder_length)

        """
        split_householder_DOFs = tf.RaggedTensor.from_row_lengths(
        householder_parameters_orthogonal_matrix,
        householder_length)#"""

        #"""
        householder_dofs_tensor_array = self.split_flat_tensor_into_householder_dofs(
        householder_parameters_orthogonal_matrix, householder_length)#"""

        # Defines the step function that will perform each update of the
        # input tensor by means of the Householder reflector

        def update_step(accumulator_vector, householder_reflector_index):

            return self.multiply_input_vector_by_householder_reflector(
            accumulator_vector, householder_reflector_index, 
            householder_first_index, householder_length, 
            householder_number_of_leading_zeros, 
            #split_householder_DOFs,
            #householder_parameters_orthogonal_matrix,
            householder_dofs_tensor_array,
            input_dimensionality)

        # Uses foldl to perform the chain multiplication

        return tf.foldl(update_step, householder_reflector_indices,
        initializer=input_vector)
    
    # Defines a function to evaluate the multiplication of the Househol-
    # der chain of the transposed B matrix of the SVD decomposition (A*
    # diag(sigma)*transpose(B)). The partial_B_matrix is a tensor [p_i, 
    # rank] where p_i is the number of neurons of the i-th layer and 
    # rank is the rank of the weight matrix

    #@tf.function
    def update_B_matrix_with_householder_chain(self, partial_B_matrix, 
    householder_reflector_indices, householder_first_index_B, 
    householder_length_B, householder_number_of_leading_zeros_B,
    householder_parameters_B_matrix, input_dimensionality):
        
        # Iterates through the indices of Householder reflectors

        for householder_reflector_index in householder_reflector_indices:
        
            # Gets the Householder vector from the Householder param-
            # eters of the B matrix. Keep in mind that the order of the 
            # Householder chain of the B matrix is reversed with respect 
            # to the A matrix, since B is transposed in the SVD

            (householder_vector_core, number_of_leading_zeros
            ) = self.method_for_householder_vector_from_parameters(
            householder_first_index_B, householder_length_B, 
            householder_number_of_leading_zeros_B, 
            householder_parameters_B_matrix, householder_reflector_index,
            input_dimensionality)

            # Gets the Householder vector by padding the core with the
            # leading zeros

            householder_vector = tf.pad(householder_vector_core, [[
            number_of_leading_zeros, 0]])

            # Multiplies the partially reconstructed B matrix by the 
            # Householder reflector to the left. However, the structure 
            # of the rank-1 projection is taken advantage of

            partial_B_matrix = partial_B_matrix-(self.two*tf.matmul(
            partial_B_matrix, householder_vector[:, None])*
            householder_vector[None,:])

        # Returns the recursively updated matrix

        return partial_B_matrix
    
    # Defines a function to evaluate the multiplication of the Househol-
    # der chain of the A matrix of the SVD decomposition (A*diag(sigma)*
    # transpose(B)). The partial_A_matrix is a tensor [n_samples, p_(i+1
    # ), rank] where p_(i+1) is the number of neurons of the (i+1)-th 
    # layer and rank is the rank of the weight matrix

    def update_A_matrix_with_householder_chain(self, partial_A_matrix, 
    householder_reflector_indices, householder_first_index_A, 
    householder_length_A, householder_number_of_leading_zeros_A,
    householder_parameters_A_matrix, input_dimensionality):
        
        # Iterates through the indices of Householder reflectors

        for householder_reflector_index in householder_reflector_indices:
        
            # Gets the Householder vector from the Householder parameters 
            # of the A matrix

            (householder_vector_core, number_of_leading_zeros
            ) = self.method_for_householder_vector_from_parameters(
            householder_first_index_A, householder_length_A, 
            householder_number_of_leading_zeros_A, 
            householder_parameters_A_matrix, householder_reflector_index,
            input_dimensionality)

            # Gets the Householder vector by padding the core with the
            # leading zeros

            householder_vector = tf.pad(householder_vector_core, [[
            number_of_leading_zeros, 0]])

            # Multiplies the partially reconstructed A matrix by the 
            # Householder reflector to the left. However, the structure 
            # of the rank-1 projection is taken advantage of. The multi-
            # plication of the Householder vector by the partial update 
            # of the A matrix is different, because A matrix has the di-
            # mension with the number of samples. The dimension of sam-
            # ples was gained due to the multiplication by the singular 
            # values coming from the accessory layer

            partial_A_matrix = partial_A_matrix-(self.two*
            householder_vector[None, :, None]*tf.einsum("p,spr->sr", 
            householder_vector, partial_A_matrix)[:, None, :])

        # Returns the recursively updated matrix

        return partial_A_matrix