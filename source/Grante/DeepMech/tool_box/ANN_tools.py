# Routine to store methods to construct ANN models

import tensorflow as tf

import numpy as np

from copy import deepcopy

from ..tool_box import differentiation_tools as diff_tools

from ..tool_box import parameters_tools

from ..tool_box.custom_activation_functions import CustomActivationFunctions

from ...PythonicUtilities import dictionary_tools, function_tools

########################################################################
#                       ANN construction classes                       #
########################################################################

# Defines a class to construct a multilayer perceptron network. Receives
# the number of neurons in the input layer and a list of dictionaries of
# names of activation functions as keys with their corresponding number
# of neurons as values. If there is one type of activation function only
# per layer, i.e. each dictionary in the list has only one key, a Dense
# Keras layer is created per default. Otherwise, a custom layer is crea-
# ted

class MultiLayerModel:

    def __init__(self, input_dimension, layers_activationInfo, 
    enforce_customLayers=False, evaluate_parameters_gradient=False,
    flat_trainable_parameters=False, verbose=False, parameters_dtype=
    "float32", accessory_layers_activationInfo=[], 
    input_size_accessory_network=None):
        
        # Instantiates the class of custom activation functions

        self.custom_activations_class = CustomActivationFunctions(dtype=
        parameters_dtype)
        
        # Retrieves the parameters

        self.input_dimension = input_dimension

        self.input_size_accessory_network = input_size_accessory_network

        self.layers_info = layers_activationInfo

        self.verbose = verbose

        self.flat_trainable_parameters = flat_trainable_parameters

        # Verifies the need for an accessory network

        if len(accessory_layers_activationInfo)==0:

            self.accessory_network = False

            self.accessory_layers_info = [{} for i in range(len(
            self.layers_info))]

        else:

            self.accessory_network = True

            self.accessory_layers_info = accessory_layers_activationInfo

        # Initializes the dictionary of live-wired activation functions

        self.live_activations = dict()

        # Sets a flag to enforce the use of custom layers even though
        # the layers have each a single type of activation functions

        self.enforce_customLayers = enforce_customLayers

        # Sets a flag to tell if the gradient of the model with respect
        # to its parameters is to be given as a function

        self.evaluate_parameters_gradient = evaluate_parameters_gradient

        # Gets the number of neurons in the output layer

        self.output_dimension = 0

        for n_neurons in self.layers_info[-1].values():

            # Verifies if the value attached to this activation function
            # is a dictionary, i.e. it has further information for key-
            # word arguments

            if isinstance(n_neurons, dict):

                # Verifies if it has the key number of neurons

                if "number of neurons" in n_neurons:

                    self.output_dimension += n_neurons["number of neur"+
                    "ons"]

                else:

                    raise KeyError("The last layer of the model, "+str(
                    self.layers_info)+", has a value for one activatio"+
                    "n function which is "+str(n_neurons)+". This is a"+
                    " dictionary, but no key 'number of neurons' was p"+
                    "rovided")

            else:

                self.output_dimension += n_neurons

        # Sets the type of the parameters

        self.parameters_dtype = parameters_dtype

    # Defines a function to verify the list of dictionaries and, then,
    # it creates the model accordingly

    def __call__(self):

        # Sets global precision for all layers' parameters

        tf.keras.mixed_precision.set_global_policy(self.parameters_dtype)

        # Initializes a flag to create a custom model or not

        flag_customLayers = False

        # Iterates through the layers to check the activation functions

        layer_counter = 1

        for i in range(len(self.layers_info)):

            self.live_activations, flag_customLayers = verify_activationDict(
            self.layers_info[i], layer_counter, self.live_activations, 
            flag_customLayers, self.custom_activations_class)

            # Verifies if the accessory layer in case of partially input-
            # convex neural networks is used

            if self.accessory_layers_info[i]:

                self.live_activations, flag_customLayers = verify_activationDict(
                self.accessory_layers_info[i], layer_counter, 
                self.live_activations, flag_customLayers, 
                self.custom_activations_class)

            layer_counter += 1

        # If the model has custom layers, uses the custom layer builder

        if flag_customLayers or self.enforce_customLayers:

            if self.verbose:

                print("Uses custom layers to build the model\n")

            return self.multilayer_modelCustomLayers()
        
        # Otherwise, uses the Keras standard model

        else:

            if self.verbose:

                print("Uses Keras layers to build the model\n")

            return self.multilayer_modelKeras()

    # Defines a function to construct a multilayer model with custom 
    # layers. The dimension of the input must be provided, as well as a 
    # list of dictionaries. Each dictionary corresponds to a layer, and 
    # each dictionary has activation functions' names as keys and the 
    # corresponding number of neurons to each activation as values

    def multilayer_modelCustomLayers(self):

        # Verifies if an accessory network is required and if the input
        # size to it has been determmined

        if self.accessory_network and ((self.input_size_accessory_network
        ) is None):
            
            raise ValueError("An accessory network has been required, "+
            "but the numebr of input neurons to it has not been provid"+
            "ed")

        # Initializes the input layer

        input_layer = tf.keras.Input(shape=(self.input_dimension,))

        # Gets the first layer. Here the class is used as a function di-
        # rectly due to the call function. It goes directly there

        output_eachLayer = MixedActivationLayer(self.layers_info[0], 
        self.custom_activations_class, live_activationsDict=
        self.live_activations, activations_accessory_layer_dict=
        self.accessory_layers_info[0], layer=0, 
        input_size_accessory_network=self.input_size_accessory_network)(
        input_layer)

        # Iterates through the other layers

        for i in range(1,len(self.layers_info)):

            output_eachLayer = MixedActivationLayer(self.layers_info[i],
            self.custom_activations_class, live_activationsDict=
            self.live_activations, activations_accessory_layer_dict=
            self.accessory_layers_info[i], layer=i)(output_eachLayer)

        # Assembles the model

        model = tf.keras.Model(inputs=input_layer, outputs=
        output_eachLayer)

        # If the gradient is to be evaluated too

        if self.evaluate_parameters_gradient:
        
            return model, diff_tools.model_jacobian(model, 
            self.output_dimension, self.evaluate_parameters_gradient)
        
        # If the flat parameters tensor vector is to be given

        if self.flat_trainable_parameters:

            return (model, 
            parameters_tools.model_parameters_to_flat_tensor_and_shapes(
            model))
        
        # If not, returns the model only

        return model
    
    # Defines a method to construct a standard Keras model using the 
    # activation functions dictionary

    def multilayer_modelKeras(self):

        # Adds the method to call the Dense keras layer giving the para-
        # meters as a 1D tensor

        tf.keras.layers.Dense.call_with_parameters = (
        parameters_tools.keras_dense_call_with_parameters)

        # Initializes the Keras model. Constructs a list of layers

        model_parameters = []

        # Creates the first layer

        keys = list(self.layers_info[0].keys())

        model_parameters.append(tf.keras.Input(shape=(
        self.input_dimension,)))

        model_parameters.append(tf.keras.layers.Dense(self.layers_info[0
        ][keys[0]], activation=keys[0]))

        # Iterates through the layers, but skips the first layer, as it
        # has already been populated

        for i in range(1,len(self.layers_info)):

            # Gets the dictionary of this layer and its keys, i.e. the
            # activation function of this layer

            layer = self.layers_info[i]

            keys = list(layer.keys())

            # Appends this information using Keras convention to the pa-
            # rameters list

            model_parameters.append(tf.keras.layers.Dense(layer[keys[0]
            ], activation=keys[0]))

        # Retuns the model and the gradient with respect to the parame-
        # ters if necessary

        if self.evaluate_parameters_gradient:

            keras_model = tf.keras.Sequential(model_parameters)
        
            return keras_model, diff_tools.model_jacobian(keras_model, 
            self.output_dimension, self.evaluate_parameters_gradient)

        else:
            
            return tf.keras.Sequential(model_parameters)

# Defines a class to construct a layer with different activation 
# functions. Receives a dictionary of activation functions, the activa-
# tion functions' names are the keys while the values are the numbers of 
# neurons with this function. The line below, before the class defini-
# tion is to ensure that TensorFlow finds the class during deserializa-
# tion. Uses the tf.keras.layers.Layer as parent class to inherit its
# methods

@tf.keras.utils.register_keras_serializable()

class MixedActivationLayer(tf.keras.layers.Layer):

    def __init__(self, activation_functionDict, custom_activations_class,
    live_activationsDict=dict(), activations_accessory_layer_dict=dict(), 
    input_size_accessory_network=None, layer=0, **kwargs):

        # Initializes the parent class, i.e. Layer. The kwargs are opti-
        # onal arguments used during layer creation and deserialization, 
        # such as layer's name, trainable flag, and so forth

        super().__init__(**kwargs)

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

            # Otherwise, gets only the activation functions from the con-
            # ventional dictionary

            else:

                self.live_activationFunctions, *_ = verify_activationDict(
                activation_functionDict, layer, {}, True,
                self.custom_activations_class)

        else:

            self.live_activationFunctions = live_activationsDict

        # Gets the dictionary of functions into the class
        
        self.functions_dict = activation_functionDict

        self.layer = layer

        # Gets the dictionary of functions of the accessory network in 
        # the case of partially input-convex networks

        self.functions_dict_acessory_network = activations_accessory_layer_dict

        # Selects the method that will give the ouput of the layer based
        # on the existence or not of an accessory layer

        if activations_accessory_layer_dict:

            self.call_from_input_method = self.call_from_input_with_accessory_layer

            # Saves the number of input neurons for the accessory network

            self.input_size_accessory_network = input_size_accessory_network

        else:

            self.call_from_input_method = self.call_from_input_no_accessory_layer

    # Defines a function to help Keras build the layer

    def build(self, input_shape):

        # Gets a list with the numbers of neurons per activation function

        self.neurons_per_activation = [value["number of neurons"] if (
        isinstance(value, dict)) else value for value in (
        self.functions_dict.values())]

        # Counts the number of neurons in the layer. But takes cares if 
        # the value attached to each name of activation function is a 
        # dictionary

        total_neurons = sum(self.neurons_per_activation)

        # Constructs a dense layer with identity activation functions

        self.dense = tf.keras.layers.Dense(total_neurons)

        # If the accessory layer is used, initializes the necessary in-
        # formation for it

        if self.functions_dict_acessory_network:

            # Gets a list with the numbers of neurons per activation 
            # function for the accessory network (u) in case of partial-
            # ly input-convex neural networks (AMOS ET AL, Input convex 
            # neural networks)

            self.neurons_per_activation_acessory_layer = [value["numbe"+
            "r of neurons"] if isinstance(value, dict) else value for (
            value) in (self.functions_dict_acessory_network.values())]

            # Creates a dense layer for the accessory network

            self.dense_accessory_layer = tf.keras.layers.Dense(sum(
            self.neurons_per_activation_acessory_layer))

            ############################################################
            #         Amos et at, Input convex neural networks         #
            ############################################################

            # Creates a dense layer for the bit of the accessory layer's
            # result that multiplies the main layer response using the 
            # Hadamard product

            self.dense_Wzu = tf.keras.layers.Dense()

        super().build(input_shape)

    # Defines a function to get the output of such a mixed layer

    def call(self, input):

        return self.call_from_input_method(input)
    
    # Defines a method for getting the layer value given the input when
    # no accessory layer is necessary

    def call_from_input_no_accessory_layer(self, input):

        # Initializes the input as dense layer and split it into the 
        # different families of activation functions. This keeps the in-
        # put as a tensor

        x_splits = tf.split(self.dense(input), 
        self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)
        
        output_activations = [self.live_activationFunctions[name](split
        ) for name, split in zip(self.functions_dict.keys(), x_splits)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row

        return tf.concat(output_activations, axis=-1)
    
    # Defines a method for getting the layer value given the input when
    # an accessory layer is necessary. In this case, the input must be a
    # tuple

    def call_from_input_with_accessory_layer(self, input):

        # If it's the first layer, the input tensor must be sliced: one
        # bit for the main network and the rest for the accessory network

        if self.layer==0:

            input = (input[..., :self.input_size_accessory_network], input[
            ..., self.input_size_accessory_network:])

        # The first element in the input tuple is the main layer. The 
        # second element is due to the accessory layer

        # Initializes the input as dense layer and split it into the 
        # different families of activation functions for the main layer. 
        # This keeps the input as a tensor

        x_splits_main_layer = tf.split(self.dense(input[0]), 
        self.neurons_per_activation,  axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)
        
        output_activations_main_layer = [self.live_activationFunctions[
        name](split) for name, split in zip(self.functions_dict.keys(), 
        x_splits_main_layer)]

        # Does the same for the accessory layer

        x_splits_accessory_layer = tf.split(self.dense_accessory_layer(
        input[1]), self.neurons_per_activation_acessory_layer,  axis=-1)
        
        output_activations_accessory_layer = [
        self.live_activationFunctions[name](split) for name, split in (
        zip(self.functions_dict_acessory_network.keys(), 
        x_splits_accessory_layer))]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row. Returns always the main layer
        # first, then the accessory layer

        return (tf.concat(output_activations_main_layer, axis=-1), 
        tf.concat(output_activations_accessory_layer, axis=-1))
    
    # Defines a function to get the output of such a mixed layer given 
    # the parameters (weights and biases) as a flat list (still a tensor)

    def call_with_parameters(self, input, parameters):

        # Gets the weights and biases

        weights, biases = parameters

        # Multiplies the weights by the inputs and adds the biases, then
        # splits by activation function family

        x_splits = tf.split(tf.matmul(input, weights)+biases, 
        self.neurons_per_activation, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)
        
        output_activations = [self.live_activationFunctions[name](split
        ) for name, split in zip(self.functions_dict.keys(), x_splits)]

        # Concatenates the response and returns it. Uses flag axis=-1 to
        # concatenate next to the last row

        return tf.concat(output_activations, axis=-1)
    
    # Defines a function to get the output of a layer that constitutes a
    # partially input-convex neural network (AMOS ET AL, Input Convex 
    # Neural Networks)

    def call_partially_convex_layer_with_parameters(self, layer_input, 
    acessory_layer_input, network_convex_input, parameters):
        
        # Gets the weights and biases

        W_tilde, b_tilde, W_z, W_zu, W_y, W_yu, W_u, b_z, b_y, b_layer = (
        parameters)

        ################################################################
        #                    Accessory layer update                    #
        ################################################################

        # Multiplies the weights of the acessory network (u) by the in-
        # puts of the acessory layer and, then, adds the biases. Finally,
        # splits by activation function family

        x_splits_u = tf.split(tf.matmul(acessory_layer_input, W_tilde)+
        b_tilde, self.neurons_per_activation_acessory_layer, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions) for the accessory layer
        
        output_activations_u = [self.live_activationFunctions[name](split
        ) for name, split in zip(self.functions_dict_acessory_network.keys(
        ), x_splits_u)]

        # Concatenates the response and saves it into u_(i+1). Uses flag 
        # axis=-1 to concatenate next to the last row

        output_u = tf.concat(output_activations_u, axis=-1)

        ################################################################
        #                      Main layer update                       #
        ################################################################

        # Gets the multiplication of the matrix W_u by the output of the
        # accessory previous layer, and adds this layer bias

        parcel_1 = tf.matmul(acessory_layer_input, W_u)+b_layer

        # Gets the parcel of the convex input multiplied by the bit made
        # of the accessory layer using the Hadamard product. Then, mul-
        # tiplies the whole by the corresponding weight matrix

        parcel_2 = tf.matmul(tf.multiply(network_convex_input, tf.matmul(
        acessory_layer_input, W_yu)+b_y), W_y)

        # Gets the parcel of the input of the main network multiplied by
        # the absolute value of the bit given by accessory previous 
        # layer, using the Hadamard product. Then, multiplies by the cor-
        # responding weight matrix

        parcel_3 = tf.matmul(tf.multiply(layer_input, tf.abs(tf.matmul(
        acessory_layer_input, W_zu)+b_z)), W_z)

        # Sums the parcels and splits it according to the families of 
        # activation functions

        x_splits_z = tf.split(parcel_1+parcel_2+parcel_3,
        self.neurons_per_activation, axis=-1)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions) for the main layer
        
        output_activations_z = [self.live_activationFunctions[name](split
        ) for name, split in zip(self.functions_dict.keys(), x_splits_z)]

        # Concatenates the response and saves it into z_(i+1). Uses flag 
        # axis=-1 to concatenate next to the last row

        output_z = tf.concat(output_activations_z, axis=-1)

        # Returns both outputs

        return output_z, output_u
    
    # Defines a function to construct a dictionary of instructions to
    # save and load the model using TensorFlow methods

    def get_config(self):

        # Calls the method get config in Layer class

        config = super().get_config()

        # Updates the instructions dictionary

        config.update({"activation_functionDict": self.functions_dict,
        "layer": self.layer, "custom_activations_config": 
        self.custom_activations_class.get_config(), "custom_activation"+
        "s_class": None})

        return config
    
    # Defines a function as a class method to reconstruct the layer from
    # a file, which contains the instructions dictionary. Defines it as
    # a class method because TensorFlow calls it as such

    @classmethod

    def from_config(cls, config):

        # Rebuilds the class of CustomActivationFunctions from its own
        # config

        custom_activations = CustomActivationFunctions.from_config(
        config.pop("custom_activations_config"))

        # Allocates it into the config dictionary

        config["custom_activations_class"] = custom_activations

        return cls(**config)

########################################################################
#                               Utilities                              #
########################################################################

# Defines a function to generate random numbers between a range

def random_inRange(x_min, x_max):

    delta_x = x_max-x_min

    return (x_min+(np.random.rand()*delta_x))

# Defines a function to test whether an activation function's name cor-
# responds to an actual activation function in TensorFlow

def verify_activationName(function_name, custom_activations_class, 
arguments):

    if function_name=="linear":

        # Verifies if arguments have been prescribed, which are not al-
        # lowed for this activation function

        if not (arguments is None):

            raise KeyError("The activation function 'linear' cannot ha"+
            "ve addtional arguments, thus, its corresponding value in "+
            "the dictionary of activation functions musn't be a dictio"+
            "nary with any other key beside 'number of neurons'")

        return True
    
    elif function_name in (
    custom_activations_class.custom_activation_functions_dict):

        return True
    
    else:

        # Verifies if arguments have been prescribed, which are not al-
        # lowed for native tensorflow activation functions

        if not (arguments is None):

            raise KeyError("The activation function '"+str(function_name
            )+"', native to tensorflow, cannot have addtional argument"+
            "s, thus, its corresponding value in the dictionary of act"+
            "ivation functions musn't be a dictionary with any other k"+
            "ey beside 'number of neurons'")

        return (hasattr(tf.nn, function_name) and callable(getattr(tf.nn, 
        function_name, None)))

# Defines a function to check if the dictionary of activation functions
# has real activation names

def verify_activationDict(activation_dict, layer, 
live_activationFunctions, flag_customLayers, custom_activations_class):

    # Verifies if the dictionary is empty

    if not activation_dict:

        raise KeyError("The layer "+str(layer)+" has no activation fun"+
        "ction information. You must provide at least one activation f"+
        "unction and the number of neurons to it.")
    
    # Checks if there is more than one key in the dictionary, i.e. if 
    # there is more than one activation function type in this layer

    if not flag_customLayers:

        if len(activation_dict.keys())>1:

            flag_customLayers = True

    # Gets the already retrieved activation functions

    live_activations = set(live_activationFunctions.keys())

    # Initializes a message

    error_message = ""

    # Iterates over the activation functions' names

    for name, activation_info in activation_dict.items():

        # Verifies if the activation_info is a dictionary, i.e. keyword 
        # arguments have been passed as well

        arguments = None

        if isinstance(activation_info, dict):

            # Verifies if the number of neurons that use this activation 
            # function has been passed

            if not ("number of neurons" in activation_info):

                raise KeyError("A dictionary has been used to set an a"+
                "ctivation function information, "+str(activation_info)+
                ", but no 'number of neurons' key has been included")
            
            # Gets the arguments and deletes the key for the number of 
            # neurons

            arguments = deepcopy(activation_info)

            del arguments["number of neurons"]

            # If this dictionary is empty, turns this variable into None
            # again

            if not arguments:

                arguments = None

        # Verifies if the name of this activation function exists

        if not verify_activationName(name, 
        custom_activations_class, arguments):

            error_message += ("\n          "+str(name)+", in layer "+
            str(layer)+", is not a name of an actual activation functi"+
            "on in TensorFlow nor in the list\n          of custom act"+
            "ivation functions of DeepMech (see the DeepMech's list:\n"+
            "          "+str(
            custom_activations_class.custom_activation_functions_dict.keys(
            ))[11:-2]+")")

        # Verifies if this activation function has not already been map-
        # ped into the dictionary of live-wired activation functions

        elif not (name in live_activations):

            live_activationFunctions[name] = get_activationFunction(
            name, custom_activations_class, arguments)

    # If the error message is not empty, raises an exception

    if error_message!="":

        raise NameError(error_message)
    
    # Returns the updated dictionary of live-wired activation functions

    return live_activationFunctions, flag_customLayers
    
# Defines a function to get the activation function by its name

def get_activationFunction(function_name, custom_activations_class, 
arguments):

    if function_name=="linear":

        return tf.identity
    
    elif function_name in (
    custom_activations_class.custom_activation_functions_dict):
        
        # Gets the pair of function and keyword arguments

        function_info = (
        custom_activations_class.custom_activation_functions_dict[
        function_name])

        # If arguments have been prescribed

        if not (arguments is None):

            # Verifies if the dictionary of arguments has arguments that
            # are allowed and adds the default values which were not 
            # prescribed

            arguments = dictionary_tools.verify_dictionary_keys(
            arguments, function_info[1], dictionary_location="at defin"+
            "ition of custom activation function '"+str(function_name)+
            "'", fill_in_keys=True)

            # Uses a wrapper to wrap the function to set the new values 
            # for the keyword arguments
        
            return function_tools.construct_lambda_function(
            function_info[0], arguments)

        # If no arguments have been prescribed, returns the function 
        # simply

        return function_info[0]
    
    else:

        return getattr(tf.nn, function_name)