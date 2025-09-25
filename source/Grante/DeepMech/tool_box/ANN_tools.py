# Routine to store methods to construct ANN models

import tensorflow as tf

import numpy as np

from copy import deepcopy

from ..tool_box import differentiation_tools as diff_tools

from ..tool_box import parameters_tools

from ..tool_box.custom_activation_functions import CustomActivationFunctions

from ...PythonicUtilities import dictionary_tools

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
    "float32"):
        
        # Instantiates the class of custom activation functions

        self.custom_activations_class = CustomActivationFunctions(dtype=
        parameters_dtype)
        
        # Retrieves the parameters

        self.input_dimension = input_dimension

        self.layers_info = layers_activationInfo

        self.verbose = verbose

        self.flat_trainable_parameters = flat_trainable_parameters

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

        for layer_dictionary in self.layers_info:

            self.live_activations, flag_customLayers = verify_activationDict(
            layer_dictionary, layer_counter, self.live_activations, 
            flag_customLayers, self.custom_activations_class)

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

        # Initializes the input layer

        input_layer = tf.keras.Input(shape=(self.input_dimension,))

        # Gets the first layer. Here the class is used as a function di-
        # rectly due to the call function. It goes directly there

        output_eachLayer = MixedActivationLayer(self.layers_info[0], 
        self.custom_activations_class, live_activationsDict=
        self.live_activations, layer=0)(input_layer)

        # Iterates through the other layers

        for i in range(1,len(self.layers_info)):

            output_eachLayer = MixedActivationLayer(self.layers_info[i],
            self.custom_activations_class, live_activationsDict=
            self.live_activations, layer=i)(output_eachLayer)

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
    live_activationsDict=dict(), layer=0, **kwargs):

        # Initializes the parent class, i.e. Layer. The kwargs are opti-
        # onal arguments used during layer creation and deserialization, 
        # such as layer's name, trainable flag, and so forth

        super().__init__(**kwargs)

        # Adds the dictionary of live-wired activation functions. But 
        # checks if if is given as argument

        if (live_activationsDict is None) or live_activationsDict=={}: 

            self.live_activationFunctions, *_ = verify_activationDict(
            activation_functionDict, layer, {}, True,
            custom_activations_class)

        else:

            self.live_activationFunctions = live_activationsDict

        # Gets the dictionary of functions into the class
        
        self.functions_dict = activation_functionDict

        self.layer = layer

    # Defines a function to help Keras build the layer

    def build(self, input_shape):

        # Counts the number of neurons in the layer

        total_neurons = sum(self.functions_dict.values())

        # Gets a list with the numbers of neurons per activation function

        self.neurons_per_activation = list(self.functions_dict.values())

        # Constructs a dense layer with identity activation functions

        self.dense = tf.keras.layers.Dense(total_neurons)

        super().build(input_shape)

    # Defines a function to get the output of such a mixed layer

    def call(self, input):

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
    
    # Defines a function to construct a dictionary of instructions to
    # save and load the model using TensorFlow methods

    def get_config(self):

        # Calls the method get config in Layer class

        config = super().get_config()

        # Updates the instructions dictionary

        config.update({"activation_functionDict": self.functions_dict,
        "layer": self.layer})

        return config
    
    # Defines a function as a class method to reconstruct the layer from
    # a file, which contains the instructions dictionary. Defines it as
    # a class method because TensorFlow calls it as such

    @classmethod

    def from_config(cls, config):

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

def verify_activationName(name, custom_activations_class):

    # Verifies if the name is a dictionary, i.e. keyword arguments have
    # been passed as well

    function_name = ""

    if isinstance(name, dict):

        # Verifies if the name of the activation function have been 
        # passed

        if not ("name" in name):

            raise KeyError("A dictionary has been used to set an activ"+
            "ation function, "+str(name)+", but no 'name' key has been"+
            " included")

        # Gets the name of the function

        function_name = name["name"]

    else:

        # Simply gets the name of the activation function

        function_name = name

    if function_name=="linear":

        return True
    
    elif function_name in (
    custom_activations_class.custom_activation_functions_dict):

        return True
    
    else:

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

    # Gets the names of the activation functions

    activation_names = list(activation_dict.keys())

    # Checks if there is more than one key in the dictionary, i.e. if 
    # there is more than one activation function type in this layer

    if not flag_customLayers:

        if len(activation_names)>1:

            flag_customLayers = True

    # Gets the already retrieved activation functions

    live_activations = set(live_activationFunctions.keys())

    # Initializes a message

    error_message = ""

    # Iterates over the activation functions' names

    for name in activation_names:

        if not verify_activationName(name, custom_activations_class):

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

            live_activationFunctions[name] = get_activationFunction(name,
            custom_activations_class)

    # If the error message is not empty, raises an exception

    if error_message!="":

        raise NameError(error_message)
    
    # Returns the updated dictionary of live-wired activation functions

    return live_activationFunctions, flag_customLayers
    
# Defines a function to get the activation function by its name

def get_activationFunction(name, custom_activations_class):

    # Verifies if the name is a dictionary, i.e. keyword arguments have
    # been passed as well

    arguments = None

    function_name = ""

    if isinstance(name, dict):

        # Verifies if the name of the activation function have been 
        # passed

        if not ("name" in name):

            raise KeyError("A dictionary has been used to set an activ"+
            "ation function, "+str(name)+", but no 'name' key has been"+
            " included")

        # Gets the name of the function

        function_name = name["name"]
        
        # Gets the arguments and deletes the key for the name

        arguments = deepcopy(name)

        del arguments["name"]

    else:

        # Simply gets the name of the activation function

        function_name = name

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

            # Uses a lambda function to set the new values for the key-
            # word arguments

            return lambda x: function_info[0](x, arguments)

        # If no arguments have been prescribed, returns the function 
        # simply

        return function_info[0]
    
    else:

        return getattr(tf.nn, function_name)