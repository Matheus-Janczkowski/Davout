# Routine to store methods to construct ANN models

import tensorflow as tf

import numpy as np

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
    enforce_customLayers=False):
        
        # Retrieves the parameters

        self.input_dimension = input_dimension

        self.layers_info = layers_activationInfo

        # Initializes the dictionary of live-wired activation functions

        self.live_activations = dict()

        # Sets a flag to enforce the use of custom layers even though
        # the layers have each a single type of activation functions

        self.enforce_customLayers = enforce_customLayers

    # Defines a function to verify the list of dictionaries and, then,
    # it creates the model accordingly

    def __call__(self):

        # Initializes a flag to create a custom model or not

        flag_customLayers = False

        # Iterates through the layers to check the activation functions

        layer_counter = 1

        for layer_dictionary in self.layers_info:

            self.live_activations, flag_customLayers = verify_activationDict(
            layer_dictionary, layer_counter, self.live_activations, 
            flag_customLayers)

            layer_counter += 1

        # If the model has custom layers, uses the custom layer builder

        if flag_customLayers or self.enforce_customLayers:

            return self.multilayer_modelCustomLayers()
        
        # Otherwise, uses the Keras standard model

        else:

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
        live_activationsDict=self.live_activations, layer=0)(input_layer)

        # Iterates through the other layers

        for i in range(1,len(self.layers_info)):

            output_eachLayer = MixedActivationLayer(self.layers_info[i],
            live_activationsDict=self.live_activations, layer=i)(
            output_eachLayer)

        # Assembles the model

        model = tf.keras.Model(inputs=input_layer, outputs=
        output_eachLayer)

        return model
    
    # Defines a method to construct a standard Keras model using the 
    # activation functions dictionary

    def multilayer_modelKeras(self):

        # Initializes the Keras model. Constructs a list of layers

        model_parameters = []

        # Creates the first layer

        keys = list(self.layers_info[0].keys())

        model_parameters[0] = tf.keras.layers.Dense(
        self.layers_info[0][keys[0]], activation=keys[0], 
        input_dim=self.input_dimension)

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

        # Retuns the model
        
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

    def __init__(self, activation_functionDict, live_activationsDict=
    dict(), layer=0, **kwargs):

        # Initializes the parent class, i.e. Layer. The kwargs are opti-
        # onal arguments used during layer creation and deserialization, 
        # such as layer's name, trainable flag, and so forth

        super().__init__(**kwargs)

        # Adds the dictionary of live-wired activation functions. But 
        # checks if if is given as argument

        if (live_activationsDict is None) or live_activationsDict=={}: 

            self.live_activationFunctions, *_ = verify_activationDict(
            activation_functionDict, layer, {}, True)

        else:

            self.live_activationFunctions = live_activationsDict

        # Gets the dictionary of functions into the class
        
        self.functions_dict = activation_functionDict

        self.layer = layer

        # Counts the number of neurons in the layer

        total_neurons = sum(self.functions_dict.values())

        # Constructs a dense layer with identity activation functions

        self.dense = tf.keras.layers.Dense(total_neurons)

    # Defines a function to help Keras build the layer

    def build(self, input_shape):

        super().build(input_shape)

    # Defines a function to get the output of such a mixed layer

    def call(self, input):

        # Gets the input in the ANN format using the dense layer

        x = self.dense(input)

        # Initializes a list of outputs for each family of neurons (or-
        # ganized by their activation functions)

        output_activations = []

        # Sets a counter of neurons that have already been evaluated

        n_evaluated = 0

        # Iterates through the dictionary of activation functions

        for activation_name, neurons_number in self.functions_dict.items():

            # Appends the result of the neurons with this activation 

            output_activations.append(self.live_activationFunctions[
            activation_name](x[:,n_evaluated:(n_evaluated+neurons_number
            )]))

            # Updates the number of evaluated neurons

            n_evaluated += neurons_number

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

def verify_activationName(name):

    if name=="linear":

        return True
    
    else:

        return (hasattr(tf.nn, name) and callable(getattr(tf.nn, name, 
        None)))

# Defines a function to check if the dictionary of activation functions
# has real activation names

def verify_activationDict(activation_dict, layer, 
live_activationFunctions, flag_customLayers):

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

        if not verify_activationName(name):

            error_message += ("\n          "+str(name)+", in layer "+
            str(layer)+", is not a name of an actual activation functi"+
            "on in TensorFlow")

        # Verifies if this activation function has not already been map-
        # ped into the dictionary of live-wired activation functions

        elif not (name in live_activations):

            live_activationFunctions[name] = get_activationFunction(name)

    # If the error message is not empty, raises an exception

    if error_message!="":

        raise NameError(error_message)
    
    # Returns the updated dictionary of live-wired activation functions

    return live_activationFunctions, flag_customLayers
    
# Defines a function to get the activation function by its name

def get_activationFunction(name):

    if name=="linear":

        return tf.identity
    
    else:

        return getattr(tf.nn, name)