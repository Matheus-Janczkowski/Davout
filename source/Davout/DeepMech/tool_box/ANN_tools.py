# Routine to store methods to construct ANN models

import tensorflow as tf

import numpy as np

from ...DeepMech import custom_architectures

from ..tool_box import differentiation_tools as diff_tools

from ..tool_box import parameters_tools

from ..tool_box.custom_activation_functions import CustomActivationFunctions

from ..tool_box.activation_function_utilities import verify_activationDict, verify_activationName

from ...PythonicUtilities.package_tools import load_classes_from_package

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
    input_size_main_network=None, custom_architecture=None, 
    regularizing_function="smooth absolute value"):
        
        # Instantiates the class of custom activation functions

        self.custom_activations_class = CustomActivationFunctions(dtype=
        parameters_dtype)
        
        # Retrieves the parameters

        self.input_dimension = input_dimension

        self.input_size_main_network = input_size_main_network

        self.layers_info = layers_activationInfo

        self.verbose = verbose

        self.flat_trainable_parameters = flat_trainable_parameters

        # Gets the available classes to special architectures

        available_architectures = load_classes_from_package(
        custom_architectures, return_dictionary_of_classes=True)

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

        # Saves the flag for input-convex models

        self.custom_architecture = custom_architecture

        # If the custom architecture is None, make it the generic feed-
        # forward 

        if self.custom_architecture is None:

            self.custom_architecture = "GenericFeedForwardNNs"

        # Verifies if it is one of the available architectures

        if not (self.custom_architecture in available_architectures):

            available_names = ""

            for name in available_architectures.keys():

                available_names += "\n'"+str(name)+"'"
            
            raise NameError("'custom_architecture' is '"+str(
            self.custom_architecture)+"'. However, it must be one of t"+
            "he following names to select a special or custom architec"+
            "ture:"+available_names)
        
        # If no error has been given, set custom architecture as one
        # of the given by the available architectures

        self.custom_architecture = available_architectures[
        self.custom_architecture]
            
        # Saves the regularizing function variable to instruct how to 
        # train input convex or partially input convex models

        self.regularizing_function = regularizing_function

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

            # Keras models cannot be used yet for input convex networks

            if (self.custom_architecture.__name__!="GenericFeedForward"+
            "NNs"):

                raise NotImplementedError("'custom_architecture' is '"+
                str(self.custom_architecture.__name__)+"', but Keras "+
                "models does not feature the custom architecture opti"+
                "on yet. Enforce the use of custom model instead")

            return self.multilayer_modelKeras()

    # Defines a function to construct a multilayer model with custom 
    # layers. The dimension of the input must be provided, as well as a 
    # list of dictionaries. Each dictionary corresponds to a layer, and 
    # each dictionary has activation functions' names as keys and the 
    # corresponding number of neurons to each activation as values

    def multilayer_modelCustomLayers(self):

        # Verifies if an accessory network is required and if the input
        # size to it has been determmined

        if self.accessory_network and ((self.input_size_main_network
        ) is None):
            
            raise ValueError("An accessory network has been required, "+
            "but the numebr of input neurons to the main network, 'inp"+
            "ut_size_main_network', has not been provided")

        # Initializes the input layer

        input_layer = tf.keras.Input(shape=(self.input_dimension,))

        # TODO add the apply_parameters_to_layer method to input_layer

        # Gets the first layer. Here the class is used as a function di-
        # rectly due to the call function. It goes directly there. Takes
        # care with the case of an accessory network

        input_size_main_layer = None

        output_eachLayer = MixedActivationLayer(self.layers_info[0], 
        self.custom_activations_class, live_activationsDict=
        self.live_activations, activations_accessory_layer_dict=
        self.accessory_layers_info[0], layer=0, 
        input_size_main_network=self.input_size_main_network,
        input_size_main_layer=self.input_size_main_network, 
        custom_architecture=self.custom_architecture, float_dtype=
        self.parameters_dtype, regularization_function=
        self.regularizing_function)(input_layer)

        # Iterates through the other layers

        for i in range(1,len(self.layers_info)):

            # Evaluates the quantities for the case of accessory layers

            if self.input_size_main_network is not None:

                # Sums up the neurons of the previous main layer

                input_size_main_layer = 0

                for value in self.layers_info[i-1].values():

                    if isinstance(value, int):

                        input_size_main_layer += value 

                    elif "number of neurons" in value:

                        input_size_main_layer += value["number of neur"+
                        "ons"]

                    else:

                        raise KeyError("The bit '"+str(value)+"' of th"+
                        "e dictionary of layer info does not have the "+
                        "key 'number of neurons'")
                    
            # Gets the layer number. If it is the last layer, gives -1
    
            layer_number = i

            if layer_number==len(self.layers_info)-1:

                layer_number = -1

            # Gets the output of this layer

            output_eachLayer = MixedActivationLayer(self.layers_info[i],
            self.custom_activations_class, live_activationsDict=
            self.live_activations, activations_accessory_layer_dict=
            self.accessory_layers_info[i], input_size_main_network=
            self.input_size_main_network, input_size_main_layer=
            input_size_main_layer, layer=layer_number, 
            custom_architecture=self.custom_architecture, 
            regularization_function=self.regularizing_function, 
            float_dtype=self.parameters_dtype)(output_eachLayer)

        # Assembles the model

        model = tf.keras.Model(inputs=input_layer, outputs=
        output_eachLayer)

        # Adds the shapes of the trainables parameters

        for layer in model.layers:

            layer.trainable_variables_shapes = tuple([tuple(
            trainable_tensor.shape) for trainable_tensor in (
            layer.trainable_variables)])

        # Adds the input convex information and the dimension of the 
        # output layer

        model.custom_architecture = self.custom_architecture

        model.output_dimension = self.output_dimension

        # Adds the regularizing function for regularizing weight matrices
        # in case of input convex models or partially input convex models

        model.regularizing_function = self.regularizing_function

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

        #tf.keras.layers.Dense.call_with_parameters = (
        #parameters_tools.keras_dense_call_with_parameters)

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

        keras_model = tf.keras.Sequential(model_parameters)

        keras_model.output_dimension = self.output_dimension

        # Updates the model with call_with_parameters method for each
        # layer

        keras_model = insert_call_with_parameters_to_keras(
        keras_model)

        if self.evaluate_parameters_gradient:
        
            return keras_model, diff_tools.model_jacobian(keras_model, 
            self.output_dimension, self.evaluate_parameters_gradient)

        else:
            
            return keras_model
        
# Defines a function to add the call with parameters method to each Keras
# layer

def insert_call_with_parameters_to_keras(model):

    # Iterates over the layers of the model

    for layer in model.layers:

        # Gets the shapes of trhe trainable parameters
        
        trainable_variables_shapes = tuple([tuple(trainable_tensor.shape
        ) for trainable_tensor in layer.trainable_variables])

        # Instantiates the class that contains the method to evaluate 
        # the layer output given the parameters as a split flat tensor.
        # This class has a __call__ method, thus it can be called direc-
        # tly

        layer.call_with_parameters = parameters_tools.KerasCallWithParameters(
        trainable_variables_shapes, layer.activation)

    # Returns the model

    return model

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
    input_size_main_network=None, input_size_main_layer=None, layer=0, 
    custom_architecture=None, regularization_function=None, float_dtype=
    tf.float32, **kwargs):

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

        # Gets all the live activation functions into a tuple

        self.activation_list = tuple([self.live_activationFunctions[name
        ] for name in self.functions_dict.keys()])

        # Saves the flag to inform if the model is supposed to be input-
        # convex or not. Instantiates the class accordingly

        self.regularization_function = regularization_function

        self.float_dtype = float_dtype

        # Guarantees that the custom architecture is not None

        if custom_architecture is None:

            # Uses GenericFeedForwardNNs as default, which is a generic
            # deep feed-forward neural network architecture

            custom_architecture = load_classes_from_package(
            custom_architectures, return_dictionary_of_classes=True)[
            "GenericFeedForwardNNs"]

        self.custom_architecture_instance = custom_architecture(self,
        activation_functionDict, custom_activations_class,
        activations_accessory_layer_dict, input_size_main_network, layer,
        regularization_function, float_dtype)

        # Gets the dictionary of functions of the accessory network in 
        # the case of partially input-convex networks

        self.functions_dict_acessory_network = activations_accessory_layer_dict

        # Gets all the live activation functions into a tuple

        self.activation_list_acessory_network = tuple([
        self.live_activationFunctions[name] for name in (
        self.functions_dict_acessory_network.keys())])

        # Defines the method that will be used to call the layer's res-
        # ponse when the trainable parameters are fixed

        self.call_from_input_method = (
        self.custom_architecture_instance.call_from_input_method)

        # Defines the method that will be used to call the layer's res-
        # ponse when the trainable parameters are given

        self.call_given_parameters = (
        self.custom_architecture_instance.call_given_parameters)

        # Saves the parameters for the case of accessory networks, even
        # when they are not used. This saving is done so that this class 
        # can be reinstantiated later when loaded from a file

        self.input_size_main_network = input_size_main_network

        self.input_size_main_layer = input_size_main_layer

        # Initializes a variable with the shapes of the trainable para-
        # meters

        self.trainable_variables_shapes = None

    # Defines a function to help Keras build the layer

    def build(self, input_shape):

        # Gets a list with the numbers of neurons per activation function

        self.neurons_per_activation = [value["number of neurons"] if (
        isinstance(value, dict)) else value for value in (
        self.functions_dict.values())]

        # Counts the number of neurons in the layer. But takes cares if 
        # the value attached to each name of activation function is a 
        # dictionary

        self.total_neurons = sum(self.neurons_per_activation)

        # Calls the custom architecture object with the instance of the 
        # class that builds the layer

        if self.custom_architecture_instance:

            self.custom_architecture_instance.mixed_layer_builder(
            input_shape)

        # Otherwise, constructs it plainly

        else:

            # Constructs a dense layer with identity activation functions

            self.dense = tf.keras.layers.Dense(self.total_neurons)

            # Constructs the layer

            super().build(input_shape)

    # Defines a function to get the output of such a mixed layer

    #@tf.function
    def call(self, input):
        
        return self.call_from_input_method(input)

    # Defines a function to get the output of such a mixed layer when 
    # the trainable parameters are given

    #@tf.function
    def call_with_parameters(self, input, parameters):

        return self.call_given_parameters(input, parameters)
    
    # Defines a function to construct a dictionary of instructions to
    # save and load the model using TensorFlow methods

    def get_config(self):

        # Calls the method get config in Layer class

        config = super().get_config()

        # Updates the instructions dictionary

        config.update({"activation_functionDict": self.functions_dict,
        "layer": self.layer, "custom_activations_config": 
        self.custom_activations_class.get_config(), "custom_activation"+
        "s_class": None, "activations_accessory_layer_dict":
        self.functions_dict_acessory_network, "input_size_main_network":
        self.input_size_main_network, "input_size_main_layer":
        self.input_size_main_layer, "custom_architecture": 
        self.custom_architecture_instance.__class__.__name__, "regular"+
        "ization_function": self.regularization_function, "float_dtype":
        self.float_dtype})

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

        # Reallocates the dictionary of custom architectures to the con-
        # fig dictionary

        architecture_name = config.pop("custom_architecture")

        architecture_map = load_classes_from_package(
        custom_architectures, return_dictionary_of_classes=True)

        config["custom_architecture"] = architecture_map[
        architecture_name]

        return cls(**config)

########################################################################
#                               Utilities                              #
########################################################################

# Defines a function to generate random numbers between a range

def random_inRange(x_min, x_max):

    delta_x = x_max-x_min

    return (x_min+(np.random.rand()*delta_x))