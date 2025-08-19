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
    enforce_customLayers=False, evaluate_parameters_gradient=False):
        
        # Retrieves the parameters

        self.input_dimension = input_dimension

        self.layers_info = layers_activationInfo

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

        # If the gradient is to be evaluated too

        if self.evaluate_parameters_gradient:
        
            return model, self.model_gradient(model)
        
        # If not, returns the model only

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

        # Retuns the model and the gradient with respect to the parame-
        # ters if necessary

        if self.evaluate_parameters_gradient:

            keras_model = tf.keras.Sequential(model_parameters)
        
            return keras_model, self.model_gradient(keras_model)

        else:
        
            return tf.keras.Sequential(model_parameters)
    
    # Defines a method to evaluate the derivative of the model with res-
    # pect to the parameters (weights and biases)

    def model_gradient(self, model, evaluate_parameters_gradient=None):

        if evaluate_parameters_gradient is None:

            evaluate_parameters_gradient = self.evaluate_parameters_gradient

        # Defines a function to compute the sum of the gradient (jacobi-
        # an of the model) with respect to the model parameters

        if evaluate_parameters_gradient=="individual samples":

            @tf.function
            def gradient_evaluator(x):

                with tf.GradientTape() as tape:

                    model_response = model(x)

                return tape.jacobian(model_response, 
                model.trainable_variables)
            
            return gradient_evaluator
        
        elif evaluate_parameters_gradient in ["derivative matrix",
        True]:
            
            # Defines a function to compute a matrix of derivatives of
            # the model w.r.t. to the vector of parameters. Each column
            # corresponds to the gradient evaluated at a sample of in-
            # puts. The structure of the matrix is:
            #
            # dy1(x1)/dp1 dy1(x2)/dp1 … dy1(xn)/dp1 
            # dy1(x1)/dp2 dy1(x2)/dp2 … dy1(xn)/dp2 
            # … 
            # dy1(x1)/dpm dy1(x2)/dpm … dy1(xn)/dpm 
            # dy2(x1)/dp1 dy2(x2)/dp1 … dy2(xn)/dp1 
            # dy2(x1)dp2 dy2(x2)/dp2 … dy2(xn)/dp2 
            # … 
            # dyl(x1)dpm dyl(x2)/dpm … dyl(xn)/dpm 
            # 
            # n: number of input samples; m: number of model trainable 
            # variables; l: number of output neurons; yi is the i-th 
            # output neuron of the model. Hence, the matrix would be
            # (l*m)xn

            @tf.function
            def gradient_evaluator(x):

                # Flattens the parameters (weights and biases) list

                params = tf.concat([tf.reshape(p, [-1]) for p in (
                model.trainable_variables)], axis=0)

                # Counts the number of parameters and the number of sam-
                # ples

                n_parameters = tf.shape(params)[0]

                n_samples = tf.shape(x)[0]

                # Initializes the sample and the output indices

                sample_index = tf.Variable(0, dtype=tf.int32)

                output_index = tf.Variable(0, dtype=tf.int32)

                # Gets the derivative tape

                with tf.GradientTape(persistent=True) as tape:

                    model_response = model(x)[sample_index, output_index]

                # Initializes the matrix of gradients as a list

                gradient_matrix = []

                # Iterates through the number of samples

                for i in range(n_samples):

                    # Iterates through the number of outputs

                    for j in range(self.output_dimension):

                        # Updates the indices

                        sample_index.assign(i)

                        output_index.assign(j)

                        # Gets the gradient of this output evaluated at
                        # this sample

                        gradient_on_sample = tape.gradient(
                        model_response, model.trainable_variables)

                        print("sample "+str(i)+", output "+str(j)+": "+
                        str(gradient_on_sample)+"\n")

                        # Flattens this gradient and appends it to the 
                        # gradients matrix

                        gradient_matrix.append(tf.concat([tf.reshape(
                        layer, [-1]) for layer in gradient_on_sample], 
                        axis=0))

                # Deletes the tape to spare memory

                del tape

                # Stacks the gradient matrix to the format of number of
                # samples times the output dimension rows

                gradient_matrix = tf.stack(gradient_matrix, axis=0)

                # Reshapes the matrix and tranposes it

                gradient_matrix = tf.reshape(gradient_matrix, (n_samples,
                self.output_dimension, n_parameters))

                gradient_matrix = tf.transpose(gradient_matrix, (1,2,0))

                gradient_matrix = tf.reshape(gradient_matrix, (
                self.output_dimension*n_parameters, n_samples))

                return gradient_matrix
            
            return gradient_evaluator
        
        elif evaluate_parameters_gradient=="teste":

            def gradient_evaluator(x):

                gradient_samples = []

                for sample_counter in range(x.shape[0]):

                    x_sample = tf.expand_dims(x[sample_counter], 0)

                    with tf.GradientTape() as tape:

                        model_response = model(x_sample)[0,0]

                    safe_grads = tape.gradient(model_response,
                    model.trainable_variables)

                    flat_grad = tf.concat([tf.reshape(g, [-1]) for g in safe_grads], axis=0)

                    gradient_samples.append(flat_grad.numpy())

                return gradient_samples

            return gradient_evaluator
        
        elif evaluate_parameters_gradient==("vectorized derivativ"+
        "e matrix"):

            @tf.function
            def gradient_evaluator(x):
                # Forward pass
                with tf.GradientTape() as tape:
                    y = model(x)                   # shape: (n, l)

                n = tf.shape(y)[0]                 # number of samples
                l = tf.shape(y)[1]                 # number of outputs

                # For each trainable variable v:
                #   J_v = dy/dv has shape (n, l, *v.shape)
                per_var_jac = [tape.jacobian(y, v) for v in model.trainable_variables]

                # Flatten parameter axes of each J_v to a single axis k_v
                # => shapes become (n, l, k_v)
                per_var_flat = [tf.reshape(Jv, (n, l, -1)) for Jv in per_var_jac]

                # Concatenate along parameters to get (n, l, m)
                J_nlm = tf.concat(per_var_flat, axis=2)

                # Reorder to (l, m, n), then flatten to (l*m, n)
                J_lmn = tf.transpose(J_nlm, (1, 2, 0))
                J_lm_n = tf.reshape(J_lmn, (-1, n))   # (l*m, n)

                return J_lm_n

            return gradient_evaluator
        
        else:

            raise NameError("The flag 'evaluate_parameters_gradient' i"+
            "n 'MultiLayerModel' can be either 'individual samples', o"+
            "r 'derivative matrix', True (defaults to 'derivative matr"+
            "ix'), or 'vectorized derivative matrix'")

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