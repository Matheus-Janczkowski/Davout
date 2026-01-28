# Routine to store methods for automatic differentiation

import tensorflow as tf

import numpy as np

from ..tool_box import parameters_tools

from ..tool_box import numerical_tools

########################################################################
#                          NN model gradient                           #
########################################################################

# Defines a class to evaluate the gradient of a scalar function with 
# respect to the model trainable parameters

class ScalarGradientWrtTrainableParams:
    
    def __init__(self, scalar_function, input_tensor, model_true_values=
        None):

        self.scalar_function = scalar_function

        self.input_tensor = input_tensor

        # Creates a dummy true value of output, because the Keras' loss 
        # functions requires y_true and y_pred as arguments

        if model_true_values is None:

            self.model_true_values = tf.constant([0.0], dtype=
            input_tensor.dtype)

        else:

            self.model_true_values = model_true_values

    # Defines a function to actually evaluate the derivative

    @tf.function
    def __call__(self, model):

        # Creates the tape

        with tf.GradientTape() as tape:

            # Evaluates the model output

            y = model(self.input_tensor)

            # Evaluates the scalar function

            phi = self.scalar_function(self.model_true_values, y)

        # Gets the gradient

        return tape.gradient(phi, model.trainable_variables)

# Defines a class to evaluate the gradient of a scalar function with 
# respect to the model trainable parameters, when the parameters are gi-
# ven as a 1D tensor

class ScalarGradientWrtTrainableParamsGivenParameters:
    
    def __init__(self, scalar_function, model, input_tensor, 
    shapes_trainable_parameters, model_true_values=None):
        
        self.scalar_function = scalar_function

        self.model = model 

        self.input_tensor = input_tensor

        self.shapes_trainable_parameters = shapes_trainable_parameters

        # Creates a dummy true value of output, because the Keras' loss 
        # functions requires y_true and y_pred as arguments

        if model_true_values is None:

            self.model_true_values = tf.constant([0.0], dtype=
            input_tensor.dtype)

        else:

            self.model_true_values = model_true_values

        # Saves a flag to tell to use the custom gradient defined within
        # the loss function class or not

        if hasattr(self.scalar_function, "custom_gradient_usage"):

            self.custom_gradient = self.scalar_function.custom_gradient_usage

        else:

            self.custom_gradient = False

        # If the custom gradient is to be used, prepare it sharing the 
        # same model and input tensor

        if self.custom_gradient:

            self.scalar_function.prepare_custom_gradient(self.model,
            self.input_tensor)

    # Defines a function to actually evaluate the derivative

    @tf.function
    def __call__(self, trainable_parameters):

        # Uses the custom implementation of the gradient defined inside
        # the class of the loss function

        if self.custom_gradient:

            # Gets the response of the model

            y = parameters_tools.model_output_given_trainable_parameters(
            self.input_tensor, self.model, trainable_parameters, 
            self.shapes_trainable_parameters)

            return self.scalar_function.custom_gradient(
            self.model_true_values, y, trainable_parameters)

        # Otherwise, uses automatic differentiation

        else:

            # Creates the tape

            with tf.GradientTape() as tape:

                tape.watch(trainable_parameters)

                # Gets the response of the model

                y = parameters_tools.model_output_given_trainable_parameters(
                self.input_tensor, self.model, trainable_parameters, 
                self.shapes_trainable_parameters)

                phi = self.scalar_function(self.model_true_values, y)

            # Gets the gradient

            return tape.gradient(phi, trainable_parameters)
        
    # Defines a function to evaluate the loss function

    def evaluate_scalar_function(self, trainable_parameters):

        # Gets the response of the model

        y = parameters_tools.model_output_given_trainable_parameters(
        self.input_tensor, self.model, trainable_parameters, 
        self.shapes_trainable_parameters)

        # Gets the scalar function value and returns it

        return self.scalar_function(self.model_true_values, y)
    
    # Defines a function to update the scalar function if it is parame-
    # terizable by externally-given quantities

    def update_function(self, *external_arguments):

        # Calls the method to update the scalar function

        self.scalar_function.update_arguments(*external_arguments)

# Defines a class to evaluate the gradient of a scalar function with 
# respect to the model trainable parameters, when the parameters are gi-
# ven as a 1D tensor and the model is input-convex

class ScalarGradientWrtTrainableParamsGivenParametersConvexModel:
    
    def __init__(self, scalar_function, model, input_tensor, 
    shapes_trainable_parameters, regularizing_function="smooth absolut"+
    "e value", model_true_values=None, parameters_type=tf.float32):
        
        self.scalar_function = scalar_function

        self.model = model 

        self.input_tensor = input_tensor

        self.shapes_trainable_parameters = shapes_trainable_parameters

        # Sets the function which is used to regularize the weights to 
        # be non-negative

        if isinstance(regularizing_function, str):

            self.regularizing_function = numerical_tools.build_tensorflow_math_expressions(
            regularizing_function, dtype=parameters_type)

        else:

            self.regularizing_function = regularizing_function

        # Creates a dummy true value of output, because the Keras' loss 
        # functions requires y_true and y_pred as arguments

        if model_true_values is None:

            self.model_true_values = tf.constant([0.0], dtype=
            input_tensor.dtype)

        else:

            self.model_true_values = model_true_values

        # Saves a flag to tell to use the custom gradient defined within
        # the loss function class or not

        if hasattr(self.scalar_function, "custom_gradient_usage"):

            self.custom_gradient = self.scalar_function.custom_gradient_usage

        else:

            self.custom_gradient = False

        # If the custom gradient is to be used, prepare it sharing the 
        # same model and input tensor

        if self.custom_gradient:

            self.scalar_function.prepare_custom_gradient(self.model,
            self.input_tensor)

        # Verifies if the type of the parameters is different to the ty-
        # pe of the input tensor

        if self.input_tensor.dtype!=parameters_type:

            raise TypeError("The type of the input tensor, "+str(
            self.input_tensor.dtype)+", is different to the type of th"+
            "e trainable parameters, "+str(parameters_type))

    # Defines a function to actually evaluate the derivative

    @tf.function
    def __call__(self, trainable_parameters):

        # Uses the custom implementation of the gradient defined inside
        # the class of the loss function

        if self.custom_gradient:

            # Gets the response of the model

            y = parameters_tools.model_output_given_trainable_parameters(
            self.input_tensor, self.model, trainable_parameters, 
            self.shapes_trainable_parameters, regularizing_function=
            self.regularizing_function)

            return self.scalar_function.custom_gradient(
            self.model_true_values, y, trainable_parameters)

        # Otherwise, uses automatic differentiation

        else:

            # Creates the tape

            with tf.GradientTape() as tape:

                tape.watch(trainable_parameters)

                # Gets the response of the model

                y = parameters_tools.model_output_given_trainable_parameters(
                self.input_tensor, self.model, trainable_parameters, 
                self.shapes_trainable_parameters, regularizing_function=
                self.regularizing_function)

                phi = self.scalar_function(self.model_true_values, y)

            # Gets the gradient

            return tape.gradient(phi, trainable_parameters)
        
    # Defines a function to evaluate the loss function

    def evaluate_scalar_function(self, trainable_parameters):

        # Gets the response of the model

        y = parameters_tools.model_output_given_trainable_parameters(
        self.input_tensor, self.model, trainable_parameters, 
        self.shapes_trainable_parameters, regularizing_function=
        self.regularizing_function)

        # Gets the scalar function value and returns it

        return self.scalar_function(self.model_true_values, y)
    
    # Defines a function to update the scalar function if it is parame-
    # terizable by externally-given quantities

    def update_function(self, *external_arguments):

        # Calls the method to update the scalar function

        self.scalar_function.update_arguments(*external_arguments)

########################################################################
#                       NN model jacobian matrix                       #
########################################################################

# Defines a method to evaluate the derivative of the model with respect
# to the parameters (weights and biases)

def model_jacobian(model, output_dimension, evaluate_parameters_gradient=
True):
        
    # Defines a function to compute a matrix of derivatives of the model
    # w.r.t. to the vector of parameters. Each column corresponds to the
    # gradient evaluated at a sample of inputs. The structure of the ma-
    # trix is:
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
    # n: number of input samples; m: number of model trainable variables;
    # l: number of output neurons; yi is the i-th output neuron of the
    # model. Hence, the matrix would be (l*m)xn
    
    # Uses the gradient function for each component and sample
    
    if evaluate_parameters_gradient=="tensorflow gradient":
        
        def gradient_evaluator(x):

            # Initializes the matrix of gradients

            gradient_matrix = []

            # Iterates through the samples of the input

            for sample_counter in range(x.shape[0]):

                # Gets the corresponding sample of the input

                x_sample = tf.expand_dims(x[sample_counter], 0)

                # Initializes the gradient for this sample

                gradient_sample = []

                # Iterates through the output neurons

                for output in range(output_dimension):

                    # Gets the tape for evaluating the gradient

                    with tf.GradientTape() as tape:

                        model_response = model(x_sample)[0,output]

                    # Evaluates the gradient with respect to the model
                    # parameters

                    full_gradient = tape.gradient(model_response,
                    model.trainable_variables)

                    # Flattens the gradient

                    flat_gradient = tf.concat([tf.reshape(layer, [-1]
                    ) for layer in full_gradient], axis=0)

                    # Concatenates this gradient to the sample list

                    gradient_sample.extend(flat_gradient.numpy())

                # Adds this gradient sample list to the gradient matrix

                gradient_matrix.append(gradient_sample)

            # Transforms into a numpy array and transposes to be a matrix
            # of (number of parameters times the number of output neurons
            # ) x (number of samples)

            return np.transpose(np.array(gradient_matrix))

        return gradient_evaluator
    
    elif evaluate_parameters_gradient=="tensorflow jacobian":

        def gradient_evaluator(x):

            # Initializes the matrix of gradients

            gradient_matrix = []

            # Iterates through the samples of the input

            for sample_counter in range(x.shape[0]):

                # Gets the corresponding sample of the input

                x_sample = tf.expand_dims(x[sample_counter], 0)

                # Gets the tape for evaluating the gradient

                with tf.GradientTape() as tape:

                    model_response = model(x_sample)[0]

                # Evaluates the gradient with respect to the model para-
                # meters using the jacobian function to capture the de-
                # rivative of all output neurons at once

                full_jacobian = tape.jacobian(model_response,
                model.trainable_variables)

                # Flattens the gradient

                flat_jacobian = tf.concat([tf.reshape(layer, (
                output_dimension, -1)) for layer in full_jacobian], 
                axis=1)

                # Adds this gradient sample list to the gradient matrix

                gradient_matrix.append(flat_jacobian.numpy().ravel())

            # Transforms into a numpy array and transposes to be a matrix
            # of (number of parameters times the number of output neurons) 
            # x (number of samples)

            return np.transpose(np.array(gradient_matrix))

        return gradient_evaluator
    
    elif evaluate_parameters_gradient in ["vectorized tensorflow jacob"+
    "ian", True]:
        
        def gradient_evaluator(x):

            # Gets the tape for evaluating the gradient
            
            with tf.GradientTape() as tape:

                model_response = model(x)

            # Evaluates the gradient with respect to the model parameters
            # using the jacobian function to capture the derivative of 
            # all output neurons at once

            full_jacobian = tape.jacobian(model_response,
            model.trainable_variables, experimental_use_pfor=True)

            # Reshape the jacobian to (number of samples, output dimen-
            # sion, number of parameters), and concatenates along the 
            # parameters
            
            reshaped_jacobian = tf.concat(tf.nest.map_structure(
            lambda J: tf.reshape(J, (tf.shape(x)[0], output_dimension, 
            -1)), full_jacobian), axis=-1)

            # Flattens it and gets as a numpy array and returns the
            # transpose

            return tf.transpose(tf.reshape(reshaped_jacobian, (tf.shape(
            x)[0], -1))).numpy()

        return gradient_evaluator
    
    else:

        raise NameError("The flag 'evaluate_parameters_gradient' in 'M"+
        "ultiLayerModel' can be either 'tensorflow gradient', or 'tens"+
        "orflow jacobian', True (defaults to 'vectorized tensorflow ja"+
        "cobian'), or 'vectorized tensorflow jacobian'")
    
########################################################################
#                              Utilities                               #
########################################################################
 
# Defines a function to convert the gradient of a scalar loss function 
# with respect to the trainable variables to a numpy array

def convert_scalar_gradient_to_numpy(gradient):

    return tf.concat([tf.reshape(layer, [-1] ) for layer in gradient], 
    axis=0)