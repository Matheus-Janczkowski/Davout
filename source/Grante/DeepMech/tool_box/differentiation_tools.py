# Routine to store methods for automatic differentiation

import tensorflow as tf

import numpy as np

########################################################################
#                          NN model gradient                           #
########################################################################

# Defines a function to evaluate the a scalar function and its gradient
# with respect to the model trainable parameters

@tf.function
def scalar_gradient_wrt_trainable_params(scalar_function, model, 
input_tensor):

    # Creates the tape

    with tf.GradientTape as tape:

        phi = scalar_function(model, input_tensor)

    # Gets the gradient

    return phi, tape.gradient(phi, model.trainable_variables)

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