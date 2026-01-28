# Routine to store mathematical operators, both algebra and calculus of
# tensors

import tensorflow as tf

# Defines a function to evaluate the gradient of a field in a point x
# where the NN model can have more arguments

def grad(model, x, extra_arguments=None, axis=1):

    # Assures that the inputs are tensorflow tensors

    x = tf.convert_to_tensor(x)

    if not (extra_arguments is None):

        extra_arguments = tf.convert_to_tensor(extra_arguments)

    # Evaluates the gradient using automatic differentiation

    with tf.GradientTape() as tape:

        # Signals to record operations with x to take the derivative la-
        # ter

        tape.watch(x)

        # Concatenates the input and generates the response of the model

        model_response = None

        if extra_arguments is None:

            model_response = model(x)

        else:

            model_response = model(tf.concat([x, extra_arguments], axis=
            axis))

        # Takes the gradient

        model_gradient = tape.jacobian(model_response, x)

    # Deletes the evaluation of the automatic differentiation to spare
    # memory

    del tape

    # Returns the gradient

    if model_gradient is None:

        raise ValueError("The model "+str(model)+" does not depend upo"+
        "n the given variables. Thus, the gradient is None")
    
    else:

        return model_gradient
    
# Defines a function to evaluate the divergent of a field

def div(model, x, extra_arguments=None, axis=1):

    # Assures that the inputs are tensorflow tensors

    x = tf.convert_to_tensor(x)

    if not (extra_arguments is None):

        extra_arguments = tf.convert_to_tensor(extra_arguments)

    # Evaluates the divergent using automatic differentiation

    with tf.GradientTape() as tape:

        # Signals to record operations with x to take the derivative la-
        # ter

        tape.watch(x)

        # Concatenates the input and generates the response of the model

        model_response = None

        if extra_arguments is None:

            model_response = model(x)

        else:

            model_response = model(tf.concat([x, extra_arguments], axis=
            axis))

        # Takes the gradient

        model_gradient = tape.gradient(model_response, x)

    # Deletes the evaluation of the automatic differentiation to spare
    # memory

    del tape

    # Returns the gradient

    if model_gradient is None:

        raise ValueError("The model "+str(model)+" does not depend upo"+
        "n the given variables. Thus, the gradient is None")
    
    else:

        return model_gradient