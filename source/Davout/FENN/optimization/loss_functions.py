# Routine to store loss functions

import tensorflow as tf

from ...PythonicUtilities.dictionary_tools import verify_obligatory_and_optional_keys

# Defines a class to evaluate the loss function corresponding only to
# the quadratic form over the residual

class QuadraticForm:

    def __init__(self, info_dictionary):

        # Verifies the keys of the dictionary of information for this
        # class

        verify_obligatory_and_optional_keys(info_dictionary, ["loss fu"+
        "nction name"], {}, "info_dictionary", "QuadraticForm")

    # Defines a function to get the vector of parameters, the residual
    # vector and the conditioned residual vector to evaluate the loss 
    # function

    @tf.function
    def __call__(self, vector_of_parameters, global_residual_vector,
    conditioned_residual_vector):

        # Returns a quadratic form over the residual vector only

        return tf.reduce_sum(global_residual_vector*
        conditioned_residual_vector)