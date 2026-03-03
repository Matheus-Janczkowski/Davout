# Routine to store methods to calculate conditioning matrices

import tensorflow as tf

from ...PythonicUtilities.dictionary_tools import verify_obligatory_and_optional_keys

# Defines a class to calculate a conditioning matrix which is a multi-
# ple of the identity matrix

class IdentityMultiple:

    def __init__(self, info_dictionary, float_dtype, number_of_dofs):

        # Verifies the keys of the dictionary of information for this
        # class

        verify_obligatory_and_optional_keys(info_dictionary, ["conditi"+
        "oner name"], {"multiple": {"type": float, "description": "Con"+
        "stant that multiplies the identity matrix to build the condit"+
        "ioning matrix"}}, "info_dictionary", "IdentityMultiple")

        # Saves the code-given information

        self.number_of_dofs = number_of_dofs

        self.float_dtype = float_dtype

        # Verifies if the constant that multiplies the identity matrix
        # is given

        self.constant = 1.0

        if "multiple" in info_dictionary:

            self.constant = info_dictionary["multiple"]

        # Converts to a tensor

        self.constant = tf.constant(self.constant, dtype=
        self.float_dtype)

    # Defines a function to construct the conditioning matrix as a ten-
    # sor, then, multiply it by the residual vector tensor

    @tf.function
    def __call__(self, vector_of_parameters, global_residual_vector):

        # Multiplies the conditioning matrix by the global residual vec-
        # tor, which is exactly multiplying the global residual vector 
        # by the constant

        return self.constant*global_residual_vector