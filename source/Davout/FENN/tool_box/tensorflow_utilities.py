# Routine to store tensorflow utilities

import tensorflow as tf

########################################################################
#                                Tensors                               #
########################################################################

# Defines a function to convert an object to tensorflow tensor with the
# designated tensorflow type

def convert_object_to_tensor(x, dtype):

    # Verifies if x is tensorflow tensor already
    
    if tf.is_tensor(x):

        # Simply uses function cast to change the type

        return tf.cast(x, dtype)
    
    # Otherwise, converts to tensor
    
    return tf.convert_to_tensor(x, dtype=dtype)