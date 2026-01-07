# Routine to store methods to process information in constitutive models

import tensorflow as tf

# Defines a function to transform a single material property in a tensor
# of that property for all elements and quadrature points

def convert_scalar_to_tensor(scalar_value):

    mu_elem = tf.gather(scalar_value, element_ids)

    mu_q = mu_elem[..., tf.newaxis]
