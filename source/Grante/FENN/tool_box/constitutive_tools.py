# Routine to store methods to process information in constitutive models

import tensorflow as tf

# Defines a function to transform a single material property in a tensor
# of that property for all elements and quadrature points

def convert_scalar_to_tensor():

    mu_elem = tf.gather(mu_per_element, element_ids)
    
    mu_q = mu_elem[..., tf.newaxis]
