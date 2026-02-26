# Routine to store parametric curves

import tensorflow as tf

# Defines a class to get a linear function zero at the origin

class linear:

    def __init__(self, current_time, final_value, final_time):
        
        self.current_time = current_time

        self.final_value = tf.convert_to_tensor(final_value, dtype=
        current_time.dtype)

        self.final_time = tf.convert_to_tensor(final_time, dtype=
        current_time.dtype)

        # Initializes the result and updates with the values initially
        # provided

    @tf.function
    def __call__(self):

        return self.final_value*(self.current_time/self.final_time)