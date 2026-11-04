# Routine to store methods to set the GPU for tensorflow

import tensorflow as tf

# Defines a function to set the access of tensorflow to the GPU

def give_access_of_gpu_to_tensorflow(flag_to_use_gpu):

    # Checks whether the flag for tensorflow to use the GPU is false

    if not flag_to_use_gpu:

        # Disables the GPU if necessary

        tf.config.set_visible_devices([], 'GPU')

    print(tf.config.list_physical_devices('CPU'))

    print(tf.config.list_physical_devices('GPU'))

    print("Visible GPUs:", tf.config.get_visible_devices("GPU"))