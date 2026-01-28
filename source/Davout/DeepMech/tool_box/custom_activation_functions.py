# Routine to store a class with custom activation functions

import tensorflow as tf

import inspect

from ...PythonicUtilities import function_tools

class CustomActivationFunctions:

    def __init__(self, dtype=tf.float32):

        # Sets the numerical type for the constants and so forth

        self.dtype = dtype

        self.update_dtype(dtype)

        # Initializes a dictionary of methods as Keras activation func-
        # tions

        self.custom_activation_functions_dict = dict()

        # Defines a list of methods that are not to be compiled within
        # this dictionary

        excepted_methods = ["__init__"]

        # Iterates through the methods defined on this class

        for method_name, method_function in inspect.getmembers(
        self, predicate=inspect.ismethod):
            
            if not (method_name in excepted_methods):

                # Adds to the dictionary all methods except those inside
                # the list of exceptions

                self.custom_activation_functions_dict[method_name] = [
                method_function, function_tools.get_functions_arguments(
                method_function)]
    
    # Defines a method to update the tensorflow type of the constants of
    # this class

    def update_dtype(self, dtype):

        if isinstance(dtype, str):

            self.dtype = tf.as_dtype(dtype)

        else:

            self.dtype = dtype

    # Defines a method to serialize this class inside Keras

    def get_config(self):

        # Returns only JSON-serializable values (dicts, lists, numbers, 
        # strings)

        return {"dtype": self.dtype.name}

    @classmethod
    def from_config(cls, config):

        # Gets dtype back into a tensorflow

        return cls(dtype=tf.as_dtype(config["dtype"]))

    ####################################################################
    #             Define special activation functions below            #
    ####################################################################

    # Defines a quadratic activation function
    
    def quadratic(self, x, a2=1.0, a1=0.0, a0=0.0):
        
        a2 = tf.constant(a2, dtype=self.dtype)
        
        a1 = tf.constant(a1, dtype=self.dtype)
        
        a0 = tf.constant(a0, dtype=self.dtype)

        return (a2*tf.square(x))+(a1*x)+a0
    
    # Defines an exponential function
    
    def exponential(self, x, a1=1.0, a0=0.0):
        
        a1 = tf.constant(a1, dtype=self.dtype)
        
        a0 = tf.constant(a0, dtype=self.dtype)

        return tf.exp((a1*x)+a0)
    
########################################################################
#                               Testing                                #
########################################################################

if __name__=="__main__":

    custom = CustomActivationFunctions()

    print(custom.custom_activation_functions_dict)