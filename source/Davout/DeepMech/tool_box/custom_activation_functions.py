# Routine to store a class with custom activation functions

import tensorflow as tf

import inspect

from ...PythonicUtilities.dictionary_tools import verify_obligatory_and_optional_keys

from ...DeepMech.tool_box.numerical_tools import FunctionMathematicalData

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

                self.custom_activation_functions_dict[method_name] = (
                method_function)
    
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
    
    def quadratic(self, arguments_dict):

        # Verifies if the dictionary of arguments has keys that are not 
        # for this activation function

        arguments_dict = verify_obligatory_and_optional_keys(
        arguments_dict, {}, {"a2": {"type": float, "description": "coe"+
        "fficient a2 of the polynomial a2*(x^2)+a1*x+a0", "default": 1.0
        }, "a1": {"type": float, "description": "coefficient a1 of the"+
        " polynomial a2*(x^2)+a1*x+a0", "default": 0.0}, "a0": {"type": 
        float, "description": "coefficient a0 of the polynomial a2*(x^"+
        "2)+a1*x+a0", "default": 0.0}}, "dictionary of activation func"+
        "tion extra information", "CustomActivationFunctions")
        
        a2 = tf.constant(arguments_dict["a2"], dtype=self.dtype)
        
        a1 = tf.constant(arguments_dict["a1"], dtype=self.dtype)
        
        a0 = tf.constant(arguments_dict["a0"], dtype=self.dtype)

        # Defines the activation function itself

        @tf.function
        def quadratic_activation(x):

            return (a2*tf.square(x))+(a1*x)+a0

        # Sets an instance of the class that stores mathematical metada-
        # ta about this function

        convex = False 

        non_negative = False

        non_negative_on_non_negative_real_line = False

        monotonically_increasing_on_non_negative_real_line = True

        if a2>0.0:

            convex = True 

            if ((4.0*a2*a0)-tf.square(a1))>=0.0:

                non_negative = True

                non_negative_on_non_negative_real_line = True

            # Tests if the derivative is positive at zero

            if a1>=0.0:

                non_negative_on_non_negative_real_line = True

                monotonically_increasing_on_non_negative_real_line = True

        origin_centered = False

        if a1==0.0 and a0==0.0:

            origin_centered = True

        mathematical_data_class = FunctionMathematicalData(convex=convex, 
        monotonically_increasing=False, non_negative=non_negative,
        origin_centered=origin_centered, 
        non_negative_on_non_negative_real_line=
        non_negative_on_non_negative_real_line,
        monotonically_increasing_on_non_negative_real_line=
        monotonically_increasing_on_non_negative_real_line)

        return quadratic_activation, mathematical_data_class
    
    # Defines an exponential function
    
    def exponential(self, arguments_dict):

        # Verifies if the dictionary of arguments has keys that are not 
        # for this activation function

        arguments_dict = verify_obligatory_and_optional_keys(
        arguments_dict, {}, {"a1": {"type": float, "description": "coe"+
        "fficient a1 of the expression exp((a1*x)+a0)", "default": 1.0}, 
        "a0": {"type": float, "description": "coefficient a0 of the ex"+
        "pression exp((a1*x)+a0)", "default": 0.0}}, "dictionary of ac"+
        "tivation function extra information", "CustomActivationFuncti"+
        "ons")
        
        a1 = tf.constant(arguments_dict["a1"], dtype=self.dtype)
        
        a0 = tf.constant(arguments_dict["a0"], dtype=self.dtype)

        # Defines the activation function itself
    
        @tf.function
        def exponential_activation(x):

            return tf.exp((a1*x)+a0)

        # Sets an instance of the class that stores mathematical metada-
        # ta about this function

        monotonically_increasing = False

        monotonically_increasing_on_non_negative_real_line = False

        if a1>0.0:

            monotonically_increasing = True

            monotonically_increasing_on_non_negative_real_line = True

        mathematical_data_class = FunctionMathematicalData(convex=True, 
        monotonically_increasing=monotonically_increasing, non_negative=
        True, origin_centered=False, 
        non_negative_on_non_negative_real_line=True,
        monotonically_increasing_on_non_negative_real_line=
        monotonically_increasing_on_non_negative_real_line)

        return exponential_activation, mathematical_data_class
    
########################################################################
#                               Testing                                #
########################################################################

if __name__=="__main__":

    custom = CustomActivationFunctions()

    print(custom.custom_activation_functions_dict)