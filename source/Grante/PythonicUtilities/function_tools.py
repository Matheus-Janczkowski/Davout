# Routine to store methods to be used with and for functions

import inspect

import functools

########################################################################
#                       Signature and arguments                        #
########################################################################

# Defines a function to get the arguments of a function and list the 
# keyword arguments into a dictionary

def get_functions_arguments(function_object, number_of_arguments_only=
False, positional_arguments_only=False):

    # Gets the signature of the function

    signature = inspect.signature(function_object)

    # If just the number of arguments is to be given

    if number_of_arguments_only:

        number_arguments = len(signature.parameters.keys())

        if positional_arguments_only:

            for default_value in signature.parameters.values():

                # If the argument is not empty, it is optional, thus, 
                # does not count it

                if default_value.default!=inspect._empty:

                    number_arguments -= 1

        return number_arguments

    # Initializes the dictionary of keyword arguments

    keyword_arguments = dict()

    # Iterates through the arguments of the function

    for argument_name, default_value in signature.parameters.items():

        # If the default value is not empty (empty default value means 
        # the argument is positional and obligatory)

        if default_value.default!=inspect._empty:

            # Saves the argument and its default value

            keyword_arguments[argument_name] = default_value.default

    # Returns the dictionary of keyword arguments

    return keyword_arguments

########################################################################
#                          Lambdas and drivers                         #
########################################################################

# Defines a function to construct a wrapper using functools instead of
# using the conventional lambda function. This is preferable for seria-
# lization

def construct_lambda_function(function_object, fixed_arguments):

    """
    Constructs a function as in 
    lambda x: function_object(x, **fixed_arguments)
    but in a fancier way using functools wrapper, to benefit of 
    serialization capabilities"""

    # If the dictionary is empty, returns the function without any modi-
    # fication

    if not fixed_arguments:

        return function_object
    
    # Otherwise, wraps the fixed arguments

    @functools.wraps(function_object)
    def wrapped_function(x):

        return function_object(x, **fixed_arguments)
    
    return wrapped_function