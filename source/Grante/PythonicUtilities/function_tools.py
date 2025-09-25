# Routine to store methods to be used with and for functions

import inspect

########################################################################
#                       Signature and arguments                        #
########################################################################

# Defines a function to get the arguments of a function and list the 
# keyword arguments into a dictionary

def get_functions_arguments(function_object):

    # Gets the signature of the function

    signature = inspect.signature(function_object)

    # Initializes the dictionary of keyword arguments

    keyword_arguments = dict()

    # Iterates through the arguments of the function

    for argument_name, default_value in signature.parameters.items():

        # If the default value is not empty

        if default_value!=inspect._empty:

            # Saves the argument and its default value

            keyword_arguments[argument_name] = default_value

    # Returns the dictionary of keyword arguments

    return keyword_arguments