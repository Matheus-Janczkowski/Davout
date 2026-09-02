# Routine to store methods to be used with and for functions

import inspect

import functools

########################################################################
#                       Signature and arguments                        #
########################################################################

# Defines a function to get the arguments of a function and list the 
# keyword arguments into a dictionary

def get_functions_arguments(function_object, number_of_arguments_only=
False, positional_arguments_only=False, return_positional_arguments_list=
False):

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

    # Initializes the list of positional arguments

    positional_arguments_list = []

    # Iterates through the arguments of the function

    for argument_name, default_value in signature.parameters.items():

        # If the default value is not empty (empty default value means 
        # the argument is positional and obligatory)

        if default_value.default!=inspect._empty:

            # Saves the argument and its default value

            keyword_arguments[argument_name] = default_value.default

        # Otherwise, saves the argument only in the list for positional
        # arguments

        else:

            positional_arguments_list.append(argument_name)

    # If the list of positional arguments is to be returned as well

    if return_positional_arguments_list:

        return keyword_arguments, positional_arguments_list

    # Returns the dictionary of keyword arguments

    return keyword_arguments

# Defines a function to verify if a dictionary can be used as argument
# to a function. In other words, does the dictionary has all positional
# arguments? Does it have any keys that are not keyword arguments?

def verify_dictionary_as_function_argument(function_object, 
input_dictionary):

    # Gets the dictionary of keyword arguments and the list of positi-
    # onal arguments

    keyword_arguments, positional_arguments_list = get_functions_arguments(
    function_object, return_positional_arguments_list=True)

    # Verifies if the dictionary has all positional arguments

    for argument_name in positional_arguments_list:

        if not (argument_name in input_dictionary):

            # Gets a string with the positional arguments

            positional_arguments_string = ""

            for arg_name in positional_arguments_list:

                positional_arguments_string += "\n'"+str(arg_name)+"'"

            raise KeyError("The 'input dictionary' to the function '"+
            str(function_object.__name__)+"' does not have a key for t"+
            "he positional argument '"+str(argument_name)+"'. Check th"+
            "e necessary positional arguments:\n"+
            positional_arguments_string)

    # Verifies if any of the keys does not correspond to a valid argu-
    # ment

    for key in input_dictionary.keys():

        if (not (key in keyword_arguments)) and (not (key in (
        positional_arguments_list))):

            # Gets a string with the positional arguments

            positional_arguments_string = ""

            for argument_name in positional_arguments_list:

                positional_arguments_string += "\n'"+str(argument_name
                )+"'"

            # Gets a string with the keyword arguments

            keyword_arguments_string = ""

            for argument_name in keyword_arguments.keys():

                keyword_arguments_string += "\n'"+str(argument_name)+"'"

            raise ValueError("The key '"+str(key)+"' in the input dict"+
            "ionary for the function '"+str(function_object.__name__)+
            "' is not a valid argument of that function.\nCheck the av"+
            "ailable positional arguments:\n"+
            positional_arguments_string+"\n\nand the available keyword"+
            " arguments:\n"+keyword_arguments_string)

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