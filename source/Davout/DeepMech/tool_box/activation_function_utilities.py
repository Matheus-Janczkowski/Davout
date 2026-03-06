# Routine to store utilities for activation functions

import tensorflow as tf 

from copy import deepcopy

from ...PythonicUtilities import dictionary_tools, function_tools

# Defines a function to test whether an activation function's name cor-
# responds to an actual activation function in TensorFlow

def verify_activationName(function_name, custom_activations_class, 
arguments):

    if function_name=="linear":

        # Verifies if arguments have been prescribed, which are not al-
        # lowed for this activation function

        if not (arguments is None):

            raise KeyError("The activation function 'linear' cannot ha"+
            "ve addtional arguments, thus, its corresponding value in "+
            "the dictionary of activation functions musn't be a dictio"+
            "nary with any other key beside 'number of neurons'")

        return True
    
    elif function_name in (
    custom_activations_class.custom_activation_functions_dict):

        return True
    
    else:

        # Verifies if arguments have been prescribed, which are not al-
        # lowed for native tensorflow activation functions

        if not (arguments is None):

            raise KeyError("The activation function '"+str(function_name
            )+"', native to tensorflow, cannot have addtional argument"+
            "s, thus, its corresponding value in the dictionary of act"+
            "ivation functions musn't be a dictionary with any other k"+
            "ey beside 'number of neurons'")

        return (hasattr(tf.nn, function_name) and callable(getattr(tf.nn, 
        function_name, None)))

# Defines a function to check if the dictionary of activation functions
# has real activation names

def verify_activationDict(activation_dict, layer, 
live_activationFunctions, flag_customLayers, custom_activations_class):

    # Verifies if the dictionary is empty

    if not activation_dict:

        raise KeyError("The layer "+str(layer)+" has no activation fun"+
        "ction information. You must provide at least one activation f"+
        "unction and the number of neurons to it.")
    
    # Checks if there is more than one key in the dictionary, i.e. if 
    # there is more than one activation function type in this layer

    if not flag_customLayers:

        if len(activation_dict.keys())>1:

            flag_customLayers = True

    # Gets the already retrieved activation functions

    live_activations = set(live_activationFunctions.keys())

    # Initializes a message

    error_message = ""

    # Iterates over the activation functions' names

    for name, activation_info in activation_dict.items():

        # Verifies if the activation_info is a dictionary, i.e. keyword 
        # arguments have been passed as well

        arguments = None

        if isinstance(activation_info, dict):

            # Verifies if the number of neurons that use this activation 
            # function has been passed

            if not ("number of neurons" in activation_info):

                raise KeyError("A dictionary has been used to set an a"+
                "ctivation function information, "+str(activation_info)+
                ", but no 'number of neurons' key has been included")
            
            # Gets the arguments and deletes the key for the number of 
            # neurons

            arguments = deepcopy(activation_info)

            del arguments["number of neurons"]

            # If this dictionary is empty, turns this variable into None
            # again

            if not arguments:

                arguments = None

        # Verifies if the name of this activation function exists

        if not verify_activationName(name, 
        custom_activations_class, arguments):

            error_message += ("\n          "+str(name)+", in layer "+
            str(layer)+", is not a name of an actual activation functi"+
            "on in TensorFlow nor in the list\n          of custom act"+
            "ivation functions of DeepMech (see the DeepMech's list:\n"+
            "          "+str(
            custom_activations_class.custom_activation_functions_dict.keys(
            ))[11:-2]+")")

        # Verifies if this activation function has not already been map-
        # ped into the dictionary of live-wired activation functions

        elif not (name in live_activations):

            live_activationFunctions[name] = get_activationFunction(
            name, custom_activations_class, arguments)

    # If the error message is not empty, raises an exception

    if error_message!="":

        raise NameError(error_message)
    
    # Returns the updated dictionary of live-wired activation functions

    return live_activationFunctions, flag_customLayers
    
# Defines a function to get the activation function by its name

def get_activationFunction(function_name, custom_activations_class, 
arguments):

    if function_name=="linear":

        return tf.identity
    
    elif function_name in (
    custom_activations_class.custom_activation_functions_dict):
        
        # Gets the pair of function and keyword arguments

        function_info = (
        custom_activations_class.custom_activation_functions_dict[
        function_name])

        # If arguments have been prescribed

        if not (arguments is None):

            # Verifies if the dictionary of arguments has arguments that
            # are allowed and adds the default values which were not 
            # prescribed

            arguments = dictionary_tools.verify_dictionary_keys(
            arguments, function_info[1], dictionary_location="at defin"+
            "ition of custom activation function '"+str(function_name)+
            "'", fill_in_keys=True)

            # Uses a wrapper to wrap the function to set the new values 
            # for the keyword arguments
        
            return function_tools.construct_lambda_function(
            function_info[0], arguments)

        # If no arguments have been prescribed, returns the function 
        # simply

        return function_info[0]
    
    else:

        return getattr(tf.nn, function_name)