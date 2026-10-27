# Routine to store methods to handle errors in python

import numpy as np

########################################################################
#                       Type verification errors                       #
########################################################################

# Defines a function to verify the type of an object and to return an 
# error if its type is not as intended

def verify_type(object_value, object_name, type_class, code_location, 
ignore_none_value=False, description=None, default_in_case_of_none=
False):

    # If None values are allowed and should not be verified

    if (ignore_none_value or default_in_case_of_none!=False) and (
    object_value is None):

        if default_in_case_of_none!=False:

            # Returns the default value

            return default_in_case_of_none

        return object_value

    # Sets a dictionary of special cases. Each value is also a dictiona-
    # ry with the type and the special function to verify the incoming
    # object

    special_cases = {"non-negative int": {"type": (int, np.integer), 
    "verification_function": check_non_negative_integer}}

    # Checks whether the type class is an option of the special cases

    type_name = None 

    special_verification_message = None

    if type_class in special_cases:

        # Saves the type name as the given type class

        type_name = str(type_class)

        # Gets the type class itself and the verification function

        type_class = special_cases[type_name]["type"]

        special_verification_function = special_cases[type_name]["veri"+
        "fication_function"]

        # Checks the verification function

        special_verification_message = special_verification_function(
        object_value)

    # Otherwise, tests the value

    if (not isinstance(object_value, type_class)) or (
    special_verification_message is not None):

        # If the special verification message is None, makes it an empty
        # string

        if special_verification_message is None:

            special_verification_message = ""

        # If description is None, simply makes it an empty string

        if description is None:

            description = ""

        else:

            description = ("\n\nThe description of "+str(object_name)+
            " is: "+str(description))

        # Tries to convert the type class to a name

        types_dictionary = {str: "string", dict: "dictionary", int: "i"+
        "nteger", float: "float (real number)", np.ndarray: "numpy arr"+
        "ay", list: "list", bool: "bool (True or False)"}

        # If the type class is a key in the dictionary of types, it will
        # be converted to the textual name of the type. Otherwise, it 
        # will remain as the type reference itself

        if type_name is None:

            if type_class in types_dictionary:

                type_name = types_dictionary[type_class]

            else:

                type_name = type_class

        raise TypeError(str(object_name)+" in "+str(code_location)+" m"+
        "ust be a "+str(type_name)+". Currently, it is "+str(
        object_value)+", whose type is "+str(type(object_value))+".\n"+
        special_verification_message+description)

    return object_value

# Defines a function to verify whether an object is a non-negative inte-
# get

def check_non_negative_integer(object_value):

    # Discards non-integer objectsm, even booleans

    if not isinstance(object_value, (int, np.integer)) or isinstance(
    object_value, (bool, np.bool_)):

        return None 

    # Checks if it is less than zero

    if object_value<0:

        return "Additionally, it is negative, which is not allowed."

    else:

        return None