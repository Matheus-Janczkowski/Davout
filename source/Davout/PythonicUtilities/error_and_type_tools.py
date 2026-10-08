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

    # Otherwise, tests the value

    if not isinstance(object_value, type_class):

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

        if type_class in types_dictionary:

            type_class = types_dictionary[type_class]

        raise TypeError(str(object_name)+" in "+str(code_location)+" m"+
        "ust be a "+str(type_class)+". Currently, it is "+str(
        object_value)+", whose type is "+str(type(object_value))+
        description)

    return object_value