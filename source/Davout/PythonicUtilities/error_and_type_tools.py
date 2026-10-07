# Routine to store methods to handle errors in python

import numpy as np

########################################################################
#                       Type verification errors                       #
########################################################################

# Defines a function to verify the type of an object and to return an 
# error if its type is not as intended

def verify_type(object_value, object_name, type_class, code_location):

    if not isinstance(object_value, type_class):

        # Tries to convert the type class to a name

        types_dictionary = {str: "string", dict: "dictionary", int: "i"+
        "nteger", float: "float (real number)", np.ndarray: "numpy arr"+
        "ay", list: "list"}

        # If the type class is a key in the dictionary of types, it will
        # be converted to the textual name of the type. Otherwise, it 
        # will remain as the type reference itself

        if type_class in types_dictionary:

            type_class = types_dictionary[type_class]

        raise TypeError(str(object_name)+" in "+str(code_location)+" m"+
        "ust be a "+str(type_class)+". Currently, it is "+str(
        object_value)+", whose type is "+str(type(object_value)))