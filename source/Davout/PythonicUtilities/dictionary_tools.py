# Routine to store methods to be used with and for dictionaries

########################################################################
#                             Verification                             #
########################################################################

def verify_dictionary_keys(dictionary: dict, master_keys: (list | dict), 
dictionary_location="at unknown location in the code", 
must_have_all_keys=False, fill_in_keys=False):
    
    """ Defines a function to verify if a dictionary has keys that are
    not listed in a list of allowable keys, 'master_keys'. If the flag 
    'must_have_all_keys' is True, the keys of the dictionary must match
    all the keys in the list of master keys. 'dictionary_location' is a
    string which tells in an eventual error, where the given dictionary
    is used. 'fill_in_keys' must be True when missing key-value pairs 
    from the 'master_keys' are missing in the dictionary; in this case,
    'master_keys' must be a dictionary"""

    # Verifies if the dictionary must have all keys

    if must_have_all_keys:

        # Transforms the keys of the dictionary and the list of keys to
        # sets, then compares them

        if set(dictionary.keys())!=set(master_keys):

            raise KeyError("The dictionary "+str(dictionary_location)+
            " must have all the keys of "+str(master_keys)+". But it h"+
            "as the following keys: "+str(list(dictionary.keys())))
        
    # Verifies if the missing key-value pairs must be filled in

    if fill_in_keys:

        # Verifies if master_keys is a dictionary

        if not isinstance(master_keys, dict):

            raise TypeError("'master_keys' must be a dictionary to fil"+
            "l in the missing key-value pairs in dictionary at "+str(
            dictionary_location))
        
        # Verifies if one of the keys are not in master keys

        for current_key in dictionary.keys():

            if not (current_key in master_keys):

                raise KeyError("The dictionary "+str(dictionary_location
                )+" has the key '"+str(current_key)+"', but it is not "+
                "in the list of master keys: "+str(master_keys))

        # Fill in the missing keys from master keys

        for master_key, master_value in master_keys.items():

            if not (master_key in dictionary.keys()):

                dictionary[master_key] = master_value

        # Returns the complemented dictionary

        return dictionary
            
    else:
        
        # Verifies if one of the keys are not in master keys

        for current_key in dictionary.keys():

            if not (current_key in master_keys):

                raise KeyError("The dictionary "+str(dictionary_location
                )+" has the key '"+str(current_key)+"', but it is not "+
                "in the list of master keys")
            
# Defines a function to verify optional and obligatory keys in a given
# dictionary

def verify_obligatory_and_optional_keys(dictionary: dict, 
obligatory_keys: (list | dict), optional_keys: (list | dict), 
dictionary_variable_name: str, location: str, check_not_expected_keys=
True):
    
    """
    Function to verify if a dictionary has all the obligatory keys, and
    if any other key not listed in obligatory or optional keys was given.
    
    dictionary: asserted dictionary
    
    obligatory_keys: list of obligatory keys. It can be either a simple 
    list such as [key1, key2, ..., keyn], or a dictionary with a key for 
    description and type {key1: {"description": description_1, "type":
    type1}, ...]
    
    optional_keys: list of optional_keys. It can be either a simple list
    such as [key1, key2, ..., keyn], or a dictionary with a key for 
    description and type {key1: {"description": description_1, "type":
    type1}, ...]
    
    dictionary_variable_name: name of the variable that possess that 
    dictionary
    
    location: function or class name where this dictionary is processed
    
    check_not_expected_keys: flag to tell if this function is to verify
    if keys in the dictionary are not in the list of obligatory or 
    optional keys. The default value is True"""

    # Verifies if dictionary is indeed a dictionary

    if not isinstance(dictionary, dict):

        raise TypeError("'"+str(dictionary_variable_name)+"' in '"+str(
        location)+"' must be a dictionary. Currently it is:\n"+str(
        dictionary)+"\nwhose type is: "+str(type(dictionary)))
    
    # Initializes a set of inspected keys

    inspected_keys = set()

    # Verifies if obligatory keys is a dictionary

    if isinstance(obligatory_keys, dict):
    
        # Verifies if any of the obligatory key is not there

        for key, info_dict in obligatory_keys.items():
                
            # Verifies if the key is there

            if not (key in dictionary):

                # Gets a string with the list of obligatory and optional 
                # keys

                available_keys = list_keys(obligatory_keys, 
                optional_keys)

                raise ValueError("Key '"+str(key)+"' was not found in "+
                "'"+str(dictionary_variable_name)+"' dictionary at '"+
                str(location)+"'. However, that key is obligatory. Che"+
                "ck the available keys:"+available_keys)
            
            # Verifies the type of the associated value

            elif "type" in info_dict:

                # Gets the necessary type

                value_type = info_dict["type"]

                # Gets the associated value in the given dictionary

                value = dictionary[key]

                # Verifies the type

                if not isinstance(value, value_type):

                    # Gets a string with the list of obligatory and op-
                    # tional keys

                    available_keys = list_keys(obligatory_keys, 
                    optional_keys)

                    raise TypeError("The value associated with key '"+
                    str(key)+"' in dictionary '"+str(
                    dictionary_variable_name)+"' at '"+str(location)+
                    "' has type "+str(type(value))+", whereas it must "+
                    "have type '"+str(value_type)+"'. Check the availa"+
                    "ble keys and their types:")
                
            # If nothing happened, updates the set of verified keys

            inspected_keys.add(key)

    # Otherwise

    else:
    
        # Verifies if any of the obligatory key is not there

        for key_element in obligatory_keys:
                
            # Verifies if the key in there

            if not (key_element in dictionary):

                # Gets a string with the list of obligatory and optional 
                # keys

                available_keys = list_keys(obligatory_keys, 
                optional_keys)

                raise ValueError("Key '"+str(key_element)+"' was not f"+
                "ound in '"+str(dictionary_variable_name)+"' dictionar"+
                "y at '"+str(location)+"'. However, that key is obliga"+
                "tory. Check the available keys:"+available_keys)
                
            # Otherwise, updates the set of verified keys

            else:

                inspected_keys.add(key_element)

    # Gets a list of optional keys

    optional_keys_list = []

    flag_check_type = False

    if isinstance(optional_keys, dict):

        optional_keys_list = list(optional_keys.keys())

        flag_check_type = True

    else:

        optional_keys_list = optional_keys
        
    # Verifies if any of the keys in the dictionary is not in the lists 
    # of obligatory or optional keys

    if check_not_expected_keys:

        # Iterates through the keys that were not inspect yet

        for key in (set(dictionary.keys())-inspected_keys):

            # Verifies if they are in the list of optional keys

            if not (key in optional_keys_list):

                # Gets a string with the list of obligatory and optional
                # keys

                available_keys = list_keys(obligatory_keys, 
                optional_keys)

                raise ValueError("Key '"+str(key)+"' is not a valid ke"+
                "y for the dictionary '"+str(dictionary_variable_name)+
                "' at '"+str(location)+"'. Check the available keys:"+
                available_keys)
            
            # Verifies the type of the associated value if needed

            elif flag_check_type:

                # Gets the value and the required type

                value_type = optional_keys[key]["type"]

                value = dictionary[key]

                # Compares them

                if not isinstance(value, value_type):

                    # Gets a string with the list of obligatory and op-
                    # tional keys

                    available_keys = list_keys(obligatory_keys, 
                    optional_keys)

                    raise TypeError("The value associated with key '"+
                    str(key)+"' in dictionary '"+str(
                    dictionary_variable_name)+"' at '"+str(location)+
                    "' has type "+str(type(value))+", whereas it must "+
                    "have type '"+str(value_type)+"'. Check the availa"+
                    "ble keys and their types:")
            
# Defines a function to list the keys in a string for error messageing

def list_keys(obligatory_keys, optional_keys):

    # Gets the list of available keys

    available_keys = "\n\nObligatory keys:"

    # Verifies if the obligatory keys are in fact a dictionary

    if isinstance(obligatory_keys, dict):

        for key, key_info in obligatory_keys.items():

            # Adds the key name

            available_keys += ("\n\n'"+str(key)+"'")

            # Verifies if type is given

            if "type" in key_info:

                available_keys += ("\nType: "+str(key_info["type"]))

            # Verifies if a description is given

            if "description" in key_info:

                available_keys += ("\nDescription:\n"+str(key_info["de"+
                "scription"]))

    # Otherwise, if it is a plain list

    else:

        for key in obligatory_keys:

            # Adds the key name and its description

            available_keys += "\n'"+str(key)+"'"

    # Adds the optional keys

    available_keys += "\n\nOptional keys:"

    if isinstance(optional_keys, dict):

        for key, key_info in optional_keys.items():

            # Adds the key name

            available_keys += ("\n\n'"+str(key)+"'")

            # Verifies if type is given

            if "type" in key_info:

                available_keys += ("\nType: "+str(key_info["type"]))

            # Verifies if a description is given

            if "description" in key_info:

                available_keys += ("\nDescription:\n"+str(key_info["de"+
                "scription"]))

    # Otherwise, if it is a plain list

    else:

        for key in optional_keys:

            # Adds the key name and its description

            available_keys += "\n'"+str(key)+"'"

    # Returns the string

    return available_keys

########################################################################
#                           Key manipulation                           #
########################################################################

# Defines a function to get the first key from a value

def get_first_key_from_value(dictionary, given_value):

    for key, value in dictionary.items():

        if value==given_value:

            return key 

    return None 

# Defines a function to delete keys off of a dictionary

def delete_dictionary_keys(dictionary, keys):

    if isinstance(keys, list):

        for key in keys:

            # Deletes the key

            dictionary.pop(key, None)

    else:

        dictionary.pop(keys, None)

    return dictionary

# Defines a function to sort a dictionary using its values

def sort_dictionary_by_values(dictionary):

    return dict(sorted(dictionary.items(), key=lambda item: item[1]))