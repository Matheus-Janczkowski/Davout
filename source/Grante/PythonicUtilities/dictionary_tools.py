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