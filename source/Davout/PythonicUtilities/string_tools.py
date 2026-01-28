# Routine to store tools for managing strings

import numpy as np

from ..PythonicUtilities import recursion_tools

########################################################################
#                   Parsing strings to other formats                   #
########################################################################

# Defines a function to convert a dictionary written as string back to a
# dictionary

def string_toDict(original_string):

    #print("Receives:", original_string)

    # Initializes the dictionary

    read_dictionary = dict()

    # Initializes a key and a value

    key = ""

    value = ""

    # Initializes one flag to inform what is being read: True for key;
    # False for the value. Initializes it as True because keys appear
    # always first

    flag_key = True

    # Iterates through the characters

    string_length = len(original_string)

    character_counter = 1

    while character_counter<string_length:

        # Takes the character

        character = original_string[character_counter]

        # Tests whether it is the last character and if it is "}"

        if character_counter==(string_length-1):

            if character=="}" or character=='}':

                # Saves the key and value only if it is not empty
                
                if (len(key)>0) and (len(value)>0):

                    # Tries to convert the key to other format

                    key = convert_string(key)

                    # Tries to convert the value

                    value = convert_value(value)

                    # Saves the pair

                    #print(key, value)

                    read_dictionary[key] = value

            # Otherwise, raises and error for the original string being
            # incomplete

            else:

                raise ValueError("The string to be converted to a dict"+
                "ionary does not end with '}', so it is improper and, "+
                "possibly, incomplete")

        # Tests whether this character is :

        elif character==":" or character==':':

            # Changes the flag to save values

            flag_key = False

        # Tests whether this character is ,

        elif character=="," or character==',':

            # Tests if the key is being saved

            if flag_key:

                # Saves to the key

                key += character

            else:

                # Changes the flag to save keys

                flag_key = True 

                # Tries to convert the key to other format

                key = convert_string(key)

                # Tries to convert the value

                value = convert_value(value)

                # Saves the pair

                read_dictionary[key] = value

                # Cleans up the key and value variables

                key = ""

                value = ""

        # Verifies if there's a dictionary inside this dictionary

        elif character=="{" or character=='{':

            #print("nested dictiopnary")

            # Initializes a counter of nested dictionaries

            n_nestedDicts = 0

            # Fast forwards to find the end of this nested dictionary

            for nested_counter in range(character_counter, string_length
            -1, 1):

                nested_character = original_string[nested_counter]

                # If it is an open bracket, a dictionary is initiated

                if nested_character=="{" or nested_character=='{':

                    n_nestedDicts += 1

                # If it is a closing bracket, a dictionary is terminated

                elif nested_character=="}" or nested_character=='}':

                    n_nestedDicts -= 1

                # If the number of nested dictionaries is 0, it means all
                # of the nested dictionaries have been found

                if n_nestedDicts==0:

                    # Saves the value

                    value = original_string[character_counter:(
                    nested_counter+1)]

                    """print("finish nested", character_counter, nested_counter)

                    print(value)"""

                    # Updates the global character counter

                    character_counter = nested_counter+0

                    break

        # Verifies if this character is a blank space put after the : 
        # symbol next to a key

        elif (character==" " or character==' ') and (original_string[
        character_counter-1]==":" or original_string[character_counter-1
        ]==':' or original_string[character_counter-1]=="," or (
        original_string[character_counter-1]==',')):

            # Does not save it

            pass

        # If the character is different than ', retrieves it

        elif character!="'":

            # Verifies which is being saved

            if flag_key:

                key += character

            else:

                value += character

        # Updates the character counter

        character_counter += 1

    # Returns the dictionary

    return read_dictionary

# Defines a function to convert a string to a list

def string_toList(saved_string, element_separator=",", print_warnings=
True):

    # Intializes the list to be read

    read_list = []
    
    # Initializes a list of indexes to inform where to append the read
    # element

    indexes_list = []

    # Initializes a counter of elements in the current sublist

    element_counterSubList = 0

    # Iterates through the characters of the string, but takes the first
    # and last characters out as they are the outer brackets

    # Initializes a string to store the element that is being read

    read_element = ""

    character_counter = 0

    #for i in range(1,len(saved_string),1):

    string_length = len(saved_string)

    while character_counter<(string_length-1):

        character_counter += 1

        # Retrieves the current character

        current_character = saved_string[character_counter]

        # If the character is a comma or a closing bracket, stops the
        # reading of the element

        if current_character==element_separator or current_character=="]":

            if len(read_element)>0:

                # Converts the element

                if read_element=="True":

                    read_element = True 

                elif read_element=="False":

                    read_element = False

                elif not isinstance(read_element, dict):

                    try:

                        # Tries to convert it to an integer

                        read_element = int(read_element)

                    except:

                        # Tries to convert it to a float

                        try:

                            read_element = float(read_element)

                        except:

                            if print_warnings:

                                print("Could not convert", read_element, 
                                "to a number\n")

                            if isinstance(read_element, str):

                                if len(read_element)>1:

                                    if read_element[0]=="'":

                                        read_element = read_element[1:]

                                    if read_element[-1]=="'":

                                        read_element = read_element[0:-1]

                # Appends the element to the list using the list of in-
                # dexes

                """
                if current_character=="]":

                    print("\nFinalizes sublist in list with indexes_li"+
                    "st="+str(indexes_list)+" and current_character="+
                    str(current_character))

                else:

                    print("\nAdds element to list with indexes list="+
                    str(indexes_list)+" and current_character="+str(
                    current_character))
                """

                read_list = recursion_tools.recursion_listAppending(
                read_element, read_list, indexes_list, -len(indexes_list
                ))

                """
                print("\nUpdated list:", read_list, "element_counterSu"+
                "bList="+str(element_counterSubList)+", indexes_list="+
                str(indexes_list)+"\n")
                """

                # Clears the read element

                read_element = ""

            # If the character is a comma, updates the number o elements
            # in the current sublist

            if current_character==element_separator:

                element_counterSubList += 1

            # If the character is a closing bracket, takes the last in-
            # dex out of the list of indexes and updates the counter of
            # elements in the sublist to the remaining last index

            if current_character=="]" and len(indexes_list)>0:

                element_counterSubList = indexes_list[-1]+0

                indexes_list = indexes_list[0:-1]

            """
            print("\ncharacter="+str(current_character)+" and element_"+
            "counterSubList="+str(element_counterSubList)+" and indexe"+
            "s_list="+str(indexes_list)+"\n")
            """

        # If the current character is an opening bracket, it sinalizes a
        # new sublist, hence, updates list of indexes and the counter of
        # elements in the sublist

        elif current_character=="[":

            """
            print("\nAdds new sublist with indexes_list="+str(
            indexes_list)+" and current_character="+str(
            current_character))
            """

            # Adds a new empty list to the read list
            
            read_list = recursion_tools.recursion_listAppending([], 
            read_list, indexes_list, -len(indexes_list))

            # Updates the list of indexes and the counter of elements in
            # the sublist (makes it 0, as a new empty list is added)

            indexes_list.append(element_counterSubList)

            """
            print("\nUpdated list:", read_list, "and the indexes_list="+
            str(indexes_list)+" and the element_counterSubList="+str(
            element_counterSubList)+"\n")
            """

            element_counterSubList = 0

        # Tests if it a bracket (signaling a dictionary)

        elif current_character=="{" or current_character=='{':

            # Initializes a counter of nested dictionaries

            n_nestedDicts = 0

            # Fast forwards to find the end of this nested dictionary

            for nested_counter in range(character_counter, string_length, 
            1):

                nested_character = saved_string[nested_counter]

                # If it is an open bracket, a dictionary is initiated

                if nested_character=="{" or nested_character=='{':

                    n_nestedDicts += 1

                # If it is a closing bracket, a dictionary is terminated

                elif nested_character=="}" or nested_character=='}':

                    n_nestedDicts -= 1

                # If the number of nested dictionaries is 0, it means all
                # of the nested dictionaries have been found

                if n_nestedDicts==0:

                    # Saves the value

                    dictionary_string = saved_string[character_counter:(
                    nested_counter+1)]

                    # Converts this to a dictionary

                    read_element = string_toDict(dictionary_string)

                    """print("finish nested", character_counter, nested_counter)

                    print(value)"""

                    # Updates the global character counter

                    character_counter = nested_counter+0

                    break 

        # Otherwise, it must be a valid element to be read

        else:

            read_element += current_character

    # Returns the read list

    return read_list

# Defines a function to convert the values found in the string_toDict
# method

def convert_value(value):

    # Verifies if the value is not already a list or a dictionary itself

    if isinstance(value, dict) or isinstance(value, list):

        return value

    # Or a dictionary in the making

    elif (value[0]=="{" and value[-1]=="}") or (value[0]=='{' and value[-1
    ]=='}'):

        value = string_toDict(value)

    # Verifies if the value is not a list itself

    elif (value[0]=="[" and value[-1]=="]") or (value[0]=='[' and value[
    -1]==']'):

        value = string_toList(value)

    # Otherwise, try to convert the value to other format

    else:

        value = convert_string(value)

    return value

########################################################################
#                  Parsing strings to simpler formats                  #
########################################################################

# Defines a function to try to convert string variables to some useful
# other formats

def convert_string(string):

    # Tries to convert it to integer

    try:

        string = int(string)

    except:

        # Tries to convert to a float

        try:

            string = float(string)

        except:

            pass

    return string

# Defines a function to convert a float to a string substituting the dot
# by an underline

def float_toString(number):

    if isinstance(number, int):

        return str(number)

    # Converts the number to string

    number = str(number)

    new_number = ""

    # Checks for dots

    for i in range(len(number)):

        if number[i]==".":

            new_number += "_"

        elif number[i]!=",":

            new_number += number[i]

    return new_number

# Defines a function to convert a float to a string in scientific nota-
# tion

def float_to_scientific_notation(number, decimal_places=5):

    # Gets the logarithm

    log_number = np.log10(number)

    # Gets the floor value

    floor_log = np.floor(log_number)

    # Gets the rest

    rest = log_number-floor_log

    # Evaluates the corresponding part to the rest

    front_number = str(10.0**rest)[0:(decimal_places+2)]

    # Returns the whole string
    
    return front_number+"e"+str(int(floor_log)) 

########################################################################
#                               Testing                                #
########################################################################

def test_stringToDict():

    t = "[0.0]"

    t = {"1":1, 2:"dois", (1,"teste"): [0.0], "dicionário": {1: "numero", "dic": {"3": 3}, 4: "4"}}

    print(t)
    
    print(string_toDict(str(t)), "\n\n")

def test_scientific_notation():

    t = 0.0123

    q = 123000.0

    print(str(t)+"->"+float_to_scientific_notation(t))

    print(str(q)+"->"+float_to_scientific_notation(q))

#test_stringToDict()

#test_scientific_notation()