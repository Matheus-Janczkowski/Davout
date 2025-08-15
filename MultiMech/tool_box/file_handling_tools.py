# Routine to store methods for in-and-out processing, like reading and
# writing files

import os

import copy

from collections import OrderedDict

from pathlib import Path

from ..tool_box import programming_tools

########################################################################
#                            Parsing tools                             #
########################################################################

# Defines a function to convert a list of values and a list of names in-
# to a lists of sublists of key and values, much like a dictionary

def named_list(values_dictionary=None, values_list=None, keys_list=None):

    # Initializes the list of pairs

    pairs_list = []

    if (not (keys_list is None)) and (not (values_list is None)):

        # Verifies if the keys and values have the same size

        if len(values_list)!=len(keys_list):

            raise ValueError("The list of values has "+str(len(
            values_list))+" elements whereas the list of keys (names) "+
            "has "+str(len(keys_list)))

        # Adds the pairs as they are ordered

        for i in range(len(values_list)):

            pairs_list.append([keys_list[i], values_list[i]])

    else:

        # Verifies if the dictionary at least is a dictionary

        if not isinstance(values_dictionary, dict):

            raise TypeError("The list of values and/or the list of key"+
            "s are None, but the expected dictionary is not a dictiona"+
            "ry either")

        for key, value in values_dictionary.items():

            pairs_list.append([key, value])

    # Returns the list

    return pairs_list

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

# Defines a function to convert a string to a list

def string_toList(saved_string):

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

        if current_character=="," or current_character=="]":

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

                read_list = recursion_listAppending(read_element, 
                read_list, indexes_list, -len(indexes_list))

                """
                print("\nUpdated list:", read_list, "element_counterSu"+
                "bList="+str(element_counterSubList)+", indexes_list="+
                str(indexes_list)+"\n")
                """

                # Clears the read element

                read_element = ""

            # If the character is a comma, updates the number o elements
            # in the current sublist

            if current_character==",":

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
            
            read_list = recursion_listAppending([], read_list, 
            indexes_list, -len(indexes_list))

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

# Defines a function to convert a list of lists of the format [[t0, A0],
# [t1, A1], ..., [tn, An]] into a dictionary, where ti are the keys and
# Ai are the values

def list_toDict(original_list):

    # Checks if the list has the required format

    if len(original_list)==0:

        raise ValueError("The original list is empty, so it cannot be "+
        "translated to a dictionary")
    
    # Initializes the dictionary

    list_dictionary = OrderedDict()

    # Iterates through the list

    for sublist in original_list:

        # Checks if the length of the list is at least two

        if len(sublist)<2:

            raise KeyError("Each sublist to be translated to a pair ke"+
            "y-value must have at least a length of 2")
        
        # Checks if the first element is not a list

        if (not (isinstance(sublist[0], float) or isinstance(sublist[0], 
        str) or isinstance(sublist[0], tuple))):
            
            raise KeyError("The first element of the sublist must be a"+
            " number, a string, or a tuple to be translated into a key")
        
        # After the last checks, pairs the key to the value

        list_dictionary[sublist[0]] = copy.deepcopy(sublist[1])

    # Returns the dictionary

    return list_dictionary

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

########################################################################
#                              Path tools                              #
########################################################################

# Defines a function to verify if a path exists or not. If not, create
# it

def verify_path(parent_path, file_name):

    if parent_path is None:

        return file_name

    # Checks if the parent path exists

    if not os.path.exists(parent_path):

        # Initializes a list of directories to create

        directories = [""]

        # Iterates through the path

        for i in range(len(parent_path)):

            # If the character is a bar, this means the last directory
            # name has been finished

            if parent_path[-(i+1)]=="/":

                # Verifies if the last directory is not empty

                if directories[-1]!="":

                    directories.append("")

            # Otherwise, saves the characters

            else:

                directories[-1] = parent_path[-(i+1)]+directories[-1]

        # Checks if the last saved directory is empty

        if directories[-1]=="":

            directories = directories[0:-1]

        # Initializes the path

        path = ""

        # Iterates through the directories

        for i in range(len(directories)):

            # Appends this bit of directory to the path

            path += "//"+directories[-(i+1)]

            # Checks if this path exists

            if not os.path.exists(path):

                # Creates the directory

                os.mkdir(path)

    # Joins everything together

    return parent_path+"//"+file_name

# Defines a function to take out the termination file name of a string

def take_outFileNameTermination(file_name, get_termination=False):

    # Initializes the new name

    clean_fileName = ""

    # Initializes the termination

    termination = ""

    termination_reading = False

    # Iterates through the file name

    for character in file_name:

        if character=="." or character=='.':

            if not get_termination:

                break 

            else:

                termination_reading = True

        else:

            if termination_reading:

                termination += character

            else:

                clean_fileName += character

    # Returns the file name without the termination

    if get_termination:

        return clean_fileName, termination
    
    else:

        return clean_fileName
    
# Defines a function to get the path to a file

def get_parent_path_of_file(file="__file__", path_bits_to_be_excluded=1):

    if file=="__file__":

        raise NameError("To get the parent path of a file, you should "+
        "type file=file, where file is an imported file; or you should"+
        " type file=__file__ to get the parent path to the current file")
    
    # Gets a list of the bits of the whole path
    
    current_path = Path(os.path.abspath(file)).parts

    # Joins them again without the last bit, which is the name of the 
    # file itself

    return Path(*current_path[0:-path_bits_to_be_excluded])

########################################################################
#                              txt files                               #
########################################################################

# Defines a function to write a list into a txt file

def list_toTxt(saved_list, file_name, add_extension=True, parent_path=
None):

    # Converts the list syntax into a string using a recursion loop 
    # through the list elements

    saved_string = ""

    saved_string = recursion_listWriting(saved_list, saved_string)

    # Takes out the last comma

    saved_string = saved_string[0:-1]

    # Adds the parent path if it is given

    if not (parent_path is None):

        file_name = verify_path(parent_path, file_name)

    # Saves the string into a txt file

    if add_extension:

        file_name = file_name+".txt"

    txt_file = 0

    try:

        txt_file = open(file_name, "w")

    except:

        raise FileNotFoundError("The path to write the file '"+str(
        file_name)+"' does not exist. It cannot be used to write the l"+
        "ist")

    txt_file.write(saved_string)

    txt_file.close()

# Defines a function to read a list from a txt file

def txt_toList(file_name, parent_path=None):

    # Adds the parent path if it is given

    if not (parent_path is None):

        file_name = verify_path(parent_path, file_name)

    # Reads the txt file

    saved_string = ""

    try:

        with open(file_name+".txt", "r") as infile:

            saved_string = infile.read()

    except:

        raise FileNotFoundError("The file "+file_name+".txt was not fo"+
        "und while evaluating txt_toList method in file_handling_tools"+
        ".py\n")

    # Converts the string to a list

    read_list = string_toList(saved_string)

    return read_list

# Defines a function to read a txt file and convert it into a dictionary

def txt_toDict(file_name, parent_path=None):

    # Reads as a list first

    read_list = txt_toList(file_name, parent_path=parent_path)

    print(read_list)

    # Converts to dictionary and returns it

    return list_toDict(read_list)

########################################################################
#                           Recursion tools                            #
########################################################################

# Defines a function to recursively access the elements of a list and
# write the list syntax into a string

def recursion_listWriting(accessed_list, saved_string):

    # Adds the initial bracket as THIS list has just begun

    saved_string += "["

    # Iterates through the element of the list

    for i in range(len(accessed_list)):

        # Accesses the i-th element

        accessed_element = accessed_list[i]

        # Tests if the element is another list. If it is, recalls the 
        # recursion function to start the process all over again

        if isinstance(accessed_element, list):

            saved_string = recursion_listWriting(accessed_element, 
            saved_string)

        # Otherwise, returns the value, as the stopping criterion of the
        # recursion is precisely when the element is no longer another 
        # list

        elif isinstance(accessed_element, str):

            # Updates the string

            saved_string += "'"+str(accessed_element)+"',"

        else:

            # Updates the string

            saved_string += str(accessed_element)+","

    # Takes out the last character, for it is the last comma and; then, 
    # adds the bracket as THIS list has ended

    if saved_string[-1]==",":

        saved_string = saved_string[0:-1]

    saved_string += "],"

    # Returns the string

    return saved_string

# Defines a function to recursively access a list using a list of inde-
# xes to append an element to the list at the last index

def recursion_listAppending(added_element, accessed_list, indexes_list, 
indexes_counter=None):
    
    """print("Added element:", added_element)

    print("Accessed list:", accessed_list)

    print("Indexes list:", indexes_list)

    print("Indexes counter:", indexes_counter)"""

    # Verifies if it was not given

    if indexes_counter==None:

        indexes_counter = -len(indexes_list)

    # Verifies whether the index counter is negative

    elif indexes_counter>0:

        indexes_counter = -1*indexes_counter

    # The indexes counter is a counter for the list of indexes, it is 
    # negative. This works well because at each index of the accessed 
    # list, the counter is added 1, thus, when it is zero, it sinalizes
    # that the position has been found, where the new element must be 
    # added

    if indexes_counter==0:

        accessed_list.append(added_element)
    
    # Otherwise, gets the list at the desired index, sends it further to
    # the recursion and brings it back rightfully appended

    else:

        sub_list = accessed_list[indexes_list[indexes_counter]]

        # Updates the index counter

        indexes_counter += 1

        # Recovers the appended sub_list and allocates it into the ori-
        # ginal list

        accessed_list[indexes_list[indexes_counter-1]] = recursion_listAppending(
        added_element, sub_list, indexes_list, indexes_counter=
        indexes_counter)

    # Returns the corrected list

    return accessed_list

# Defines a function to get the combinations of possible indexes given
# a list of the number of possible indexes per index

def get_indexesCombinations(n_indexesList):

    return get_indexesCombinationationsRecursion(n_indexesList, 
    index_combinations=[])

# Defines a function to do the recursion activity for the construction 
# of the list of possible index combinations

def get_indexesCombinationationsRecursion(n_indexesList, 
index_combination=[], index_combinations=[], component=0):

    if len(index_combination)==0:

        index_combination = [0 for i in range(len(n_indexesList))]

    # Verifies if the component if the component is larger than the 
    # length of the list

    if component>=len(n_indexesList):

        # Adds another combination

        index_combinations.append(copy.deepcopy(index_combination))

        return index_combinations
    
    # Otherwise, iterates through the indexes

    else:

        for i in range(n_indexesList[component]):

            # Adds this index

            index_combination[component] = i+0

            # Sends the index combination down the line

            index_combinations = get_indexesCombinationationsRecursion(
            n_indexesList, index_combination=index_combination, 
            index_combinations=index_combinations, component=(component+
            1))

    return index_combinations

# Defines a function to build a list full of zeros given a list of di-
# mensions

@programming_tools.optional_argumentsInitializer({'zeros_list': lambda: 
[]})

def initialize_listFromDimensions(dimensions_list, zeros_list=None):

    print(dimensions_list, zeros_list)

    # If the list has no dimensions left, returns the list of zeros

    if len(dimensions_list)==1:

        for i in range(dimensions_list[0]):

            zeros_list.append(0.0)

        return zeros_list
    
    # Otherwise populates the dimension with zeros

    for i in range(dimensions_list[0]):

        print("for:", zeros_list)

        zeros_list.append(initialize_listFromDimensions(
        dimensions_list[1:]))

    return zeros_list

########################################################################
#                               Testing                                #
########################################################################

def general_test():

    t=[1,[2,[3,4], 5, [0,10]]]

    print("Original t:", t)

    indexes_counter = [1, 1]

    t = recursion_listAppending(6, t, indexes_counter)

    list_toTxt(t, "test")

    print("Appended t:", t)

    t = txt_toList("test")

    print("Read t:    ", t, "\n\n")

def test_indexBuilder():

    n_indexesList = [3]

    indexes = get_indexesCombinations(n_indexesList)

    print(len(indexes), "combinations:", indexes, "\n\n")

    n_indexesList = [3,3]

    indexes = get_indexesCombinations(n_indexesList)

    print(len(indexes), "combinations:", indexes, "\n\n")

    n_indexesList = [3,3,3]

    indexes = get_indexesCombinations(n_indexesList)

    print(len(indexes), "combinations:", indexes, "\n\n")

def test_nullListBuilder():

    dimensionality = [3]

    print("\ndimensionality: ", dimensionality)

    print(initialize_listFromDimensions(dimensionality))

    dimensionality = [3,3]

    print("\ndimensionality: ", dimensionality)

    print(initialize_listFromDimensions(dimensionality))

    dimensionality = (3,2,3)

    print("\ndimensionality: ", dimensionality)

    print(initialize_listFromDimensions(dimensionality), "\n\n")

def tensor_test():

    t=[[0.0,[[0.0, 0.0, 0.0],[0.0, 0.0], [0.0]]], [1.0,[[1.0,2.0],[3.0,4.0]]]]

    print("Original t:", t)

    list_toTxt(t, "test")

    t = txt_toList("test")

    print("Read t:    ", t, "\n\n")

def dict_tensor():

    list_sample = [[0.0,[0.0,0.0,0.0]],[0.041666666666666664,[3.2514527856755824e-05,0.028971553093453284,-0.000196579173227914]],[0.08333333333333333,[6.48960468681511e-05,0.057919688360002206,-0.0007827088154507426]],[0.125,[9.710502819223095e-05,0.08682111832092808,-0.001756757546712193]],[0.16666666666666666,[0.0001291026796825274,0.11565280011836387,-0.003116022907317753]]]

    print(list_sample, "\n")

    print(list_toDict(list_sample), "\n\n")

def test_stringToDict():

    t = "[0.0]"

    t = {"1":1, 2:"dois", (1,"teste"): [0.0], "dicionário": {1: "numero", "dic": {"3": 3}, 4: "4"}}

    print(t)
    
    print(string_toDict(str(t)), "\n\n")

def test_txtToListWithDict():

    q = {"1":1, 2:"dois", (1,"teste"): [0.0], "dicionário": {1: "numero", "dic": {"3": 3}, 4: "4"}}

    t = [[0.0,[[0.0, 0.0, 0.0],[0.0, 0.0], [0.0]]], [1.0,[[1.0,2.0],[3.0,4.0]]], [2.0, q]]

    print("Original t:", t)

    list_toTxt(t, "test")

    t = txt_toList("test")

    print("Read t:    ", t, "\n\n")

#general_test()

#test_indexBuilder()

#test_nullListBuilder()

#tensor_test()

#dict_tensor()

#test_stringToDict()

#test_txtToListWithDict()