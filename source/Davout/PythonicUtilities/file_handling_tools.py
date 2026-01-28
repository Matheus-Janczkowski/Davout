# Routine to store methods for in-and-out processing, like reading and
# writing files

import copy

from collections import OrderedDict

from ..PythonicUtilities import string_tools

from ..PythonicUtilities import recursion_tools

from ..PythonicUtilities import path_tools

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

########################################################################
#                              txt files                               #
########################################################################

# Defines a function to write a list into a txt file

def list_toTxt(saved_list, file_name, add_extension=True, parent_path=
None):

    # Converts the list syntax into a string using a recursion loop 
    # through the list elements

    saved_string = ""

    saved_string = recursion_tools.recursion_listWriting(saved_list, 
    saved_string)

    # Takes out the last comma

    saved_string = saved_string[0:-1]

    # Adds the parent path if it is given

    if not (parent_path is None):

        file_name = path_tools.verify_path(parent_path, file_name)

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

        file_name = path_tools.verify_path(parent_path, file_name)

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

    read_list = string_tools.string_toList(saved_string)

    return read_list

# Defines a function to read a txt file and convert it into a dictionary

def txt_toDict(file_name, parent_path=None):

    # Reads as a list first

    read_list = txt_toList(file_name, parent_path=parent_path)

    print(read_list)

    # Converts to dictionary and returns it

    return list_toDict(read_list)

########################################################################
#                               Testing                                #
########################################################################

if __name__=="__main__":

    def general_test():

        t=[1,[2,[3,4], 5, [0,10]]]

        print("Original t:", t)

        indexes_counter = [1, 1]

        t = recursion_tools.recursion_listAppending(6, t, indexes_counter)

        list_toTxt(t, "test")

        print("Appended t:", t)

        t = txt_toList("test")

        print("Read t:    ", t, "\n\n")

    def test_indexBuilder():

        n_indexesList = [3]

        indexes = recursion_tools.get_indexesCombinations(n_indexesList)

        print(len(indexes), "combinations:", indexes, "\n\n")

        n_indexesList = [3,3]

        indexes = recursion_tools.get_indexesCombinations(n_indexesList)

        print(len(indexes), "combinations:", indexes, "\n\n")

        n_indexesList = [3,3,3]

        indexes = recursion_tools.get_indexesCombinations(n_indexesList)

        print(len(indexes), "combinations:", indexes, "\n\n")

    def test_nullListBuilder():

        dimensionality = [3]

        print("\ndimensionality: ", dimensionality)

        print(recursion_tools.initialize_listFromDimensions(
        dimensionality))

        dimensionality = [3,3]

        print("\ndimensionality: ", dimensionality)

        print(recursion_tools.initialize_listFromDimensions(
        dimensionality))

        dimensionality = (3,2,3)

        print("\ndimensionality: ", dimensionality)

        print(recursion_tools.initialize_listFromDimensions(
        dimensionality), "\n\n")

    def tensor_test():

        t=[[0.0,[[0.0, 0.0, 0.0],[0.0, 0.0], [0.0]]], [1.0,[[1.0,2.0],[3.0,4.0]]]]

        print("Original t:", t)

        list_toTxt(t, "test")

        t = txt_toList("test")

        print("Read t:    ", t, "\n\n")

    def dict_tensor():

        list_sample = [[0.0,[0.0,0.0,0.0]],[0.041666666666666664,[
        3.2514527856755824e-05,0.028971553093453284,-0.000196579173227914
        ]],[0.08333333333333333,[6.48960468681511e-05,
        0.057919688360002206,-0.0007827088154507426]],[0.125,[
        9.710502819223095e-05,0.08682111832092808,-0.001756757546712193]
        ],[0.16666666666666666,[0.0001291026796825274,
        0.11565280011836387,-0.003116022907317753]]]

        print(list_sample, "\n")

        print(list_toDict(list_sample), "\n\n")

    def test_stringToDict():

        t = "[0.0]"

        t = {"1":1, 2:"dois", (1,"teste"): [0.0], "dicionário": {1: "numero", "dic": {"3": 3}, 4: "4"}}

        print(t)
        
        print(string_tools.string_toDict(str(t)), "\n\n")

    def test_txtToListWithDict():

        q = {"1":1, 2:"dois", (1,"teste"): [0.0], "dicionário": {1: "numero", "dic": {"3": 3}, 4: "4"}}

        t = [[0.0,[[0.0, 0.0, 0.0],[0.0, 0.0], [0.0]]], [1.0,[[1.0,2.0],[3.0,4.0]]], [2.0, q]]

        print("Original t:", t)

        list_toTxt(t, "test")

        t = txt_toList("test")

        print("Read t:    ", t, "\n\n")

    general_test()

    test_indexBuilder()

    test_nullListBuilder()

    tensor_test()

    dict_tensor()

    test_stringToDict()

    test_txtToListWithDict()