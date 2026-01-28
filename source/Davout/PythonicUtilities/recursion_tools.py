# Routine to store functions and methods for recursion

import copy

from ..PythonicUtilities import programming_tools

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

if __name__=="__main__":

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

    test_indexBuilder()

    test_nullListBuilder()