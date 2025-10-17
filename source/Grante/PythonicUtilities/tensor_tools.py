# Routine to store some functions for tensors

########################################################################
#                       Useful and famous tensors                      #
########################################################################

# Defines a function to get the Kronecker's delta

def kroneckers_delta(i,j):

    if i==j:

        return 1.0
    
    else:

        return 0.0

# Defines a function to get the third order permutation tensor in compo-
# nents

def third_order_permutation_tensor_components(i,j,k):

    # Gets the set

    index_set = str(i)+str(j)+str(k)

    # Checks for positive component

    if index_set in ["123", "231", "312"]:

        return 1.0
    
    elif index_set in ["132", "213", "321"]:

        return -1.0
    
    else:

        return 0.0