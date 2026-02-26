# Routine to store methods that wrap interisting numpy functionalities

import numpy as np

########################################################################
#                             Array search                             #
########################################################################

# Defines a function to get rows and their indices from a numpy array gi-
# ven the expected values of a set of columns

def get_rows_given_column_values(array: np.ndarray, inspected_columns:
list, target_values: (list | np.ndarray)):

    """
    Function to get rows and their indices from a numpy array whose given
    subset of columns equals an array of values. Returns rows and, then,
    the rows indices.
    
    array: 2D numpy array that is going to be searched
    
    inspected_columns: list of indices of columns that will be inspected
    for the searched values
    
    target_values: list or array of values searched in those columns"""

    # Verifies if the array is a 2D numpy array

    if not isinstance(array, np.ndarray):

        raise ValueError("'array' in 'get_rows_given_column_values' mu"+
        "st be a 2D numpy array. Currently its type is: "+str(type(array
        )))

    elif len(array.shape)!=2:

        raise ValueError("'array' in 'get_rows_given_column_values' mu"+
        "st be a 2D numpy array. Currently its shape is: "+str(
        array.shape))

    # If target values is None, returns the proper array

    if target_values is None:

        return array, np.arange(array.shape[0], dtype=int)

    # Verifies if inspected columns is not a list

    if not isinstance(inspected_columns, list):

        raise TypeError("'inspected_columns' in 'get_rows_given_column"+
        "_values' must be a list with the indices of the columns to be"+
        " inspected. Currenty, it is:\n"+str(inspected_columns)+"\nwho"+
        "se type is: "+str(type(inspected_columns)))
    
    # Verifies if target values is a list

    if isinstance(target_values, list):

        # Converts to an array

        target_values = np.asarray(target_values)

    # If not a numpy array, throws an error

    if not isinstance(target_values, np.ndarray):

        raise TypeError("'target_values' in 'get_rows_given_column_val"+
        "ues' must be a list or a numpy array with the values of the c"+
        "olumns to be inspected. Currently, it is:\n"+str(target_values
        )+"\nwhose type is: "+str(type(target_values)))
    
    # Verifies if target values and inspected columns have the same 
    # length

    elif len(inspected_columns)!=target_values.shape[0]:

        raise IndexError("In 'get_rows_given_column_values', 'target_v"+
        "alues' has "+str(target_values.shape[0])+" elements, whereas "+
        "'inspected_columns' has "+str(len(inspected_columns))+". They"+
        " must have the same size")

    # Creates a mask to tell which rows of the array have the values cor-
    # responding to the inspected columns equal to the target values

    mask = np.all(array[:,inspected_columns]==target_values, axis=1)

    # Gets the indices of the rows who comply

    row_indices = np.where(mask)[0]

    # And the row values themselves

    rows = array[row_indices] 

    return rows, row_indices