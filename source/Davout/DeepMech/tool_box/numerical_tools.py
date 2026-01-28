# Routine to store numerical tools for the use with TensorFlow

import numpy as np

import tensorflow as tf 

from ...PythonicUtilities import dictionary_tools

########################################################################
#                            Linear Algebra                            #
########################################################################

# Defines a function to convert a scipy sparse matrix to a TensorFlow 
# sparse tensor. The provided sparse matrix can be a matrix constructed
# using scipy formats, or it can be a list of those formats. If a list
# is provided, then a big block-diagonal tensor will be created

def scipy_sparse_to_tensor_sparse(sparse_matrix, number_type="float32",
reorder_indices=True, block_multiplication=True, n_samples=None):

    # Verifies if the required type for the data is available

    number_types = ["float32", "float64", "int32", "int64"]

    dtype = None

    if number_type in number_types:

        # Gets the type from numpy

        dtype = getattr(np, number_type)

    else:

        raise TypeError("The 'number_type' argument in scipy_to_tensor"+
        "_sparse function is not one of the following options: "+str(
        number_types))

    # Tests if a block-diagonal tensor is to be created

    if isinstance(sparse_matrix, list) and block_multiplication:

        # Initializes the lists for the indices and for the values of the
        # non-zero positions of the sparse matrices inside the list 

        non_zero_indices = []

        non_zero_values = []

        # Initializes the number of dimensions that have already been ex-
        # plored

        explored_rows = 0

        explored_columns = 0

        # Iterates through the sparse matrices inside the list

        for sub_matrix in sparse_matrix:

            # Enforces the COOrdinate sparse format

            sub_matrix = sub_matrix.tocoo(copy=False)

            # Gets the indices of the non-zero positions and sums the 
            # number of dimensions of the previously tracked matrices

            indices = (np.vstack((sub_matrix.row+explored_rows, 
            sub_matrix.col+explored_columns)).T.astype(np.int64))
            
            # Gets the values of these positions

            values = sub_matrix.data.astype(dtype)

            # Adds to the indices and values arrays

            non_zero_indices.append(indices)

            non_zero_values.append(values)

            # Updates the number of explored dimensions

            explored_rows += sub_matrix.shape[0]

            explored_columns += sub_matrix.shape[1]

        # Returns the TensorFlow sparse tensor in a specific number type.
        # If the tensor is to be reordered in lexigrographic indices, 
        # which is useful for TensorFlow operations

        if reorder_indices:

            return tf.sparse.reorder(tf.SparseTensor(indices=np.concatenate(
            non_zero_indices, axis=0), values=np.concatenate(non_zero_values, 
            axis=0), dense_shape=(explored_rows, explored_columns)))
        
        else:

            return tf.SparseTensor(indices=np.concatenate(
            non_zero_indices, axis=0), values=np.concatenate(
            non_zero_values, axis=0), dense_shape=(explored_rows, 
            explored_columns))
        
    elif (not (n_samples is None)) and block_multiplication:
        
        # Gets the COOrdinate sparse format

        sparse_matrix = sparse_matrix.tocoo(copy=False)

        # Gets the indices of the non-zero positions, into a nx2 list, 
        # where n is the number of non-zero positions

        basic_indices = np.vstack((sparse_matrix.row, sparse_matrix.col)
        ).T.astype(np.int64)
            
        # Gets the values of these positions

        values = sparse_matrix.data.astype(dtype)

        # Initializes the lists for the indices and for the values of the
        # non-zero positions of the sparse matrices inside the list 

        non_zero_indices = []

        non_zero_values = []

        # Gets the number of dimensions

        n_dimensions = sparse_matrix.shape[0]

        # Iterates through the sparse matrices inside the list

        for i in range(n_samples):

            # Adds to the indices and values arrays

            non_zero_indices.append(basic_indices+(n_dimensions*i))

            non_zero_values.append(values)

        # Returns the TensorFlow sparse tensor in a specific number type.
        # If the tensor is to be reordered in lexigrographic indices, 
        # which is useful for TensorFlow operations

        if reorder_indices:

            return tf.sparse.reorder(tf.SparseTensor(indices=np.concatenate(
            non_zero_indices, axis=0), values=np.concatenate(non_zero_values, 
            axis=0), dense_shape=(n_dimensions*n_samples, n_dimensions
            *n_samples)))
        
        else:

            return tf.SparseTensor(indices=np.concatenate(
            non_zero_indices, axis=0), values=np.concatenate(
            non_zero_values, axis=0), dense_shape=(n_dimensions*
            n_samples, n_dimensions*n_samples))

    # Tests if a tensor with a third index is to be created. The first 
    # index is to get the sparse matrix of the corresponding batch

    elif isinstance(sparse_matrix, list):

        # Iterates through the sparse matrices inside the list

        for i in range(len(sparse_matrix)):

            # Enforces the COOrdinate sparse format

            sub_matrix = sparse_matrix[i].tocoo(copy=False)

            # Gets the indices of the non-zero positions

            indices = (np.vstack((sub_matrix.row, sub_matrix.col)
            ).T.astype(np.int64))

            # Adds the TensorFlow sparse tensor in a specific number ty-
            # pe. If the tensor is to be reordered in lexigrographic in-
            # dices, which is useful for TensorFlow operations

            if reorder_indices:

                sparse_matrix[i] = tf.sparse.reorder(tf.SparseTensor(
                indices=indices, values=sub_matrix.data.astype(dtype), 
                dense_shape=sub_matrix.shape))
            
            else:

                sparse_matrix[i] = tf.SparseTensor(indices=indices, 
                values=sub_matrix.data.astype(dtype), dense_shape=
                sub_matrix.shape)

        return sparse_matrix

    else:

        # Gets the COOrdinate sparse format

        sparse_matrix = sparse_matrix.tocoo(copy=False)

        # Gets the indices of the non-zero positions, into a nx2 list, 
        # where n is the number of non-zero positions

        non_zero_indices = np.vstack((sparse_matrix.row, 
        sparse_matrix.col)).T.astype(np.int64)

        # Returns the TensorFlow sparse tensor in a specific number type

        if reorder_indices:

            return tf.sparse.reorder(tf.SparseTensor(indices=
            non_zero_indices, values=sparse_matrix.data.astype(dtype), 
            dense_shape=sparse_matrix.shape))
        
        else:

            return tf.SparseTensor(indices=non_zero_indices, values=
            sparse_matrix.data.astype(dtype), dense_shape=
            sparse_matrix.shape)
        
########################################################################
#                        Regularizing functions                        #
########################################################################

# Defines a function to get a string with the name of the regularizing 
# function to be used and return the live function. If a dictionary is
# given, optional parameters may be taken

def build_tensorflow_math_expressions(expression_name, dtype=tf.float32
):

    """Builds a mathematical expression with tensorflow operations to
    facilitate differentiation. The argument is:
    
    expression_name: a string with one of the expressions are to be
    built with their default parameters; otherwise, a dictionary with
    key 'name' for the name of the function and other string keys for 
    the respective parameters """

    # Verifies if the expression name is just a string

    if isinstance(expression_name, str):

        # Turns it into a dictionary with the expression name as the va-
        # lue for the key 'name'

        expression_name = {"name": expression_name}

    # Verifies if it is not a dictionary

    elif not isinstance(expression_name, dict):

        raise TypeError("The argument 'expression_name' for function '"+
        "build_tensorflow_math_expressions' must be a string or a dict"+
        "ionary")

    # Verifies if the expression to be used is the smooth absolute value

    if expression_name["name"]=="smooth absolute value":

        # Verifies if the dictionary has keys that are not for this ex-
        # pression

        expression_name = dictionary_tools.verify_dictionary_keys(
        expression_name, {"name": "", "eps": tf.constant(1E-6, dtype=
        dtype)}, dictionary_location="at the builder of tensorflow mat"+
        "h expressions", fill_in_keys=True)

        # Returns the smooth absolute value

        eps = expression_name["eps"]

        eps_squared = tf.square(eps)

        #def smooth_abs(x, eps=eps, eps_squared=eps_squared):

            #print(x.shape, eps.shape, eps_squared.shape)

            #print(x.dtype, eps.dtype, eps_squared.dtype)

            #return tf.sqrt(tf.square(x)+eps_squared)-eps

        #return smooth_abs

        return lambda x: tf.sqrt(tf.square(x)+eps_squared)-eps

    else:

        raise KeyError("The name of the expression for the function 'b"+
        "uild_tensorflow_math_expressions' must be one of the availabl"+
        "e options: 'smooth absolute value'")