# Routine to store numerical tools for the use with TensorFlow

import numpy as np

import tensorflow as tf 

########################################################################
#                            Linear Algebra                            #
########################################################################

# Defines a function to convert a scipy sparse matrix to a TensorFlow 
# sparse tensor. The provided sparse matrix can be a matrix constructed
# using scipy formats, or it can be a list of those formats. If a list
# is provided, then a big block-diagonal tensor will be created

def scipy_sparse_to_tensor_sparse(sparse_matrix, number_type="float32",
reorder_indices=True, block_multiplication=True):

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