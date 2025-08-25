# Routine to store numerical tools for the use with TensorFlow

import numpy as np

import tensorflow as tf 

########################################################################
#                            Linear Algebra                            #
########################################################################

# Defines a function to convert a scipy sparse matrix to a TensorFlow 
# sparse tensor

def scipy_sparse_to_tensor_sparse(sparse_matrix, number_type="float32"):

    # Gets the COOrdinate sparse format

    sparse_matrix = sparse_matrix.tocoo(copy=False)

    # Gets the indices of the non-zero positions, into a nx2 list, where
    # n is the number of non-zero positions

    non_zero_indices = np.vstack((sparse_matrix.row, sparse_matrix.col)
    ).T

    # Returns the TensorFlow sparse tensor in a specific number type

    return tf.SparseTensor(indices=non_zero_indices, values=
    sparse_matrix.data.astype(getattr(np, number_type)), dense_shape=
    sparse_matrix.shape)