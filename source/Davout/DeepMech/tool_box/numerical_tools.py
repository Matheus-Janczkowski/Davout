# Routine to store numerical tools for the use with TensorFlow

import numpy as np

import tensorflow as tf 

from ...PythonicUtilities import dictionary_tools

from ...PythonicUtilities import programming_tools

from ...PythonicUtilities.function_tools import verify_function_arguments

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

# Defines a class to get a string with the name of the regularizing 
# function to be used and return the live function in its call method. 
# If a dictionary is given, optional parameters may be taken

class BuildTensorflowMathExpressions:

    def __init__(self, dtype=tf.float32):
        
        # Saves the float type

        self.dtype = dtype

        # Automatically gets a dictionary of the methods defined in this
        # class

        self.available_methods = programming_tools.get_attribute(self,
        None, None, dictionary_of_methods=True, delete_init_key=True,
        reserved_methods=["__call__"])

        # Checks if the available methods have the necessary common ar-
        # guments

        for method_name, method_object in self.available_methods.items():

            # Gets the tensorflow function constructed by this method

            tensorflow_expression = method_object({"name": method_name})

            # Verifies the necessary arguments common to all tensorflow
            # expressions

            verify_function_arguments(tensorflow_expression, ["dimensi"+
            "on_axis"], "BuildTensorflowMathExpressions")

        # Defines a flag that tells if the output has unit norm along 
        # the desired axis. The default value for safety is false

        self.unit_norm_along_desired_axis = False

    # Defines the method that selects the expression name and converts 
    # it to a live expression

    def __call__(self, expression_name):

        """Builds a mathematical expression with tensorflow operations to
        facilitate differentiation. The argument is:
        
        expression_name: a string with one of the expressions are to be
        built with their default parameters; otherwise, a dictionary with
        key 'name' for the name of the function and other string keys for 
        the respective parameters """

        # Verifies if the expression name is just a string

        if isinstance(expression_name, str):

            # Turns it into a dictionary with the expression name as the 
            # value for the key 'name'

            expression_name = {"name": expression_name}

        # Verifies if it is not a dictionary

        elif not isinstance(expression_name, dict):

            raise TypeError("The argument 'expression_name' the method"+
            " __call__ in class 'BuildTensorflowMathExpressions' must "+
            "be a string or a dictionary. Currently, it is: "+str(
            expression_name))
        
        # Verifies is expression name has the key 'name'

        if not ("name" in expression_name):

            raise ValueError("The dictionary 'expression_name' in 'Bui"+
            "ldTensorflowMathExpressions' does not have the key 'name'"+
            ", which is obligatory. The given dictionary is:\n"+str(
            expression_name))

        # Verifies if the expression to be used is in the dictionary of
        # available methods

        if expression_name["name"] in self.available_methods:

            # Calls the method and returns it

            return self.available_methods[expression_name["name"]](
            expression_name)

        else:

            available_methods_names = ""

            for name in self.available_methods.keys():

                available_methods_names += "\n'"+str(name)+"'"

            raise ValueError("The name of the expression for the class"+
            " 'BuildTensorflowMathExpressions' must be one of the avai"+
            "lable options:"+available_methods_names+"\n\nThe given na"+
            "me was: '"+str(expression_name["name"])+"'")
        
    ####################################################################
    #                              Methods                             #
    ####################################################################

    # Defines a function to create a tensorflow expression for a smooth
    # absolute value

    def smooth_absolute_value(self, expression_name):

        # Verifies if the dictionary has keys that are not for this ex-
        # pression

        expression_name = dictionary_tools.verify_dictionary_keys(
        expression_name, {"name": "", "eps": tf.constant(1E-6, dtype=
        self.dtype)}, dictionary_location="at the builder of tensorflo"+
        "w math expressions", fill_in_keys=True)

        # Returns the smooth absolute value

        eps = expression_name["eps"]

        eps_squared = tf.square(eps)

        # Defines a flag that tells if the output has unit norm along 
        # the desired axis. The smooth absolute value is computed in a
        # component-wise fashion, thus it does NOT yield unit norm vec-
        # tors in the desired axis

        self.unit_norm_along_desired_axis = False

        # Defines the tensorflow expression

        @tf.function
        def smooth_abs(x, eps=eps, eps_squared=eps_squared, 
        dimension_axis=1):

            return tf.sqrt(tf.square(x)+eps_squared)-eps

        return smooth_abs

    # Defines a function that maps vectors in the n-dimensional real 
    # space to the positive orthant of this space. This transformation
    # is smooth almost everywhere (except for the direction along the 
    # negative identity line). This function can handle batched vectors
    # in a (n_samples, n_dimensions) tensor or a (n_dimensions, 
    # n_samples) tensor. Thus, the axis parameter tells which index of
    # the incoming tensor contains the dimensionality of the real space

    def contractive_positive_orthant_mapping(self, expression_name):

        # Verifies if the dictionary has keys that are not for this ex-
        # pression

        expression_name = dictionary_tools.verify_dictionary_keys(
        expression_name, {"name": "", "eps": tf.constant(1E-12, dtype=
        self.dtype)}, dictionary_location="at the builder of tensorflo"+
        "w math expressions", fill_in_keys=True)

        # Precomputes some useful tensors and constants

        eps = expression_name["eps"]

        eps_squared = tf.square(eps)

        constant_half = tf.cast(0.5, dtype=self.dtype)

        constant_one = tf.cast(1.0, dtype=self.dtype)

        constant_two = tf.cast(2.0, dtype=self.dtype)

        # Defines a flag that tells if the output has unit norm along 
        # the desired axis. This contractive mapping does yield unit 
        # norm vectors in the desired axis

        self.unit_norm_along_desired_axis = True

        # Defines the function

        @tf.function
        def contractive_mapping(w_vectors_tensor, epsilon=eps, 
        epsilon_squared=eps_squared, dimension_axis=1, dtype=self.dtype,
        constant_one=constant_one, constant_two=constant_two, 
        constant_half=constant_half):

            # Gets the space dimension
            
            space_dimension = tf.shape(w_vectors_tensor)[dimension_axis]
        
            # Gets the dot product of the given u vector by the positive 
            # identity vector
        
            dimensionality_reciprocal_square_root = tf.math.rsqrt(
            tf.cast(space_dimension, dtype=dtype))
        
            u_dot_d = (tf.reduce_sum(w_vectors_tensor, axis=
            dimension_axis)*dimensionality_reciprocal_square_root)
        
            # Computes the square norm of u
        
            u_dot_u = tf.reduce_sum(tf.square(w_vectors_tensor), axis=
            dimension_axis)
        
            # Computes beta, that is the argument of the arccosine func-
            # tion
        
            beta = u_dot_d*tf.math.rsqrt(u_dot_u+epsilon_squared)
        
            # Evaluates the common denominator of the coefficients to 
            # generate a vector perpendicular to the identity line in 
            # the subspace spanned by u and d
            
            denominator = tf.math.sqrt(u_dot_u-tf.square(u_dot_d)+
            epsilon_squared)
        
            # Computes the coefficients of the combination of the given 
            # vector u and of the identity vector d. If u is colinear to 
            # the identity, makes the first coefficient 0 and the second 
            # 1
        
            coefficient_u = tf.expand_dims(tf.math.reciprocal(
            denominator), axis=dimension_axis)
        
            coefficient_d = -(coefficient_u*u_dot_d)
        
            # Builds the vector colinear to the identity line
        
            d_vector = tf.expand_dims(tf.reshape(
            dimensionality_reciprocal_square_root, [1]), axis=dimension_axis)
        
            # Gets the orthonormal vector to the identity line. The sha-
            # pe of the coefficients must be expanded to include the di-
            # mension of the space, that was lost during summation ope-
            # rations
        
            c_vector = (coefficient_u*w_vectors_tensor)+(coefficient_d*
            d_vector)
        
            # Evaluates the inequation to determine the mu factor
        
            inequation_numerator = tf.abs(c_vector)-c_vector
        
            inequation_denominator = (constant_two*(d_vector-c_vector))
        
            mu_inequation = tf.math.divide_no_nan(inequation_numerator, 
            inequation_denominator)
        
            # Gets the maximum mu factor and the corresponding boundary 
            # vector
        
            mu = tf.reduce_max(mu_inequation, axis=dimension_axis)
        
            # Gets the relative position of the vector u with respect to 
            # the identity line. The relative position must be 1 when it 
            # lies on the identity line and 0 when it lies on the nega-
            # tive identity line. For this purpose, we use beta, since 
            # it is contained within the interval [-1,1]
        
            ratio = constant_half*(beta+constant_one)
        
            # Gets the mu corresponding to the final vector inside the 
            # positive orthant
        
            final_mu = tf.where(ratio<epsilon, tf.ones_like(ratio), (mu*
            (constant_one-ratio))+ratio)
        
            denominator = tf.math.rsqrt(tf.square(final_mu)+tf.square((
            constant_one-final_mu)))
        
            # Expands the shape of the final mu coefficient and of the 
            # common denominator to account for the space dimension that 
            # was lost during summation operations
        
            final_mu = tf.expand_dims(final_mu, axis=dimension_axis)
        
            denominator = tf.expand_dims(denominator, axis=
            dimension_axis)
        
            # Constructs the final vector as a linear interpolation of 
            # the identity line and the orthonormal vector c. Returns an 
            # array (n_samples, space_dimension)
        
            return ((final_mu*denominator)*d_vector)+(((constant_one-
            final_mu)*denominator)*c_vector)

        return contractive_mapping