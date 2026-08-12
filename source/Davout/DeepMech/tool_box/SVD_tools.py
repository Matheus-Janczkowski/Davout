# Routine to store methods to work with the singular value decomposition
# (SVD)

import tensorflow as tf

import numpy as np

########################################################################
#             Assembly of chains of Householder reflectors             #
########################################################################

# Defines a function to create the tensors of Householder parameters 
# (degrees of freedom) and of indices

def create_householder_tensors(input_dimension, output_dimension,
float_type, integer_type, householder_epsilon_squared):

    # Gets the rank of the corresponding orthogonal transformation

    rank = min(input_dimension, output_dimension)

    # Counts the number of degrees of freedom (DOFs) of the left-ortho-
    # gonal matrix to be created using CHR

    n_dofs = int(0.5*(rank*((2*input_dimension)-rank-1)))

    # Creates a list of tuples of the indices to recreate the vectors of
    # DOFs of the Householder vectors

    householder_first_index = []

    householder_length = []

    householder_number_of_leading_zeros = []

    parameters_counter = 0

    for i in range(min(rank, input_dimension-1)):

        # Appends the index of the first parameter for the corresponding
        # Householder vector; the number of parameters necessary for 
        # this vector (length), and the number of leading zeros.
        # Starts with the last Householder reflector since it will be
        # the first to multiply any vector to the right of the corres-
        # ponding orthogonal matrix. Appends to the simple flat tensors

        householder_first_index.append(parameters_counter)

        householder_length.append(input_dimension-i-1)

        householder_number_of_leading_zeros.append(i)

        # Updates the parameter counter

        parameters_counter += input_dimension-i-1

    # Converts the flat vectors to flat tensors

    householder_first_index = tf.constant(householder_first_index, dtype
    =tf.as_dtype(integer_type))

    householder_length = tf.constant(householder_length, dtype=
    tf.as_dtype(integer_type))

    householder_number_of_leading_zeros = tf.constant(
    householder_number_of_leading_zeros, dtype=tf.as_dtype(integer_type))

    # Creates a variable with the Householder DOFs

    householder_parameters = tf.Variable(np.random.randn((
    n_dofs)), dtype=tf.as_dtype(float_type))

    # Defines the Householder epsilon for the regularization of the
    # first non-zero component of each Householder vector

    householder_epsilon_squared = tf.constant(
    householder_epsilon_squared, dtype=tf.as_dtype(float_type))

    return (householder_parameters, householder_first_index, 
    householder_length, householder_number_of_leading_zeros,
    householder_epsilon_squared, n_dofs, rank)

########################################################################
#           Application of chains of Householder reflectors            #
########################################################################

# Defines a function to parse a single Householder vector from a vector
# of all Householder vectors to construct an orthogonal matrix. The in-
# coming vector is a tensor [0.5*(m_rank*((2*n_rows)-m_rank-1))], where 
# m_rank is the number of columns of the final orthogonal matrix, and 
# n_rows is the number of rows. Receives the index of the Householder 
# reflector in the Householder chain to collect the corresponding House-
# holder vector

def get_householder_vector_from_parameters(householder_first_index,
householder_length, householder_number_of_leading_zeros,
householder_parameters, householder_reflector_index, 
householder_epsilon_squared):
    
    # Gets the first and last indices of the Householder vector, then
    # the vector that will become the Householder vector

    initial_index = householder_first_index[householder_reflector_index] 

    length = householder_length[householder_reflector_index]

    number_of_leading_zeros = householder_number_of_leading_zeros[
    householder_reflector_index]

    raw_vector = tf.slice(householder_parameters, begin=tf.expand_dims(
    initial_index, axis=0), size=tf.expand_dims(length, axis=0))

    # Computes the average of the raw vector

    average_raw_vector = tf.reduce_mean(raw_vector)

    # Concatenates the non-free component of the Householder vector to
    # the first position

    unnormalized_first_component = tf.sqrt(tf.square(average_raw_vector
    )+householder_epsilon_squared)

    appended_raw_vector = tf.concat([unnormalized_first_component[None], 
    raw_vector], axis=0)

    # Rescales the raw vector to have unit norm, then adds the leading
    # zeros and returns it

    alpha = tf.math.rsqrt(tf.reduce_sum(tf.square(appended_raw_vector)))

    # Constructs the padding tensor with the leading zeros dynamically 
    # for graph execution

    paddings = tf.stack([tf.stack([number_of_leading_zeros, 0])])
    
    householder_vector = tf.pad(alpha*appended_raw_vector, paddings)

    return (householder_vector, average_raw_vector, alpha, raw_vector, 
    unnormalized_first_component, length)

# Defines a function to evaluate the multiplication of one Householder
# reflector of the Householder chain of one of the two orthogonal matri-
# ces of the SVD decomposition (A*diag(sigma)*transpose(B)) by the input 
# vector of the corresponding layer. The input vector is a tensor 
# [n_samples, p_i] where p_i is the number of neurons of the i-th layer

def multiply_input_vector_by_householder_reflector(input_vector, 
householder_reflector_index, householder_first_index,
householder_length, householder_number_of_leading_zeros, 
householder_parameters_orthogonal_matrix, constant_two, 
householder_epsilon_squared):
    
    # Gets the Householder vector from the Householder parameters of the
    # B matrix. Keep in mind that the order of the Householder chain of 
    # the B matrix is reversed with respect to the A matrix (in the SVD-
    # based architecture), since B is transposed in the SVD

    householder_vector, _, _, _, _, _ = get_householder_vector_from_parameters(
    householder_first_index, householder_length, 
    householder_number_of_leading_zeros, 
    householder_parameters_orthogonal_matrix, 
    householder_reflector_index, householder_epsilon_squared)

    # Multiplies the input vector by the Householder reflector (the ope-
    # ration is already broken down into the rank-1 calculation)

    return input_vector-(constant_two*tf.einsum('bi,i->b', input_vector, 
    householder_vector)[:,None]*householder_vector[None,:])

# Defines a function to create a wrapper for the method that multiplies
# the input vector by an orthogonal matrix recursively using the House-
# holder chain

def multiply_input_vector_by_householder_chain(input_vector, 
householder_first_index, householder_length, 
householder_number_of_leading_zeros,
householder_parameters_orthogonal_matrix, constant_two,
householder_epsilon_squared, output_dimension):

    # Gets the number of reflectors

    number_of_reflectors = tf.shape(householder_first_index)[0]
    
    # Iterates through the indices of Householder reflectors in reverse
    # order since the orthogonal matrix is given by the QR decomposition
    # as Q = H_1*H_2*...*H_m

    for householder_reflector_index in tf.range(number_of_reflectors-1,
    -1,-1):

        # Updates the input vector by recursive multiplication of re-
        # flectors of the Householder chain

        input_vector = multiply_input_vector_by_householder_reflector(
        input_vector, householder_reflector_index, 
        householder_first_index, householder_length, 
        householder_number_of_leading_zeros, 
        householder_parameters_orthogonal_matrix, constant_two,
        householder_epsilon_squared)

    # Returns the updated input vector. Returns only the indices corres-
    # ponding to the output dimension

    return input_vector[:,:output_dimension]

# Defines a function to create a wrapper for the method that multiplies
# the input vector [n_samples, input_dimensionality] by an orthogonal 
# matrix recursively using the Householder chain. This function, however, 
# already possesses its custom gradient. The output is a tensor [
# n_samples, input_dimensionality]

@tf.custom_gradient
def multiply_input_vector_by_householder_chain_with_custom_gradient(
input_vector, householder_first_index, householder_length, 
householder_number_of_leading_zeros,
householder_parameters_orthogonal_matrix, constant_two,
householder_epsilon_squared, output_dimension):

    # Gets a tensor to keep a memory pointer to the original input tensor

    X_input = input_vector

    # Evaluates the gradient with respect to the Householder parameters
    # (DOFs)
    
    def gradient(upstream_gradient):

        # Gets the dimensionality and rank information

        rank = tf.shape(householder_first_index)[0]

        dimensionality = tf.shape(X_input)[1]

        # Initializes the matrix Q as the product of all Householder re-
        # flectors

        Q = tf.eye(rank, dimensionality, dtype=
        householder_parameters_orthogonal_matrix.dtype)

        """for i in tf.range(rank):

            # Get the Householder vector of the i-th reflector

            householder_vector, _, _, _, _, _ = get_householder_vector_from_parameters(
            householder_first_index, householder_length, 
            householder_number_of_leading_zeros, 
            householder_parameters_orthogonal_matrix, i, 
            householder_epsilon_squared)

            # Multiplies this Householder reflector to the right of Q u-
            # sing the corresponding rank-1 update

            Q -= constant_two*tf.einsum('ik,k,j->ij', Q, 
            householder_vector, householder_vector)"""

        def while_function(i, Q_tensor):

            # Get the Householder vector of the i-th reflector
            
            householder_vector, _, _, _, _, _ = get_householder_vector_from_parameters(
            householder_first_index, householder_length, 
            householder_number_of_leading_zeros, 
            householder_parameters_orthogonal_matrix, i, 
            householder_epsilon_squared)

            # Multiplies this Householder reflector to the right of Q u-
            # sing the corresponding rank-1 update

            Q_updated = Q_tensor-constant_two*tf.einsum('ik,k,j->ij', Q_tensor, 
            householder_vector, householder_vector)

            return i+1, Q_updated

        def while_condition(i, vec):
        
                return i<rank

        _, Q = tf.while_loop(while_condition, while_function, loop_vars=(
        tf.constant(0, dtype=tf.int32), Q))

        # Calculates the gradient of the application of the Householder
        # chain with respect to the input tensor. The double contraction
        # with the upstream gradient due to the chain rule is also per-
        # formed

        gradient_wrt_input_tensor = tf.einsum('ki,im->km', 
        upstream_gradient, Q)

        # Calculates the gradient of the application of the Householder 
        # chain with respect to the DOFs of the Householder chain

        gradient_wrt_householder_parameters = tf.einsum('ki,kim->m',
        upstream_gradient, 
        evaluate_derivative_of_chain_of_householder_reflectors_application(
        X_input, householder_first_index, householder_length,
        householder_number_of_leading_zeros, 
        householder_parameters_orthogonal_matrix, 
        householder_epsilon_squared, tf.shape(X_input)[0], 
        householder_parameters_orthogonal_matrix.dtype, constant_two, 
        Q))

        return (gradient_wrt_input_tensor, None, None, None, 
        gradient_wrt_householder_parameters, None, None, None)

    """# Performs the forward pass

    forward_pass_output = multiply_input_vector_by_householder_chain(
    input_vector, householder_first_index, householder_length, 
    householder_number_of_leading_zeros,
    householder_parameters_orthogonal_matrix, constant_two,
    householder_epsilon_squared, output_dimension)

    return forward_pass_output, gradient"""

    # Gets the number of reflectors

    number_of_reflectors = tf.shape(householder_first_index)[0]
    
    # Iterates through the indices of Householder reflectors in reverse
    # order since the orthogonal matrix is given by the QR decomposition
    # as Q = H_1*H_2*...*H_m

    """for householder_reflector_index in tf.range(number_of_reflectors-1,
    -1,-1):

        # Updates the input vector by recursive multiplication of re-
        # flectors of the Householder chain

        input_vector = multiply_input_vector_by_householder_reflector(
        input_vector, householder_reflector_index, 
        householder_first_index, householder_length, 
        householder_number_of_leading_zeros, 
        householder_parameters_orthogonal_matrix, constant_two,
        householder_epsilon_squared)"""

    householder_reflector_index = number_of_reflectors-1

    def forward_function(householder_reflector_index, input_vector):
    
        # Updates the input vector by recursive multiplication of re-
        # flectors of the Householder chain

        updated_input_vector = multiply_input_vector_by_householder_reflector(
        input_vector, householder_reflector_index, 
        householder_first_index, householder_length, 
        householder_number_of_leading_zeros, 
        householder_parameters_orthogonal_matrix, constant_two,
        householder_epsilon_squared)

        return householder_reflector_index-1, updated_input_vector

    def fwd_cond(idx, vec):

        return idx >= 0

    _, input_vector = tf.while_loop(fwd_cond, forward_function, loop_vars=(
    householder_reflector_index, input_vector))

    # Returns the updated input vector and the gradient. Returns only
    # the dimensionality corresponding to the output

    return input_vector[:,:output_dimension], gradient

########################################################################
#                        Analytical derivatives                        #
########################################################################

# Defines a function to evaluate the derivative of the component of a
# Householder vector that is not a degree of freedom (\tilde{v}^{i}) with
# respect to the vector of degrees of freedom of the Householder vector.
# This derivative will be a tensor [n_dofs]. This function also evaluates
# the derivative of the normalization factor alpha with respect to the 
# vector of DOFs.
#
# householder_index is the index of the corresponding Householder re-
# flector
#
# v_bar is a tensor of shape (), i.e. a scalar
#
# alpha is also a scalar
#
# vector_of_dofs is a tensor [n_dofs], i.e. [dimensionality-
# householder_index-1]
#
# unnormalized_first_component_householder_vector is a scalar equal to
# sqrt((v_bar^2)+(householder_epsilon^2))

def evaluate_derivative_of_v_tilde_and_alpha(v_bar, alpha, n_dofs, 
vector_of_dofs, unnormalized_first_component_householder_vector):

    # Calculates the bit of the derivative of the component of the
    # Householder vector that is not a DOF (v_bar) that is common to all 
    # components of the derivative of v_dof with respect to the vector
    # of DOFs

    alpha_cubed = alpha*alpha*alpha

    renormalized_first_component = (v_bar/n_dofs)

    kappa = renormalized_first_component*((alpha/
    unnormalized_first_component_householder_vector)-(alpha_cubed*
    unnormalized_first_component_householder_vector))

    # Calculates the derivative of the component of the Householder vec-
    # tor that is not a DOF

    v_tilde = (kappa-(alpha_cubed*
    unnormalized_first_component_householder_vector*vector_of_dofs))

    # Calculates the derivative of alpha (normalization factor) with 
    # respect to the vector of DOFs

    alpha_derivative = -alpha_cubed*(vector_of_dofs+
    renormalized_first_component)

    return v_tilde, alpha_derivative

# Defines a function to assemble the derivative of a Householder reflec-
# tor and multiply it by the tensor [n_samples, dimensionality] that is
# multiplied to its right. What we want to evaluate is 
#
# (d/dv)[Q*(I-2v_hat\otimes v_hat)*X] = 
# -2[Q_ik*(dv_hat_k/dv_n)*v_hat_m*X_mj + 
#    Q_ik*v_hat_k*(dv_hat_m/v_n)*X_mj]*e_i \otimes e_j \otimes e_n
#
# such that Q is [dimensionality, dimensionality]; I is [dimensionality,
# dimensionality]; v_hat is [dimensionality]; v is [n_dofs]; X is 
# [n_samples, dimensionality]. We will define
# 
# term_1: Q_ik*(dv_hat_k/dv_n)*v_hat_m*X_mj
# term_2: Q_ik*v_hat_k*(dv_hat_m/v_n)*X_mj
#
# The output is a tensor of third-order [n_samples, dimensionality,
# n_dofs]

def evaluate_derivative_of_householder_reflector_application(v_bar, 
householder_vector, alpha, n_dofs, vector_of_dofs, reflector_index,
right_tensor_to_be_multiplied, left_tensor_to_be_multiplied,
unnormalized_first_component_householder_vector, constant_two, dtype):

    # Gets the derivatives of the first non-zero component of the House-
    # holder vector (v_tilde, and of the normalizing factor with respect 
    # to the vector of degrees of freedom

    derivative_v_tilde, alpha_derivative = evaluate_derivative_of_v_tilde_and_alpha(
    v_bar, alpha, tf.cast(n_dofs, dtype=dtype), vector_of_dofs, 
    unnormalized_first_component_householder_vector)

    # The derivative of the Householder reflector is composed of two 
    # terms. Each must be computed individually. The first term multi-
    # plies the Householder vector by the tensor to be multiplied. First
    # result is a tensor [n_samples]

    term_1 = tf.einsum('i,ji->j', householder_vector, 
    right_tensor_to_be_multiplied)

    # The second term begins with the multiplication of the transposed
    # derivative of the Householder vector by the tensor to be multiplied
    # to the right. The first result is a tensor [n_samples, n_dofs].
    # Remember that the last index must represent the DOFs with respect
    # to we are differentiating

    term_2 = tf.einsum('i,ji,k->jk', vector_of_dofs, 
    right_tensor_to_be_multiplied[:,(-n_dofs):], alpha_derivative)+(
    #
    alpha*right_tensor_to_be_multiplied[:,(-n_dofs):])

    # Adds the bit related to the derivative of v_tilde with respect to 
    # the vector of DOFs

    term_2 += tf.einsum('i,j->ji', derivative_v_tilde, 
    right_tensor_to_be_multiplied[:,reflector_index])

    # Multiplies the first term by the derivative of the Householder
    # vector with respect to the vector of DOFs to the right. The order
    # of the indices is due to the fact that the last index accounts for
    # the DOFs to which the Householder reflector is differentiated 

    final_term_1 = tf.einsum('i,j,k->jik', vector_of_dofs, term_1,
    alpha_derivative)

    # Adds the contribution of the identity times the normalization fac-
    # tor, which came from the derivative of the components of v_bar

    final_term_1 += (alpha*tf.eye(n_dofs, dtype=dtype)[None,:,:]*term_1[
    :,None,None])

    # Evaluates the bit of the derivative of v_tilde with respect to the 
    # vector of DOFs. Inverts the order because the index of the DOFs is 
    # the last one

    contribution_of_v_tilde_derivative = tf.einsum('i,j->ji', 
    derivative_v_tilde, term_1)

    dimensionality = tf.shape(left_tensor_to_be_multiplied)[1]

    n_samples = tf.shape(right_tensor_to_be_multiplied)[0]

    # Gets the indices of the DOFs to be updated

    number_of_indices = tf.shape(vector_of_dofs)[0]

    scatter_indices = (reflector_index+1+tf.range(number_of_indices))[:, 
    None]

    # Defines all indices as the concatenation of the index of the re-
    # flector with the indices of the scatter update

    all_indices = tf.concat([[[reflector_index]], scatter_indices], 
    axis=0)

    # To optimize scattered update, transposes the tensor final_term_1
    # to the shape [dimensionality, n_samples, n_dofs]. Scattered upda-
    # tes work best when the indexed dimension is the first one; on the
    # other hand, transposing is very very cheap. Also initializes the
    # transposed version with the right dimensionality, instead of just
    # the number of DOFs

    # Concatenates the update of the derivative of v_tilde and the al-
    # ready calculated first term

    transposed_updates = tf.concat([contribution_of_v_tilde_derivative[
    None,:,:], tf.transpose(final_term_1, [1, 0, 2])], axis=0)

    # Creates the transposed first term in a single operation using 
    # scatter operations

    transposed_first_term = tf.scatter_nd(all_indices, 
    transposed_updates, shape=[dimensionality, n_samples, n_dofs])

    # Tranposes the final term back to the original shape of [n_samples,
    # dimensionality, n_dofs]

    final_term_1 = tf.transpose(transposed_first_term, [1, 0, 2])

    # Performs the contraction of the first term with the tensor to the 
    # left

    final_term_1 = tf.einsum('ik,jkn->jin', left_tensor_to_be_multiplied,
    final_term_1)

    # Multiplies the second term by the Householder vector to yield a 
    # third-order tensor. Note that the tensor to be multiplied to the
    # left is already added here

    term_2 = tf.einsum('ik,k,jn->jin', left_tensor_to_be_multiplied, 
    householder_vector, term_2)

    # Returns the sum of the two

    return -constant_two*(final_term_1+term_2)

# Defines a function to evaluate the derivative of the application of a
# chain of Householder reflectors (CHR) with respect to the DOFs of each
# Householder reflector. The operation that is being differentiated is
#
# y = H_1*H_2*...*H_m*X = H_1*H_2*...*H_{i-1}*H_{i}*H_{i+1}*...*H_m*X
#   = Q_{i}*H_{i}*X_{i}
#
# such that Q_{i+1} = Q_{i}*H_{i} and X_{i} = H_{i+1}*X_{i+1} 

def evaluate_derivative_of_chain_of_householder_reflectors_application(
right_tensor_to_be_multiplied, householder_first_index,
householder_length, householder_number_of_leading_zeros, 
householder_parameters, householder_epsilon_squared, n_samples, dtype, 
constant_two, Q):

    # Gets the number of reflectors (which is equivalent to the rank of
    # the resulting left-orthogonal matrix)

    rank = tf.shape(householder_first_index)[0]

    # Initializes the right tensor to be multiplied

    X = right_tensor_to_be_multiplied

    # Calculates the number of DOFs of the whole Householder chain

    n_DOFS_chain = tf.shape(householder_parameters)[0]

    # Initializes the derivative tensor with shape [n_dofs_chain, 
    # n_samples, rank] to force the DOFs dimension to be the first index.
    # This choice takes advantage of computational efficiency to scatter
    # update in the first axis

    chain_application_derivative_transposed = tf.zeros((n_DOFS_chain,
    n_samples, rank), dtype=dtype)

    # Iterates over the Householder reflectors in reverse order to eval-
    # uate the derivative of the output y with respect to the DOFs of 
    # each Householder reflector. Uses the reverse order since the or-
    # thogonal matrix is given by the QR decomposition as Q = H_1*H_2*
    # ...*H_m

    """for i in tf.range(rank-1,-1,-1):

        # Gets the Householder vector from the flat tensor of Householder
        # DOFs

        (householder_vector, v_bar, alpha, vector_of_dofs,
        unnormalized_first_component, local_n_dofs) = get_householder_vector_from_parameters(
        householder_first_index, householder_length, 
        householder_number_of_leading_zeros, householder_parameters, 
        i, householder_epsilon_squared)

        # Updates the left matrix by canceling the right-most House-
        # holder reflector

        Q -= constant_two*tf.einsum('ik,k,j->ij', Q, householder_vector,
        householder_vector)

        # Evaluates the derivative of y with respect to the DOFs of this
        # Householder reflector. Transposes the result to [n_dofs, 
        # n_samples, dimensionality] to take advantage of the computa-
        # tional performance of scatter update in the first axis

        dy_dDOFs_transposed = tf.transpose(
        evaluate_derivative_of_householder_reflector_application(
        v_bar, householder_vector, alpha, local_n_dofs, vector_of_dofs, 
        i, X, Q, unnormalized_first_component, constant_two, dtype), [2,
        0, 1])

        # Determines the range of DOFs of this Householder vector in the
        # flat tensor of DOFs of the Householder chain

        dof_start_index = householder_first_index[i]

        reflector_dofs_indices = tf.range(dof_start_index, 
        dof_start_index+local_n_dofs)[:,None]

        # Updates the derivative tensor using the scattering update me-
        # thod

        chain_application_derivative_transposed = tf.tensor_scatter_nd_update(
        chain_application_derivative_transposed, reflector_dofs_indices, 
        dy_dDOFs_transposed)

        # Updates the right matrix by multiplying by the right-most 
        # Householder reflector

        X -= constant_two*tf.einsum('ij,j,k->ik', X, householder_vector,
        householder_vector)"""

    def while_function(i, Q, X, chain_application_derivative_transposed):

        # Gets the Householder vector from the flat tensor of Householder
        # DOFs

        (householder_vector, v_bar, alpha, vector_of_dofs,
        unnormalized_first_component, local_n_dofs) = get_householder_vector_from_parameters(
        householder_first_index, householder_length, 
        householder_number_of_leading_zeros, householder_parameters, 
        i, householder_epsilon_squared)

        # Updates the left matrix by canceling the right-most House-
        # holder reflector

        Q -= constant_two*tf.einsum('ik,k,j->ij', Q, householder_vector,
        householder_vector)

        # Evaluates the derivative of y with respect to the DOFs of this
        # Householder reflector. Transposes the result to [n_dofs, 
        # n_samples, dimensionality] to take advantage of the computa-
        # tional performance of scatter update in the first axis

        dy_dDOFs_transposed = tf.transpose(
        evaluate_derivative_of_householder_reflector_application(
        v_bar, householder_vector, alpha, local_n_dofs, vector_of_dofs, 
        i, X, Q, unnormalized_first_component, constant_two, dtype), [2,
        0, 1])

        # Determines the range of DOFs of this Householder vector in the
        # flat tensor of DOFs of the Householder chain

        dof_start_index = householder_first_index[i]

        reflector_dofs_indices = tf.range(dof_start_index, 
        dof_start_index+local_n_dofs)[:,None]

        # Updates the derivative tensor using the scattering update me-
        # thod

        chain_application_derivative_transposed = tf.tensor_scatter_nd_update(
        chain_application_derivative_transposed, reflector_dofs_indices, 
        dy_dDOFs_transposed)

        # Updates the right matrix by multiplying by the right-most 
        # Householder reflector

        X -= constant_two*tf.einsum('ij,j,k->ik', X, householder_vector,
        householder_vector)

        return i-1, Q, X, chain_application_derivative_transposed

    def fwd_cond(idx, Q, X, chain):
    
        return idx >= 0

    _, Q, X, chain_application_derivative_transposed = tf.while_loop(
    fwd_cond, while_function, loop_vars=(rank-1, Q, X, 
                                         chain_application_derivative_transposed))

    # Transposes the result back to the shape [n_samples, rank, 
    # n_dofs_chain]

    return tf.transpose(chain_application_derivative_transposed, [1,2,0])