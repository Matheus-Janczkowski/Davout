# Routine to store methods to work with the singular value decomposition
# (SVD)

import tensorflow as tf

########################################################################
#             Assembly of chains of Householder reflectors             #
########################################################################

# Defines a function to parse a single Householder vector from a vector
# of all Householder vectors to construct an orthogonal matrix. The in-
# coming vector is a tensor [0.5*(m_rank*((2*n_rows)-m_rank-1))], where 
# m_rank is the number of columns of the final orthogonal matrix, and 
# n_rows is the number of rows. Receives the index of the Householder 
# reflector in the Householder chain to collect the corresponding House-
# holder vector

def get_householder_vector_from_parameters(householder_indices,
householder_parameters, householder_reflector_index, 
householder_epsilon_squared):
    
    # Gets the first and last indices of the Householder vector, then
    # the vector that will become the Householder vector

    initial_index, length, number_of_leading_zeros = (
    householder_indices[householder_reflector_index])

    raw_vector = tf.slice(householder_parameters, [initial_index], [
    length])

    # Computes the average of the raw vector

    average_raw_vector = tf.reduce_mean(raw_vector)

    # Concatenates the non-free component of the Householder vector to
    # the first position

    raw_vector = tf.concat([tf.sqrt((average_raw_vector*
    average_raw_vector)+householder_epsilon_squared)[None], raw_vector], 
    axis=0)

    # Rescales the raw vector to have unit norm, then adds the leading
    # zeros and returns it

    return tf.pad(tf.math.l2_normalize(raw_vector), [[
    number_of_leading_zeros, 0]])

# Defines a function to evaluate the multiplication of one Householder
# reflector of the Householder chain of one of the two orthogonal matri-
# ces of the SVD decomposition (A*diag(sigma)*transpose(B)) by the input 
# vector of the corresponding layer. The input vector is a tensor 
# [n_samples, p_i] where p_i is the number of neurons of the i-th layer

def multiply_input_vector_by_householder_reflector(input_vector, 
householder_reflector_index, householder_indices_orthogonal_matrix, 
householder_parameters_orthogonal_matrix, constant_two, 
householder_epsilon_squared):
    
    # Gets the Householder vector from the Householder parameters of the
    # B matrix. Keep in mind that the order of the Householder chain of 
    # the B matrix is reversed with respect to the A matrix (in the SVD-
    # based architecture), since B is transposed in the SVD

    householder_vector = get_householder_vector_from_parameters(
    householder_indices_orthogonal_matrix, 
    householder_parameters_orthogonal_matrix, 
    householder_reflector_index, householder_epsilon_squared)

    # Multiplies the input vector by the Householder reflector (the ope-
    # ration is already broken down into the rank-1 calculation)

    return input_vector-(constant_two*tf.reduce_sum(input_vector*
    householder_vector, axis=-1, keepdims=True)*householder_vector)

# Defines a function to create a wrapper for the method that multiplies
# the input vector by an orthogonal matrix recursively using the House-
# holder chain

def multiply_input_vector_by_householder_chain(input_vector, 
householder_reflector_indices, householder_indices_orthogonal_matrix, 
householder_parameters_orthogonal_matrix, constant_two,
householder_epsilon_squared):
    
    # Iterates through the indices of Householder reflectors

    for householder_reflector_index in householder_reflector_indices:

        # Updates the input vector by recursive multiplication of re-
        # flectors of the Householder chain

        input_vector = multiply_input_vector_by_householder_reflector(
        input_vector, householder_reflector_index, 
        householder_indices_orthogonal_matrix, 
        householder_parameters_orthogonal_matrix, constant_two,
        householder_epsilon_squared)

    # Returns the updated input vector

    return input_vector

########################################################################
#                        Analytical derivatives                        #
########################################################################

# Defines a function to evaluate the derivative of the component of a
# Householder vector that is not a degree of freedom (\bar{v}^{i}) with
# respect to the vector of degrees of freedom of the Householder vector.
# This derivative will be a tensor [n_dofs]. This function also evaluates
# the derivative of the normalization factor alpha with respect to the 
# vector of DOFs

def evaluate_derivative_of_v_bar_and_alpha(v_bar, alpha, n_dofs, 
vector_of_dofs, unnormalized_first_component_householder_vector, dtype):

    # Calculates the bit of the derivative of the component of the
    # Householder vector that is not a DOF (v_bar) that is common to all 
    # components of the derivative of v_dof with respect to the vector
    # of DOFs

    alpha_cubed = tf.pow(alpha, 3)

    renormalized_first_component = (v_bar/n_dofs)

    kappa = renormalized_first_component*((alpha/
    unnormalized_first_component_householder_vector)-(alpha_cubed*
    unnormalized_first_component_householder_vector))

    # Calculates the derivative of the component of the Householder vec-
    # tor that is not a DOF

    first_component_derivative = ((kappa*tf.ones((n_dofs), dtype=dtype))
    -alpha_cubed*unnormalized_first_component_householder_vector*
    vector_of_dofs)

    # Calculates the derivative of alpha (normalization factor) with 
    # respect to the vector of DOFs

    alpha_derivative = -alpha_cubed*(vector_of_dofs+(
    renormalized_first_component*tf.ones((n_dofs), dtype=dtype)))

    return first_component_derivative, alpha_derivative

# Defines a function to assemble the derivative of a Householder reflec-
# tor and multiply it by the tensor [dimensionality, n_samples] that is
# multiplied to its right. What we want to evaluate is 
#
# (d/dv)[Q*(I-2v_hat\otimes v_hat)*X] = 
# -2[Q_ik*(dv_hat_k/dv_n)*v_hat_m*X_mj + 
#    Q_ik*v_hat_k*(dv_hat_m/v_n)*X_mj]*e_i \otimes e_j \otimes e_n
#
# such that Q is [dimensionality, dimensionality]; I is [dimensionality,
# dimensionality]; v_hat is [dimensionality]; v is [n_dofs]; X is 
# [dimensionality, n_samples]. We will define
# 
# term_1: Q_ik*(dv_hat_k/dv_n)*v_hat_m*X_mj
# term_2: Q_ik*v_hat_k*(dv_hat_m/v_n)*X_mj

def evaluate_derivative_of_householder_reflector_and_multiply(v_bar, 
householder_vector, alpha, n_dofs, vector_of_dofs, reflector_index,
tensor_to_be_multiplied, unnormalized_first_component_householder_vector, 
constant_two, dtype):

    # Gets the derivatives of the first non-zero component of the House-
    # holder vector and of the normalizing factor with respect to the
    # vector of degrees of freedom

    first_component_derivative, alpha_derivative = evaluate_derivative_of_v_bar_and_alpha(
    v_bar, alpha, n_dofs, vector_of_dofs, 
    unnormalized_first_component_householder_vector, dtype)

    # The derivative of the Householder reflector is composed of two 
    # terms. Each must be computed individually. The first term multi-
    # plies the Householder vector by the tensor to be multiplied. First
    # result is a tensor [dimensionality, n_samples]

    term_1 = tf.einsum('i,ij->j', householder_vector, 
    tensor_to_be_multiplied)

    # The second term begins with the multiplication of the transposed
    # derivative of the Householder vector by the tensor to be multiplied
    # to the right. The first result is a tensor [n_dofs, n_samples]

    term_2 = (tf.einsum('i,ij->j', vector_of_dofs, 
    tensor_to_be_multiplied[(tensor_to_be_multiplied.shape[0]-n_dofs):,:
    ])*alpha_derivative)+(
    #
    alpha*tensor_to_be_multiplied[(
    tensor_to_be_multiplied.shape[0]-n_dofs):,:])+(
    #
    first_component_derivative*tensor_to_be_multiplied[
    tensor_to_be_multiplied.shape[0]-n_dofs,:])

    # Multiplies the first term by the derivative of the Householder
    # vector with respect to the vector of DOFs to the right

    term_1 = (tf.einsum('i,ij->j', alpha_derivative, term_1)*
    vector_of_dofs)+(
    #
    alpha*term_1)+(
    #
    tf.one_hot(reflector_index, depth=tensor_to_be_multiplied.shape[0], 
    dtype=dtype)*tf.einsum('i,ij->j', first_component_derivative, term_1))

    # Multiplies the second term by the Householder vector

    term_2 = householder_vector*term_2

    # Returns the sum of the two

    return -constant_two*(term_1+term_2)