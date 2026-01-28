# Routine to store some mathematical utilities

import tensorflow as tf

########################################################################
#                            Linear Algebra                            #
########################################################################

# Defines a function to take the inverse of a batched tensor by solving
# a linear system with the identity

@tf.function
def get_inverse(tensor_to_be_inverted, identity_tensor):

    return tf.linalg.solve(tensor_to_be_inverted, identity_tensor)

########################################################################
#                               Mappings                               #
########################################################################

# Defines a function to calculate the determinant and inverse of a 3D 
# jacobian transformation given shape functions and nodal coordinates

@tf.function
def jacobian_3D_element(natural_derivatives_shape_functions, x, y, z):

    """
    Computes the determinant and inverse of the jacobian transformation
    of a 3D element.
    
    natural_derivatives_shape_functions: [n_quadrature_points, n_nodes, 
    natural_coordinates] tensor with the derivatives of the shape func-
    tions in natural coordinates with respect each natural coordinate
    
    x: [n_elements, n_nodes] tensor with the x coordinates of the mesh

    y: [n_elements, n_nodes] tensor with the y coordinates of the mesh

    z: [n_elements, n_nodes] tensor with the z coordinates of the mesh
    """

    # Computes the jacobian of the transformation from the original co-
    # ordinates to the natural ones. To compute the jacobian, a isopara-
    # metric formulation is used. The function einsum is used to perform
    # tensor contraction using Einstein notation

    # J_11 = dx/dr

    J_11 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,0], x)

    # J_12 = dy/dr

    J_12 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,0], y)

    # J_13 = dz/dr

    J_13 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,0], z)

    # J_21 = dx/ds

    J_21 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,1], x)

    # J_22 = dy/ds

    J_22 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,1], y)

    # J_23 = dz/ds

    J_23 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,1], z)

    # J_31 = dx/dt

    J_31 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,2], x)

    # J_32 = dy/dt

    J_32 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,2], y)

    # J_33 = dz/dt

    J_33 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,2], z)

    # Computes the determinant of the jacobian 

    eps = tf.keras.backend.epsilon()

    det_J = ((J_11*((J_22*J_33)-(J_23*J_32)))+(J_12*((J_23*J_31)-(
    J_21*J_33)))+(J_13*((J_21*J_32)-(J_22*J_31)))+eps)

    # Computes the inverse of the jacobian transformation using Cramer's
    # rule

    J_inv_11 = (((J_22*J_33)-(J_23*J_32))/det_J)

    J_inv_12 = (((J_13*J_32)-(J_12*J_33))/det_J)

    J_inv_13 = (((J_12*J_23)-(J_13*J_22))/det_J)

    J_inv_21 = (((J_23*J_31)-(J_21*J_33))/det_J)

    J_inv_22 = (((J_11*J_33)-(J_13*J_31))/det_J)

    J_inv_23 = (((J_13*J_21)-(J_11*J_23))/det_J)

    J_inv_31 = (((J_21*J_32)-(J_22*J_31))/det_J)

    J_inv_32 = (((J_12*J_31)-(J_11*J_32))/det_J)

    J_inv_33 = (((J_11*J_22)-(J_12*J_21))/det_J)

    # Stacks the jacobian inverse by column first to be easier to opera-
    # te afterwards

    J_row_1 = tf.stack([J_inv_11, J_inv_12, J_inv_13], axis=-1)

    J_row_2 = tf.stack([J_inv_21, J_inv_22, J_inv_23], axis=-1)

    J_row_3 = tf.stack([J_inv_31, J_inv_32, J_inv_33], axis=-1)

    # Stacks them by row. Uses axis -2 to put the index of the row befo-
    # re the index of column

    J_inv = tf.stack([J_row_1, J_row_2, J_row_3], axis=-2)

    return det_J, J_inv

# Defines a function to calculate the determinant and inverse of a 2D 
# jacobian transformation given shape functions and nodal coordinates

@tf.function
def jacobian_2D_element(natural_derivatives_shape_functions, x, y):

    """
    Computes the determinant and inverse of the jacobian transformation
    of a 2D element.
    
    natural_derivatives_shape_functions: [n_quadrature_points, n_nodes, 
    natural_coordinates] tensor with the derivatives of the shape func-
    tions in natural coordinates with respect each natural coordinate
    
    x: [n_elements, n_nodes] tensor with the x coordinates of the mesh

    y: [n_elements, n_nodes] tensor with the y coordinates of the mesh
    """

    # Computes the jacobian of the transformation from the original co-
    # ordinates to the natural ones. To compute the jacobian, a isopara-
    # metric formulation is used. The function einsum is used to perform
    # tensor contraction using Einstein notation

    # J_11 = dx/dr

    J_11 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,0], x)

    # J_12 = dy/dr

    J_12 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,0], y)

    # J_21 = dx/ds

    J_21 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,1], x)

    # J_22 = dy/ds

    J_22 = tf.einsum('qn,en->eq', natural_derivatives_shape_functions[
    ...,1], y)

    # Computes the determinant of the jacobian 

    eps = tf.keras.backend.epsilon()

    det_J = (J_11*J_22)-(J_12*J_21)+eps

    # Computes the inverse of the jacobian transformation using Cramer's
    # rule

    J_inv_11 = (J_22/det_J)

    J_inv_12 = (-J_12/det_J)

    J_inv_21 = (-J_21/det_J)

    J_inv_22 = (J_11/det_J)

    # Stacks the jacobian inverse by column first to be easier to opera-
    # te afterwards

    J_row_1 = tf.stack([J_inv_11, J_inv_12], axis=-1)

    J_row_2 = tf.stack([J_inv_21, J_inv_22], axis=-1)

    # Stacks them by row. Uses axis -2 to put the index of the row befo-
    # re the index of column

    J_inv = tf.stack([J_row_1, J_row_2], axis=-2)

    return det_J, J_inv