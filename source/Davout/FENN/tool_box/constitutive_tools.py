# Routine to store methods to process information in and for constituti-
# ve models

import tensorflow as tf

########################################################################
#                              Kinematics                              #
########################################################################

# Defines a class to compute the deformation gradient as a tensor [
# n_elements, n_quadrature_points, 3, 3]

class DeformationGradient:

    def __init__(self, vector_of_parameters, indexing_dofs_tensor, 
    shape_functions_derivatives, identity_tensor):
        
        """
        Defines a class to compute the batched deformation gradient

        indexing_dofs_tensor: indices of DOFs of the global vector of 
        parameters as a tensor [n_elements, n_nodes, 3]

        shape_function_derivatives: derivatives of the shape functions 
        with respect to the original coordinates (x, y, z) as a tensor
        [n_elements, n_quadrature_points, n_nodes, 3]

        identity_tensor: identity tensor as a tensor [n_elements, 
        n_quadrature_points, 3, 3]
        """
        
        self.indexing_dofs_tensor = indexing_dofs_tensor

        self.shape_functions_derivatives = shape_functions_derivatives

        self.identity_tensor = identity_tensor

        self.vector_of_parameters = vector_of_parameters

    @tf.function
    def compute_batched_deformation_gradient(self):
        
        # Gathers the vector of DOFs for this mesh

        field_dofs = tf.gather(self.vector_of_parameters, 
        self.indexing_dofs_tensor)

        # Contracts the DOFs to get the material displacement gradient as 
        # a tensor [n_elements, n_quadrature_points, 3, 3]. THen, adds 
        # the identity tensor and returns

        return (tf.einsum('eqnj,eni->eqij', 
        self.shape_functions_derivatives, field_dofs)+self.identity_tensor)