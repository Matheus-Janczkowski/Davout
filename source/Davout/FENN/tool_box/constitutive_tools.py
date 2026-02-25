# Routine to store methods to process information in and for constituti-
# ve models

import tensorflow as tf

########################################################################
#                              Kinematics                              #
########################################################################

# Defines a class to compute the deformation gradient as a tensor [
# n_elements, n_quadrature_points, 3, 3]

class DeformationGradient:

    def __init__(self, indexing_dofs_tensor, mesh_data, identity_tensor):
        
        """
        Defines a class to compute the batched deformation gradient

        indexing_dofs_tensor: indices of DOFs of the global vector of 
        parameters as a tensor [n_elements, n_nodes, 3]

        mesh_data: instance of a finite element class that has, in 
        particular the tensor of derivatives of the shape functions 
        with respect to the original coordinates (x, y, z) as a tensor
        [n_elements, n_quadrature_points, n_nodes, 3]

        identity_tensor: identity tensor as a tensor [n_elements, 
        n_quadrature_points, 3, 3]
        """
        
        self.indexing_dofs_tensor = indexing_dofs_tensor

        self.identity_tensor = identity_tensor

        # Mesh data can be a list if multiple realizations of the mesh
        # were generated

        if isinstance(mesh_data, list):

            # Gets the tensors of derivatives of shape functions across
            # the realizations of the mesh and stacks them into a single
            # tensor

            self.shape_functions_derivatives = tf.stack([
            single_mesh_data.shape_functions_derivatives for (
            single_mesh_data) in mesh_data], axis=0)

            # Sets the appropriate function to contract the tensor of 
            # derivatives of the shape functions with the DOFs of the
            # field

            self.appropriate_contraction = self.contract_multiple_meshes

        # Otherwise, if there is a single mesh for all realizations of
        # the BVP

        else:

            # Takes the tensor of derivatives of the shape functions di-
            # rectly from the mesh data

            self.shape_functions_derivatives = (
            mesh_data.shape_functions_derivatives)

            # Sets the appropriate function to contract the tensor of 
            # derivatives of the shape functions with the DOFs of the
            # field

            self.appropriate_contraction = self.contract_single_mesh

    # Defines a function to contract the tensor of derivatives of the
    # shape functions with the tensor of DOFs of the field, in case of a
    # single mesh realization

    @tf.function
    def contract_single_mesh(self, field_dofs):

        return tf.einsum('eqnj,peni->peqij', 
        self.shape_functions_derivatives, field_dofs)

    # Defines a function to contract the tensor of derivatives of the
    # shape functions with the tensor of DOFs of the field, in case of 
    # multiple mesh realizations

    @tf.function
    def contract_multiple_meshes(self, field_dofs):

        return tf.einsum('peqnj,peni->peqij', 
        self.shape_functions_derivatives, field_dofs)

    @tf.function
    def compute_batched_deformation_gradient(self, vector_of_parameters):
        
        # Gathers the vector of DOFs for this mesh. Uses axis=1 to ensure
        # that the gathering is done in the DOFs index, as the first in-
        # dex is related to batching for multiple BVP instances

        field_dofs = tf.gather(vector_of_parameters, 
        self.indexing_dofs_tensor, axis=1)

        # Contracts the DOFs to get the material displacement gradient as 
        # a tensor [n_realizations, n_elements, n_quadrature_points, 3, 
        # 3]. Then, adds the identity tensor and returns

        return (self.appropriate_contraction(field_dofs)+
        self.identity_tensor)