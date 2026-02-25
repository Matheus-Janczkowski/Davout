# Routine to store methods to calculate the local (individual finite el-
# ement level) residual vector due to internal forces

import tensorflow as tf 

from ..tool_box.constitutive_tools import DeformationGradient

from ..tool_box.mesh_info_tools import get_volume_info_from_mesh_data_class

########################################################################
#               Internal work in reference configuration               #
########################################################################

# Defines a class to get the constitutive model dictionary and transform 
# it into a compiled evaluation of the residual vector due to the inter-
# nal work in the reference configuration, considering compressible hy-
# perelasticity

class CompressibleInternalWorkReferenceConfiguration:

    def __init__(self, vector_of_parameters, constitutive_models_dict, 
    mesh_data_class):
        
        # Gets the number of batched BVP instances

        self.n_realizations = vector_of_parameters.shape[0]
        
        # Initializes the list of instances of the class to compute the
        # deformation gradient 

        self.deformation_gradient_list = []

        # Initializes a list with the function to evaluate the first Pi-
        # ola-Kirchhoff stress tensor at each physical group

        self.first_piola_kirchhoff_list = []

        # Creates a tensor [n_physical_groups, n_elements, 
        # n_quadrature_points, 3, 3] containing the shape functions of
        # the elements at each physical group multiplied by the integra-
        # tion measure

        self.variation_gradient_dx = []

        # Creates a list of indices to update the global residual vector
        # across the different realizations of the BVP 

        self.updates_indices = []

        # Iterates through the dictionary of constitutive models

        for physical_group, constitutive_class in (
        constitutive_models_dict.items()):
            
            # Gets necessary information from the mesh data class, such
            # as the dictionary of domain finite elements

            (integer_dtype, float_dtype, mesh_data, physical_group_tag
            ) = get_volume_info_from_mesh_data_class(mesh_data_class,
            physical_group, "constitutive models", "CompressibleIntern"+
            "alWorkReferenceConfiguration", "Displacement")

            # Verifies if the number of realizations given is consistent
            # with the global number of realizations. If the number of 
            # realization of the material parameters are zero, it is no
            # problem

            if (constitutive_class.n_material_realizations!=0 and (
            self.n_realizations!=constitutive_class.n_material_realizations)):

                raise ValueError("The number of realizations of tracti"+
                "ons in 'CompressibleInternalWorkReferenceConfiguratio"+
                "n' in the constitutive model at physical group '"+str(
                physical_group)+"' is "+str(
                constitutive_class.n_material_realizations)+"; whereas"+
                " the global number of realizations is "+str(
                self.n_realizations)+". They must be the same")

            # Gets the identity tensor for the mesh within this physical
            # group

            identity_tensor = tf.eye(3, batch_shape=[self.n_realizations,
            mesh_data.number_elements, mesh_data.number_quadrature_points
            ], dtype=mesh_data.dtype)

            # Puts the identity tensor into the constitutive model class 
            # if it has the attribute

            if hasattr(constitutive_class, "identity_tensor"):

                constitutive_class.identity_tensor = identity_tensor

            # Instantiates the class to calculate the deformation gradi-
            # ent

            self.deformation_gradient_list.append(DeformationGradient(
            vector_of_parameters, mesh_data.dofs_per_element, 
            mesh_data.shape_functions_derivatives, identity_tensor))

            # Adds the function to evaluate the first Piola-Kirchhoff 
            # stress tensor

            self.first_piola_kirchhoff_list.append(
            constitutive_class.first_piola_kirchhoff)

            # Adds the derivatives of the shape functions multiplied by
            # the integration measure

            self.variation_gradient_dx.append(tf.einsum('eqnj,eq->eqnj',
            mesh_data.shape_functions_derivatives, mesh_data.dx))

            # Creates the indices for updating the global residual vector
            # batched along the different realizations of the BVP

            # Creates the realizations indices and expands for broadcas-
            # ting

            realization_indices = tf.range(self.n_realizations, dtype=
            mesh_data.integer_dtype)[:, None, None, None]

            # Broadcasts to match the realizations dimension as the 
            # trailing dimension

            realization_indices = tf.broadcast_to(realization_indices, [
            self.n_realizations, tf.shape(mesh_data.dofs_per_element)[0
            ], tf.shape(mesh_data.dofs_per_element)[1], tf.shape(
            mesh_data.dofs_per_element)[2]])

            # Expands DOF indices and broadcasts to the realization di-
            # mension

            dofs_indices = tf.broadcast_to(tf.expand_dims(
            mesh_data.dofs_per_element, axis=0), tf.shape(
            realization_indices))

            # Stacks them both to form a concatenation (cartesian product 
            # of realization and DOFs indices)

            self.updates_indices.append(tf.stack([realization_indices,
            dofs_indices], axis=-1))

        # Gets the number of materials

        self.n_materials = len(self.first_piola_kirchhoff_list)

        # Stacks the derivatives of the shape functions multiplied by the
        # integration measure into a tensor [n_physical_groups, 
        # n_elements, n_quadrature_points, n_nodes, n_physical_dimensions]

        self.variation_gradient_dx = tf.stack(self.variation_gradient_dx,
        axis=0)

        # Makes deformation_gradient_list and first_piola_kirchhoff_list
        # tuples to show their immutability

        self.first_piola_kirchhoff_list = tuple(
        self.first_piola_kirchhoff_list)

        self.deformation_gradient_list = tuple(
        self.deformation_gradient_list)

    # Defines a function to assemble the residual vector

    @tf.function
    def assemble_residual_vector(self, global_residual_vector):

        # Iterates through the physical groups

        for i in range(self.n_materials):

            # Gets the batched tensor [n_realizations, n_elements, 
            # n_quadrature_points, 3, 3] of the deformation gradient and,
            # then, calculates the first Piola-Kirchhoff stress as [
            # n_realizations, n_elements, n_quadrature_points, 3, 3]

            P = self.first_piola_kirchhoff_list[i](self.deformation_gradient_list[
            i].compute_batched_deformation_gradient())

            # Contracts the first Piola-Kirchhoff stress with the deri-
            # vatives of the shape functions multiplied by the integra-
            # tion measure to get the integration of the internal work 
            # of the variational form. Then, sums over the quadrature 
            # points, that are the third dimension (index 2 in python 
            # convention).
            # The result is a tensor [n_realizations, n_elements, 
            # n_nodes, n_physical_dimensions]

            internal_work = tf.reduce_sum(tf.einsum('peqij,eqnj->peqni', 
            P, self.variation_gradient_dx[i]), axis=2)

            # Adds the contribution of this physical group to the global
            # residual vector. Uses the own tensor of DOF indexing to
            # scatter the local residual. Another dimension is added to
            # the indexing tensor to make it compatible with tensorflow
            # tensor_scatter_nd_add. Performs this change in place, as
            # global_residual_vector is a variable

            global_residual_vector.scatter_nd_add(self.updates_indices[i], 
            internal_work)