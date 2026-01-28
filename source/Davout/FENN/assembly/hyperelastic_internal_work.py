# Routine to store methods to calculate the local (individual finite el-
# ement level) residual vector due to internal forces

import tensorflow as tf 

from ..tool_box.constitutive_tools import DeformationGradient

########################################################################
#               Internal work in reference configuration               #
########################################################################

# Defines a class to get the constitutive model dictionary and transform 
# it into a compiled evaluation of the residual vector due to the inter-
# nal work in the reference configuration, considering compressible hy-
# perelasticity

class CompressibleInternalWorkReferenceConfiguration:

    def __init__(self, vector_of_parameters, constitutive_models_dict, 
    mesh_dict, domain_physical_groups_dict):
        
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

        # Iterates through the dictionary of constitutive models

        for physical_group, constitutive_class in (
        constitutive_models_dict.items()):
            
            # Gets the physical group tag

            physical_group_tag = None 

            if not (physical_group in domain_physical_groups_dict):

                raise NameError("The physical group name '"+str(
                physical_group)+"' in the dictionary of constitutive m"+
                "odels is not a valid physical group name. Check the a"+
                "vailable names:\n"+str(list(
                domain_physical_groups_dict.keys())))
            
            else:

                physical_group_tag = domain_physical_groups_dict[
                physical_group]

            # Gets the instance of the mesh data

            if isinstance(mesh_dict[physical_group_tag], dict):

                raise NotImplementedError("Multiple element types per "+
                "physical group has not yet been updated to compute Co"+
                "mpressibleInternalWorkReferenceConfiguration")
            
            # Gets the first element type

            mesh_data = mesh_dict[physical_group_tag]

            # Gets the identity tensor for the mesh within this physical
            # group

            identity_tensor = tf.eye(3, batch_shape=[
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

        # Gets the number of materials

        self.n_materials = len(self.first_piola_kirchhoff_list)

        # Stacks the derivatives of the shape functions multiplied by the
        # integration measure into a tensor [n_physical_groups, 
        # n_elements, n_quadrature_points, n_nodes, n_physical_dimensions]

        self.variation_gradient_dx = tf.stack(self.variation_gradient_dx,
        axis=0)

    # Defines a function to assemble the residual vector

    @tf.function
    def assemble_residual_vector(self, global_residual_vector):

        # Iterates through the physical groups

        for i in range(self.n_materials):

            # Gets the batched tensor [n_elements, n_quadrature_points,
            # 3, 3] of the first Piola-Kirchhoff stress

            P = self.first_piola_kirchhoff_list[i](self.deformation_gradient_list[
            i].compute_batched_deformation_gradient())

            # Contracts the first Piola-Kirchhoff stress with the deri-
            # vatives of the shape functions multiplied by the integra-
            # tion measure to get the integration of the internal work 
            # of the variational form. Then, sums over the quadrature 
            # points, that are the second dimension (index 1 in python 
            # convention).
            # The result is a tensor [n_elements,  n_nodes, 
            # n_physical_dimensions]

            internal_work = tf.reduce_sum(tf.einsum('eqij,eqnj->eqni', 
            P, self.variation_gradient_dx[i]), axis=1)

            # Adds the contribution of this physical group to the global
            # residual vector. Uses the own tensor of DOF indexing to
            # scatter the local residual. Another dimension is added to
            # the indexing tensor to make it compatible with tensorflow
            # tensor_scatter_nd_add. Performs this change in place, as
            # global_residual_vector is a variable

            global_residual_vector.scatter_nd_add(tf.expand_dims(
            self.deformation_gradient_list[i].indexing_dofs_tensor, axis=
            -1), internal_work)