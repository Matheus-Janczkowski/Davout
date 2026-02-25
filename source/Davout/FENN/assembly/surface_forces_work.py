# Routine to store methods to calculate the residual vector due to sur-
# face forces

import tensorflow as tf

from ...PythonicUtilities.package_tools import load_classes_from_module

from ..tool_box import neumann_loading_tools

from ..tool_box.mesh_info_tools import get_boundary_info_from_mesh_data_class

# Defines a class to compute the contribution to the residual vector due
# to the surface tractions

class ReferentialTractionWork:

    def __init__(self, n_realizations, traction_dict, 
    mesh_data_class, field_name, time):
        
        # Gets the number of batched BVP instances

        self.n_realizations = n_realizations

        # Creates a list of the variation of the primal field at the 
        # surfaces multiplied by the surface integration measure

        self.variation_field_ds = []

        # Initializes a list of instances of classes that generate ten-
        # sors of traction vectors

        self.traction_classes = []

        # Initializes a list of indices to update the global residual
        # vector across different realizations of the BVP

        self.updates_indices = []

        # Gets the available classes to construct the traction tensors

        available_traction_classes = load_classes_from_module(
        neumann_loading_tools, return_dictionary_of_classes=True)

        # Verifies if traction dictionary is None

        if traction_dict is None:

            traction_dict = {}

        elif not isinstance(traction_dict, dict):

            raise TypeError("The dictionary of tractions must be a dic"+
            "tionary. Currently, it is: "+str(traction_dict))

        # Iterates through the dictionary of tractions

        for physical_group, traction_vector in traction_dict.items():
            
            # Gets necessary information from the mesh data class, such
            # as the dictionary of boundary finite elements

            (mesh_data, physical_group_tag, mesh_common_info
            ) = get_boundary_info_from_mesh_data_class(mesh_data_class, 
            physical_group, "Neumann boundary conditions", "Referentia"+
            "lTractionWork", field_name)

            # Checks if the traction vector is a dictionary

            if not isinstance(traction_vector, dict):

                raise TypeError("The dictionary of tractions must be d"+
                "ictionary of dictionaries, i.e. the keys are the surf"+
                "ace physical groups names and the values are dictiona"+
                "ries with instructions for the classes of Neumann bou"+
                "ndary conditions. Each one of these dictionaries must"+
                " have the key 'load case', whose value is the string "+
                "with the name of the class to apply Neumann boundary "+
                "conditions")

            # Verifies if the traction vector has the key to store the
            # name of the class to create the traction tensor
                        
            elif not ('load case' in traction_vector):

                raise KeyError("At physical group '"+str(physical_group)+
                "', the traction dictionary does not have the key 'loa"+
                "d case'. Check ou the given dictionary: "+str(
                traction_vector))
            
            # Verifies if the load case is a valid name

            elif not (traction_vector["load case"] in (
            available_traction_classes)):
                
                names = ""

                for name in available_traction_classes.keys():

                    names += "\n"+str(name)
                
                raise ValueError("The 'load case' given is '"+str(
                traction_vector["load case"])+"', but this is not an a"+
                "vailable method to construct Neumann boundary conditi"+
                "ons. Check the available methods to create tractions "+
                "on the boundary:"+names)

            # Instantiates and adds the class with the mesh data

            self.traction_classes.append(available_traction_classes[
            traction_vector["load case"]](mesh_data, traction_vector,
            physical_group, self.n_realizations, mesh_common_info, time))

            # Verifies if there are multiple realizations of the mesh

            if isinstance(mesh_data, list):

                # Gets the shape functions multiplied by the surface in-
                # tegration measure. Uses the attribute dx, because the 
                # element is 2D, thus the ds for the 3D mesh is the ele-
                # ment's dx.
                # Note, however, that, in this case, the integration
                # measure must be stacked together to include the reali-
                # zations dimension

                self.variation_field_ds.append(tf.einsum('qn,peq->peqn',
                mesh_common_info.shape_functions_tensor, tf.stack([
                mesh_data_instance.dx for mesh_data_instance in (
                mesh_data)], axis=0)))

                # Selects the approapriate function to contract the 
                # traction tensor with the variation of the field

                self.appropiate_contraction = self.contract_multiple_meshes

            # Otherwise, there a single instance of the mesh

            else:

                # Gets the shape functions multiplied by the surface in-
                # tegration measure. Uses the attribute dx, because the 
                # element is 2D, thus the ds for the 3D mesh is the ele-
                # ment's dx

                self.variation_field_ds.append(tf.einsum('qn,eq->eqn',
                mesh_common_info.shape_functions_tensor, mesh_data.dx))

                # Selects the approapriate function to contract the 
                # traction tensor with the variation of the field

                self.appropiate_contraction = self.contract_single_mesh

            # Creates the indices for updating the global residual vector
            # batched along the different realizations of the BVP

            # Creates the realizations indices and expands for broadcas-
            # ting

            realization_indices = tf.range(self.n_realizations, dtype=
            mesh_common_info.integer_dtype)[:, None, None, None]

            # Broadcasts to match the realizations dimension as the 
            # trailing dimension

            realization_indices = tf.broadcast_to(realization_indices, [
            self.n_realizations, tf.shape(
            mesh_common_info.dofs_per_element)[0], tf.shape(
            mesh_common_info.dofs_per_element)[1], tf.shape(
            mesh_common_info.dofs_per_element)[2]])

            # Expands DOF indices and broadcasts to the realization di-
            # mension

            dofs_indices = tf.broadcast_to(tf.expand_dims(
            mesh_common_info.dofs_per_element, axis=0), tf.shape(
            realization_indices))

            # Stacks them both to form a concatenation (cartesian product 
            # of realization and DOFs indices)

            self.updates_indices.append(tf.stack([realization_indices,
            dofs_indices], axis=-1))

        # Gets the number of surfaces under load

        self.n_surfaces_under_load = len(self.variation_field_ds)

        # Stacks the shape functions multiplied by the integration mea-
        # sure into a tensor [n_physical_groups, n_elements, 
        # n_quadrature_points, n_nodes] or [n_physical_groups, 
        # n_realizations, n_elements, n_quadrature_points, n_nodes] in 
        # case of multiple realizations of the mesh

        self.variation_field_ds = tf.stack(self.variation_field_ds,
        axis=0)

        # Makes traction_classes a tuple to show its immutability

        self.traction_classes = tuple(self.traction_classes)

        # Stacks all tensors of DOF indices of all traction classes
        
        self.all_indices = tf.concat(self.updates_indices, axis=0)

        # Iterates through the loaded surfaces to stack all external work
        # at once

        # Contracts the referential traction vector with the shape func-
        # tions multiplied by the integration measure to get the inte-
        # gration of the variation of the external work due to surface
        # tractions. Then, sums over the quadrature points, that are the
        # second dimension (index 1 in python convention). The result is 
        # a tensor [n_realizations, n_elements, n_nodes, 
        # n_physical_dimensions]

        self.all_external_work = tf.Variable(tf.concat([
        self.appropiate_contraction(i) for i in range(
        self.n_surfaces_under_load)], axis=0))

    # Defines a function to contract the traction tensor with the varia-
    # tion of the field if the mesh is the same across all realizations

    @tf.function
    def contract_single_mesh(self, i):

        return -tf.reduce_sum(tf.einsum('peqi,eqn->peqni', 
        self.traction_classes[i].traction_tensor, 
        self.variation_field_ds[i]), axis=2)

    # Defines a function to contract the traction tensor with the varia-
    # tion of the field if there are multiple realizations of the mesh

    @tf.function
    def contract_multiple_meshes(self, i):

        return -tf.reduce_sum(tf.einsum('peqi,peqn->peqni', 
        self.traction_classes[i].traction_tensor, 
        self.variation_field_ds[i]), axis=2)

    # Defines a function to update the loads

    @tf.function
    def update_boundary_conditions(self, vector_of_parameters):

        # Iterates through the loaded surfaces to update the calculation
        # of the traction tensor

        for i in range(self.n_surfaces_under_load):

            self.traction_classes[i].compute_traction(
            vector_of_parameters)

        # Iterates through the loaded surfaces to stack all external work
        # at once

        # Contracts the referential traction vector with the shape func-
        # tions multiplied by the integration measure to get the inte-
        # gration of the variation of the external work due to surface
        # tractions. Then, sums over the quadrature points, that are the
        # second dimension (index 1 in python convention). The result is 
        # a tensor [n_realizations, n_elements, n_nodes, 
        # n_physical_dimensions]

        self.all_external_work.assign(tf.concat([
        self.appropiate_contraction(i) for i in range(
        self.n_surfaces_under_load)], axis=0))

    # Defines a function to assemble the residual vector

    @tf.function
    def assemble_residual_vector(self, global_residual_vector):
        
        # Adds the contribution of this physical group to the global re-
        # sidual vector. Uses the own tensor of DOF indexing to scatter
        # the local residual. Another dimension was added to the indexing 
        # tensor to make it compatible with tensorflow 
        # tensor_scatter_nd_add. Performs this change in place, as 
        # global_residual_vector is a variable
        
        global_residual_vector.scatter_nd_add(self.all_indices,
        self.all_external_work)