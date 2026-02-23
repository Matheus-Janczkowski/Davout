# Routine to store methods to calculate the residual vector due to sur-
# face forces

import tensorflow as tf

from ...PythonicUtilities.package_tools import load_classes_from_module

from ..tool_box import neumann_loading_tools

# Defines a class to compute the contribution to the residual vector due
# to the surface tractions

class ReferentialTractionWork:

    def __init__(self, vector_of_parameters, traction_dict, mesh_dict,
    boundary_physical_groups_dict):
        
        # Gets the number of batched BVP instances

        self.n_realizations = vector_of_parameters.shape[0]

        # Creates a list of the variation of the primal field at the 
        # surfaces multiplied by the surface integration measure

        self.variation_field_ds = []

        # Initializes a list of instances of classes that generate ten-
        # sors of traction vectors

        self.traction_classes = []

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
            
            # Gets the physical group tag

            physical_group_tag = None 

            if not (physical_group in boundary_physical_groups_dict):

                raise NameError("The physical group name '"+str(
                physical_group)+"' in the dictionary of tractions is n"+
                "ot a valid physical group name. Check the available n"+
                "ames:\n"+str(list(boundary_physical_groups_dict.keys())))
            
            else:

                physical_group_tag = boundary_physical_groups_dict[
                physical_group]

            # Gets the instance of the mesh data

            if isinstance(mesh_dict[physical_group_tag], dict):

                raise NotImplementedError("Multiple element types per "+
                "physical group has not yet been updated to compute Re"+
                "ferentialTractionWork")
            
            # Gets the first element type

            mesh_data = mesh_dict[physical_group_tag]

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
            vector_of_parameters, physical_group, self.n_realizations))

            # Gets the shape functions multiplied by the surface inte-
            # gration measure. Uses the attribute dx, because the el-
            # ement is 2D, thus the ds for the 3D mesh is the element's
            # dx

            self.variation_field_ds.append(tf.einsum('qn,eq->eqn',
            mesh_data.shape_functions_tensor, mesh_data.dx))

        # Gets the number of surfaces under load

        self.n_surfaces_under_load = len(self.variation_field_ds)

        # Stacks the shape functions multiplied by the integration mea-
        # sure into a tensor [n_physical_groups, n_elements, 
        # n_quadrature_points, n_nodes]

        self.variation_field_ds = tf.stack(self.variation_field_ds,
        axis=0)

        # Makes traction_classes a tuple to show its immutability

        self.traction_classes = tuple(self.traction_classes)

    # Defines a function to assemble the residual vector

    @tf.function
    def assemble_residual_vector(self, global_residual_vector):

        # Iterates through the physical groups of the surfaces under load

        for i in range(self.n_surfaces_under_load):

            # Contracts the referential traction vector with the shape 
            # functions multiplied by the integration measure to get the 
            # integration of the variation of the external work due to 
            # surface tractions. Then, sums over the quadrature points, 
            # that are the second dimension (index 1 in python conventi-
            # on). The result is a tensor [n_realizations, n_elements, 
            # n_nodes, n_physical_dimensions]

            external_work = tf.reduce_sum(tf.einsum('peqi,eqn->peqni', 
            self.traction_classes[i].traction_tensor,
            self.variation_field_ds[i]), axis=2)

            # Adds the contribution of this physical group to the global
            # residual vector. Uses the own tensor of DOF indexing to
            # scatter the local residual. Another dimension was added to
            # the indexing tensor to make it compatible with tensorflow
            # tensor_scatter_nd_add. Performs this change in place, as
            # global_residual_vector is a variable

            dofs_per_element = self.traction_classes[i
            ].surface_mesh_data.dofs_per_element

            global_residual_vector.scatter_nd_add(tf.broadcast_to(
            tf.expand_dims(dofs_per_element, axis=0), [
            self.n_realizations, *dofs_per_element.shape]), 
            -external_work)