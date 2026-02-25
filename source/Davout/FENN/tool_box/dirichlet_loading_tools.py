# Routine to store methods to prescribe and enforce Dirichlet boundary
# conditions

import tensorflow as tf

from ..tool_box import parametric_curves_tools

from ..tool_box.mesh_info_tools import get_boundary_info_from_mesh_data_class

from ...PythonicUtilities.package_tools import load_classes_from_module

from ...PythonicUtilities.dictionary_tools import verify_obligatory_and_optional_keys

########################################################################
#                             Fixed support                            #
########################################################################

# Defines a class to strongly enforce homogeneous displacements in the
# nodes of a region

class FixedSupportDirichletBC:

    def __init__(self, mesh_data_class, dirichlet_information, 
    vector_of_parameters, physical_group_name, time, n_realizations):

        # Verifies the keys of the dictionary of boundary condition in-
        # formation

        verify_obligatory_and_optional_keys(dirichlet_information, ["B"+
        "C case", "field name"], {"list of realizations with this BC": {
        "type": list}}, "dirichlet_information", "FixedSupportDirichle"+
        "tBC")
        
        # Gets information from the mesh data class, such as numerical
        # types and DOFs indices
        
        (integer_dtype, float_dtype, mesh_data, physical_group_tag
        ) = get_boundary_info_from_mesh_data_class(mesh_data_class, 
        physical_group_name, "Dirichlet boundary conditions", "Dirichl"+
        "etBoundaryConditions", dirichlet_information["field name"])
        
        # Recovers the tensor of DOFs per element [n_element, n_nodes,
        # n_dofs_per_node], and flattens it to [number_of_dofs, 1]

        self.dofs_per_element = []

        # Iterates through the 3 dimensions

        for dof in range(3):

            # Gets a unique tensor of DOFs (no repetition for different
            # elements and nodes)

            unique_tensor_dofs, _ = tf.unique(tf.reshape(
            mesh_data.dofs_per_element[..., dof], (-1,)))

            self.dofs_per_element.append(unique_tensor_dofs)

        # Verifies if the user has given any list of realizations that
        # do have this boundary condition

        realizations_range = None

        if "list of realizations with this BC" in dirichlet_information:

            realizations_range = dirichlet_information["list of realiz"+
            "ations with this BC"]
            
            # Transforms it to an integer tensor

            realizations_range = tf.constant(realizations_range, dtype=
            integer_dtype)

        # Otherwise, creates a simple range

        else:

            realizations_range = tf.range(n_realizations, dtype=
            integer_dtype)

        # Stack all fixed DOFs into [n_fixed_dofs]

        self.dofs_per_element = tf.reshape(tf.stack(
        self.dofs_per_element, axis=0), [-1])

        self.n_fixed_dofs = tf.shape(self.dofs_per_element)[0]

        self.n_selected_realizations = tf.shape(realizations_range)[0]

        # Verifies if the maximum realization number is within bounds to
        # the global number of realizations

        tf.debugging.assert_less(tf.reduce_max(realizations_range),
        n_realizations, message="The maximum realization index in 'Fix"+
        "edSupportDirichletBC' is bigger than the provided global numb"+
        "er of realizations, which is "+str(n_realizations))

        # Precomputes thes zeros to write into the vector of parameters,
        # the shape of this tensor is first [n_realizations, 
        # n_fixed_dofs], then, [n_realizations*n_fixed_dofs]

        self.prescribed_values = tf.reshape(tf.zeros([
        self.n_selected_realizations, self.n_fixed_dofs], dtype=
        float_dtype), [-1])

        # Broadcasts DOF indices along the realizations axis

        dofs_indices = tf.broadcast_to(self.dofs_per_element[None, :], [
        self.n_selected_realizations, self.n_fixed_dofs])

        # Broadcasts the realization indices to make pairs with DOF in-
        # dices

        realization_indices = tf.broadcast_to(realizations_range[:, None
        ], [self.n_selected_realizations, self.n_fixed_dofs])

        # Stacks indices for scatter_nd_update: [n_realizations, 
        # n_fixed_dofs, 2]

        self.scatter_indices = tf.reshape(tf.stack([realization_indices, 
        dofs_indices], axis=-1), [-1, 2])

    # Defines a function to update loads

    @tf.function
    def update_load_curve(self):

        pass

########################################################################
#                    Prescribed value and direction                    #
########################################################################

# Defines a class to apply particular values of displacement to some de
# grees of freedom

class PrescribedDirichletBC:

    def __init__(self, mesh_data_class, dirichlet_information, 
    vector_of_parameters, physical_group_name, time, n_realizations):

        # Verifies the dictionary keys

        verify_obligatory_and_optional_keys(dirichlet_information, {"d"+
        "egrees_ofFreedomList": {"description": "list of integers with"+
        " the local indices of the degrees of freedom to be prescribed"+
        " (the first index is 0), at physical group '"+str(
        physical_group_name)+"'"}, 
        "BC case": {"description": "key to store the name of the class"+
        " to apply a particular boundary condition"},
        "field name": {"type": str, "description": "key to store the n"+
        "ame of the field to which the BC must be applied"},
        "end_point": {"type": list, "description": "list of a value co"+
        "rresponding to the final time at the first component and a li"+
        "st of prescribed values at the second component. At physical "+
        "group '"+str(physical_group_name)+"'"}}, {"list of realizatio"+
        "ns with this BC": {"type": list},
        "load_function": {"type": str, "description": "name of the par"+
        "ametric curve to generate load steps"}}, "dirichlet_informati"+
        "on", "PrescribedDirichletBC")
        
        # Gets information from the mesh data class, such as numerical
        # types and DOFs indices
        
        (integer_dtype, float_dtype, mesh_data, physical_group_tag
        ) = get_boundary_info_from_mesh_data_class(mesh_data_class, 
        physical_group_name, "Dirichlet boundary conditions", "Dirichl"+
        "etBoundaryConditions", dirichlet_information["field name"])
        
        # Gets the available parametric curves

        available_parametric_curves = load_classes_from_module(
        parametric_curves_tools, return_dictionary_of_classes=True)

        # Verifies if the user has given any list of realizations that
        # do have this boundary condition

        realizations_range = None

        if "list of realizations with this BC" in dirichlet_information:

            realizations_range = dirichlet_information["list of realiz"+
            "ations with this BC"]
            
            # Transforms it to an integer tensor

            realizations_range = tf.constant(realizations_range, dtype=
            integer_dtype)

        # Otherwise, creates a simple range

        else:

            realizations_range = tf.range(n_realizations, dtype=
            integer_dtype)

        # Gets the number of selected realizations

        self.n_selected_realizations = tf.shape(realizations_range)[0]

        # Verifies if the maximum realization number is within bounds to
        # the global number of realizations

        tf.debugging.assert_less(tf.reduce_max(realizations_range),
        n_realizations, message="The maximum realization index in 'Pre"+
        "scribedDirichletBC' is bigger than the provided global number"+
        " of realizations, which is "+str(n_realizations))

        # Gets the keys
        
        prescribed_dofs_list = dirichlet_information["degrees_ofFreedo"+
        "mList"]
        
        value_prescription = dirichlet_information["end_point"]

        final_time = None

        # Verifies if the value description is a list with the proper
        # size

        if len(value_prescription)!=2:

            raise TypeError("'end_point' provided to 'PrescribedDirich"+
            "letBC' at physical group '"+str(physical_group_name)+"' i"+
            "s a list with length different than 2. It must be a list "+
            "with the final time at the first component and a list of "+
            "prescribed values at the second component. Currently it i"+
            "s: "+str(value_prescription))
        
        # Gets the final time

        final_time = value_prescription[0]

        # Verifies if the second component is a list or an float

        if isinstance(value_prescription[1], float):

            value_prescription[1] = [value_prescription[1]]

        if not isinstance(value_prescription[1], list):

            raise TypeError("'end_point' provided to 'PrescribedDirich"+
            "letBC' at physical group '"+str(physical_group_name)+"' h"+
            "as its second component not as a list. It must be a list "+
            "prescribed values at the second component, and they must "+
            "correspond to the list of prescribed DOFs. Currently it i"+
            "s: "+str(value_prescription[1]))
        
        # Transforms the value prescription to its second component

        value_prescription = value_prescription[1]

        # Verifies if the list of prescribed DOFs is an integer

        if isinstance(prescribed_dofs_list, int):

            # Puts it into a list

            prescribed_dofs_list = [prescribed_dofs_list]

        # Verifies if a parametric curve is asked for

        load_class = None

        if "load_function" in dirichlet_information:

            # Checks if it is an available curve

            load_name = dirichlet_information["load_function"]

            if load_name in available_parametric_curves:

                load_class = available_parametric_curves[load_name]

            else:

                names = ""

                for name in available_parametric_curves:

                    names += "\n"+str(name)

                raise NameError("'load_function' provided to 'Prescrib"+
                "edDirichletBC' at physical group '"+str(
                physical_group_name)+"' has the name '"+str(load_name)+
                "', but it is not an available parametric load. Check "+
                "the available methods:"+names)
            
        else:

            load_class = available_parametric_curves["linear"]

        # Creates a list of load instances

        self.list_of_load_instances = []

        # Initializes a list of DOFs to be prescribed

        dofs_list = []

        # Verifies if it is a list. elif is not used to assert the value
        # if an integer was given

        if isinstance(prescribed_dofs_list, list):

            # Verifies if it is empty

            if len(prescribed_dofs_list)==0:

                raise ValueError("The list of 'degrees_ofFreedomList' "+
                "provided to 'PrescribedDirichletBC' at physical group"+
                " '"+str(physical_group_name)+"' is empty. At leat one"+
                " degree of freedom (local index) must be given and ut"+
                "most 3")
            
            # Verifies if it exceeds three

            elif len(prescribed_dofs_list)>3:

                raise ValueError("The list of 'degrees_ofFreedomList' "+
                "provided to 'PrescribedDirichletBC' at physical group"+
                " '"+str(physical_group_name)+"' is has length of "+str(
                len(prescribed_dofs_list))+". Utmost 3 degrees of free"+
                "dom are allowed")
            
            # Verifies if the list of prescribed values have the same 
            # number of values as the number of prescribed DOFs

            elif len(prescribed_dofs_list)!=len(value_prescription):

                raise IndexError("The list of 'degrees_ofFreedomList' "+
                "provided to 'PrescribedDirichletBC' at physical group"+
                " '"+str(physical_group_name)+"' is has length of "+str(
                len(prescribed_dofs_list))+", whereas the list of pres"+
                "cribed values has length of "+str(len(
                value_prescription))+". They must have the same length"+
                ". Check the list of prescribed DOFs: "+str(
                prescribed_dofs_list)+"\nand the list of prescribed va"+
                "lues: "+str(value_prescription))
            
            # Verifies each component if they are integers between 0 and
            # 2

            for dof, value in zip(prescribed_dofs_list, value_prescription):

                if (not isinstance(dof, int)) or dof<0 or dof>2:

                    raise ValueError("the DOF "+str(dof)+" given in th"+
                    "e list of 'degrees_ofFreedomList' provided to 'Pr"+
                    "escribedDirichletBC' at physical group '"+str(
                    physical_group_name)+"' is not allowed. Each DOF m"+
                    "ust be either 0, 1, or 2. The given list is: "+str(
                    prescribed_dofs_list))
                
                # Appends all DOFs of the mesh that possess this local
                # index. But gets just one occurence of each DOF

                unique_tensor_dofs, _ = tf.unique(tf.reshape(
                mesh_data.dofs_per_element[..., dof], (-1,)))

                dofs_list.append(unique_tensor_dofs)

                # Gets the value and transforms it into a load. Multi-
                # plied by the tensor already

                load_class_instance = load_class(time, value*tf.ones(
                unique_tensor_dofs.shape, dtype=float_dtype), 
                final_time)

                # Updates the value and appends this instance to a load
                # instances list

                self.list_of_load_instances.append(load_class_instance)

        else:

            raise TypeError("'degrees_ofFreedomList' provided to 'Pres"+
            "cribedDirichletBC' at physical group '"+str(
            physical_group_name)+"' is not a list nor an integer. Curr"+
            "ently it is: "+str(prescribed_dofs_list))

        # Stacks the list of prescribed DOFs back into a tensor, and re-
        # shapes it to a flat tensor

        prescribed_dofs = tf.reshape(tf.stack(dofs_list, axis=0), [
        -1])

        self.n_prescribed_dofs = tf.shape(prescribed_dofs)[0]

        # Broadcasts the DOFs indices to account for the realizations

        dofs_indices = tf.broadcast_to(prescribed_dofs[None, :], [
        self.n_selected_realizations, self.n_prescribed_dofs])

        # Broadcasts the realizations indices

        realization_indices = tf.broadcast_to(realizations_range[:, None
        ], [self.n_selected_realizations, self.n_prescribed_dofs])

        # Builds the scatter indices tensor by concatenating realization
        # and DOF index informations

        self.scatter_indices = tf.reshape(tf.stack([realization_indices, 
        dofs_indices], axis=-1), [-1, 2])
                
        # Stacks the list of prescribed values in the same fashion

        # Gets the values by calling the loading classes

        values = tf.reshape(tf.stack([load_instance() for (
        load_instance) in self.list_of_load_instances], axis=0), [-1])

        # Broadcasts the values across realizations

        values = tf.broadcast_to(values[None, :], [
        self.n_selected_realizations, self.n_prescribed_dofs])

        # Creates a variable for the prescribed values

        self.prescribed_values = tf.Variable(tf.reshape(values, [-1]))

    # Defines a function to update loads

    @tf.function
    def update_load_curve(self):

        # Gets the values by calling the loading classes

        values = tf.reshape(tf.stack([load_instance() for (
        load_instance) in self.list_of_load_instances], axis=0), [-1])

        # Broadcasts the values across realizations

        values = tf.broadcast_to(values[None, :], [
        self.n_selected_realizations, self.n_prescribed_dofs])

        # Creates a variable for the prescribed values

        self.prescribed_values.assign(tf.reshape(values, [-1]))