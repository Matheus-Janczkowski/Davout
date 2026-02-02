# Routine to store methods to prescribe and enforce Dirichlet boundary
# conditions

import tensorflow as tf

from ..tool_box import parametric_curves_tools

from ...PythonicUtilities.package_tools import load_classes_from_module

########################################################################
#                             Fixed support                            #
########################################################################

# Defines a class to strongly enforce homogeneous displacements in the
# nodes of a region

class FixedSupportDirichletBC:

    def __init__(self, mesh_data_class, dirichlet_information, 
    vector_of_parameters, physical_group_name, time):
        
        # Recovers the tensor of DOFs per element [n_element, n_nodes,
        # n_dofs_per_node], and flattens it to [number_of_dofs, 1]

        self.dofs_per_element = []

        # Iterates through the 3 dimensions

        for dof in range(3):

            # Gets a unique tensor of DOFs (no repetition for different
            # elements and nodes)

            unique_tensor_dofs, _ = tf.unique(tf.reshape(
            mesh_data_class.dofs_per_element[..., dof], (-1,)))

            self.dofs_per_element.append(unique_tensor_dofs)

        # Stacks and flattens the tensor of DOFs

        self.dofs_per_element = tf.reshape(tf.stack(
        self.dofs_per_element, axis=0), (-1,1))

        # Creates a null tensor of shape [n_dofs]

        self.null_tensor = tf.zeros((self.dofs_per_element.shape[0],), 
        dtype=mesh_data_class.dtype)

    # Defines a function to apply such boundary conditions

    @tf.function
    def apply(self, vector_of_parameters):

        # Assigns values to the vector of parameters in place

        vector_of_parameters.scatter_nd_update(self.dofs_per_element, 
        self.null_tensor)

########################################################################
#                    Prescribed value and direction                    #
########################################################################

# Defines a class to apply particular values of displacement to some de
# grees of freedom

class PrescribedDirichletBC:

    def __init__(self, mesh_data_class, dirichlet_information, 
    vector_of_parameters, physical_group_name, time):
        
        # Gets the available parametric curves

        available_parametric_curves = load_classes_from_module(
        parametric_curves_tools, return_dictionary_of_classes=True)
        
        # Verifies if the dictionary of information has the key for the 
        # DOFs to be prescribed

        if not ("degrees_ofFreedomList" in dirichlet_information):

            raise KeyError("The dictionary of information for 'Prescri"+
            "bedDirichletBC' at physical group '"+str(physical_group_name
            )+"' does not have the key 'degrees_ofFreedomList' whose v"+
            "alue is a list of an integer with the local indices of th"+
            "e degrees of freedom to be prescribed (the first index is"+
            " 0)")
        
        prescribed_dofs_list = dirichlet_information["degrees_ofFreedo"+
        "mList"]
        
        # Verifies if the dictionary of information has the key for the 
        # values of the DOFs to be prescribed

        if not ("end_point" in dirichlet_information):

            raise KeyError("The dictionary of information for 'Prescri"+
            "bedDirichletBC' at physical group '"+str(physical_group_name
            )+"' does not have the key 'end_point', whose value is a l"+
            "ist of a value corresponding to the final time at the fir"+
            "st component and a list of prescribed values at the secon"+
            "d component")
        
        value_prescription = dirichlet_information["end_point"]

        final_time = None

        # Verifies if the value description is a list

        if not isinstance(value_prescription, list):

            raise TypeError("'end_point' provided to 'PrescribedDirich"+
            "letBC' at physical group '"+str(physical_group_name)+"' i"+
            "s not a list. It must be a list with the final time at th"+
            "e first component and a list of prescribed values at the "+
            "second component. Currently it is: "+str(value_prescription))
        
        else:

            if len(value_prescription)!=2:

                raise TypeError("'end_point' provided to 'PrescribedDi"+
                "richletBC' at physical group '"+str(physical_group_name
                )+"' is a list with length different than 2. It must b"+
                "e a list with the final time at the first component a"+
                "nd a list of prescribed values at the second componen"+
                "t. Currently it is: "+str(value_prescription))
            
            # Gets the final time

            final_time = value_prescription[0]

            # Verifies if the second component is a list or an float

            if isinstance(value_prescription[1], float):

                value_prescription[1] = [value_prescription[1]]

            if not isinstance(value_prescription[1], list):

                raise TypeError("'end_point' provided to 'PrescribedDi"+
                "richletBC' at physical group '"+str(physical_group_name
                )+"' has its second component not as a list. It must b"+
                "e a list prescribed values at the second component, a"+
                "nd they must correspond to the list of prescribed DOF"+
                "s. Currently it is: "+str(value_prescription[1]))
            
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
                mesh_data_class.dofs_per_element[..., dof], (-1,)))

                dofs_list.append(unique_tensor_dofs)

                # Gets the value and transforms it into a load. Multi-
                # plied by the tensor already

                load_class_instance = load_class(time, value*tf.ones(
                unique_tensor_dofs.shape, dtype=mesh_data_class.dtype), 
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

        self.prescribed_dofs = tf.reshape(tf.stack(dofs_list, axis=0), (
        -1,1))
        
        # Stacks the list of prescribed values in the same fashion

        self.prescribed_values = tf.Variable(tf.reshape(tf.stack(
        [load_instance() for load_instance in (
        self.list_of_load_instances)], axis=0), (-1,)))

    # Defines a function to update loads

    def update_load_curve(self):
        
        # Stacks the list of prescribed values in the same fashion

        self.prescribed_values.assign(tf.reshape(tf.stack(
        [load_instance() for load_instance in (
        self.list_of_load_instances)], axis=0), (-1,)))

    # Defines a function to apply such boundary conditions

    @tf.function
    def apply(self, vector_of_parameters):

        # Assigns values to the vector of parameters in place

        vector_of_parameters.scatter_nd_update(self.prescribed_dofs, 
        self.prescribed_values)