# Routine to apply Dirichlet boundary conditions

import tensorflow as tf

from ...PythonicUtilities.package_tools import load_classes_from_module

from ..tool_box import dirichlet_loading_tools

# Defines a class to impose Dirichlet boundary conditions

class DirichletBoundaryConditions:

    def __init__(self, vector_of_parameters, boundary_conditions_dict, 
    mesh_data_class, time):
        
        # Gets the number of batched BVP instances

        self.n_realizations = vector_of_parameters.shape[0]

        # Initializes a list of instances of classes that apply Dirichlet
        # boundary conditions

        self.BCs_classes = []

        # Gets the available classes to construct boundary conditions

        available_BCs_classes = load_classes_from_module(
        dirichlet_loading_tools, return_dictionary_of_classes=True)

        # Verifies if the dictionary of boundary conditions is None

        if boundary_conditions_dict is None:

            boundary_conditions_dict = {}

        elif not isinstance(boundary_conditions_dict, dict):

            raise TypeError("The dictionary of Dirichlet boundary cond"+
            "itions must be a dictionary. Currently, it is: "+str(
            boundary_conditions_dict))

        # Iterates through the dictionary of boundary conditions

        for physical_group, boundary_condition in boundary_conditions_dict.items():

            # Checks if the boundary condition is a dictionary

            if not isinstance(boundary_condition, dict):

                raise TypeError("The dictionary of Dirichlet boundary "+
                "conditions must be dictionary of dictionaries, i.e. t"+
                "he keys are the surface physical groups names and the"+
                " values are dictionaries with instructions for the cl"+
                "asses of Dirichlet boundary conditions. Each one of t"+
                "hese dictionaries must have the key 'BC case', whose "+
                "value is the string with the name of the class to app"+
                "ly Dirichlet boundary conditions")

            # Verifies if the boundary condition has the key to store the
            # name of the class to create it
                        
            elif not ('BC case' in boundary_condition):

                raise KeyError("At physical group '"+str(physical_group)+
                "', the dictionary of Dirichlet boundary condition doe"+
                "s not have the key 'BC case'. Check ou the given dict"+
                "ionary: "+str(boundary_condition))
            
            # Verifies if the BC case is a valid name

            elif not (boundary_condition["BC case"] in (
            available_BCs_classes)):
                
                names = ""

                for name in available_BCs_classes.keys():

                    names += "\n"+str(name)
                
                raise ValueError("The 'BC case' given is '"+str(
                boundary_condition["BC case"])+"', but this is not an "+
                "available method to construct Dirichlet boundary cond"+
                "itions. Check the available methods to create Dirichl"+
                "et boundary conditions:"+names)

            # Instantiates and adds the class with the mesh data

            self.BCs_classes.append(available_BCs_classes[
            boundary_condition["BC case"]](mesh_data_class, 
            boundary_condition, vector_of_parameters, physical_group, 
            time, self.n_realizations))

        # Gets the number of boundary conditions

        self.n_surfaces_under_load = len(self.BCs_classes)

        # Makes BCs_classes a tuple to show its immutability

        self.BCs_classes = tuple(self.BCs_classes)

        # Stacks all tensors of DOF indices of all BC classes
        
        self.all_indices = tf.concat([bc.scatter_indices for bc in (
        self.BCs_classes)], axis=0)

        # Iterates through the boundary conditions to update the loads

        for BC_class in self.BCs_classes:

            # Calls the method of the class to update the tensor with 
            # the prescribed values

            BC_class.update_load_curve()

        # Stacks all prescribed values of all BC classes into a variable

        self.all_values = tf.Variable(tf.concat([(bc.prescribed_values
        ) for bc in self.BCs_classes], axis=0))

        # Applies boundary conditions using the initial information

        self.apply_boundary_conditions(vector_of_parameters)

    # Defines a function to update the loads

    @tf.function
    def update_boundary_conditions(self):

        # Iterates through the boundary conditions

        for BC_class in self.BCs_classes:

            # Calls the method of the class to update the tensor with 
            # the prescribed values

            BC_class.update_load_curve()

        # Stacks all prescribed values of all BC classes

        self.all_values.assign(tf.concat([bc.prescribed_values for (bc
        ) in self.BCs_classes], axis=0))

    # Defines a function to apply boundary conditions

    @tf.function
    def apply_boundary_conditions(self, vector_of_parameters):

        # Updates the vector of parameters in place with the stacked 
        # tensors of indices and of prescribed values

        vector_of_parameters.scatter_nd_update(self.all_indices, 
        self.all_values)

        # TODO alter scatter_nd_update to tensor_scatter_nd_update to
        # return vector_of_parameters instead of modifying it in place