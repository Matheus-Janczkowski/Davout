# Routine to store methods to calculate the loss function given a tensor
# of batched residual vectors

import tensorflow as tf

from ..optimization import conditioning_matrices

from ..optimization import loss_functions

from ...PythonicUtilities.package_tools import load_classes_from_module

# Defines a class to store the conditioning matrices and to calculate 
# the loss function

class LossFunction:

    def __init__(self, conditioning_matrix_list, loss_function_dict,
    vector_of_parameters):
        
        # Saves code-given information

        self.n_realizations = vector_of_parameters.shape[0]

        self.n_global_dofs = vector_of_parameters.shape[1]

        float_dtype = vector_of_parameters.dtype

        # Verifies if the list of conditioning matrices is a dictionary

        if isinstance(conditioning_matrix_list, dict):

            # Converts to a list

            conditioning_matrix_list = [conditioning_matrix_list]

        # Verifies if the list of conditioning matrices is not a list

        elif not isinstance(conditioning_matrix_list, list):

            raise TypeError("'conditioning_matrix_list' in 'LossFuncti"+
            "on' must be a list of dictionaries whose keys are: the na"+
            "me of the class that construct such conditioning matrix a"+
            "nd other keys for additional information particular to th"+
            "e conditioner method")

        # Gets the available classes to construct the conditioning ma-
        # trices

        available_conditioning_classes = load_classes_from_module(
        conditioning_matrices, return_dictionary_of_classes=True)

        # Initializes a list of class instances for the conditioning ma-
        # trices 

        self.conditioning_matrices_instances = []

        # Iterates through the list of names of conditioning matrices

        for index, condition_info in enumerate(conditioning_matrix_list):

            # Verifies if condition info is a dictionary

            if (not isinstance(condition_info, dict)) or (not ("condit"+
            "ioner name" in condition_info)):

                raise TypeError("The "+str(index+1)+"-th condition inf"+
                "o inside the list 'conditioning_matrix_list' in 'Loss"+
                "Function' is not a dictionary or does not have the ke"+
                "y 'conditioner name'. It must be a dictionary with, a"+
                "t least, the key 'conditioner name' to set the method"+
                " that computes the conditioning matrix. Currently it "+
                "is: "+str(condition_info))

            # Verifies if it is a valid conditioning matrix

            condition_name = condition_info["conditioner name"]

            if not (condition_name in available_conditioning_classes):

                available_names = ""

                for name in available_conditioning_classes.keys():

                    available_names += "\n'"+str(name)+"'"

                raise ValueError("'"+str(condition_name)+"' in 'LossFu"+
                "nction' is not a valid name for conditioning matrix. "+
                "Check the available names:"+available_names)
            
            # Instantiates the class that generates the conditioning ma-
            # trix and appends it to the list of class instances

            self.conditioning_matrices_instances.append(
            available_conditioning_classes[condition_name](
            condition_info, float_dtype, self.n_global_dofs))

        # Verifies if there are conditioning matrices at all

        if len(self.conditioning_matrices_instances)==0:

            raise ValueError("No conditioning matrices in 'LossFunctio"+
            "n' have been given. Check 'conditioning_matrix_list':\n"+
            str(conditioning_matrix_list))

        # Verifies if there is a single conditioning matrix for all BVP
        # realizations

        elif len(self.conditioning_matrices_instances)==1:

            # Sets the method to evaluate the loss function

            self.evaluate_loss = self.broadcast_single_matrix

            # Converts the list of conditioning matrices to its unique 
            # value

            self.conditioning_matrices_instances = (
            self.conditioning_matrices_instances[0])

        # Verifies if the length of the list of instances is equal to 
        # the number of realizations

        elif len(self.conditioning_matrices_instances
        )!=self.n_realizations:
            
            raise IndexError(str(len(self.conditioning_matrices_instances
            ))+" conditioning matrices have been requested; but there "+
            "are "+str(self.n_realizations)+" BVP realizations. If mul"+
            "tiple conditioning matrices are required, they are realiz"+
            "ation-specific and, for this reason, there must be as man"+
            "y conditioning matrices as realizations")
        
        # If the number of realizations is consistent with the number of
        # conditioning matrices, gets the appropriate function to stack
        # all conditioned residual vectors and evaluate the loss function

        else:

            self.evaluate_loss = self.broadcast_multiple_matrices

            # Converts the list of instances of the classes to build 
            # conditioning matrices to a tuple

            self.conditioning_matrices_instances = tuple(
            self.conditioning_matrices_instances)

        # Gets the available loss function classes

        available_loss_classes = load_classes_from_module(
        loss_functions, return_dictionary_of_classes=True)

        # Verifies if the loss function name is present

        if not ("loss function name" in loss_function_dict):

            available_names = ""

            for name in available_loss_classes:

                available_names += "\n'"+str(name)+"'"

            raise KeyError("The key 'loss function name' is not presen"+
            "t in 'loss_function_dict' at 'LossFunction' class. This k"+
            "ey is obligatory and should have one of the following val"+
            "ues:"+available_names)
        
        # Instantiates the loss function class

        self.loss_function_class = available_loss_classes[
        loss_function_dict["loss function name"]](loss_function_dict)

    # Defines a function to broadcast a single conditioning matrix for
    # all realizations

    @tf.function
    def broadcast_single_matrix(self, vector_of_parameters, 
    global_residual_vector):

        # Does not gathers any values from the vector of parameters or 
        # global residual vectors. But send them as whole to the condi-
        # tioning matrix class. Then, get the conditioned residual vector

        conditioned_residual_vector = self.conditioning_matrices_instances(
        vector_of_parameters, global_residual_vector)

        # Gets the loss function value

        return self.loss_function_class(vector_of_parameters, 
        global_residual_vector, conditioned_residual_vector)

    # Defines a function to send each column of the residual vector to 
    # the corresponding conditioning matrix class

    @tf.function
    def broadcast_multiple_matrices(self, vector_of_parameters, 
    global_residual_vector):
        
        # Gets the conditioned residual vector

        conditioned_residual_vector = tf.stack([matrix_class(
        vector_of_parameters[index,:], global_residual_vector[index,:]
        ) for index, matrix_class in enumerate(
        self.conditioning_matrices_instances)], axis=0)

        # Gets the loss function value

        return self.loss_function_class(vector_of_parameters, 
        global_residual_vector, conditioned_residual_vector)