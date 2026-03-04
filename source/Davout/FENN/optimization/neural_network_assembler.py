# Routine to store methods to assemble a neural networks model to compu-
# te the vector of parameters and the loss function

import tensorflow as tf

import numpy as np

from ...DeepMech.tool_box.ANN_tools import MultiLayerModel

# Defines a class to store the neural network model and the evaluation 
# of the loss function

class AgentModel:

    def __init__(self, agent_dofs_list: (list | np.ndarray), 
    input_dimension: int, activation_functions_per_layer: list,  
    float_dtype, integer_dtype, vector_of_parameters, 
    loss_function_class, verbose=False):
        
        # Verifies if activation_functions_per_layer is a list of dicti-
        # onaries

        if not isinstance(activation_functions_per_layer, list):

            raise TypeError("'activation_functions_per_layer' at 'Agen"+
            "tModel' must be a list with dictionaries. Each dictionary"+
            "tells the activations functions and the number of neurons"+
            " for each activation function in the corresponding layer")
        
        for layer_index, layer_info in enumerate(
        activation_functions_per_layer):

            # Verifies if this layer info is a dictionary

            if not isinstance(layer_info, dict):

                raise TypeError("The "+str(layer_index+1)+"-th layer i"+
                "nformation in 'activation_functions_per_layer' is not"+
                " a dictionary. It must be a dictionary whose keys are"+
                " the names of the activation functions of the said la"+
                "yer and the corresponding values are the number of ne"+
                "urons for those activation functions")
        
        # Creates the artificial neural networks model for this agent

        agent_class = MultiLayerModel(input_dimension, 
        activation_functions_per_layer, parameters_dtype=float_dtype,
        verbose=verbose)

        self.agent_model = agent_class()

        # Verifies if agent_dofs_list is a list or a numpy array

        if (not isinstance(agent_dofs_list, list)) and (not isinstance(
        agent_dofs_list, np.ndarray)):
            
            raise TypeError("'agent_dofs_list' at 'AgentModel' is not "+
            "a list, nor a numpy array. It must be a list or a numpy a"+
            "rray with the indices of the DOFs given by this agent mod"+
            "el")
        
        # Gets the number of realizations and the number of DOFs in the
        # whole mesh

        self.n_realizations = vector_of_parameters.shape[0]

        self.n_dofs = vector_of_parameters.shape[1]

        # Verifies if the list of agent DOFs is within bounds of the 
        # numbering of DOFs in the whole mesh

        agent_dofs_list = np.asarray(agent_dofs_list)

        if np.max(agent_dofs_list)>=self.n_dofs:

            raise IndexError("The maximum DOF index for this agent at "+
            "'AgentModel' is "+str(np.max(agent_dofs_list))+", which i"+
            "s out of bounds, for the DOF with the maximum index numbe"+
            "r is "+str(self.n_dofs-1))
        
        self.n_agent_dofs = agent_dofs_list.shape[0]
        
        # Broadcasts DOF indices of the agent along the realizations axis

        dofs_indices = tf.constant(agent_dofs_list, dtype=integer_dtype)

        dofs_indices = tf.broadcast_to(dofs_indices[None,:], [
        self.n_realizations, self.n_agent_dofs])

        # Broadcasts the realization indices to make pairs with DOF in-
        # dices

        realizations_range = tf.range(self.n_realizations, dtype=
        integer_dtype)

        realization_indices = tf.broadcast_to(realizations_range[:, None
        ], [self.n_realizations, self.n_agent_dofs])

        # Stacks indices for scatter_nd_update: [n_realizations*
        # n_agent_dofs, 2]

        self.scatter_indices = tf.reshape(tf.stack([realization_indices, 
        dofs_indices], axis=-1), [-1, 2])

        # Stores the instance of the loss function class

        self.loss_function_class = loss_function_class

    # Defines a function to get the result of the agent model and, then,
    # to plug it to the global vector of parameters

    @tf.function
    def evaluate_model(self, model_input, vector_of_parameters):

        # Gets the result of the agent model, then, applies it to the 
        # vector of parameters. Creates a new updated vector of parame-
        # ters. Reshapes the model output to [n_realizations*
        # n_agent_dofs]

        return tf.tensor_scatter_nd_update(vector_of_parameters, 
        self.scatter_indices, tf.reshape(self.agent_model(model_input),
        [-1]))

    # Defines a function to get the result of the agent model and, then,
    # to compute the loss function

    @tf.function
    def evaluate_loss_function(self, model_input, vector_of_parameters,
    residual_vector):

        # Gets the result of the agent model, then, applies it to the 
        # vector of parameters

        updated_vector_of_parameters = self.evaluate_model(model_input,
        vector_of_parameters)

        # Evaluates the loss function after the update of the vector of
        # parameters

        return self.loss_function_class.evaluate_loss(
        updated_vector_of_parameters, residual_vector)
    
    # Defines a function to compute the gradient of the loss function