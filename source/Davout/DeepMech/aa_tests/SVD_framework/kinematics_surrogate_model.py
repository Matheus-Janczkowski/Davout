# Routine to train a neural network model as surrogate for the RVE kine-
# matics

import numpy as np

import tensorflow as tf

from time import time

from .....Davout.DeepMech.tool_box import ANN_tools, training_tools

from .....Davout.DeepMech.tool_box.numerical_tools import DataMatrixRegularization

from .....Davout.DeepMech.tool_box.loss_assembler_classes import MaximumAbsoluteError

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a class to construct and train a surrogate model for the kine-
# matics of the RVE

class RVEKinematicsSurrogateModel:

    def __init__(self, activations_list_main_network, 
    activations_list_auxiliar_network, quotient_space_dimension, 
    modulating_function, displacement_data_file, input_data_file, 
    results_path, loss_metric, n_training_samples, saved_model_file,
    n_monte_carlo_realizations, n_best_models, n_best_samples,
    maximum_training_iterations, verbose_delta_iterations,
    mesh_file_name, optimizer, subdofs_to_learn, parameters_dtype="flo"+
    "at32", householder_epsilon=1.0, verbose=True, integer_dtype="int3"+
    "2", non_orthogonal_matrices=False, device_type="CPU", 
    input_data_regularization_method=None):

        # Saves the needed information

        self.non_orthogonal_matrices = non_orthogonal_matrices

        self.device_type = device_type

        self.activations_list_main_network = activations_list_main_network

        self.activations_list_auxiliar_network = (
        activations_list_auxiliar_network)

        self.quotient_space_dimension = quotient_space_dimension

        self.modulating_function = modulating_function

        self.displacement_data_file = displacement_data_file

        self.input_data_file = input_data_file

        self.results_path = results_path

        self.loss_metric = loss_metric

        self.optimizer = optimizer

        self.n_training_samples = n_training_samples

        self.saved_model_file = saved_model_file

        self.n_monte_carlo_realizations = n_monte_carlo_realizations

        self.n_best_models = n_best_models

        self.n_best_samples = n_best_samples

        self.mesh_file_name = mesh_file_name

        self.householder_epsilon = householder_epsilon

        self.verbose = verbose

        self.maximum_training_iterations = maximum_training_iterations

        self.verbose_delta_iterations = verbose_delta_iterations

        # Saves the tensorflow dtypes

        self.parameters_dtype = parameters_dtype

        self.integer_dtype = integer_dtype

        # Converts the tensor of DOFs to a tensorflow integer tensor

        self.subdofs_to_learn = tf.convert_to_tensor(subdofs_to_learn,
        dtype=tf.as_dtype(self.integer_dtype))

        # Gets the number of output neurons from the tensor of the indi-
        # ces of the DOFs that will actually be learnt

        self.n_output_neurons = self.subdofs_to_learn.shape[0]

        # Reads the output data 

        self.whole_output_data = np.load(self.results_path+"//"+
        self.displacement_data_file)

        # Gets the numbers of samples and of total of DOFs

        self.number_of_samples, self.total_number_of_dofs = (
        self.whole_output_data.shape)

        # Checks whether subdofs to learn has any DOF index that is lar-
        # ger than the number of DOFs of the output data

        if tf.reduce_max(self.subdofs_to_learn).numpy(
        )>=self.total_number_of_dofs:

            raise IndexError("The maximum DOF index requested by 'subd"+
            "ofs_to_learn' is "+str(tf.reduce_max(self.subdofs_to_learn
            ).numpy())+". But the output data at\n"+str(
            self.results_path+"//"+self.displacement_data_file)+"\nhas"+
            " only "+str(self.total_number_of_dofs)+" columns")

        # Otherwise, uses the subdofs to learn to retrieve the needed 
        # DOFs for training

        self.output_data = self.whole_output_data[:,
        self.subdofs_to_learn]

        # Gets the input data

        input_data = np.load(self.results_path+"//"+self.input_data_file)

        # Checks whether there is sufficient samples in the dataset
        
        if self.n_training_samples>=input_data.shape[0]:

            raise IndexError(str(self.n_training_samples)+" samples we"+
            "re asked for training, but there are only "+str(
            input_data.shape[0])+" samples in the input data matrix at"+
            ":\n"+str(self.results_path+"//"+self.input_data_file))

        if self.n_training_samples>=self.output_data.shape[0]:

            raise IndexError(str(self.n_training_samples)+" samples we"+
            "re asked for training, but there are only "+str(
            input_data.shape[0])+" samples in the output data matrix a"+
            "t:\n"+str(self.results_path+"//"+
            self.displacement_data_file))

        if self.output_data.shape[0]!=input_data.shape[0]:

            raise IndexError("The output data matrix has "+str(
            self.output_data.shape[0])+" samples, whereas the input da"+
            "ta matrix has "+str(input_data.shape[0])+" samples.\n\nTh"+
            "e output data matrix is at:\n"+str(self.results_path+"//"+
            self.displacement_data_file)+"\n\nthe input data matrix is"+
            "at:\n"+str(self.results_path+"//"+self.input_data_file))

        # Checks if a regularization technique was requested for the in-
        # put data matrix

        if input_data_regularization_method is not None:

            # Instantiates the data regularization class

            data_regularization_class = DataMatrixRegularization(
            regularization_function=input_data_regularization_method)

            # Regularizes the data matrix and saves the regularization
            # data using this input data

            self.input_data, _ = data_regularization_class(input_data, 
            True)

        # Otherwise, simply saves the input data into the class
        
        else:

            self.input_data = input_data

        # Gets the number of input neurons

        self.n_input_neurons = self.input_data.shape[1]

        # Verifies if the output layer of the main network has the same
        # number of output neurons

        n_output_neurons_main_network = 0

        for value in self.activations_list_main_network[-1].values():

            n_output_neurons_main_network += value

        if n_output_neurons_main_network!=self.n_output_neurons:

            raise ValueError("The main network has "+str(
            n_output_neurons_main_network)+" neurons in the output lay"+
            "er; whereas the data indicate that there must be "+str(
            self.n_output_neurons)+" neurons in the output layer")

        # Sets the architecture dictionary
    
        self.custom_architecture = {"name": "SVDQuotientSpace", "weigh"+
        "ts modulating function": self.modulating_function, "Household"+
        "er epsilon": self.householder_epsilon, "activations accessory"+
        " layer list": self.activations_list_auxiliar_network, "non-or"+
        "thogonal matrices": self.non_orthogonal_matrices, "hardware d"+
        "evice": self.device_type, "convex with respect to quotient sp"+
        "ace": False}

    # Defines a function to train the neural network model

    def train_surrogate_model(self):

        # Sets the training data

        training_data = self.input_data[0:self.n_training_samples,:]

        training_true_values = self.output_data[0:self.n_training_samples
        ,:]

        # Creates the class of neural network information

        ANN_class = ANN_tools.MultiLayerModel(self.n_input_neurons,
        self.activations_list_main_network, enforce_customLayers=True, 
        verbose=self.verbose, parameters_dtype=self.parameters_dtype, 
        custom_architecture=self.custom_architecture, 
        input_size_main_network=self.quotient_space_dimension)

        self.custom_model = ANN_class()

        # Defines the loss metric

        """
        self.live_loss_metric = None

        if isinstance(self.loss_metric, str):

            # Verifies if the loss metric is in the keras loss

            if not hasattr(tf.keras.losses, self.loss_metric):

                raise NameError("The keras losses do not own any loss "+
                "metric by the name '"+str(self.loss_metric)+"'")

            self.live_loss_metric = getattr(tf.keras.losses, 
            self.loss_metric)(dtype=tf.as_dtype(self.parameters_dtype))"""

        self.live_loss_metric = self.loss_metric

        # Sets the optimization class for training

        training_class = training_tools.ModelCustomTraining(
        self.custom_model, training_data, training_true_values, 
        self.live_loss_metric, verbose=self.verbose, n_iterations=
        self.maximum_training_iterations, verbose_deltaIterations=
        self.verbose_delta_iterations, save_model_file=
        self.saved_model_file, match_data_float_type_to_trainables=True, 
        parent_path=self.results_path, optimizer=self.optimizer)

        t_initial = time()

        training_class()

        elapsed_time = time()-t_initial

        print("\nTrains at "+str(elapsed_time)+" seconds")

        # Tests Monte Carlo training

        training_class.monte_carlo_training(n_realizations=
        self.n_monte_carlo_realizations, best_models_rank_size=
        self.n_best_models, show_reinitialization_distance=True, 
        model_base_name=self.saved_model_file)

        # Checks the loss again with the best model of the Monte Carlo
        # training

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data for the best model is "+str(
        training_class.loss_unseen_data(training_true_values, 
        training_data, output_as_numpy=True)))

    # Defines a function to test the model

    def test_surrogate_model(self):

        # Reads the two files

        output_data = np.load(self.results_path+"//"+
        self.displacement_data_file)

        input_data = np.load(self.results_path+"//"+self.input_data_file)

        # Reshufles data to put the displacement gradient as the first
        # columns. This is a requirement of the implementation of the
        # SVD architecture

        input_data = np.hstack((input_data[:,(input_data.shape[1]-
        self.quotient_space_dimension):], input_data[:,:(
        input_data.shape[1]-self.quotient_space_dimension)]))

        # Sets the test data

        test_data = input_data[self.n_training_samples:,:]

        test_true_values = output_data[self.n_training_samples:,:]

        # Defines the loss metric if any has been given
        
        if self.live_loss_metric is None:

            if isinstance(self.loss_metric, str):

                # Verifies if the loss metric is in the keras loss

                if not hasattr(tf.keras.losses, self.loss_metric):

                    raise NameError("The keras losses do not own any l"+
                    "oss metric by the name '"+str(self.loss_metric)+"'")

                self.live_loss_metric = getattr(tf.keras.losses, 
                self.loss_metric)(dtype=tf.as_dtype(
                self.parameters_dtype))

        # Iterates through the best models

        for i in range(self.n_best_models):

            # Loads the i-th best model back

            loaded_model = tf.keras.models.load_model(self.results_path+
            "//"+str(i+1)+"_best_model.keras")

            # Gets the output of the loaded model

            output_model = loaded_model(test_data)

            # Gets the loss of the test data

            test_loss = self.live_loss_metric(test_true_values, 
            output_model)

            print("Loads the "+str(i+1)+"-th best model")

            print("Loss function on test set:", format(test_loss.numpy(),
            '.5e')+"\n")

            # Verifies with the maximum absolute error

            maximum_absolute_error = MaximumAbsoluteError()

            maximum_absolute_value = maximum_absolute_error(
            test_true_values, output_model)

            print("Maximum absolute error on test set:"+str(format(
            maximum_absolute_value.numpy(),'.5e'))+"\nwhereas the mini"+
            "mum absolute error is "+str(format(
            maximum_absolute_error.minimum_absolute_error(
            test_true_values, output_model).numpy(), '.5e'))+"\n")

# Testing block

if __name__=="__main__":

    # Sets the number of transfinite divisions in the RVE mesh

    n_divisions = 6

    # Sets a dictionary for the number of DOFs in the mesh for each num-
    # ber of divisions

    dofs_per_number_of_divisions = {6: 3993, 8: 10125, 10: 20577}

    # Defines the array with the DOFs to be learnt

    subdofs_to_learn = np.arange(0, dofs_per_number_of_divisions[
    n_divisions], 3)

    # Defines the number of input neurons of the main network. Remember
    # that the data matrix is a tensor [n_samples, n_input_variables]. 
    # The first quotient_space_dimension columns must be dedicated to 
    # the variables dedicated to the quotient space, i.e. the load-based
    # set of variables

    quotient_space_dimension = 6

    # Defines the number of neurons of the different hidden layers and of
    # the output layer

    n_neurons_per_layer = [100, 1000, int(dofs_per_number_of_divisions[
    n_divisions]/3)]

    # Sets a list of layers and the activation functions

    activations_list_main_network = [{"elu": {"number of neurons":
    n_neurons_per_layer[0]}}, {"elu": {"number of neurons":
    n_neurons_per_layer[1]}}, {"linear": n_neurons_per_layer[2]}]

    activations_list_auxiliar_network = [{"quadratic": {"number of neu"+
    "rons": min(quotient_space_dimension, n_neurons_per_layer[0]), "a2": 
    1.0}}, 
    {"quadratic":  {"number of neurons": min(n_neurons_per_layer[0], 
    n_neurons_per_layer[1]), "a2": 1.0}}, 
    {"quadratic": {"number of neurons": min(n_neurons_per_layer[1], 
    n_neurons_per_layer[2])}}]

    # Sets the modulating function for the weights matrices

    modulating_function = "identity"

    # Sets the training information

    optimizer = "Adam"

    n_training_samples = 10000

    maximum_training_iterations = 1000

    verbose_delta_iterations = 50

    n_monte_carlo_realizations = 10

    n_best_models = 5

    n_best_samples = 10

    loss_metric = "MeanSquaredError"

    loss_metric = {"name": "LpNormError", "p": 12, "use_stable_impleme"+
    "ntation": True}

    parameters_dtype = "float32"

    saved_model_file = "RVE_kinematics_model.keras"

    # Sets the path for the data

    results_path = (get_parent_path_of_file()+"//results_n_divisions_"+
    str(n_divisions))
   
    displacement_data_file = "00_successful_displacement_matrix.npy"

    input_data_file = "00_successful_complete_and_filtered_data_matrix.npy"

    mesh_file_name = get_parent_path_of_file()+"//box_mesh_mecsol"

    # Sets the method to regularize the input data matrix

    input_data_regularization_method = ("maximum_absolute_order_normal"+
    "ization")

    # Sets the device type where the training will be performed

    device_type = "GPU"

    # Sets whether the constraint on orthogonality of the SVD matrices
    # will not be enforced

    non_orthogonal_matrices = True

    # Instantiates the class that owns all information about training 
    # and testing models

    modeling_class = RVEKinematicsSurrogateModel(
    activations_list_main_network, activations_list_auxiliar_network, 
    quotient_space_dimension, modulating_function, 
    displacement_data_file, input_data_file, results_path, loss_metric, 
    n_training_samples, saved_model_file, n_monte_carlo_realizations, 
    n_best_models, n_best_samples, maximum_training_iterations, 
    verbose_delta_iterations, mesh_file_name, optimizer, 
    subdofs_to_learn, parameters_dtype=parameters_dtype, 
    householder_epsilon=1.0, verbose=True, device_type=
    device_type, non_orthogonal_matrices=non_orthogonal_matrices,
    input_data_regularization_method=input_data_regularization_method)

    training_flag = True

    testing_flag = False

    if training_flag:

        modeling_class.train_surrogate_model()

    if testing_flag:

        modeling_class.test_surrogate_model()