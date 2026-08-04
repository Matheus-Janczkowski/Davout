# Routine to train a neural network model as surrogate for the RVE kine-
# matics

import numpy as np

import tensorflow as tf

from time import time

from .....Davout.DeepMech.tool_box import ANN_tools, training_tools

from .....Davout.DeepMech.tool_box.loss_assembler_classes import MaximumAbsoluteError

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a class to construct and train a surrogate model for the kine-
# matics of the RVE

class RVEKinematicsSurrogateModel:

    def __init__(self, activations_list_main_network, 
    activations_list_auxiliar_network, quotient_space_dimension, 
    modulating_function, displacement_data_file, input_data_file, 
    results_path, loss_metric, n_training_samples, saved_model_file,
    n_monte_carlo_realizations, n_best_models, 
    maximum_training_iterations, verbose_delta_iterations,
    parameters_dtype="float32", householder_epsilon=1.0, verbose=True):

        # Saves the needed information

        self.activations_list_main_network = activations_list_main_network

        self.activations_list_auxiliar_network = (
        activations_list_auxiliar_network)

        self.quotient_space_dimension = quotient_space_dimension

        self.modulating_function = modulating_function

        self.displacement_data_file = displacement_data_file

        self.input_data_file = input_data_file

        self.results_path = results_path

        self.loss_metric = loss_metric

        self.n_training_samples = n_training_samples

        self.saved_model_file = saved_model_file

        self.n_monte_carlo_realizations = n_monte_carlo_realizations

        self.n_best_models = n_best_models

        self.parameters_dtype = parameters_dtype

        self.householder_epsilon = householder_epsilon

        self.verbose = verbose

        self.maximum_training_iterations = maximum_training_iterations

        self.verbose_delta_iterations = verbose_delta_iterations

    # Defines a function to train the neural network model

    def train_surrogate_model(self):

        # Reads the two files

        output_data = np.load(self.results_path+"//"+
        self.displacement_data_file)

        input_data = np.load(self.results_path+"//"+self.input_data_file)

        # Checks whether there is sufficient samples in the dataset

        if self.n_training_samples<=input_data.shape[0]:

            raise IndexError(str(self.n_training_samples)+" samples we"+
            "re asked for training, but there are only "+str(
            input_data.shape[0])+" samples in the input data matrix at"+
            ":\n"+str(self.results_path+"//"+self.input_data_file))

        if self.n_training_samples<=output_data.shape[0]:

            raise IndexError(str(self.n_training_samples)+" samples we"+
            "re asked for training, but there are only "+str(
            input_data.shape[0])+" samples in the output data matrix a"+
            "t:\n"+str(self.results_path+"//"+
            self.displacement_data_file))

        if output_data.shape[0]!=input_data.shape[0]:

            raise IndexError("The output data matrix has "+str(
            output_data.shape[0])+" samples, whereas the input data ma"+
            "trix has "+str(input_data.shape[0])+" samples.\n\nThe out"+
            "put data matrix is at:\n"+str(self.results_path+"//"+
            self.displacement_data_file)+"\n\nthe input data matrix is"+
            "at:\n"+str(self.results_path+"//"+self.input_data_file))

        # Sets the training data

        training_data = input_data[0:self.n_training_samples,:]

        training_true_values = output_data[0:self.n_training_samples,:]

        # Gets the number of input and output neurons

        n_input_neurons = input_data.shape[1]

        n_output_neurons = output_data.shape[1]

        # Verifies if the output layer of the main network has the same
        # number of output neurons

        n_output_neurons_main_network = 0

        for value in self.activations_list_main_network[-1].values():

            n_output_neurons_main_network += value

        if n_output_neurons_main_network!=n_output_neurons:

            raise ValueError("The main network has "+str(
            n_output_neurons_main_network)+" neurons in the output lay"+
            "er; whereas the data indicate that there must be "+str(
            n_output_neurons)+" neurons in the output layer")

        # Sets the architecture dictionary
    
        custom_architecture = {"name": "SVDQuotientSpace", "weights mo"+
        "dulating function": self.modulating_function, "Householder ep"+
        "silon": self.householder_epsilon, "activations accessory laye"+
        "r list": self.activations_list_auxiliar_network}

        # Creates the class of neural network information

        ANN_class = ANN_tools.MultiLayerModel(n_input_neurons,
        self.activations_list_main_network, enforce_customLayers=True, 
        verbose=self.verbose, parameters_dtype=self.parameters_dtype, 
        custom_architecture=custom_architecture, input_size_main_network=
        self.quotient_space_dimension)

        self.custom_model = ANN_class()

        # Defines the loss metric

        self.live_loss_metric = None

        if isinstance(self.loss_metric, str):

            # Verifies if the loss metric is in the keras loss

            if not hasattr(tf.keras.losses, self.loss_metric):

                raise NameError("The keras losses do not own any loss "+
                "metric by the name '"+str(self.loss_metric)+"'")

            self.live_loss_metric = getattr(tf.keras.losses, 
            self.loss_metric)(dtype=tf.as_dtype(self.parameters_dtype))

        # Sets the optimization class for training

        training_class = training_tools.ModelCustomTraining(
        self.custom_model, training_data, training_true_values, 
        self.live_loss_metric, verbose=self.verbose, n_iterations=
        self.maximum_training_iterations, verbose_deltaIterations=
        self.verbose_delta_iterations, save_model_file=
        self.saved_model_file, match_data_float_type_to_trainables=True, 
        parent_path=self.results_path)

        t_initial = time()

        training_class()

        elapsed_time = time()-t_initial

        print("\nTrains at "+str(elapsed_time)+" seconds")

        # Tests Monte Carlo training

        training_class.monte_carlo_training(n_realizations=
        self.n_monte_carlo_realizations, best_models_rank_size=
        self.n_best_models, show_reinitialization_distance=True)

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

    # Defines the number of input neurons of the main network. Remember
    # that the data matrix is a tensor [n_samples, n_input_variables]. 
    # The first quotient_space_dimension columns must be dedicated to 
    # the variables dedicated to the quotient space, i.e. the load-based
    # set of variables

    quotient_space_dimension = 6

    # Defines the number of neurons of the different hidden layers and of
    # the output layer

    n_neurons_per_layer = [100, 1000, 20577]

    # Sets a list of layers and the activation functions

    activations_list_main_network = [{"elu": {"number of neurons":
    n_neurons_per_layer[0]}}, {"elu": {"number of neurons":
    n_neurons_per_layer[1]}}, {"linear": n_neurons_per_layer[2]}]

    activations_list_auxiliar_network = [{"quadratic": {"number"+
    " of neurons": min(quotient_space_dimension, n_neurons_per_layer[0]),
    "a2": 1.0}}, {"quadratic":  {"number of neurons": min(
    n_neurons_per_layer[0], n_neurons_per_layer[1]), "a2": 1.0}}, {"qu"+
    "adratic": {"number of neurons": min(n_neurons_per_layer[1],
    n_neurons_per_layer[2])}}]

    # Sets the modulating function for the weights matrices

    modulating_function = "identity"

    # Sets the training information

    n_training_samples = 10000

    maximum_training_iterations = 1000

    verbose_delta_iterations = 50

    n_monte_carlo_realizations = 10

    n_best_models = 5

    loss_metric = "MeanSquaredError"

    parameters_dtype = "float32"

    saved_model_file = "RVE_kinematics_model.keras"

    # Sets the path for the data

    results_path = get_parent_path_of_file()+"//results"
   
    displacement_data_file = "00_succesful_displacement_matrix.npy"

    input_data_file = "00_successful_complete_data_matrix.npy"

    # Instantiates the class that owns all information about training 
    # and testing models

    modeling_class = RVEKinematicsSurrogateModel(
    activations_list_main_network, activations_list_auxiliar_network, 
    quotient_space_dimension, modulating_function, 
    displacement_data_file, input_data_file, results_path, loss_metric, 
    n_training_samples, saved_model_file, n_monte_carlo_realizations, 
    n_best_models, maximum_training_iterations, verbose_delta_iterations,
    parameters_dtype=parameters_dtype, householder_epsilon=1.0, verbose=
    True)

    training_flag = True

    if training_flag:

        modeling_class.train_surrogate_model()

    modeling_class.test_surrogate_model()