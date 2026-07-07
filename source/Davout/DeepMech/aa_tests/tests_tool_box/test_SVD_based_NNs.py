# Routine to store some tests to evaluate SVD-based neural networks

import unittest

import time

import os

import tensorflow as tf

from ...tool_box import ANN_tools

from ...tool_box import training_tools

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        # Defines the test data for the gradient tests

        self.whole_input_dimension = 7

        self.quotient_space_dimension = 3

        self.output_dimension = 2

        self.activation_list_main_network = [{"quadratic": {"number of"+
        " neurons": 100, "a2": 1.0}}, {"linear": self.output_dimension}]

        self.accessory_activation_list_tests = [{"quadratic": {"number"+
        " of neurons": 3, "a2": 1.0}}, {"linear":self.output_dimension}]

        self.n_samples_tests = 1000

        self.n_samples_quotient_test = 10

        self.maximum_iterations = 5000

        n_samplesTraining = 6

        # Defines a function to get the true values

        def true_function(x):

            value = 0.0

            for x_i in x:

                value += x_i**2

            return [value for j in range(self.output_dimension)]
        
        self.true_function = true_function

        # Sets the training and test data

        # Creates the new test data

        self.x_min = -1.0

        self.x_max = 1.0

        data_matrix = []

        true_values = []

        for i in range(self.n_samples_tests):

            data_matrix.append([ANN_tools.random_inRange(self.x_min, 
            self.x_max) for j in range(self.whole_input_dimension
            )])

            # Evaluaets the true function

            true_values.append(self.true_function(data_matrix[-1]))

        self.training_data = data_matrix[:n_samplesTraining]

        self.training_trueValues = true_values[:n_samplesTraining]

        self.test_data = data_matrix[n_samplesTraining:]

        self.test_trueValues = true_values[n_samplesTraining:]

        # Converts thet data to tensors

        self.dtype = tf.float64

        # Defines the loss function metric

        self.loss_metric = tf.keras.losses.MeanAbsoluteError()

        # Sets the optimizer

        self.optimizer = "CG"

        self.verbose_deltaIterations = 1000

        # Sets where to save the model

        self.save_model_file = "saved_model.keras"

    # Defines a function to test the gated architecture

    def test_svd_nn(self):

        print("\n#####################################################"+
        "###################\n#                    Tests SVD-based neu"+
        "ral network                    #\n###########################"+
        "#############################################\n")

        # Tests now with custom layers

        ANN_class = ANN_tools.MultiLayerModel(
        self.whole_input_dimension, 
        self.activation_list_main_network, enforce_customLayers=True, 
        verbose=True, parameters_dtype="float64", custom_architecture={
        "name": "SVDQuotientSpace", "weights modulating function": 
        "identity", "Householder epsilon": 1.0, 
        "activations accessory layer list": self.accessory_activation_list_tests}, 
        input_size_main_network=self.quotient_space_dimension)

        custom_model = ANN_class()

        # Sets the optimization class for training

        training_class = training_tools.ModelCustomTraining(custom_model,
        self.training_data, self.training_trueValues, 
        self.loss_metric, verbose=True, n_iterations=
        self.maximum_iterations, verbose_deltaIterations=
        self.verbose_deltaIterations, save_model_file=
        self.save_model_file)

        t_initial = time.time()

        training_class()

        elapsed_time = time.time()-t_initial

        print("\nTrains at "+str(elapsed_time)+" seconds")

        # Checks the loss again with the model with the regularized pa-
        # rameters

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data is "+str(training_class.loss_unseen_data(
        self.training_trueValues, self.training_data, output_as_numpy=
        True)))

        # Tests Monte Carlo training

        training_class.monte_carlo_training(n_realizations=5, 
        best_models_rank_size=5, show_reinitialization_distance=True)

        # Checks the loss again with the best model of the Monte Carlo
        # training

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data for the best model is "+str(
        training_class.loss_unseen_data(self.training_trueValues, 
        self.training_data, output_as_numpy=True)))

        # Generates an input tensor with zeros for the input neurons 
        # corresponding to the quotient space

        coefficient = 1E-10

        x = tf.concat([coefficient*tf.random.uniform((
        self.n_samples_quotient_test, self.quotient_space_dimension), 
        minval=-1.0, maxval=1.0, dtype=self.dtype),
        tf.random.uniform((self.n_samples_quotient_test, 
        self.whole_input_dimension-self.quotient_space_dimension
        ), minval=-1.0, maxval=1.0, dtype=self.dtype)],axis=-1)

        model_output = training_class.model(x)

        print("\nTests the SVD model against the following input:")

        print(x.numpy())

        print("\nThere follow the SVD model output:")

        print(model_output)

# Runs all tests

if __name__=="__main__":

    unittest.main()