# Routine to store some tests to evaluate convex input neural networks
#
# source bin/activate
# python3 -m source.Grante.DeepMech.aa_tests.tests_tool_box.test_convex_and_partially_convex_input_NNs

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

        self.input_dimension_gradient_tests = 3

        self.input_size_main_network = 2

        self.output_dimension_gradient_tests = 1

        self.activation_list_gradient_tests = [{"quadratic": {"number "+
        "of neurons": 100, "a2": 1.0}}, {"linear": 
        self.output_dimension_gradient_tests}]

        self.accessory_activation_list_gradient_tests = [{"relu": 10}, 
        {"linear": 2}]

        self.n_samples_gradient_tests = 1000

        self.maximum_iterations = 5000

        # Defines a function to get the true values

        def true_function(x):

            value = 0.0

            for x_i in x:

                value += x_i**2

            return value
        
        self.true_function = true_function

        # Sets the training and test data

         # Creates the new test data

        self.x_min = -1.0

        self.x_max = 1.0

        data_matrix = []

        true_values = []

        for i in range(self.n_samples_gradient_tests):

            data_matrix.append([ANN_tools.random_inRange(self.x_min, 
            self.x_max) for j in range(self.input_dimension_gradient_tests
            )])

            # Evaluaets the true function

            true_values.append(self.true_function(data_matrix[-1]))

        n_samplesTraining = 6

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

    # Defines a function to test the fully convex-input neural networks

    def test_fully_convex_input_nn(self):

        print("\n#####################################################"+
        "###################\n#                Tests fully convex-inpu"+
        "t neural network               #\n###########################"+
        "#############################################\n")

        # Tests now with custom layers

        ANN_class = ANN_tools.MultiLayerModel(
        self.input_dimension_gradient_tests, 
        self.activation_list_gradient_tests, enforce_customLayers=True, 
        evaluate_parameters_gradient=False, verbose=True, 
        parameters_dtype="float64", input_convex_model="fully")

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

        # Checks the hessian matrices

        hessian_matrices = training_class.get_hessian_outputs_model(
        eigenvalues=True)

        print("\nThere follow the hessian matrices")

        print(hessian_matrices)

        # Tests Monte Carlo training

        training_class.monte_carlo_training(n_realizations=5, 
        best_models_rank_size=5, show_reinitialization_distance=True)

        # Checks the loss again with the best model of the Monte Carlo
        # training

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data for the best model is "+str(
        training_class.loss_unseen_data(self.training_trueValues, 
        self.training_data, output_as_numpy=True)))

    # Defines a function to test the partially convex-input neural net-
    # works

    def test_partially_convex_input_nn(self):

        print("\n#####################################################"+
        "###################\n#             Tests partially convex-inp"+
        "ut neural network              #\n###########################"+
        "#############################################\n")

        # Tests now with custom layers

        ANN_class = ANN_tools.MultiLayerModel(
        self.input_dimension_gradient_tests, 
        self.activation_list_gradient_tests, enforce_customLayers=True, 
        evaluate_parameters_gradient=False, verbose=True, 
        parameters_dtype="float64", accessory_layers_activationInfo=
        self.accessory_activation_list_gradient_tests, 
        input_size_main_network=self.input_size_main_network,
        input_convex_model="partially")

        custom_model = ANN_class()

        # Sets the optimization class for training

        training_class = training_tools.ModelCustomTraining(custom_model,
        self.training_data, self.training_trueValues, self.loss_metric, 
        verbose=True, n_iterations=self.maximum_iterations, 
        verbose_deltaIterations=self.verbose_deltaIterations, 
        save_model_file=self.save_model_file)

        t_initial = time.time()

        training_class()

        elapsed_time = time.time()-t_initial

        print("\nTrains at "+str(elapsed_time)+" seconds")

        # Checks the loss again with the model with the regularized pa-
        # rameters

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data is "+str(training_class.loss_unseen_data(
        self.training_trueValues, self.training_data, 
        output_as_numpy=True)))

        # Checks the hessian matrices

        hessian_matrices = training_class.get_hessian_outputs_model(
        eigenvalues=True)

        print("\nThere follow the hessian matrices")

        print(hessian_matrices)

        # Tests Monte Carlo training

        training_class.monte_carlo_training(n_realizations=5, 
        best_models_rank_size=5, show_reinitialization_distance=True)

        # Checks the loss again with the best model of the Monte Carlo
        # training

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data for the best model is "+str(
        training_class.loss_unseen_data(self.training_trueValues, 
        self.training_data, output_as_numpy=True)))

# Runs all tests

if __name__=="__main__":

    unittest.main()