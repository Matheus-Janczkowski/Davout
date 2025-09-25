# Routine to store some tests to evaluate convex input neural networks

import unittest

import time

import tensorflow as tf

from ...tool_box import ANN_tools

from ...tool_box import training_tools

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        # Defines the test data for the gradient tests

        self.input_dimension_gradient_tests = 3

        self.output_dimension_gradient_tests = 1

        self.activation_list_gradient_tests = [{"quadratic": {"number "+
        "of neurons": 100, "a2": 2.0}}, {"linear": 
        self.output_dimension_gradient_tests}]

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

        self.training_inputTensor = tf.constant(self.training_data, 
        dtype=self.dtype)

        self.test_inputTensor = tf.constant(self.test_data, dtype=
        self.dtype)

        self.training_trueTensor = tf.constant(self.training_trueValues, 
        dtype=self.dtype)

        self.test_trueTensor = tf.constant(self.test_trueValues, dtype=
        self.dtype)

        # Defines the loss function metric

        self.loss_metric = tf.keras.losses.MeanAbsoluteError()

        # Sets the optimizer

        self.optimizer = "CG"

        self.verbose_deltaIterations = 1000

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
        evaluate_parameters_gradient=False, verbose=True, parameters_dtype=
        "float64")

        custom_model = ANN_class()

        # Sets the optimization class for training

        training_class = training_tools.ModelCustomTraining(custom_model,
        self.training_inputTensor, self.training_trueTensor, 
        self.loss_metric, convex_input_model=True, verbose=True,
        n_iterations=self.maximum_iterations, verbose_deltaIterations=
        self.verbose_deltaIterations)

        t_initial = time.time()

        training_class()

        elapsed_time = time.time()-t_initial

        print("\nTrains at "+str(elapsed_time)+" seconds")

# Runs all tests

if __name__=="__main__":

    unittest.main()