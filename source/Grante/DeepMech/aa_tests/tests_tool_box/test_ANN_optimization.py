# Routine to store some tests

import unittest

import os

import time

import tensorflow as tf

import numpy as np

from ...tool_box import ANN_tools

from ...tool_box import training_tools

from ...tool_box import differentiation_tools as diff_tools

from ...tool_box import parameters_tools

from ...tool_box import loss_tools

from ...tool_box import loss_assembler_classes as loss_assemblers

from ....MultiMech.tool_box import file_handling_tools

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        self.input_tensor = tf.constant([[1.0, 2.0]], dtype=tf.float32)

        self.input_dimension = 2

        self.dummy_input = tf.random.normal((10, self.input_dimension))

        # Sets the number of optimization iterations

        self.n_iterations = 10000

        # Sets the convergence tolerance

        self.gradient_tolerance = 1E-3

        # Defines the function to be approximated

        def true_function(x):

            return ((x[0]**2)+(2.0*(x[1]**2)))
        
        self.true_function = true_function

        # the problem consists of approximating the function z=(x**2)+(2
        # *(y**2)). Thus, firstly, generates the training matrix, and 
        # the true values list

        true_values = []
        
        data_matrix = []

        n_samples = 10

        x_min = -1.0

        x_max = 1.0

        y_min = -1.5

        y_max = 1.5

        for i in range(n_samples):

            data_matrix.append([ANN_tools.random_inRange(x_min, x_max
            ), ANN_tools.random_inRange(y_min, y_max)])

            # Evaluaets the true function

            true_values.append(true_function(data_matrix[-1]))

        # Sets the training and test data

        n_samplesTraining = 6

        self.training_data = data_matrix[:n_samplesTraining]

        self.training_trueValues = true_values[:n_samplesTraining]

        self.test_data = data_matrix[n_samplesTraining:]

        self.test_trueValues = true_values[n_samplesTraining:]

        # Converts thet data to tensors

        self.training_inputTensor = tf.constant(self.training_data, 
        dtype=tf.float32)

        self.test_inputTensor = tf.constant(self.test_data, dtype=
        tf.float32)

        self.training_trueTensor = tf.constant(self.training_trueValues, 
        dtype=tf.float32)

        self.test_trueTensor = tf.constant(self.test_trueValues, dtype=
        tf.float32)

        # Defines the loss function metric

        self.loss_metric = tf.keras.losses.MeanAbsoluteError()

        # Sets the optimizer

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, 
        momentum=0.9, nesterov=True)

        self.verbose_deltaIterations = 1000

        # Sets the architecture

        self.layers_information = [{"sigmoid": 100}, {"sigmoid":100},
        {"linear": 1}]

        # Defines the test data for the gradient tests

        self.input_dimension_gradient_tests = 9

        self.output_dimension_gradient_tests = 100

        self.activation_list_gradient_tests = [{"sigmoid": 100}, {"lin"+
        "ear": self.output_dimension_gradient_tests}]

        self.n_samples_gradient_tests = 1000 

    def test_linear_loss(self):

        print("\n#####################################################"+
        "###################\n#                      Tests linear loss"+
        " function                      #\n###########################"+
        "#############################################\n")
        
        # Creates the new test data

        x_min = -1.0

        x_max = 1.0

        data_matrix = []

        for i in range(self.n_samples_gradient_tests):

            data_matrix.append([ANN_tools.random_inRange(x_min, x_max
            ) for j in range(self.input_dimension_gradient_tests)])

        # Converts the data to tensors

        input_test_data = tf.constant(data_matrix, dtype=
        tf.float32)

        # Creates the custom model with custom layers

        evaluate_parameters_gradient=False

        ANN_class = ANN_tools.MultiLayerModel(
        self.input_dimension_gradient_tests, 
        self.activation_list_gradient_tests, enforce_customLayers=True, 
        evaluate_parameters_gradient=evaluate_parameters_gradient,
        verbose=True)

        custom_model = ANN_class()

        # Gets the coefficient matrix

        coefficient_matrix = tf.random.normal((
        self.n_samples_gradient_tests, 
        self.output_dimension_gradient_tests))

        # Sets the loss function

        """loss = lambda model_response: loss_tools.linear_loss(model_response, 
        coefficient_matrix)"""

        loss = loss_assemblers.LinearLossAssembler(coefficient_matrix, 
        check_tensors=True)

        # Sets a function to capture the value and the gradient of the
        # loss 

        gradient_class = diff_tools.ScalarGradientWrtTrainableParams(
        loss, input_test_data)

        def objective_function(custom_model=custom_model, loss=loss,
        input_test_data=input_test_data):

            gradient = gradient_class(custom_model)

            # Converts to numpy

            return diff_tools.convert_scalar_gradient_to_numpy(gradient)
        
        # Sets the same function but enabling the model parameters as 
        # argument from a numpy array

        objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, loss, input_test_data, trainable_variables_type=
        "numpy")

        # Gets the value using the model parameters as input

        t_initial = time.time()

        result2 = objective_function_with_parameters(model_params*2.0)

        elapsed_time = time.time()-t_initial

        print("Elapsed time with model parameters: "+str(elapsed_time)+
        ". Loss function and gradient:")

        # Gets the value using the model parameters as input

        t_initial = time.time()

        result2 = objective_function_with_parameters(model_params*3.0)

        elapsed_time = time.time()-t_initial

        print("Elapsed time with model parameters: "+str(elapsed_time)+
        ". Loss function and gradient:")

        # Gets the value using the model parameters as input

        t_initial = time.time()

        result2 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("Elapsed time with model parameters: "+str(elapsed_time)+
        ". Loss function and gradient:")

        # Gets the value

        result = objective_function()

        t_initial = time.time()

        result = objective_function()

        elapsed_time = time.time()-t_initial

        print("Elapsed time: "+str(elapsed_time)+". Loss function and "+
        "gradient:")

        print(np.linalg.norm(result-result2))

        # Sets the same function but enabling the model parameters as 
        # argument from a tensorflow 1D tensor

        objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, loss, input_test_data)

        result3 = objective_function_with_parameters(model_params)

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("Elapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters. The difference to the gradient without au"+
        "tomatic function assembly: "+str(np.linalg.norm(result3-result)
        ))

        # Tests now with Keras layers

        ANN_class = ANN_tools.MultiLayerModel(
        self.input_dimension_gradient_tests, 
        self.activation_list_gradient_tests, enforce_customLayers=False, 
        evaluate_parameters_gradient=evaluate_parameters_gradient,
        verbose=True)

        custom_model = ANN_class()

        # Sets the same function but enabling the model parameters as 
        # argument from a tensorflow 1D tensor

        objective_function_with_parameters, model_paramsKeras = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, loss, input_test_data)

        result4 = objective_function_with_parameters(model_params)

        t_initial = time.time()

        result4 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("Elapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters and Keras layers. The difference of the gr"+
        "adient between using Keras and custom layer is "+str(
        np.linalg.norm(result3-result4))+"\n")

        # Updates the coefficient matrix

        objective_function_with_parameters.update_function(
        coefficient_matrix)

        t_initial = time.time()

        objective_function_with_parameters.update_function(
        coefficient_matrix)

        elapsed_time = time.time()-t_initial

        print("The elapsed time to update the coefficient matrix of th"+
        "e loss function through the class method is: "+str(elapsed_time
        )+"\n")

        t_initial = time.time()

        result4 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("Elapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters and Keras layers. This time is counted for"+
        " evaluating the gradient after updating the coefficient matri"+
        "x")

# Runs all tests

if __name__ == "__main__":

    unittest.main()