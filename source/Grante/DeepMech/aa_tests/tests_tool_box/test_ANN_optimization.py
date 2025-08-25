# Routine to store some tests

import unittest

import scipy as sp

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

        ANN_class = ANN_tools.MultiLayerModel(
        self.input_dimension_gradient_tests, 
        self.activation_list_gradient_tests, enforce_customLayers=True, 
        verbose=True)

        custom_model = ANN_class()

        # Gets the A matrix and the b vector

        A = sp.sparse.lil_matrix((self.output_dimension_gradient_tests,
        self.output_dimension_gradient_tests))

        A[0,0] = 0.5

        A[0,1] = 0.5

        A[self.output_dimension_gradient_tests-1, 
        self.output_dimension_gradient_tests-2] = 0.5

        A[self.output_dimension_gradient_tests-1, 
        self.output_dimension_gradient_tests-1] = 0.5

        for i in range(1,self.output_dimension_gradient_tests-1):

            A[i,i-1] = 0.5

            A[i,i] = 1.0

            A[i,i+1] = 0.5

        A_list = [A for i in range(self.n_samples_gradient_tests)]

        b = np.ones(self.output_dimension_gradient_tests)*10.0

        b_list = np.array([b for i in range(self.n_samples_gradient_tests)]).T

        # Sets the loss function

        loss = loss_assemblers.QuadraticLossOverLinearResidualAssembler(
        A_list, b_list)

        objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, loss, input_test_data)

        result3 = objective_function_with_parameters(model_params)

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("Elapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters")

# Runs all tests

if __name__ == "__main__":

    unittest.main()