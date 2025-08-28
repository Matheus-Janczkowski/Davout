# Routine to store some tests

import unittest

import scipy as sp

import time

import tensorflow as tf

import numpy as np

from ...tool_box import ANN_tools

from ...tool_box import loss_tools

from ...tool_box import loss_assembler_classes as loss_assemblers

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        # Defines the test data for the gradient tests

        self.input_dimension_gradient_tests = 9

        self.output_dimension_gradient_tests = 1000

        self.activation_list_gradient_tests = [{"sigmoid": 500}, {"lin"+
        "ear": self.output_dimension_gradient_tests}]

        self.n_samples_gradient_tests = 1000

    # Defines a test the new loss function as the multiplication of a
    # coefficient matrix by the model output

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

        # Gets the coefficient matrix

        coefficient_matrix = tf.random.normal((
        self.n_samples_gradient_tests, 
        self.output_dimension_gradient_tests))

        loss = loss_assemblers.LinearLossAssembler(coefficient_matrix, 
        check_tensors=True)

        # Tests now with custom layers

        ANN_class = ANN_tools.MultiLayerModel(
        self.input_dimension_gradient_tests, 
        self.activation_list_gradient_tests, enforce_customLayers=True, 
        evaluate_parameters_gradient=False, verbose=True)

        custom_model = ANN_class()

        # Sets the same function but enabling the model parameters as 
        # argument from a tensorflow 1D tensor

        objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, loss, input_test_data)

        print("\nWarms up")

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nFinishes warming up after "+str(elapsed_time))

        t_initial = time.time()

        objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nElapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters")

        print("\nChanges parameters")

        objective_function_with_parameters.update_function(coefficient_matrix*3.14)

        print("\nWarms up again")

        t_initial = time.time()

        objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nFinishes warming up again after "+str(elapsed_time))

        t_initial = time.time()

        objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nElapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters")

    def test_quadratic_loss(self):

        print("\n#####################################################"+
        "###################\n#                     Tests quadratic lo"+
        "ss function                    #\n###########################"+
        "#############################################\n")

        # Sets a flag to test batch multiplication or not

        flag_batch_multiplication = False
        
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

        b_list = np.array([b for i in range(self.n_samples_gradient_tests)])

        # Sets the loss function

        loss = loss_assemblers.QuadraticLossOverLinearResidualAssembler(
        A_list, b_list, block_multiplication=False)

        objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, loss, input_test_data)

        if flag_batch_multiplication:

            print("###### Batch multiplication ######")

            print("\nWarms up")

            t_initial = time.time()

            result3 = objective_function_with_parameters(model_params)

            elapsed_time = time.time()-t_initial

            print("\nFinishes warming up after "+str(elapsed_time))

            t_initial = time.time()

            result3 = objective_function_with_parameters(model_params)

            elapsed_time = time.time()-t_initial

            print("\nElapsed time: "+str(elapsed_time)+". Using automa"+
            "tic call with parameters")

            A_list = [A*1.5 for i in range(self.n_samples_gradient_tests
            )]

            b_list = b_list*2.5 

            print("\nChanges parameters")

            objective_function_with_parameters.update_function(A_list, 
            b_list)

            print("\nWarms up again")

            t_initial = time.time()

            result3 = objective_function_with_parameters(model_params)

            elapsed_time = time.time()-t_initial

            print("\nFinishes warming up again after "+str(elapsed_time
            ))

            t_initial = time.time()

            result3 = objective_function_with_parameters(model_params)

            elapsed_time = time.time()-t_initial

            print("\nElapsed time: "+str(elapsed_time)+". Using automa"+
            "tic call with parameters")

        print("\n\n###### Block multiplication ######")

        A_list = [A for i in range(self.n_samples_gradient_tests)]

        loss_block = loss_assemblers.QuadraticLossOverLinearResidualAssembler(
        A_list, b_list, block_multiplication=True)

        objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, loss_block, input_test_data)

        print("\nWarms up")

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nFinishes warming up after "+str(elapsed_time))

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nElapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters")

        A_list = [A*1.5 for i in range(self.n_samples_gradient_tests)]

        b_list = b_list*2.5 

        print("\nChanges parameters")

        objective_function_with_parameters.update_function(A_list, b_list)

        print("\nWarms up again")

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nFinishes warming up again after "+str(elapsed_time))

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nElapsed time: "+str(elapsed_time)+". Using automatic "+
        "call with parameters")

        if flag_batch_multiplication:

            print("\n\n###### Batch multiplication with helper loss ##"+
            "####")

            # Sets the loss function

            loss = loss_assemblers.QuadraticLossOverLinearResidualAssembler(
            A_list, b_list, block_multiplication=False, 
            custom_gradient_usage=True)

            objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
            custom_model, loss, input_test_data)

            print("\nWarms up")

            t_initial = time.time()

            result3 = objective_function_with_parameters(model_params)

            elapsed_time = time.time()-t_initial

            print("\nFinishes warming up after "+str(elapsed_time))

            t_initial = time.time()

            result3 = objective_function_with_parameters(model_params)

            elapsed_time = time.time()-t_initial

            print("\nElapsed time: "+str(elapsed_time)+". Using automa"+
            "tic call with parameters")

            A_list = [A*1.5 for i in range(self.n_samples_gradient_tests
            )]

            b_list = b_list*2.5 

            print("\nChanges parameters")

            objective_function_with_parameters.update_function(A_list, 
            b_list)

            print("\nWarms up again")

            t_initial = time.time()

            result3 = objective_function_with_parameters(model_params)

            elapsed_time = time.time()-t_initial

            print("\nFinishes warming up again after "+str(elapsed_time
            ))

            t_initial = time.time()

            result3 = objective_function_with_parameters(model_params)

            elapsed_time = time.time()-t_initial

            print("\nElapsed time: "+str(elapsed_time)+". Using automa"+
            "tic call with parameters")

        print("\n\n###### Block multiplication with helper loss ######")

        A_list = [A for i in range(self.n_samples_gradient_tests)]

        loss_block = loss_assemblers.QuadraticLossOverLinearResidualAssembler(
        A_list, b_list, block_multiplication=True, custom_gradient_usage=
        True)

        objective_function_with_parameters, model_params = loss_tools.build_loss_gradient_varying_model_parameters(
        custom_model, loss_block, input_test_data)

        print("\nWarms up")

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nFinishes warming up after "+str(elapsed_time))

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nElapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters")

        A_list = [A*1.5 for i in range(self.n_samples_gradient_tests)]

        b_list = b_list*2.5 

        print("\nChanges parameters")

        objective_function_with_parameters.update_function(A_list, b_list)

        print("\nWarms up again")

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nFinishes warming up again after "+str(elapsed_time))

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("\nElapsed time: "+str(elapsed_time)+". Using automatic "+
        "call with parameters")

# Runs all tests

if __name__ == "__main__":

    unittest.main()