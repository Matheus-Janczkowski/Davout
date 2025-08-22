# Routine to test the implementation of optimization methods

import unittest

import numpy as np

import time

from ...tool_box import optimization_tools

from ....DeepMech.tool_box import ANN_tools, loss_tools

from ....DeepMech.tool_box import differentiation_tools as diff_tools

# Defines a function to test the ANN optimization wrappers

class TestOptimizationWrappers(unittest.TestCase):

    def setUp(self):

        # Sets the number of optimization iterations

        self.n_iterations = 10000

        # Sets the convergence tolerance

        self.gradient_tolerance = 1E-3

    # Defines a function to test the Rosenbrock objective function
    
    def test_rosenbrock(self):

        print("\n#####################################################"+
        "###################\n#                              Rosenbroc"+
        "k                              #\n###########################"+
        "#############################################\n")

        # Sets the optimization function

        a=1.0
        
        b=100.0

        def rosenbrock(x):

            return ((a-x[0])**2)+(b*((x[1]-(x[0]**2))**2))
        
        # Sets the gradient

        def grad_rosenbrock(x):

            return [(-2.0*(a-x[0]))-(4.0*b*(x[1]-(x[0]**2))*x[0
            ]), 2.0*b*(x[1]-(x[0]**2))]
        
        # Sets the true minimum

        true_minimum = np.array([1.0, 1.0])

        # Sets the optimization problem's parameters

        optimization_method = "CG"

        initial_guess = np.array([0.0, 3.0])

        # Sets the optimization class

        optimization_class = optimization_tools.UnconstrainedFirstOrderMinimization(
        objective_function=rosenbrock, objective_gradient=
        grad_rosenbrock, optimization_method=optimization_method, 
        initial_guess=initial_guess)
        
        # Solves

        optimization_class()

        print("The minimum found for the Rosenbrock function is: "+str(
        optimization_class.design_variables)+"\n")

        self.assertEqual(np.testing.assert_allclose(true_minimum, 
        optimization_class.design_variables), None)

    # Defines a function to test if a tensorflow NN model can be succes-
    # sfully trained using scipy

    def test_train_NN_model(self):

        print("\n#####################################################"+
        "###################\n#                             NN trainin"+
        "g                              #\n###########################"+
        "#############################################\n")

        ANN_class = ANN_tools.MultiLayerModel(
        self.input_dimension_gradient_tests, 
        self.activation_list_gradient_tests, enforce_customLayers=True)

        custom_model = ANN_class()

        # Gets the coefficient matrix

        coefficient_matrix = tf.random.normal((
        self.n_samples_gradient_tests, 
        self.output_dimension_gradient_tests))
        
        # Sets the loss function

        loss = lambda model_response: loss_tools.linear_loss(model_response, 
        coefficient_matrix)

        # Sets the same function but enabling the model parameters as 
        # argument from a tensorflow 1D tensor

        objective_function_with_parameters, model_params = loss_tools.build_loss_varying_model_parameters(
        custom_model, loss, input_test_data)

        result = objective_function_with_parameters(model_params)

        t_initial = time.time()

        result = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("Elapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters")

# Runs all tests

if __name__=="__main__":

    unittest.main()