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

    # 1. Defines a function to test the custom layer class
    
    def test_customLayer(self):

        print("\n#####################################################"+
        "###################\n#                           Custom layer"+
        " test                          #\n###########################"+
        "#############################################\n")

        activation_dict = {"relu": 2, "sigmoid": 3}

        live_activations, *_ = ANN_tools.verify_activationDict(
        activation_dict, 0, dict(), False)

        mixed_layer = ANN_tools.MixedActivationLayer(activation_dict,
        live_activationsDict=live_activations)

        output_direct = mixed_layer(self.input_tensor)

        print("Output with direct call:", output_direct.numpy())

    # 2. Defines a function to test the multilayered model

    def test_multilayeredModel(self):

        print("\n#####################################################"+
        "###################\n#                       Tests multilayer"+
        "ed model                       #\n###########################"+
        "#############################################\n")

        activation_functionsEachLayer = [{"relu": 2, "sigmoid": 3},
        {"relu": 3, "sigmoid": 4}, {"sigmoid": 2}]

        model = ANN_tools.MultiLayerModel(self.input_dimension,
        activation_functionsEachLayer)()

        output = model(self.dummy_input)

        print("Input multilayered model:", self.dummy_input.numpy())

        print("Output multilayered model:", output.numpy())

        self.assertEqual(output.shape[0], self.dummy_input.shape[0])

    # 3. Defines a function to test saving and laoding of a model

    def test_savingAndLoading(self):

        print("\n#####################################################"+
        "###################\n#                       Tests saving and"+
        " loading                       #\n###########################"+
        "#############################################\n")

        config = [{"relu": 2, "sigmoid": 3}, {"relu": 3, "sigmoid": 2}]

        model = ANN_tools.MultiLayerModel(2, config)()

        model_path = os.path.join(file_handling_tools.get_parent_path_of_file(
        file=__file__), "mixed_activation_mo"+
        "del.keras")

        print(model_path)

        # Saves the model

        model.save(model_path)

        # Loads it back

        loaded_model = tf.keras.models.load_model(model_path)

        # Tests the loaded model

        dummy_input = tf.convert_to_tensor(np.random.rand(1, 2).astype(
        np.float32))

        output_original = model(dummy_input)

        output_loaded = loaded_model(dummy_input)

        print("Output from saved model:", output_original)

        print("Output from loaded model:", output_loaded)

        np.testing.assert_allclose(output_original.numpy(), 
        output_loaded.numpy(), rtol=1e-5)

    # 4. Defines a function to test the trainability of such custom mo-
    # dels

    def test_trainability(self):

        print("\n#####################################################"+
        "###################\n#                  Tests trainability of"+
        " custom layer                  #\n###########################"+
        "#############################################\n")

        # Creates the custom model with custom layers

        custom_model = ANN_tools.MultiLayerModel(2, 
        self.layers_information, enforce_customLayers=True)()

        # Iterates through the optimization loop

        custom_model, *_ = training_tools.ModelTraining(custom_model, 
        self.training_data, self.training_trueValues, loss_metric
        =self.loss_metric, optimizer=self.optimizer, n_iterations=
        self.n_iterations, gradient_tolerance=self.gradient_tolerance, 
        verbose_deltaIterations=self.verbose_deltaIterations, verbose=
        True)()

        # Evaluates the model on the test set

        y_test = custom_model(self.test_inputTensor)

        test_loss = self.loss_metric(self.test_trueTensor, y_test)

        print("Loss function on test set:", format(test_loss.numpy(), 
        '.5e'))

    # 5. Defines a function to test the trainability of the equivalent 
    # dense model

    def test_trainabilityKeras(self):

        print("\n#####################################################"+
        "###################\n#                       Tests trainabili"+
        "ty Keras                       #\n###########################"+
        "#############################################\n")

        # Initializes the Keras model

        model_parameters = []

        for layer in self.layers_information:

            keys = list(layer.keys())

            model_parameters.append(tf.keras.layers.Dense(layer[keys[0]
            ], activation=keys[0]))

        keys = list(self.layers_information[0].keys())

        model_parameters[0] = tf.keras.layers.Dense(
        self.layers_information[0][keys[0]], activation=keys[0], 
        input_dim=2)

        tf_model = tf.keras.Sequential(model_parameters)

        # Iterates through the optimization loop

        tf_model, *_ = training_tools.ModelTraining(tf_model, 
        self.training_data, self.training_trueTensor, loss_metric
        =self.loss_metric, optimizer=self.optimizer, n_iterations=
        self.n_iterations, gradient_tolerance=self.gradient_tolerance, 
        verbose_deltaIterations=self.verbose_deltaIterations, verbose=
        True)()

        # Evaluaets the model on the test set

        y_test = tf_model(self.test_inputTensor)

        test_loss = self.loss_metric(self.test_trueTensor, y_test)

        print("Loss function on test set:", format(test_loss.numpy(), 
        '.5e'))

    # 6. Defines a function to test evaluation of the gradient of a model

    def test_gradient_evaluation(self):

        print("\n#####################################################"+
        "###################\n#                       Tests gradient e"+
        "valuation                      #\n###########################"+
        "#############################################\n")
        
        # Creates the new test data

        input_dimension = 9

        output_dimension = 100

        n_samples = 100

        x_min = -1.0

        x_max = 1.0

        data_matrix = []

        for i in range(n_samples):

            data_matrix.append([ANN_tools.random_inRange(x_min, x_max
            ) for j in range(input_dimension)])

        # Converts the data to tensors

        input_test_data = tf.constant(data_matrix, dtype=
        tf.float32)

        # Creates the custom model with custom layers

        evaluate_parameters_gradient=True

        ANN_class = ANN_tools.MultiLayerModel(input_dimension, [{"sigm"+
        "oid": 3}, {"linear": output_dimension}], enforce_customLayers=
        True, evaluate_parameters_gradient=evaluate_parameters_gradient)

        custom_model, gradient = ANN_class()

        #print("Original input tensor:")

        #print(input_test_data)

        """gradient = ANN_class.model_jacobian(custom_model, 
        evaluate_parameters_gradient="tensorflow gradient")

        t_initial = time.time()

        test_gradient = gradient(input_test_data)

        elapsed_time = time.time()-t_initial

        print("\nelapsed time: "+str(elapsed_time)+". Gradient as a ma"+
        "trix computing the gradient function from tensorflow:")

        print(test_gradient)"""

        """gradient = ANN_class.model_jacobian(custom_model, 
        evaluate_parameters_gradient="tensorflow jacobian")

        t_initial = time.time()

        test_gradient = gradient(input_test_data)

        elapsed_time = time.time()-t_initial

        print("\nelapsed time: "+str(elapsed_time)+". Gradient as a ma"+
        "trix computing the jacobian function from tensorflow:")

        print(test_gradient)"""

        gradient = diff_tools.model_jacobian(custom_model, 
        ANN_class.output_dimension, evaluate_parameters_gradient="vect"+
        "orized tensorflow jacobian")

        t_initial = time.time()

        test_gradient = gradient(input_test_data)

        elapsed_time = time.time()-t_initial

        print("\nelapsed time: "+str(elapsed_time)+". Gradient as a matrix computing the vectorized jacobia"+
        "n function from tensorflow:")

        #print(test_gradient)

    # Defines a test to pick up the model parameters as a numpy array and
    # reassign them

    def test_parameters_conversion(self):

        print("\n#####################################################"+
        "###################\n#                      Tests parameters "+
        "conversion                     #\n###########################"+
        "#############################################\n")
        
        # Creates the new test data

        input_dimension = 2

        output_dimension = 2

        n_samples = 2

        x_min = -1.0

        x_max = 1.0

        data_matrix = []

        for i in range(n_samples):

            data_matrix.append([ANN_tools.random_inRange(x_min, x_max
            ) for j in range(input_dimension)])

        # Converts the data to tensors

        input_test_data = tf.constant(data_matrix, dtype=
        tf.float32)

        # Creates the custom model with custom layers

        evaluate_parameters_gradient=False

        ANN_class = ANN_tools.MultiLayerModel(input_dimension, [{"sigm"+
        "oid": 3}, {"linear": output_dimension}], enforce_customLayers=
        True, evaluate_parameters_gradient=evaluate_parameters_gradient)

        custom_model = ANN_class()

        # Gets the model parameters as a list

        model_params = parameters_tools.model_parameters_to_numpy(
        custom_model)

        print("Original model parameters:")

        print(model_params)

        print("Response of the model:")

        print(custom_model(input_test_data))

        # Reassigns the same model parameters

        custom_model = parameters_tools.update_model_parameters(
        custom_model, model_params)

        # Gets the model again and shows them

        print("\nReassigned model parameters:")

        print(parameters_tools.model_parameters_to_numpy(custom_model))

        print("Response of the model:")

        print(custom_model(input_test_data))

    # Defines a test the new loss function as the multiplication of a
    # coefficient matrix by the model output

    def test_linear_loss(self):

        print("\n#####################################################"+
        "###################\n#                      Tests linear loss"+
        " function                      #\n###########################"+
        "#############################################\n")
        
        # Creates the new test data

        input_dimension = 9

        output_dimension = 100

        n_samples = 1000

        x_min = -1.0

        x_max = 1.0

        data_matrix = []

        for i in range(n_samples):

            data_matrix.append([ANN_tools.random_inRange(x_min, x_max
            ) for j in range(input_dimension)])

        # Converts the data to tensors

        input_test_data = tf.constant(data_matrix, dtype=
        tf.float32)

        # Creates the custom model with custom layers

        evaluate_parameters_gradient=False

        ANN_class = ANN_tools.MultiLayerModel(input_dimension, [{"sigm"+
        "oid": 100}, {"linear": output_dimension}], enforce_customLayers=
        True, evaluate_parameters_gradient=evaluate_parameters_gradient,
        verbose=True)

        custom_model = ANN_class()

        # Gets the coefficient matrix

        coefficient_matrix = tf.random.normal((n_samples, 
        output_dimension))

        # Sets the loss function

        loss = lambda model_response: loss_tools.linear_loss(model_response, 
        coefficient_matrix)

        # Sets a function to capture the value and the gradient of the
        # loss 

        def objective_function(custom_model=custom_model, loss=loss,
        input_test_data=input_test_data):

            gradient = diff_tools.scalar_gradient_wrt_trainable_params(
            loss, custom_model, input_test_data)

            # Converts to numpy

            return diff_tools.convert_scalar_gradient_to_numpy(gradient)
        
        # Sets the same function but enabling the model parameters as 
        # argument from a numpy array

        objective_function_with_parameters, model_params = loss_tools.build_loss_varying_model_parameters(
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

        t_initial = time.time()

        result = objective_function()

        elapsed_time = time.time()-t_initial

        print("Elapsed time: "+str(elapsed_time)+". Loss function and "+
        "gradient:")

        print(result)

        theta0, shapes = parameters_tools.model_parameters_to_flat_tensor_and_shapes(
        custom_model)

        L, g = parameters_tools.loss_and_grad(theta0, custom_model, input_test_data, coefficient_matrix, shapes)

        t_initial = time.time()

        L, g = parameters_tools.loss_and_grad(theta0, custom_model, input_test_data, coefficient_matrix, shapes)

        elapsed_time = time.time()-t_initial
        print("Elapsed time: "+str(elapsed_time)+". Using call with pa"+
        "rameters")

        print(np.linalg.norm(result-g))

        print(np.linalg.norm(result-result2))

        # Sets the same function but enabling the model parameters as 
        # argument from a tensorflow 1D tensor

        objective_function_with_parameters, model_params = loss_tools.build_loss_varying_model_parameters(
        custom_model, loss, input_test_data)

        result3 = objective_function_with_parameters(model_params)

        t_initial = time.time()

        result3 = objective_function_with_parameters(model_params)

        elapsed_time = time.time()-t_initial

        print("Elapsed time: "+str(elapsed_time)+". Using automatic ca"+
        "ll with parameters. The difference to the gradient without au"+
        "tomatic function assembly: "+str(np.linalg.norm(result3-g)))

# Runs all tests

if __name__ == "__main__":

    unittest.main()