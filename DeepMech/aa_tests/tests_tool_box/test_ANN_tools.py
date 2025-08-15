# Routine to store some tests

import unittest

import tensorflow as tf

import numpy as np

from ...tool_box import ANN_tools

from ...tool_box import training_tools

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

        self.layers_information = [{"sigmoid": 1000}, {"sigmoid":1000},
        {"linear": 1}]

    # Defines a function to test the custom layer class
    
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

        output_call = mixed_layer.call(self.input_tensor)

        output_direct = mixed_layer(self.input_tensor)

        print("Output with call:", output_call.numpy())

        print("Output with direct call:", output_direct.numpy())

        self.assertEqual(output_call.shape, output_direct.shape)

    # Defines a function to test the multilayered model

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

    # Defines a function to test saving and laoding of a model

    def test_savingAndLoading(self):

        print("\n#####################################################"+
        "###################\n#                       Tests saving and"+
        " loading                       #\n###########################"+
        "#############################################\n")

        config = [{"relu": 2, "sigmoid": 3}, {"relu": 3, "sigmoid": 2}]

        model = ANN_tools.MultiLayerModel(2, config)()

        model_path = "mixed_activation_model.keras"

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

    # Defines a function to test the trainability of such custom models

    def test_trainability(self):

        print("\n#####################################################"+
        "###################\n#                          Tests trainab"+
        "ility                          #\n###########################"+
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

    # Defines a function to test the trainability of the equivalent den-
    # se model

    def test_trainabilityKeras(self):

        print("\n#####################################################"+
        "###################\n#                          Tests trainab"+
        "ility                          #\n###########################"+
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

# Runs all tests

if __name__ == "__main__":

    unittest.main()