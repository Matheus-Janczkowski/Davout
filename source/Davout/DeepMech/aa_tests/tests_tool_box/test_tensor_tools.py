# Routine to store some tests

import unittest

import tensorflow as tf

import numpy as np

from ...tool_box import tensor_tools

from ...tool_box import ANN_tools

from ...tool_box import training_tools

# Defines a function to test the tensor tools methods

class TestTensorTools(unittest.TestCase):

    def setUp(self):

        self.input_tensor = tf.constant([[1.0, 2.0]], dtype=tf.float32)

        self.input_dimension = 2

        self.n_samples = 1

        self.dummy_input = tf.random.normal((self.n_samples, 
        self.input_dimension))

    def test_gradient(self):

        print("\n#####################################################"+
        "###################\n#                            Tests gradi"+
        "ent                            #\n###########################"+
        "#############################################\n")

        activation_functionsEachLayer = [{"relu": 2, "sigmoid": 3},
        {"relu": 3, "sigmoid": 4}, {"sigmoid": 2}]

        model = ANN_tools.MultiLayerModel(self.input_dimension,
        activation_functionsEachLayer)()

        output = model(self.dummy_input)

        # Evaluates the gradient

        gradient_model = tensor_tools.grad(model, self.dummy_input)

        print("\nThe gradient given by the 'grad' method is:\n"+str(
        gradient_model)+"\n")

    def test_divergent(self):

        print("\n#####################################################"+
        "###################\n#                            Tests diver"+
        "gent                           #\n###########################"+
        "#############################################\n")

        activation_functionsEachLayer = [{"relu": 2, "sigmoid": 3},
        {"relu": 3, "sigmoid": 4}, {"sigmoid": 2}]

        model = ANN_tools.MultiLayerModel(self.input_dimension,
        activation_functionsEachLayer)()

        output = model(self.dummy_input)

        # Evaluates the divergent

        div_model = tensor_tools.div(model, self.dummy_input)

        print("\nThe divergent given by the 'div' method is:\n"+str(
        div_model)+"\n")

# Runs all tests

if __name__ == "__main__":

    unittest.main()