# Routine to test the implementation of optimization methods

import unittest

import numpy as np

from copy import deepcopy

# Defines a function to test the ANN optimization wrappers

class TestsClass(unittest.TestCase):

    def setUp(self):

        # Sets the unimodal data

        self.unimodal_x_data = np.sort(np.random.rand(15))

        self.unimodal_y_data = [np.exp(x) for x in self.unimodal_x_data]

        self.unimodal_y_data2 = [np.exp(1.5*x) for x in self.unimodal_x_data]

        # Sets the multimodal data

        self.multimodal_x_data = []

        self.multimodal_y_data = []

        self.n_curves = 3

        for i in range(self.n_curves):

            self.multimodal_x_data.append(np.sort(np.random.rand(15)))

            self.multimodal_y_data.append([np.exp((i+1)*x) for x in (
            self.multimodal_x_data[-1])])

    def test_multimodal_plot(self):

        print("\n#####################################################"+
        "###################\n#                           Multimodal c"+
        "urve                           #\n###########################"+
        "#############################################\n")

# Runs all tests

if __name__=="__main__":

    unittest.main()