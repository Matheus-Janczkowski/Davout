# Routine to test the implementation of optimization methods

import unittest

import numpy as np

from copy import deepcopy

from ...PythonicUtilities import plotting_tools

# Defines a function to test the ANN optimization wrappers

class TestPlots(unittest.TestCase):

    def setUp(self):

        # Sets the unimodal data

        self.unimodal_x_data = np.sort(np.random.rand(15))

        self.unimodal_y_data = [np.exp(x) for x in self.unimodal_x_data]

        # Sets the multimodal data

        self.multimodal_x_data = []

        self.multimodal_y_data = []

        self.n_curves = 3

        for i in range(self.n_curves):

            self.multimodal_x_data.append(np.sort(np.random.rand(15)))

            self.multimodal_y_data.append([np.exp((i+1)*x) for x in (
            self.multimodal_x_data[-1])])

    # Defines a function to test the plot of a curve with error bar
    
    def test_error_bar(self):

        print("\n#####################################################"+
        "###################\n#                              Error bar"+
        "                               #\n###########################"+
        "#############################################\n")

        # Initializes the error bar with the confidence intervals

        error_bar = [0.1 for i in range(len(self.unimodal_x_data))]

        # Calls the plotter

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=deepcopy(error_bar), file_name=
        "test_error_bar", plot_type="scatter")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=deepcopy(error_bar), file_name=
        "test_error_region", plot_type="line")

        # Calls with multimodal data

        error_bar = []

        for i in range(self.n_curves):

            error_bar.append([((i+1)/10) for j in range(len(
            self.multimodal_x_data[i]))])

        plotting_tools.plane_plot(x_data=self.multimodal_x_data, y_data=
        self.multimodal_y_data, error_bar=deepcopy(error_bar), file_name=
        "test_error_region_multimodal", plot_type="line")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="t-Student", file_name="test_e"+
        "rror_region_t_student", plot_type="line")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="normal distribution", 
        file_name="test_error_region_z_score", plot_type="line")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="normal distribution", 
        file_name="test_error_bar_z_score", plot_type="scatter")

# Runs all tests

if __name__=="__main__":

    unittest.main()