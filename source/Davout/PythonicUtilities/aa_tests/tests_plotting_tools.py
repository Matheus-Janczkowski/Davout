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

        x_data = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])]

        y_data = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 6.0])]

        plotting_tools.plane_plot(x_data=x_data, y_data=y_data, 
        file_name="test_two_curves_numpy", plot_type=["line", "scatter"],
        element_size=[1.5, 10.0], color=["yellow", "black"])

        x_data = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        y_data = [[1.0, 2.0, 3.0], [1.0, 4.0, 6.0]]

        plotting_tools.plane_plot(x_data=x_data, y_data=y_data, 
        file_name="test_two_curves_list")

    # Defines a function to test the plot of a curve with error bar
    
    def test_error_bar(self):

        print("\n#####################################################"+
        "###################\n#                              Error bar"+
        "                               #\n###########################"+
        "#############################################\n")

        # Initializes the error bar with the confidence intervals

        error_bar = [0.1 for i in range(len(self.unimodal_x_data))]

        # Calls the plotter

        # Scatter single curve given the error bar but separately plotted

        print("\nTests error plot with error bar being separately plot"+
        "ted")

        plot_object = plotting_tools.plane_plot(x_data=
        self.unimodal_x_data, y_data=self.unimodal_y_data, file_name=
        "test_error_bar_separate_plotting", plot_type="scatter", 
        error_bar=error_bar, color="black")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data2, file_name="test_error_bar_separate_plot"+
        "ting", plot_type="line", plot_object=plot_object)

        # Scatter single curve given the error bar

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=error_bar, file_name=
        "test_error_bar", plot_type="scatter")

        # Scatter single curve given the error bar with upper and lower
        # bounds for error

        error_bar_lower_upper = [[0.1, 0.05] for i in range(len(
        self.unimodal_x_data))]

        print("\nTests error plot with error bar that has upper and lo"+
        "wer bounds")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=error_bar_lower_upper, 
        file_name="test_error_bar_lower_upper", plot_type="scatter")

        # Scatter single curve given the error bar with upper and lower
        # bounds for error

        error_bar_lower_upper = [[0.1, 0.05] for i in range(len(
        self.unimodal_x_data))]

        print("\nTests error plot with error region that has upper and lo"+
        "wer bounds")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=error_bar_lower_upper, 
        file_name="test_error_bar_line_lower_upper", plot_type="line")

        # Continuous single curve given the error bar

        print("\nTests error plot for a continuous curve")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=error_bar, file_name=
        "test_error_region", plot_type="line")

        # Calls with multimodal data

        error_bar = []

        error_bar_lower_upper = []

        for i in range(self.n_curves):

            error_bar.append([((i+1)/10) for j in range(len(
            self.multimodal_x_data[i]))])

            error_bar_lower_upper.append([[((i+1)/10), ((i+1)/5)] for (
            j) in range(len(self.multimodal_x_data[i]))])

        # Continuous multiple curves given the error bar

        print("\nTests error plot for multiple continuous curve")

        plotting_tools.plane_plot(x_data=self.multimodal_x_data, y_data=
        self.multimodal_y_data, error_bar=error_bar, file_name=
        "test_error_region_multimodal", plot_type="line")

        # Continuous multiple curves given the error bar with upper and
        # lower bounds

        print("\nTests error plot for multiple continuous curve with u"+
        "pper and lower bounds")

        plotting_tools.plane_plot(x_data=self.multimodal_x_data, y_data=
        self.multimodal_y_data, error_bar=error_bar_lower_upper, file_name=
        "test_error_region_multimodal_upper_lower", plot_type="line")

        # Continuous single curve automatically evaluating the error bar
        # for the t-Student distribution

        print("\nTests error plot for a continuous curve and automatic"+
        " evaluation of the error bar following t-Student")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="t-Student", file_name="test_e"+
        "rror_region_t_student", plot_type="line")

        # Continuous single curve automatically evaluating the error bar
        # for the normal distribution

        print("\nTests error plot for a continuous curve and automatic"+
        " evaluation of the error bar following normal distribution")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="normal distribution", 
        file_name="test_error_region_z_score", plot_type="line")

        # Scatter single curve automatically evaluating the error bar
        # for the normal distribution

        print("\nTests error plot for a scatter curve and automatic ev"+
        "aluation of the error bar following t-Student")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="normal distribution", 
        file_name="test_error_bar_z_score", plot_type="scatter")

        # Continuous single curve automatically evaluating the error bar
        # for the t-Student distribution asking for a 90% confidence

        print("\nTests error plot for a scatter curve and automatic ev"+
        "aluation of the error bar following t-Student and 90 percent "+
        "of confidence")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar={"name": "t-Student", "confi"+
        "dence": 0.9}, file_name="test_error_region_t_student_0_90", 
        plot_type="line")

        # Continuous single curve automatically evaluating the error bar
        # for the normal distribution asking for a 90% confidence

        print("\nTests error plot for a scatter curve and automatic ev"+
        "aluation of the error bar following normal distribution and 9"+
        "0 percent of confidence")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar={"name": "normal distribution",
        "confidence": 0.9}, file_name="test_error_region_z_score_0_90", 
        plot_type="line")

# Runs all tests

if __name__=="__main__":

    unittest.main()