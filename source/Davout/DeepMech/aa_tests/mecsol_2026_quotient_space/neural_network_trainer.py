# Routine to train a neural network model as surrogate for the RVE kine-
# matics

import numpy as np 

import tensorflow as tf

from time import time

from .....Davout.DeepMech.tool_box import ANN_tools, training_tools

from .....Davout.DeepMech.tool_box.loss_assembler_classes import MaximumAbsoluteError

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.MultiMech.tool_box.binary_tools import read_field_from_binary

from .....Davout.GraphUtilities import paraview_tools

# Defines a class for the training and testing procedures

class SurrogateModel:

    def __init__(self, displacement_data_file, input_data_file, 
    saved_model_file, results_path, n_training_samples, 
    quotient_space_dimension, n_monte_carlo_realizations, n_best_models,
    n_best_samples, mesh_file_name, screenshots_path, optimizer):

        # Stores the data

        self.displacement_data_file = displacement_data_file

        self.input_data_file = input_data_file

        self.saved_model_file = saved_model_file

        self.results_path = results_path

        self.n_training_samples = n_training_samples

        self.quotient_space_dimension = quotient_space_dimension

        self.n_monte_carlo_realizations = n_monte_carlo_realizations

        self.n_best_models = n_best_models

        self.n_best_samples = n_best_samples

        self.mesh_file_name = mesh_file_name

        self.screenshots_path = screenshots_path

        self.optimizer = optimizer

        # Reads the two files

        self.output_data = np.load(self.results_path+"//"+
        self.displacement_data_file)

        self.n_output_neurons = self.output_data.shape[1]

        # Sets a list of layers and the activation functions

        self.activations_list = [{"elu": {"number of neurons": 100}}, {
        "elu": {"number of neurons": 100}}, {"linear": 
        self.n_output_neurons}]

    # Defines a function to train the neural network model

    def train_surrogate_model(self):

        input_data = np.load(self.results_path+"//"+self.input_data_file)

        # Reshufles data to put the displacement gradient as the first
        # columns. This is a requirement of the implementation of the 
        # GatedQuotientSpace architecture

        input_data = np.hstack((input_data[:,(input_data.shape[1]-
        self.quotient_space_dimension):], input_data[:,:(
        input_data.shape[1]-self.quotient_space_dimension)]))

        # Sets the training data

        training_data = input_data[0:self.n_training_samples,:]

        training_true_values = self.output_data[0:(
        self.n_training_samples),:]

        # Gets the number of input and output neurons

        n_input_neurons = input_data.shape[1]

        # Creates the class of neural network information

        ANN_class = ANN_tools.MultiLayerModel(n_input_neurons, 
        self.activations_list, enforce_customLayers=True, verbose=True, 
        parameters_dtype="float32", custom_architecture={"name": "Gate"+
        "dQuotientSpace", "quotient space dimension": 
        self.quotient_space_dimension})

        custom_model = ANN_class()

        # Sets the number of training iterations and the number of iter-
        # ations to plot results in the terminal

        maximum_iterations = 1000
        
        verbose_delta_iterations = 50

        # Sets the optimization class for training

        training_class = training_tools.ModelCustomTraining(custom_model,
        training_data, training_true_values, 
        tf.keras.losses.MeanAbsoluteError(), verbose=True, n_iterations=
        maximum_iterations, verbose_deltaIterations=
        verbose_delta_iterations, save_model_file=self.saved_model_file, 
        match_data_float_type_to_trainables=True, parent_path=
        self.results_path, optimizer=self.optimizer)

        t_initial = time()

        training_class()

        elapsed_time = time()-t_initial

        print("\nTrains at "+str(elapsed_time)+" seconds")

        # Tests Monte Carlo training

        training_class.monte_carlo_training(n_realizations=
        self.n_monte_carlo_realizations, best_models_rank_size=
        self.n_best_models, show_reinitialization_distance=True)

        # Checks the loss again with the best model of the Monte Carlo
        # training

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data for the best model is "+str(
        training_class.loss_unseen_data(training_true_values, 
        training_data, output_as_numpy=True)))

    # Defines a function to test the model

    def test_surrogate_model(self):

        # Reads the input file

        input_data = np.load(self.results_path+"//"+self.input_data_file)

        print("The data has a total of "+str(input_data.shape[0])+"sam"+
        "ples\n")

        # Reshufles data to put the displacement gradient as the first
        # columns. This is a requirement of the implementation of the 
        # GatedQuotientSpace architecture

        input_data = np.hstack((input_data[:,(input_data.shape[1]-
        self.quotient_space_dimension):], input_data[:,:(
        input_data.shape[1]-self.quotient_space_dimension)]))

        # Sets the test data

        test_data = input_data[self.n_training_samples:,:]

        test_true_values = self.output_data[self.n_training_samples:,:]

        # Defines the loss function metric

        loss_metric = tf.keras.losses.MeanAbsoluteError()

        maximum_absolute_error = MaximumAbsoluteError()

        # Iterates through the best models

        for i in range(self.n_best_models):

            # Loads this model

            loaded_model = tf.keras.models.load_model(self.results_path+
            "//"+str(i+1)+"_best_model.keras")

            # Gets the output of the loaded model

            output_model = loaded_model(test_data)

            # Gets the loss of the test data

            test_loss = loss_metric(test_true_values, output_model)

            print("Loads the "+str(i+1)+"-th best model")

            print("Loss function on test set:", format(test_loss.numpy(
            ), '.5e'))

            # Verifies with the maximum absolute error

            maximum_absolute_value = maximum_absolute_error(
            test_true_values, output_model)

            print("Maximum absolute error on test set: "+str(format(
            maximum_absolute_value.numpy(),'.5e'))+"\nwhereas the mini"+
            "mum absolute error is "+str(format(
            maximum_absolute_error.minimum_absolute_error(
            test_true_values, output_model).numpy(), '.5e')))

            # Gets the mean absolute error for each row

            loss_per_sample = tf.reduce_mean(tf.abs(test_true_values-
            output_model), axis=1).numpy()

            # Gets the samples indices with the lowest error

            best_samples_indices = np.argsort(loss_per_sample)[
            :self.n_best_samples]

            print("The "+str(self.n_best_samples)+" best samples have "+
            "the following mean absolute error:\n"+str(loss_per_sample[
            best_samples_indices])+"\n")

            # Saves as binary files the true data and the output data of 
            # the best preserving samples

            np.save(self.results_path+"//true_best_samples_of_"+str(i+1
            )+"_best_model.npy", test_true_values[best_samples_indices,:
            ])

            np.save(self.results_path+"//surrogate_best_samples_of_"+
            str(i+1)+"_best_model.npy", output_model.numpy()[
            best_samples_indices,:])

    # Defines a function to insert the results into a function space and
    # plot it for the samples of the training set

    def plot_training_response(self):

        # Gets the input data

        input_data = np.load(self.results_path+"//"+self.input_data_file)
    
        # Reshufles data to put the displacement gradient as the first
        # columns. This is a requirement of the implementation of the 
        # GatedQuotientSpace architecture

        input_data = np.hstack((input_data[:,(input_data.shape[1]-
        self.quotient_space_dimension):], input_data[:,:(
        input_data.shape[1]-self.quotient_space_dimension)]))

        # Sets the training data

        training_data = input_data[0:self.n_training_samples,:]

        training_true_values = self.output_data[0:(
        self.n_training_samples),:]

        # Defines the loss function metric

        loss_metric = tf.keras.losses.MeanAbsoluteError()

        maximum_absolute_error = MaximumAbsoluteError()

        # Iterates through the best models

        for i in range(self.n_best_models):

            # Loads this model

            loaded_model = tf.keras.models.load_model(self.results_path+
            "//"+str(i+1)+"_best_model.keras")

            # Gets the output of the loaded model

            output_model = loaded_model(training_data)

            # Gets the loss of the test data

            test_loss = loss_metric(training_true_values, output_model)

            print("Loads the "+str(i+1)+"-th best model")

            print("Loss function on training set:", format(
            test_loss.numpy(), '.5e'))

            # Verifies with the maximum absolute error

            maximum_absolute_value = maximum_absolute_error(
            training_true_values, output_model)

            print("Maximum absolute error on test set: "+str(format(
            maximum_absolute_value.numpy(),'.5e'))+"\nwhereas the mini"+
            "mum absolute error is "+str(format(
            maximum_absolute_error.minimum_absolute_error(
            training_true_values, output_model).numpy(), '.5e')))

            # Gets the mean absolute error for each row

            loss_per_sample = tf.reduce_mean(tf.abs(training_true_values
            -output_model), axis=1).numpy()

            # Gets the samples indices with the lowest error

            best_samples_indices = np.argsort(loss_per_sample)[
            :self.n_best_samples]

            print("The "+str(self.n_best_samples)+" best samples have "+
            "the following mean absolute error:\n"+str(loss_per_sample[
            best_samples_indices])+"\n")

            # Recovers the name of the file that contains the displace-
            # ment DOFs of this model

            displacement_output_file = (self.results_path+"//surrogate"+
            "_best_training_samples_of_"+str(i+1)+"_best_model.npy")

            # Recovers the name of the file with the true values of dis-
            # placement

            true_displacement_output_file = (self.results_path+"//true"+
            "_best_training_samples_of_"+str(i+1)+"_best_model.npy")

            # Saves as binary files the true data and the output data of 
            # the best preserving samples

            np.save(true_displacement_output_file, training_true_values[
            best_samples_indices,:])

            np.save(displacement_output_file, output_model.numpy()[
            best_samples_indices,:])

            # Reads the binary file directly and converts it to a FEniCS 
            # function space data class. Selects the flag 
            # 'data_matrix_has_time_point_per_row' as False, since the 
            # first column of the data matrix is composed of actual dis-
            # placement DOFs, not time points.
            # On the other hand, the argument 'time_step' is set to 0 to 
            # capture the first row of the data matrix, which corres-
            # ponds to the best sample of the surrogate model. Thus, 
            # this variable has no meaning of time in the context of the 
            # particular application of this code

            _, _, xdmf_field_file = read_field_from_binary(
            displacement_output_file, self.mesh_file_name, {"Displacem"+
            "ent": {"field type": "vector", "interpolation function": 
            "CG", "polynomial degree": 2}}, 
            data_matrix_has_time_point_per_row=False, time_step=0, 
            return_visualization_file_name=True, save_to_xdmf=True)

            # Makes a visualization copy for the true displacement as 
            # well

            _, _, xdmf_true_field_file = read_field_from_binary(
            true_displacement_output_file, self.mesh_file_name, {"Disp"+
            "lacement": {"field type": "vector", "interpolation functi"+
            "on": "CG", "polynomial degree": 2}}, 
            data_matrix_has_time_point_per_row=False, time_step=0, 
            return_visualization_file_name=True, save_to_xdmf=True)

            # Sets the name of the screenshot file

            screenshot_file = ("surrogate_best_training_sample_of_"+str(
            i+1)+"_best_model.png")

            # Sets the name of the screenshot file for the true displa-
            # cement

            screenshot_true_file = ("true_best_training_sample_of_"+str(
            i+1)+"_best_model.png")

            # Takes a snapshot of a simulation saved in the xdmf file 
            # whose field is called as 'Displacement' inside FEniCS. The 
            # edges of the finite elements will be shown
            
            paraview_tools.frozen_snapshots(xdmf_field_file, "Displace"+
            "ment", time=0.0, representation_type="Surface With Edges", 
            axes_color="black", legend_bar_font="latex", zoom_factor=1.0, 
            component_to_plot="Magnitude", warp_by_vector=True, 
            resolution_ratio=10, background_color="WhiteBackground", 
            display_reference_configuration=False, 
            transparent_background=True, legend_bar_font_color="black", 
            set_camera_interactively=False, 
            #color_bar_min_value=0.1, color_bar_max_value=0.6,
            output_imageFileName=screenshot_file, output_path=
            self.screenshots_path, read_camera_settings_dictionary=True, 
            legend_bar_visibility=True)

            # Makes the same screenshot for the true values of displace-
            # ment
            
            paraview_tools.frozen_snapshots(xdmf_true_field_file, "Dis"+
            "placement", time=0.0, representation_type="Surface With E"+
            "dges", axes_color="black", legend_bar_font="latex", 
            zoom_factor=1.0, component_to_plot="Magnitude", 
            warp_by_vector=True, resolution_ratio=10, background_color=
            "WhiteBackground", display_reference_configuration=False, 
            transparent_background=True, legend_bar_font_color="black", 
            set_camera_interactively=False, 
            #color_bar_min_value=0.1, color_bar_max_value=0.6,
            output_imageFileName=screenshot_true_file, output_path=
            self.screenshots_path, read_camera_settings_dictionary=True, 
            legend_bar_visibility=True)

    # Defines a function to insert the results into a function space and
    # plot it for the samples of the test set

    def plot_test_response(self):

        # Iterates over the best models

        for i in range(n_best_models):

            # Recovers the name of the file that contains the displace-
            # ment DOFs of this model

            displacement_output_file = (self.results_path+"//surrogate"+
            "_best_samples_of_"+str(i+1)+"_best_model.npy")

            # Recovers the name of the file with the true values of dis-
            # placement

            true_displacement_output_file = (self.results_path+"//true"+
            "_best_samples_of_"+str(i+1)+"_best_model.npy")

            # Reads the binary file directly and converts it to a FEniCS 
            # function space data class. Selects the flag 
            # 'data_matrix_has_time_point_per_row' as False, since the 
            # first column of the data matrix is composed of actual dis-
            # placement DOFs, not time points.
            # On the other hand, the argument 'time_step' is set to 0 to 
            # capture the first row of the data matrix, which corres-
            # ponds to the best sample of the surrogate model. Thus, 
            # this variable has no meaning of time in the context of the 
            # particular application of this code

            _, _, xdmf_field_file = read_field_from_binary(
            displacement_output_file, self.mesh_file_name, {"Displacem"+
            "ent": {"field type": "vector", "interpolation function": 
            "CG", "polynomial degree": 2}}, 
            data_matrix_has_time_point_per_row=False, time_step=0, 
            return_visualization_file_name=True, save_to_xdmf=True)

            # Makes a visualization copy for the true displacement as 
            # well

            _, _, xdmf_true_field_file = read_field_from_binary(
            true_displacement_output_file, self.mesh_file_name, {"Disp"+
            "lacement": {"field type": "vector", "interpolation functi"+
            "on": "CG", "polynomial degree": 2}}, 
            data_matrix_has_time_point_per_row=False, time_step=0, 
            return_visualization_file_name=True, save_to_xdmf=True)

            # Sets the name of the screenshot file

            screenshot_file = ("surrogate_best_sample_of_"+str(i+1)+"_"+
            "best_model.png")

            # Sets the name of the screenshot file for the true displa-
            # cement

            screenshot_true_file = ("true_best_sample_of_"+str(i+1)+"_"+
            "best_model.png")

            # Takes a snapshot of a simulation saved in the xdmf file 
            # whose field is called as 'Displacement' inside FEniCS. The 
            # edges of the finite elements will be shown
            
            paraview_tools.frozen_snapshots(xdmf_field_file, "Displace"+
            "ment", time=0.0, representation_type="Surface With Edges", 
            axes_color="black", legend_bar_font="latex", zoom_factor=1.0, 
            component_to_plot="Magnitude", warp_by_vector=True, 
            resolution_ratio=10, background_color="WhiteBackground", 
            display_reference_configuration=False, 
            transparent_background=True, legend_bar_font_color="black", 
            set_camera_interactively=False, 
            #color_bar_min_value=0.1, color_bar_max_value=0.6,
            output_imageFileName=screenshot_file, output_path=
            self.screenshots_path, read_camera_settings_dictionary=True, 
            legend_bar_visibility=True)

            # Makes the same screenshot for the true values of displace-
            # ment
            
            paraview_tools.frozen_snapshots(xdmf_true_field_file, "Dis"+
            "placement", time=0.0, representation_type="Surface With E"+
            "dges", axes_color="black", legend_bar_font="latex", 
            zoom_factor=1.0, component_to_plot="Magnitude", 
            warp_by_vector=True, resolution_ratio=10, background_color=
            "WhiteBackground", display_reference_configuration=False, 
            transparent_background=True, legend_bar_font_color="black", 
            set_camera_interactively=False, 
            #color_bar_min_value=0.1, color_bar_max_value=0.6,
            output_imageFileName=screenshot_true_file, output_path=
            self.screenshots_path, read_camera_settings_dictionary=True, 
            legend_bar_visibility=True)

# Execution block block

if __name__=="__main__":

    results_path = get_parent_path_of_file()+"//results_lagrange"
    
    displacement_data_file = ("00_succesful_displacement_matrix_pc_mat"+
    "heus.npy")

    input_data_file = ("00_successful_complete_data_matrix_pc_matheus."+
    "npy")

    saved_model_file = "saved_model"

    n_training_samples = 10000

    quotient_space_dimension = 9

    # Trains a new model

    n_monte_carlo_realizations = 500

    n_best_models = 30

    n_best_samples = 10

    optimizer = "Adam"

    training_flag = True 

    test_flag = False

    # Gets the mesh of the RVE

    mesh_file_name = get_parent_path_of_file()+"//box_mesh_mecsol"

    # Gets the path to the directory of screenshots

    screenshots_path = results_path+"//screenshots"

    # Instantiates the class of the surrogate model

    surrogate_model_class = SurrogateModel(displacement_data_file, 
    input_data_file, saved_model_file, results_path, n_training_samples, 
    quotient_space_dimension, n_monte_carlo_realizations, n_best_models,
    n_best_samples, mesh_file_name, screenshots_path, optimizer)

    # Sets training forth if it is the case

    if training_flag:

        surrogate_model_class.train_surrogate_model()

    # Sets testing forth if it is the case

    if test_flag:

        surrogate_model_class.test_surrogate_model()

    # Plots the best models

    surrogate_model_class.plot_training_response()

    #surrogate_model_class.plot_test_response()