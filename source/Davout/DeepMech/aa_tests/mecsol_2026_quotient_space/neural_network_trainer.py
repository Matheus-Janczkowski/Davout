# Routine to train a neural network model as surrogate for the RVE kine-
# matics

import numpy as np 

import tensorflow as tf

from time import time

from .....Davout.DeepMech.tool_box import ANN_tools, training_tools

from .....Davout.DeepMech.tool_box.loss_assembler_classes import MaximumAbsoluteError

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a function to train the neural network model

def train_surrogate_model(displacement_data_file, input_data_file, 
saved_model_file, results_path, n_training_samples, 
quotient_space_dimension):

    # Reads the two files

    output_data = np.load(results_path+"//"+displacement_data_file)

    input_data = np.load(results_path+"//"+input_data_file)

    # Reshufles data to put the displacement gradient as the first
    # columns. This is a requirement of the implementation of the 
    # GatedQuotientSpace architecture

    input_data = np.hstack((input_data[:,(input_data.shape[1]-
    quotient_space_dimension):], input_data[:,:(input_data.shape[1]-
    quotient_space_dimension)]))

    # Sets the training data

    training_data = input_data[0:n_training_samples,:]

    training_true_values = output_data[0:n_training_samples,:]

    # Gets the number of input and output neurons

    n_input_neurons = input_data.shape[1]

    n_output_neurons = output_data.shape[1]

    # Sets a list of layers and the activation functions

    activations_list = [{"elu": {"number of neurons": 100}}, {"elu": 
    {"number of neurons": 1000}}, {"linear": n_output_neurons}]

    # Creates the class of neural network information

    ANN_class = ANN_tools.MultiLayerModel(n_input_neurons, 
    activations_list, enforce_customLayers=True, verbose=True, 
    parameters_dtype="float32", custom_architecture={"name": "GatedQuo"+
    "tientSpace", "quotient space dimension": quotient_space_dimension})

    custom_model = ANN_class()

    # Sets the number of training iterations and the number of iterations
    # to plot results in the terminal

    maximum_iterations = 1000
    
    verbose_delta_iterations = 50

    # Sets the optimization class for training

    training_class = training_tools.ModelCustomTraining(custom_model,
    training_data, training_true_values, 
    tf.keras.losses.MeanAbsoluteError(), verbose=True, n_iterations=
    maximum_iterations, verbose_deltaIterations=
    verbose_delta_iterations, save_model_file=saved_model_file, 
    match_data_float_type_to_trainables=True, parent_path=results_path)

    t_initial = time()

    training_class()

    elapsed_time = time()-t_initial

    print("\nTrains at "+str(elapsed_time)+" seconds")

# Defines a function to test the model

def test_surrogate_model(displacement_data_file, input_data_file, 
saved_model_file, results_path, n_training_samples, 
quotient_space_dimension):

    # Reads the two files

    output_data = np.load(results_path+"//"+displacement_data_file)

    input_data = np.load(results_path+"//"+input_data_file)

    # Reshufles data to put the displacement gradient as the first
    # columns. This is a requirement of the implementation of the 
    # GatedQuotientSpace architecture

    input_data = np.hstack((input_data[:,(input_data.shape[1]-
    quotient_space_dimension):], input_data[:,:(input_data.shape[1]-
    quotient_space_dimension)]))

    # Sets the test data

    test_data = input_data[n_training_samples:,:]

    test_true_values = output_data[n_training_samples:,:]

    # Loads it back

    loaded_model = tf.keras.models.load_model(results_path+"//"+
    saved_model_file)

    # Defines the loss function metric

    loss_metric = tf.keras.losses.MeanAbsoluteError()

    # Gets the output of the loaded model

    output_model = loaded_model(test_data)

    # Gets the loss of the test data

    test_loss = loss_metric(test_true_values, output_model)

    print("\nLoss function on test set:", format(test_loss.numpy(),'.5'+
    'e')+"\n")

    # Verifies with the maximum absolute error

    maximum_absolute_error = MaximumAbsoluteError()

    maximum_absolute_value = maximum_absolute_error(test_true_values, 
    output_model)

    print("\nMaximum absolute error on test set:"+str(format(
    maximum_absolute_value.numpy(),'.5e'))+"\nwhereas the minimum abso"+
    "lute error is "+str(format(
    maximum_absolute_error.minimum_absolute_error(test_true_values, 
    output_model).numpy(), '.5e'))+"\n")

# Testing block

if __name__=="__main__":

    results_path = get_parent_path_of_file()+"//results"
    
    displacement_data_file = "00_succesful_displacement_matrix.npy"

    input_data_file = "00_successful_complete_data_matrix.npy"

    saved_model_file = "saved_model.keras"

    n_training_samples = 10000

    quotient_space_dimension = 9

    # Trains a new model

    training_flag = False 

    if training_flag:

        train_surrogate_model(displacement_data_file, input_data_file, 
        saved_model_file, results_path, n_training_samples,
        quotient_space_dimension)

    test_surrogate_model(displacement_data_file, input_data_file,
    saved_model_file, results_path, n_training_samples,
    quotient_space_dimension)