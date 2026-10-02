# Routine to store methods to test ANN models that learn discrete apro-
# ximations given by the FEM

import tensorflow as tf

import numpy as np

from ...PythonicUtilities.path_tools import join_path_and_verify_existence

# Defines a class to evaluate and compare the response of a FEM surroga-
# te ANN model against the original FEM data

class FEMSurrogateEvaluator:

    def __init__(self, input_data_file_name, output_true_data_file_name, 
    indices_of_training_samples, parent_path=None, subdofs_to_learn=None):

        # Checks whether the file paths exist and returns the file paths
        # updated with the parent path. Additionally, the given files 
        # must be numpy binaries

        self.input_file_name, self.output_true_file_name = join_path_and_verify_existence(
        [input_data_file_name, output_true_data_file_name], parent_path=
        parent_path, path_bits_to_be_excluded=3, required_termination=
        "npy")

        # Reads the input data and the output true data

        self.input_data = np.load(self.input_file_name)

        self.output_true_data = np.load(self.output_true_file_name)

        # Checks if both input and output data have the same number of 
        # rows (samples)

        if self.input_data.shape[0]!=self.output_true_data.shape[0]:

            raise IndexError("The input data given by:\n'"+
            self.input_file_name+"\nhas "+str(self.input_data.shape[0])+
            " rows (samples); whereas the output true data given by:\n"+
            self.output_true_file_name+"\nhas "+str(
            self.output_true_data.shape[0])+" rows. They must have the"+
            " same number of samples")

        # If the indices of training samples is an integer, the range is
        # from the first sample to the given number of training samples,
        # as per convention

        if isinstance(indices_of_training_samples, int):

            # Converts to a numpy range

            indices_of_training_samples = np.arange(0, 
            indices_of_training_samples)

        elif not isinstance(indices_of_training_samples, np.ndarray):

            raise TypeError("'indices_of_training_samples' in 'FEMSurr"+
            "ogateEvaluator' must be an integer or a numpy array. If i"+
            "t is an integer, the training samples will be considered "+
            "the rows of the input data from the first to the row of i"+
            "dex given by the integer-1.\nCurrently, 'indices_of_train"+
            "ing_samples' is:\n"+str(indices_of_training_samples))

        elif len(indices_of_training_samples.shape)!=1:

            raise IndexError("'indices_of_training_samples' in 'FEMSur"+
            "rogateEvaluator' is an array of shape "+str(
            indices_of_training_samples.shape)+". The shape must be (n"+
            "_samples) instead")

        # Checks if there is an index larger than the number of rows of
        # the input data

        elif indices_of_training_samples.max()>=self.input_data.shape[0]:

            raise IndexError("The maximum index in 'indices_of_trainin"+
            "g_samples' in 'FEMSurrogateEvaluator' is "+str(
            indices_of_training_samples.max())+" which is larger or eq"+
            "ual to the number of rows in the input data and in the ou"+
            "tput data, which is "+str(self.input_data.shape[0]))

        # Saves the indices of the training samples into the class

        self.indices_of_training_samples = indices_of_training_samples

        # If subdofs to be learned are None, all DOFs must be learned by
        # the surrogate model

        if subdofs_to_learn is None:

            # Gets a range to the number of columns of the output data,
            # since each column corresponds to a DOF of the FEM data

            subdofs_to_learn = np.arange(0, self.output_true_data.shape[
            1])

        # Checks if subdofs to be learned by the surrogate model is a 
        # numpy array

        elif not isinstance(subdofs_to_learn, np.ndarray):

            raise TypeError("'subdofs_to_learn' in 'FEMSurrogateEvalua"+
            "tor' must be a numpy array with the indices of the DOFs t"+
            "o be learned by the surrogate model. Currently, it is\n"+
            str(subdofs_to_learn)+"\nwhose type is "+str(type(
            subdofs_to_learn)))

        # Checks if subdofs to be learned is a flat array

        elif len(subdofs_to_learn.shape)!=1:
        
            raise IndexError("'subdofs_to_learn' in 'FEMSurrogateEvalu"+
            "ator' is an array of shape "+str(subdofs_to_learn.shape)+
            ". The shape must be (n_dofs) instead")

        # Checks if subdofs is limited to the number of DOFs available
        # in the output data

        elif subdofs_to_learn.max()>=self.output_true_data.shape[1]:
        
            raise IndexError("The maximum index in 'subdofs_to_learn' "+
            "in 'FEMSurrogateEvaluator' is "+str(subdofs_to_learn.max()
            )+" which is larger or equal to the number of columns in t"+
            "he output true data, which is "+str(
            self.output_true_data.shape[1]))

        # Saves the subdofs to be learned in the class

        self.subdofs_to_learn = subdofs_to_learn