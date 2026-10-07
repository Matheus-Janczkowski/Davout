# Routine to store methods to test ANN models that learn discrete apro-
# ximations given by the FEM

import tensorflow as tf

import numpy as np

from ...PythonicUtilities.path_tools import join_path_and_verify_existence, take_outFileNameTermination

from ...PythonicUtilities.error_and_type_tools import verify_type

from ...DeepMech.tool_box.training_tools import verify_loss_metric

from ...DeepMech.tool_box.loss_assembler_classes import MaximumAbsoluteError

from ...MultiMech.tool_box.binary_tools import read_field_from_binary

from ...GraphUtilities.paraview_tools import frozen_snapshots

# Defines a class to evaluate and compare the response of a FEM surroga-
# te ANN model against the original FEM data

class FEMSurrogateEvaluator:

    def __init__(self, input_data_file_name, output_true_data_file_name, 
    indices_of_training_samples, saved_model_file_name, field_name,
    maximum_number_of_models_to_be_evaluated=None, parent_path=None, 
    subdofs_to_learn=None, loss_metric="MeanAbsoluteError", verbose=
    False, number_of_best_samples=None):

        self.verbose = verbose

        self.field_name = field_name

        # Checks whether the file paths exist and returns the file paths
        # updated with the parent path. Additionally, the given files 
        # must be numpy binaries

        updated_files_list, parent_path = join_path_and_verify_existence(
        [input_data_file_name, output_true_data_file_name], parent_path=
        parent_path, path_bits_to_be_excluded=3, required_termination=
        "npy", return_parent_path=True)

        # Unpacks the updated files list into their new attributes owned
        # by this class

        self.input_file_name, self.output_true_file_name = (
        updated_files_list)

        # Saves the parent path into the class

        self.parent_path = parent_path

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

        # Verifies and saves the indices of the training samples into 
        # the class

        self.indices_of_training_samples = self.check_sample_indices(
        indices_of_training_samples, indices_name="indices_of_training"+
        "_samples")

        # Verifies and saves the subdofs to be learned in the class

        self.subdofs_to_learn = self.check_subdofs_to_learn(
        subdofs_to_learn)

        # Gets the loss metric and verifies if it is a valid option

        self.loss_metric = verify_loss_metric(loss_metric)

        # Instantiates the class to compute the average maximum absolute
        # error over a set of samples

        self.maximum_absolute_error_class = MaximumAbsoluteError()

        # Verifies if the maximum number of models to be evaluated was
        # given

        if maximum_number_of_models_to_be_evaluated is not None:

            # Verifies if this number is an integer

            if not isinstance(maximum_number_of_models_to_be_evaluated,
            int):

                raise TypeError("'maximum_number_of_models_to_be_evalu"+
                "ated' in 'FEMSurrogateEvaluator' is not None, instead"+
                " it is: "+str(maximum_number_of_models_to_be_evaluated
                )+". It must be an integer")

            # Checks if the maximum model number is available with the
            # template of model file

            last_model_file = join_path_and_verify_existence(
            str(maximum_number_of_models_to_be_evaluated)+"_"+
            saved_model_file_name, parent_path=parent_path, 
            path_bits_to_be_excluded=3, required_termination="keras")

        # If the maximum number of models to be evaluated is None, it
        # indicates that a single model must be evaluated

        else:

            # Checks if this single model file trully exists

            join_path_and_verify_existence(saved_model_file_name, 
            parent_path=parent_path, path_bits_to_be_excluded=3, 
            required_termination="keras")

        # Saves the maximum number of models to be evaluated and the
        # template for the model file name

        self.maximum_number_of_models_to_be_evaluated = (
        maximum_number_of_models_to_be_evaluated)

        self.saved_model_file_name = saved_model_file_name

        # Verifies if the numebr of best samples to which each model 
        # will be assessed is an integer or if it was no provided

        if number_of_best_samples is not None:

            if not isinstance(number_of_best_samples, int):

                raise TypeError("'number_of_best_samples' in 'FEMSurro"+
                "gateEvaluator' must be an integer. Currently, it is "+
                str(number_of_best_samples)+", and its type is "+str(
                type(number_of_best_samples)))

        # If it is indeed None, makes it 1

        else:

            number_of_best_samples = 1

        self.number_of_best_samples = number_of_best_samples

    ####################################################################
    #                   Evaluation of common datasets                  #
    ####################################################################

    # Defines a function to get the whole dataset, take the samples used
    # for training, and, then, evaluate the model there

    def evaluate_model_on_training_set(self):

        # Gets the training data set

        training_input_set = self.input_data[
        self.indices_of_training_samples,:]

        training_true_output_set = self.output_true_data[
        self.indices_of_training_samples,:]

        # Checks whether a single model was trained 

        if self.maximum_number_of_models_to_be_evaluated is None:

            # Calls the function that evaluates the performance of that
            # model

            self.evaluate_single_model(self.saved_model_file_name, 
            training_input_set, training_true_output_set, "training", 
            self.field_name)

        # Otherwise, iterates over the different models that were trained
        # in a Monte Carlo-like procedure

        else:

            for i in range(self.maximum_number_of_models_to_be_evaluated):

                # Gets the name of the model

                model_file_name = str(i+1)+"_"+self.saved_model_file_name

                # Calls the method to evaluate the performance of this
                # model

                self.evaluate_single_model(model_file_name, 
                training_input_set, training_true_output_set, "trainin"+
                "g", self.field_name)

    ####################################################################
    #                        Evaluation methods                        #
    ####################################################################

    # Defines a function to test a single model. This function yields 
    # the performance of the model across different samples of the data-
    # set and orders the samples from the best result to the poorest.
    # The tensors model_input_data and true_output_data must have the
    # same number of samples and they must have the number of samples 
    # expected for the intended evaluation. Thus, if the intended evalu-
    # ation in a subset of the dataset, the incoming data must be sam-
    # pled before-hand

    def evaluate_single_model(self, model_file_name, model_input_data, 
    true_output_data, dataset_name, field_name):

        # Removes the termination of the model file name just in case

        model_file_name_without_termination = take_outFileNameTermination(
        model_file_name)

        # Checks if the number of samples is the same in the input and 
        # output data

        number_of_samples = tf.shape(true_output_data)[0]

        if tf.shape(model_input_data)[0]!=number_of_samples:

            raise IndexError("In 'evaluate_single_model' method at 'FE"+
            "MSurrogateEvaluator', 'model_input_data' has shape "+str(
            tf.shape(model_input_data))+" whereas 'true_output_data' h"+
            "as "+str(number_of_samples)+" samples. They must have the"+
            " same number of samples, i.e., the same shape at the firs"+
            "t axis")

        # Gets the values of the true DOFs that are learned by the model

        true_output_of_subdofs = true_output_data[:, 
        self.subdofs_to_learn]

        # Loads this model

        loaded_model = tf.keras.models.load_model(self.parent_path+
        "//"+model_file_name_without_termination+".keras")

        # Gets the output of the loaded model

        model_output = loaded_model(model_input_data)

        # Gets the loss of the model

        dataset_loss = self.loss_metric(true_output_of_subdofs, 
        model_output)

        # Verifies the maximum absolute error

        maximum_absolute_error = self.maximum_absolute_error_class(
        true_output_of_subdofs, model_output)

        if self.verbose:

            print("\nLoss function on "+str(dataset_name)+":", format(
            dataset_loss.numpy(), '.5e')+"\n\nMaximum absolute error o"+
            "n "+str(dataset_name)+": "+str(format(
            maximum_absolute_error.numpy(),'.5e'))+"\nwhereas the mini"+
            "mum absolute error is "+str(format(
            maximum_absolute_error.minimum_absolute_error(
            true_output_of_subdofs, model_output).numpy(), '.5e')))

        # Gets the mean absolute error for each row

        loss_per_sample = self.get_mean_absolute_error_per_sample(
        true_output_of_subdofs, model_output)

        # Gets the samples indices with the lowest mean absolute error.
        # The average is evaluated across the dimensions of each output
        # sample

        best_samples_indices = np.argsort(loss_per_sample)[
        :self.number_of_best_samples]

        if self.verbose:

            loss_per_sample_string = ""

            for loss_value in loss_per_sample:

                loss_per_sample += "\n"+str(loss_value)

            print("\nThe "+str(self.number_of_best_samples)+" best sam"+
            "ples have the following mean absolute error:\n"+
            loss_per_sample_string)

        # Recovers the name of the file that contains the DOFs of the 
        # field given by this model

        field_output_file = (self.parent_path+"//surrogate_"+str(
        field_name)+"_best_samples_on_"+str(dataset_name)+"_of_"+
        model_file_name_without_termination+".npy")

        # Recovers the name of the file with the true values of the DOFs
        # of this field

        true_field_output_file = (self.parent_path+"//true_"+str(
        field_name)+"_best_samples_on_"+str(dataset_name)+"_of_"+
        model_file_name_without_termination+".npy")

        # Gets the field of the model into a null tensor. In other words,
        # only the predicted DOFs are updated

        model_field = tf.zeros_like(true_output_data)

        number_of_updated_dofs = tf.shape(self.subdofs_to_learn)[0]

        # Creates grid of row indices [0, 1, ... num_samples-1] ex- 
        # panded to match subdofs

        row_indices = tf.repeat(tf.range(number_of_samples)[:, None], 
        repeats=number_of_updated_dofs, axis=1)

        column_indices = tf.tile(tf.constant(self.subdofs_to_learn)[
        None,:], multiples=[number_of_samples,1])

        # Stacks into shape (number_of_samples*number_of_updated_dofs, 
        # 2)

        indices = tf.cast(tf.reshape(tf.stack([row_indices, 
        column_indices], axis=-1), [-1, 2]), dtype=tf.int32)

        # Flattens model_output to match indices

        updates = tf.reshape(model_output, [-1])

        # Performs scatter update to get the initially null tensor into
        # a tensor with the model output for each sample

        model_field = tf.tensor_scatter_nd_update(model_field, indices, 
        updates)

        # Saves as binary files the true data and the output data of 
        # the best preserving samples

        np.save(true_field_output_file, true_output_data[
        best_samples_indices,:])

        np.save(field_output_file, model_field.numpy()[
        best_samples_indices,:])

    # Defines a function to create snapshots of the visualization of the
    # results of the models in paraview

    def get_snapshots_for_single_model(self, field_output_file, 
    true_field_output_file, field_name, field_type, ):

        # Checks if field name is a string

        verify_type(field_name, "field_name", str, "'get_snapshots_for"+
        "_single_model' at 'FEMSurrogateEvaluator'")

        # Reads the binary file directly and converts it to a FEniCS
        # function space data class. Selects the flag 'data_matrix_has_
        # time_point_per_row' as False, since the first column of the 
        # data matrix is composed of actual displacement DOFs, not time 
        # points.
        # On the other hand, the argument 'time_step' is set to 0 to
        # capture the first row of the data matrix, which corresponds to 
        # the best sample of the surrogate model. Thus, this variable 
        # has no meaning of time in the context of the particular appli-
        # cation of this code

        _, _, xdmf_field_file = read_field_from_binary(
        field_output_file, self.mesh_file_name, {field_name: {"field type": "vector", "interpolation function": 
        "CG", "polynomial degree": 2}}, 
        data_matrix_has_time_point_per_row=False, time_step=0, 
        return_visualization_file_name=True, save_to_xdmf=True)

        # Makes a visualization copy for the true displacement as 
        # well

        _, _, xdmf_true_field_file = read_field_from_binary(
        true_field_output_file, self.mesh_file_name, {"Disp"+
        "lacement": {"field type": "vector", "interpolation functi"+
        "on": "CG", "polynomial degree": 2}}, 
        data_matrix_has_time_point_per_row=False, time_step=0, 
        return_visualization_file_name=True, save_to_xdmf=True)

        # Sets the name of the screenshot file

        screenshot_file = ("surrogate_best_training_sample_of_"+str(
        i+1)+"_"+self.saved_model_file+".png")

        # Sets the name of the screenshot file for the true displa-
        # cement

        screenshot_true_file = ("true_best_training_sample_of_"+str(
        i+1)+"_best_model.png")

        # Takes a snapshot of a simulation saved in the xdmf file 
        # whose field is called as 'Displacement' inside FEniCS. The 
        # edges of the finite elements will be shown
        
        frozen_snapshots(xdmf_field_file, "Displace"+
        "ment", time=0.0, representation_type="Surface With Edges", 
        axes_color="black", legend_bar_font="latex", zoom_factor=1.0, 
        component_to_plot=self.displacement_component_to_plot,
        warp_by_vector=False, resolution_ratio=10, background_color=
        "WhiteBackground", display_reference_configuration=True, 
        transparent_background=True, legend_bar_font_color="black", 
        set_camera_interactively=False, 
        #color_bar_min_value=0.1, color_bar_max_value=0.6,
        output_imageFileName=screenshot_file, output_path=
        self.screenshots_path, read_camera_settings_dictionary=True, 
        legend_bar_visibility=True)

        # Makes the same screenshot for the true values of displace-
        # ment
        
        frozen_snapshots(xdmf_true_field_file, "Dis"+
        "placement", time=0.0, representation_type="Surface With E"+
        "dges", axes_color="black", legend_bar_font="latex", 
        zoom_factor=1.0, component_to_plot=
        self.displacement_component_to_plot, warp_by_vector=False, 
        resolution_ratio=10, background_color="WhiteBackground", 
        display_reference_configuration=True, 
        transparent_background=True, legend_bar_font_color="black", 
        set_camera_interactively=False, 
        #color_bar_min_value=0.1, color_bar_max_value=0.6,
        output_imageFileName=screenshot_true_file, output_path=
        self.screenshots_path, read_camera_settings_dictionary=True, 
        legend_bar_visibility=True)

        print("\nThe input data for the "+str(i+1)+"-th model is: "+
        str(training_data[best_samples_indices[0],:])+"\n")

    # Defines a function to compute the mean absolute error per sample

    def get_mean_absolute_error_per_sample(self, true_output, 
    model_output):

        # Reduces the mean alongside the axis of columns, since the rows
        # are the samples

        return tf.reduce_mean(tf.abs(true_output-model_output), axis=1
        ).numpy()

    ####################################################################
    #                      Verification utilities                      #
    ####################################################################

    # Defines a function to verify if the indices of samples to be eva-
    # luated are consistent

    def check_sample_indices(self, samples_indices, indices_name="indi"+
    "ces_of_training_samples"):

        # If the indices of training samples is an integer, the range is
        # from the first sample to the given number of training samples,
        # as per convention

        if isinstance(samples_indices, int):

            # Converts to a numpy range

            samples_indices = np.arange(0, samples_indices)

        elif not isinstance(samples_indices, np.ndarray):

            raise TypeError("'"+str(indices_name)+"' in 'FEMSurrogateE"+
            "valuator' must be an integer or a numpy array. If it is a"+
            "n integer, the training samples will be considered the ro"+
            "ws of the input data from the first to the row of idex gi"+
            "ven by the integer-1.\nCurrently, '"+str(indices_name)+"'"+
            " is:\n"+str(samples_indices))

        elif len(samples_indices.shape)!=1:

            raise IndexError("'"+str(indices_name)+"' in 'FEMSurrogate"+
            "Evaluator' is an array of shape "+str(samples_indices.shape
            )+". The shape must be (n_samples) instead")

        # Checks if there is an index larger than the number of rows of
        # the input data

        elif samples_indices.max()>=self.input_data.shape[0]:

            raise IndexError("The maximum index in '"+str(indices_name)+
            "' in 'FEMSurrogateEvaluator' is "+str(samples_indices.max()
            )+" which is larger or equal to the number of rows in the "+
            "input data and in the output data, which is "+str(
            self.input_data.shape[0]))

        return samples_indices

    # Defines a function to verify if the given subdofs to learn are
    # consistent

    def check_subdofs_to_learn(self, subdofs_to_learn):

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

        return subdofs_to_learn