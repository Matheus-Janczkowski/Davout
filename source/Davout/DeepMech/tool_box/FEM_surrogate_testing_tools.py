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
    indices_of_dataset_samples, saved_model_file_name, field_name,
    maximum_number_of_models_to_be_evaluated=None, parent_path=None, 
    subdofs_to_learn=None, loss_metric="MeanAbsoluteError", verbose=
    False, number_of_best_samples=None, take_snapshots=False, 
    number_of_samples_to_be_snapshot=1, mesh_file_name=None, 
    field_component_to_plot=None, screenshots_path=None, 
    representation_type="Surface With Edges", warp_by_vector=False,
    set_camera_interactively=False, color_bar_min_value=None, 
    color_bar_max_value=None, field_type=None, polynomial_degree=None,
    interpolation_function=None):

        self.verbose = verbose

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

        # Verifies and saves the indices of the dataset samples into 
        # the class

        self.indices_of_dataset_samples = self.check_sample_indices(
        indices_of_dataset_samples, indices_name="indices_of_dataset"+
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

            verify_type(maximum_number_of_models_to_be_evaluated, "max"+
            "imum_number_of_models_to_be_evaluated", int, "'FEMSurroga"+
            "teEvaluator' is not None, thus it")

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

        self.number_of_best_samples = verify_type(number_of_best_samples, 
        "number_of_best_samples", int, "'FEMSurrogateEvaluator'", 
        default_in_case_of_none=1)

        # If a mesh file has been given

        if mesh_file_name is not None:

            mesh_file_name = join_path_and_verify_existence(
            mesh_file_name, parent_path=parent_path, 
            path_bits_to_be_excluded=3, required_termination="msh")

        self.mesh_file_name = mesh_file_name

        # Saves the flag to take snapshots or not and the number of sam-
        # ples to be shot

        self.take_snapshots = take_snapshots

        self.number_of_samples_to_be_snapshot = verify_type(
        number_of_samples_to_be_snapshot, "number_of_samples_to_be_sna"+
        "pshot", int, "'FEMSurrogateEvaluator'")

        # Saves functional data information. But, if snapshots are re-
        # quired, the functional data must be given

        if self.take_snapshots:

            field_type = verify_type(field_type, "field_type", str, "'"+
            "FEMSurrogateEvaluator'", description="a string with the t"+
            "ype of the field, such as 'scalar', 'vector', and 'tensor"+
            "'")

            polynomial_degree = verify_type(polynomial_degree, "polyno"+
            "mial_degree", int, "'FEMSurrogateEvaluator'", description=
            "an integer with the order of the polynomial that interpol"+
            "ates the field in a finite-element space. Such as 1 and 2")

            interpolation_function = verify_type(interpolation_function, 
            "interpolation_function", str, "'FEMSurrogateEvaluator'", 
            description="a string with the type of the interpolation f"+
            "unction, such as 'CG' and 'DG'")

            mesh_file_name = verify_type(mesh_file_name, "mesh_file_na"+
            "me", str, "'FEMSurrogateEvaluator'", description="a strin"+
            "g with the path to the mesh where the field will be inter"+
            "polated upon")

        self.field_name = field_name

        self.field_type = field_type
        
        self.polynomial_degree = polynomial_degree

        self.interpolation_function = interpolation_function

        # Verifies if there are a larger number of best samples than the
        # number of samples in the list of indices of samples of the da-
        # taset. Then, checks if the number of best samples to be shot
        # in ParaView is larger than the number of best samples that 
        # were actually recorded

        self.check_numbers_of_best_samples(
        self.indices_of_dataset_samples)

        # Checks if the number of samples to be shot is less than the
        # number of best samples

        # Stores the component to plot in the screenshot 

        self.field_component_to_plot = field_component_to_plot

        # If no path to save the screenshots was given, reuses the pa-
        # rent path

        if screenshots_path is None:

            screenshots_path = str(self.parent_path)

        self.screenshots_path = screenshots_path

        # Checks if the given representation type is a string and stores
        # it

        self.representation_type = verify_type(representation_type, "r"+
        "epresentation_type", str, "'FEMSurrogateEvaluator'")

        # Checks if the warp by vector flag is boolean and stores it

        self.warp_by_vector = verify_type(warp_by_vector, "warp_by_vec"+
        "tor", bool, "'FEMSurrogateEvaluator'")

        # Checks if the flag for setting the snapshot camera iteratively
        # has been set 

        self.set_camera_interactively = verify_type(
        set_camera_interactively, "set_camera_interactively", bool, "'"+
        "FEMSurrogateEvaluator'")

        # Checks if the color bounds of the legend of the snapshot were
        # given (not None) and if they are float values

        self.color_bar_min_value = verify_type(color_bar_min_value, "c"+
        "olor_bar_min_value", float, "'FEMSurrogateEvaluator", 
        ignore_none_value=True)

        self.color_bar_max_value = verify_type(color_bar_max_value, "c"+
        "olor_bar_max_value", float, "'FEMSurrogateEvaluator",
        ignore_none_value=True)

    ####################################################################
    #                   Evaluation of common datasets                  #
    ####################################################################

    # Defines a function to get the whole dataset, take the samples used
    # for this dataset, and, then, evaluate the model there

    def evaluate_model_on_dataset(self, indices_of_dataset_samples=None, 
    dataset_name="training"):

        # Initializes the dataset values

        dataset_input_matrix = None 

        dataset_true_output_matrix = None

        # Checks if indices of dataset samples were provided

        if indices_of_dataset_samples is not None:

            # Checks the indices of samples against the available whole
            # dataset

            checked_indices_of_dataset_samples = self.check_sample_indices(
            indices_of_dataset_samples, indices_name="indices_of_datas"+
            "et_samples")

            # Checks if they are compatible with the numbers of best 
            # samples asked during the instantiation of the class

            self.check_numbers_of_best_samples(
            checked_indices_of_dataset_samples)

            dataset_input_matrix = self.input_data[
            checked_indices_of_dataset_samples,:]

            dataset_true_output_matrix = self.output_true_data[
            checked_indices_of_dataset_samples,:]

        # Otherwise, uses the indices that were stored and tested during
        # instantiation of this class

        else:

            dataset_input_matrix = self.input_data[
            self.indices_of_dataset_samples,:]

            dataset_true_output_matrix = self.output_true_data[
            self.indices_of_dataset_samples,:]

        # Checks whether a single model was trained 

        if self.maximum_number_of_models_to_be_evaluated is None:

            # Calls the function that evaluates the performance of that
            # model

            self.evaluate_single_model(self.saved_model_file_name, 
            dataset_input_matrix, dataset_true_output_matrix, 
            dataset_name, self.field_name)

        # Otherwise, iterates over the different models that were trained
        # in a Monte Carlo-like procedure

        else:

            for i in range(self.maximum_number_of_models_to_be_evaluated):

                # Gets the name of the model

                model_file_name = str(i+1)+"_"+self.saved_model_file_name

                # Calls the method to evaluate the performance of this
                # model

                self.evaluate_single_model(model_file_name, 
                dataset_input_matrix, dataset_true_output_matrix, 
                dataset_name, self.field_name)

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

        if self.verbose:

            print("\nEvaluates the model saved at '"+self.parent_path+
            "//"+model_file_name_without_termination+".keras'")

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

        # Checks the shape consistency between model's and true outputs

        if true_output_of_subdofs.shape[0]!=tf.shape(model_output
        ).numpy()[0]:

            raise IndexError("The shape of the true output given the s"+
            "ubdofs to learn is "+str(true_output_of_subdofs.shape)+
            "\nwhereas the shape of the model's output is "+str(
            tf.shape(model_output).numpy())+"\n\nThey must have the sa"+
            "me shape. Check the number samples in the true output: "+
            str(true_output_of_subdofs.shape[0])+"\nand the number of "+
            "samples in the model's output: "+str(tf.shape(model_output
            )[0]))

        if true_output_of_subdofs.shape[1]!=tf.shape(model_output
        ).numpy()[1]:

            raise IndexError("The shape of the true output given the s"+
            "ubdofs to learn is "+str(true_output_of_subdofs.shape)+
            "\nwhereas the shape of the model's output is "+str(
            tf.shape(model_output).numpy())+"\n\nThey must have the sa"+
            "me shape. Check the number of DOFs in the true output: "+
            str(true_output_of_subdofs.shape[1])+"\nand the number of "+
            "DOFs in the model's output: "+str(tf.shape(model_output
            ).numpy()[1])+"\nThe provided 'subdofs_to_learn' is:\n"+str(
            self.subdofs_to_learn))

        # Gets the loss of the model

        dataset_loss = self.loss_metric(true_output_of_subdofs, 
        model_output)

        # Verifies the maximum absolute error

        maximum_absolute_value = self.maximum_absolute_error_class(
        true_output_of_subdofs, model_output)

        if self.verbose:

            print("\nLoss function on "+str(dataset_name)+":", format(
            dataset_loss.numpy(), '.5e')+"\n\nMaximum absolute error o"+
            "n "+str(dataset_name)+": "+str(format(
            maximum_absolute_value.numpy(),'.5e'))+"\nwhereas the mini"+
            "mum absolute error is "+str(format(
            self.maximum_absolute_error_class.minimum_absolute_error(
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

            for loss_value in loss_per_sample[best_samples_indices]:

                loss_per_sample_string += "\n"+str(loss_value)

            print("\nThe "+str(self.number_of_best_samples)+" best sam"+
            "ples have the following mean absolute error:"+
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

        updates = tf.cast(tf.reshape(model_output, [-1]), dtype=
        model_field.dtype)

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

        # Verifies if snapshots are to be taken

        if self.take_snapshots:

            # Iterates over the number of samples that will be snapshot

            for sample_index in range(
            self.number_of_samples_to_be_snapshot):

                # Calls the function that creates the snapshot for this
                # sample

                self.get_single_snapshot_for_single_model(
                field_output_file, true_field_output_file, sample_index)

    ####################################################################
    #             Snapshooting of the solution in ParaView             #
    ####################################################################

    # Defines a function to create snapshots of the visualization of the
    # results of the models in paraview

    def get_single_snapshot_for_single_model(self, field_output_file, 
    true_field_output_file, sample_index):

        # Removes the termination of the file names just in case

        field_output_file = take_outFileNameTermination(
        field_output_file)

        true_field_output_file = take_outFileNameTermination(
        true_field_output_file)

        # Checks if all functional data are strings

        verify_type(self.field_name, "'field_name'", str, "'get_snapsh"+
        "ots_for_single_model' at 'FEMSurrogateEvaluator'", description=
        "name of the field that must be provided to build a class of F"+
        "EniCS functional data")

        verify_type(self.field_type, "'field_type'", str, "'get_snapsh"+
        "ots_for_single_model' at 'FEMSurrogateEvaluator'", description=
        "type of the field that must be provided to build a class of F"+
        "EniCS functional data. Examples: 'scalar', 'vector', 'tensor'")

        verify_type(self.interpolation_function, "'interpolation_funct"+
        "ion'", str, "'get_snapshots_for_single_model' at 'FEMSurrogat"+
        "eEvaluator'", description="interpolation function of the fiel"+
        "d that must be provided to build a class of FEniCS functional"+
        " data. Examples: 'CG', 'DG'")

        verify_type(self.polynomial_degree, "'polynomial_degree'", int, 
        "'get_snapshots_for_single_model' at 'FEMSurrogateEvaluator'",
        description="polynomial degree of the interpolation function o"+
        "f the field that must be provided to build a class of FEniCS "+
        "functional data. Example: 1, 2")

        # Checks whether the mesh file name is None

        if self.mesh_file_name is None:

            raise ValueError("'mesh_file_name' in 'get_single_snapshot"+
            "_for_single_model' at 'FEMSurrogateEvaluator' is None. A "+
            "valid .msh mesh must be provided to make snapshots in Par"+
            "aView")

        # Reads the binary file directly and converts it to a FEniCS
        # function space data class. Selects the flag 'data_matrix_has_
        # time_point_per_row' as False, since the first column of the 
        # data matrix is composed of actual displacement DOFs, not time 
        # points.
        # On the other hand, the argument 'time_step' is set to the in-
        # dex of the sample to capture the corresponding row of the data 
        # matrix, which corresponds to the i-th best sample of the sur-
        # rogate model. Thus, this variable has no meaning of time in 
        # the context of the particular application of this code

        _, _, xdmf_field_file = read_field_from_binary(
        field_output_file+".npy", self.mesh_file_name, {self.field_name: 
        {"field type": self.field_type, "interpolation function": 
        self.interpolation_function, "polynomial degree": 
        self.polynomial_degree}}, data_matrix_has_time_point_per_row=
        False, time_step=sample_index, return_visualization_file_name=
        True, save_to_xdmf=True)

        # Makes a visualization copy for the true field as well

        _, _, xdmf_true_field_file = read_field_from_binary(
        true_field_output_file+".npy", self.mesh_file_name, {
        self.field_name: {"field type": self.field_type, "interpolatio"+
        "n function": self.interpolation_function, "polynomial degree": 
        self.polynomial_degree}}, data_matrix_has_time_point_per_row=
        False, time_step=sample_index, return_visualization_file_name=
        True, save_to_xdmf=True)

        # Sets the name of the screenshot file with the sample index in
        # it

        screenshot_file = ("screenshot_of_"+field_output_file+"_sample_"
        +str(sample_index+1)+".png")

        # Sets the name of the screenshot file for the true field DOFs

        screenshot_true_file = ("screenshot_of_"+true_field_output_file+
        "_sample_"+str(sample_index+1)+".png")

        # If the warp by vector functionality is unavailable, makes the
        # snapshot display the reference configuration. Otherwise, no-
        # thing will be shown

        display_reference_configuration = False

        if not self.warp_by_vector:

            display_reference_configuration = True

        # Takes a snapshot of a simulation saved in the xdmf file 
        # whose field is called by the given field name inside FEniCS. 
        # The edges of the finite elements will be shown
        
        frozen_snapshots(xdmf_field_file, self.field_name, time=0.0, 
        representation_type=self.representation_type, axes_color="blac"+
        "k", legend_bar_font="latex", zoom_factor=1.0, 
        component_to_plot=self.field_component_to_plot, warp_by_vector=
        self.warp_by_vector, resolution_ratio=10, background_color=
        "WhiteBackground", display_reference_configuration=
        display_reference_configuration, transparent_background=True, 
        legend_bar_font_color="black", set_camera_interactively=
        self.set_camera_interactively, color_bar_min_value=
        self.color_bar_min_value, color_bar_max_value=
        self.color_bar_max_value, output_imageFileName=screenshot_file, 
        output_path=self.screenshots_path, 
        read_camera_settings_dictionary=True, legend_bar_visibility=True)

        # Makes the same screenshot for the true values of the field in-
        # terpolation in a finite-element space
        
        frozen_snapshots(xdmf_true_field_file, self.field_name, time=0.0, 
        representation_type=self.representation_type, axes_color="black", 
        legend_bar_font="latex", zoom_factor=1.0, component_to_plot=
        self.field_component_to_plot, warp_by_vector=self.warp_by_vector, 
        resolution_ratio=10, background_color="WhiteBackground", 
        display_reference_configuration=display_reference_configuration, 
        transparent_background=True, legend_bar_font_color="black", 
        set_camera_interactively=self.set_camera_interactively, 
        color_bar_min_value=self.color_bar_min_value, 
        color_bar_max_value=self.color_bar_max_value,
        output_imageFileName=screenshot_true_file, output_path=
        self.screenshots_path, read_camera_settings_dictionary=True, 
        legend_bar_visibility=True)

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
    "ces_of_dataset_samples"):

        # If the indices of dataset samples is an integer, the range is
        # from the first sample to the given number of dataset samples,
        # as per convention

        if isinstance(samples_indices, int):

            # Converts to a numpy range

            samples_indices = np.arange(0, samples_indices)

        elif not isinstance(samples_indices, np.ndarray):

            raise TypeError("'"+str(indices_name)+"' in 'FEMSurrogateE"+
            "valuator' must be an integer or a numpy array. If it is a"+
            "n integer, the dataset samples will be considered the row"+
            "s of the input data from the first to the row of idex giv"+
            "en by the integer-1.\nCurrently, '"+str(indices_name)+"' "+
            "is:\n"+str(samples_indices))

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

        # Initializes the retrieved subdofs_to_learn

        retrieved_subdofs_to_learn = None

        # If subdofs to be learned are None, all DOFs must be learned by
        # the surrogate model

        if subdofs_to_learn is None:

            # Gets a range to the number of columns of the output data,
            # since each column corresponds to a DOF of the FEM data

            retrieved_subdofs_to_learn = np.arange(0, 
            self.output_true_data.shape[1])

        # Checks if subdofs has an attribute .numpy, such as tensorflow
        # tensor

        elif hasattr(subdofs_to_learn, "numpy"):

            # Converts it to numpy array

            retrieved_subdofs_to_learn = subdofs_to_learn.numpy()

        else:

            retrieved_subdofs_to_learn = subdofs_to_learn

        # Checks if subdofs to be learned by the surrogate model is a 
        # numpy array

        if not isinstance(retrieved_subdofs_to_learn, np.ndarray):

            raise TypeError("'subdofs_to_learn' in 'FEMSurrogateEvalua"+
            "tor' must be a numpy array with the indices of the DOFs t"+
            "o be learned by the surrogate model. Currently, it is\n"+
            str(subdofs_to_learn)+"\nwhose type is "+str(type(
            subdofs_to_learn)))

        # Checks if subdofs to be learned is a flat array

        elif len(retrieved_subdofs_to_learn.shape)!=1:
        
            raise IndexError("'subdofs_to_learn' in 'FEMSurrogateEvalu"+
            "ator' is an array of shape "+str(
            retrieved_subdofs_to_learn.shape)+". The shape must be (n_"+
            "dofs) instead")

        # Checks if subdofs is limited to the number of DOFs available
        # in the output data

        elif retrieved_subdofs_to_learn.max()>=(
        self.output_true_data.shape[1]):
        
            raise IndexError("The maximum index in 'subdofs_to_learn' "+
            "in 'FEMSurrogateEvaluator' is "+str(
            retrieved_subdofs_to_learn.max())+" which is larger or equ"+
            "al to the number of columns in the output true data, whic"+
            "h is "+str(self.output_true_data.shape[1]))

        return retrieved_subdofs_to_learn

    # Defines a function to check if the numbers of best samples are
    # consistent

    def check_numbers_of_best_samples(self, indices_of_dataset_samples):

        # Verifies if there are a larger number of best samples than the
        # number of samples in the list of indices of samples of the da-
        # taset

        if self.number_of_best_samples>(
        indices_of_dataset_samples.shape[0]):

            raise IndexError("'number_of_best_samples', "+str(
            self.number_of_best_samples)+", is larger than the number "+
            "of samples provided by 'indices_of_dataset_samples', "+str(
            indices_of_dataset_samples.shape[0])+", at 'FEMSurrogateEv"+
            "aluator'. This is not allowed, for the best samples are t"+
            "aken from this dataset")

        # Checks if the number of best samples to be shot in ParaView is
        # larger than the number of best samples that were actually re-
        # corded

        if self.number_of_samples_to_be_snapshot>(
        self.number_of_best_samples):

            raise IndexError("'number_of_samples_to_be_snapshot', "+str(
            self.number_of_samples_to_be_snapshot)+", is larger than t"+
            "he number of best samples provided by 'number_of_best_sam"+
            "ples', "+str(self.number_of_best_samples)+", at 'FEMSurro"+
            "gateEvaluator'. This is not allowed, for the samples to b"+
            "e snapshot are taken from the best-performing samples")