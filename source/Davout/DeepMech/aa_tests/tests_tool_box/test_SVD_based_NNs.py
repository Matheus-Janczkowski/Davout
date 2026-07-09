# Routine to store some tests to evaluate SVD-based neural networks

import time

import os

import tensorflow as tf

from ...tool_box import ANN_tools, training_tools, parameters_tools, differentiation_tools

from ....PythonicUtilities.testing_tools import run_class_of_tests

from ....PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a function to test the SVD-based architecture

class TestSVDArchitecture:

    def __init__(self):

        # Defines the test data for the gradient tests

        self.whole_input_dimension = 7

        self.quotient_space_dimension = 3

        self.number_of_neurons_hidden_layer_main_network = 100

        self.output_dimension = 2

        self.activation_list_main_network = [{"quadratic": {"number of"+
        " neurons": self.number_of_neurons_hidden_layer_main_network, 
        "a2": 1.0}}, {"linear": self.output_dimension}]

        self.accessory_activation_list_tests = [{"quadratic": {"number"+
        " of neurons": min(self.quotient_space_dimension,
        self.number_of_neurons_hidden_layer_main_network), "a2": 1.0}}, 
        {"linear": min(self.output_dimension, 
        self.number_of_neurons_hidden_layer_main_network)}]

        # Sets the parameters of the custom SVD-based architecture

        self.modulating_function = "smooth_absolute_value"

        self.custom_architecture = {
        "name": "SVDQuotientSpace", "weights modulating function": 
        self.modulating_function, "Householder epsilon": 1.0, 
        "activations accessory layer list": self.accessory_activation_list_tests}

        self.n_samples_tests = 1000

        self.n_samples_quotient_test = 10

        self.maximum_iterations = 5000

        n_samples_training = 6

        # Defines a function to get the true values

        def true_function(x):

            value = 0.0

            for x_i in x:

                value += x_i**2

            return [value for j in range(self.output_dimension)]
        
        self.true_function = true_function

        # Sets the training and test data

        # Creates the new test data

        self.x_min = -1.0

        self.x_max = 1.0

        data_matrix = []

        true_values = []

        for i in range(self.n_samples_tests):

            data_matrix.append([ANN_tools.random_inRange(self.x_min, 
            self.x_max) for j in range(self.whole_input_dimension
            )])

            # Evaluaets the true function

            true_values.append(self.true_function(data_matrix[-1]))

        self.training_data = data_matrix[:n_samples_training]

        self.training_true_values = true_values[:n_samples_training]

        self.test_data = data_matrix[n_samples_training:]

        self.test_true_values = true_values[n_samples_training:]

        # Converts thet data to tensors

        self.parameters_dtype = "float64"

        # Defines the loss function metric

        self.loss_metric = tf.keras.losses.MeanSquaredError(
        dtype=tf.as_dtype(self.parameters_dtype))

        # Sets the optimizer

        self.optimizer = "CG"

        self.verbose_delta_iterations = 1000

        # Sets where to save the model

        self.save_model_file = "saved_model.keras"

    # Defines a function to test assembling a model with this architec-
    # ture

    def test_assembling_model(self):

        print("\n#####################################################"+
        "###################\n#               Tests assembling SVD-bas"+
        "ed neural network              #\n###########################"+
        "#############################################\n")

        # Tests now with custom layers

        ANN_class = ANN_tools.MultiLayerModel(
        self.whole_input_dimension, self.activation_list_main_network, 
        enforce_customLayers=True, verbose=True, parameters_dtype=
        self.parameters_dtype, custom_architecture=
        self.custom_architecture, input_size_main_network=
        self.quotient_space_dimension)

        self.custom_model = ANN_class()

    # Defines a function to test saving and reloading the model

    def test_saving_and_loading(self):

        print("\n#####################################################"+
        "###################\n#                       Tests saving and"+
        " loading                       #\n###########################"+
        "#############################################\n")
        
        # Sets the path to save the model

        model_path = os.path.join(get_parent_path_of_file(), "SVD_model.keras")

        print("\nSaves the model at: "+str(model_path)+"\n")

        # Saves the model

        self.custom_model.save(model_path)

        # Loads it back

        loaded_model = tf.keras.models.load_model(model_path)

        # Gets the training data as a tensorflow array

        self.training_input_constant = tf.constant(self.training_data, 
        dtype=tf.as_dtype(self.parameters_dtype))

        # Tests the loaded model

        output_original = self.custom_model(self.training_input_constant)

        output_loaded = loaded_model(self.training_input_constant)

        print("Output from saved model:\n"+str(output_original))

        print("\nOutput from loaded model:\n"+str(output_loaded)+"\n\n"+
        "The maximum absolute difference between components of the ori"+
        "ginal and loaded models is: "+str(tf.reduce_max(tf.abs(
        output_original-output_loaded)).numpy())+"\n")

    # Defines a function to test evaluating the model from input and 
    # with trainable parameters to assert if the result is the same in
    # both cases

    def test_input_and_parameters(self):

        # Gets the model trainable parameters and their shapes

        self.initial_model_parameters, self.parameters_shapes = parameters_tools.model_parameters_to_flat_tensor_and_shapes(
        self.custom_model)

        # Sets the class to evaluate the model output given the traina-
        # ble parameters

        self.model_output_given_parameters = parameters_tools.ModelOutputGivenTrainableParameters(
        self.custom_model, self.parameters_shapes)

        # Gets the response of the model using the trainable parameters
        # as an argument

        y_given_parameters = self.model_output_given_parameters(
        self.training_input_constant, self.initial_model_parameters)

        # Gets the response of the model using the built-in parameters

        y_from_input = self.custom_model(self.training_input_constant)

        print("\nThe response of the model given parameters is:\n"+str(
        y_given_parameters)+"\n\nwhereas the response of the model whe"+
        "n evaluated from input is: "+str(y_from_input)+"\n\nThe maxim"+
        "um absolute difference between components is: "+str(
        tf.reduce_max(tf.abs(y_from_input-y_given_parameters)).numpy())+
        "\n")

    # Defines a function to test the derivative of the loss function to 
    # assert if automatic differentiation is working properly

    def test_derivative(self):

        # Converts the training true values to a tensorflow constant

        self.training_true_values_constant = tf.constant(
        self.training_true_values, dtype=tf.as_dtype(
        self.parameters_dtype))

        # Gets an instance of the class that evaluates the loss function
        # and its derivative from the trainable parameters

        gradient_class = differentiation_tools.ScalarGradientWrtTrainableParamsGivenParameters(
        self.loss_metric, self.custom_model, 
        self.training_input_constant, self.parameters_shapes, 
        self.initial_model_parameters, model_true_values=
        self.training_true_values_constant, parameters_type=
        self.initial_model_parameters.dtype)

        # Evaluates the loss function and the derivative with respect to
        # the trainable parameters
        
        loss_value, gradient_wrt_params = gradient_class(
        self.initial_model_parameters) 

        # Evaluates the loss function with respect to the trainable pa-
        # rameters directly from the trainable variables stored in the
        # model

        with tf.GradientTape() as tape:

            loss = self.loss_metric(self.training_true_values_constant,
            self.custom_model(self.training_input_constant))

        gradient_input = tape.gradient(loss, 
        self.custom_model.trainable_variables)

        # Prints the gradient for each trainable variable

        gradient_input_string = ""

        for tensor, trainable_variable in zip(gradient_input, 
        self.custom_model.trainable_variables):

            gradient_input_string += "\n\n"+str(trainable_variable.name
            )+"\n"+str(tensor)

        print("\nThe gradient evaluated from the input directly is: "+
        gradient_input_string+"\n")

        # Evaluates the gradient using central finite differences

        cfd_gradient = []

        cfd_step_length = tf.constant(1E-6, dtype=tf.as_dtype(
        self.parameters_dtype))

        for parameter_index in range(
        self.initial_model_parameters.shape[0]):
            
            # Evaluates the loss function one step forward

            initial_parameters_forward = tf.tensor_scatter_nd_add(
            self.initial_model_parameters, indices=[[parameter_index]],
            updates=[cfd_step_length])

            loss_forward = self.loss_metric(
            self.training_true_values_constant,
            self.model_output_given_parameters(
            self.training_input_constant, initial_parameters_forward))

            # Evaluates the loss function one step backward

            initial_parameters_backward = tf.tensor_scatter_nd_add(
            self.initial_model_parameters, indices=[[parameter_index]],
            updates=[-cfd_step_length])

            loss_backward = self.loss_metric(
            self.training_true_values_constant,
            self.model_output_given_parameters(
            self.training_input_constant, initial_parameters_backward)) 

            # Appends the derivative to the gradient

            cfd_gradient.append((loss_forward-loss_backward)/(2.0*
            cfd_step_length))

        # Converts the gradient evaluated using central finite differen-
        # ces to a tensor

        cfd_gradient = tf.stack(cfd_gradient)

        # Flattens the gradient to compare with the one calculated from
        # given parameters

        flat_gradient_input = tf.concat([tf.reshape(gradient, [-1]
        ) for gradient in gradient_input], axis=0)

        # Stacks the two gradients into a single matrix for comparison

        gradient_matrix = tf.stack([flat_gradient_input, 
        gradient_wrt_params, cfd_gradient], axis=0)

        print("The three methods of evaluating the gradient are presen"+
        "ted below. The first\nrow shows the gradient evaluated when t"+
        "he model is called from input; the\nsecond row shows the grad"+
        "ient when the model is called with the given\ntrainable param"+
        "eters as well using tensorflow AD. the third row shows\nthe g"+
        "radient evaluated from parameters using central finite differ"+
        "ences\n\n"+str(gradient_matrix)+"\n\nThe maximum absolute dif"+
        "ference between components of the gradient from\ninput to tho"+
        "se of the gradient from parameters is: "+str(tf.reduce_max(
        tf.abs(flat_gradient_input-gradient_wrt_params)).numpy())+"\nT"+
        "he difference between components of the gradient from paramet"+
        "ers using\ntensorflow AD to the gradient using central finite"+
        " differences is: "+str(tf.reduce_max(tf.abs(gradient_wrt_params
        -cfd_gradient)).numpy())+"\n")

    # Defines a function to test training a model with this architecture

    def test_training_model(self):

        print("\n#####################################################"+
        "###################\n#                Tests training SVD-base"+
        "d neural network               #\n###########################"+
        "#############################################\n")

        # Sets the optimization class for training

        self.training_class = training_tools.ModelCustomTraining(
        self.custom_model, self.training_data, self.training_true_values, 
        self.loss_metric, verbose=True, n_iterations=
        self.maximum_iterations, verbose_deltaIterations=
        self.verbose_delta_iterations, save_model_file=
        self.save_model_file)

        t_initial = time.time()

        self.training_class()

        elapsed_time = time.time()-t_initial

        print("\nTrains at "+str(elapsed_time)+" seconds")

        # Checks the loss again with the model with the regularized pa-
        # rameters

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data is "+str(self.training_class.loss_unseen_data(
        self.training_true_values, self.training_data, output_as_numpy=
        True)))

    # Defines a function to test training a model with this architecture
    # for many realizations in a Monte Carlo scheme. At each realiza-
    # tion, a new set of trainable parameters is initialized and used 
    # for training

    def test_monte_carlo_training_model(self):

        print("\n#####################################################"+
        "###################\n#     Tests Monte Carlo training with th"+
        "e SVD-based neural network     #\n###########################"+
        "#############################################\n")

        # Tests Monte Carlo training

        self.training_class.monte_carlo_training(n_realizations=5, 
        best_models_rank_size=5, show_reinitialization_distance=True)

        # Checks the loss again with the best model of the Monte Carlo
        # training

        print("\nThe loss function evaluated again over the set of tra"+
        "ining data for the best model is "+str(
        self.training_class.loss_unseen_data(self.training_true_values, 
        self.training_data, output_as_numpy=True)))

    # Defines a function to test the invariance of the model when the
    # quotient space slice of the input vector is null

    def test_quotient_space_invariance_model(self):

        print("\n#####################################################"+
        "###################\n#    Tests quotient space invariance of "+
        "the SVD-based neural network   #\n###########################"+
        "#############################################\n")

        # Generates an input tensor with zeros for the input neurons 
        # corresponding to the quotient space

        coefficient = 1E-10

        x = tf.concat([coefficient*tf.random.uniform((
        self.n_samples_quotient_test, self.quotient_space_dimension), 
        minval=-1.0, maxval=1.0, dtype=tf.as_dtype(self.parameters_dtype
        )), tf.random.uniform((self.n_samples_quotient_test, 
        self.whole_input_dimension-self.quotient_space_dimension
        ), minval=-1.0, maxval=1.0, dtype=tf.as_dtype(
        self.parameters_dtype))],axis=-1)

        model_output = self.training_class.model(x)

        print("\nTests the SVD model against the following input:\n"+
        str(x.numpy())+"\n\nThere follows the SVD model output:\n"+str(
        model_output)+"\n")

    # Defines a function to test if the SVD model is convex to variables
    # of the quotient space

    def test_convexity_in_quotient_space(self):

        # Sets a flag to evaluate the eigenvalues of the hessian matrix

        return_eigenvalues = True

        # Checks the hessian matrices

        hessian_info = self.training_class.get_hessian_outputs_model(
        eigenvalues=False)

        # Converts the hessian info to a string

        hessian_matrices = ""

        # Iterates over the output neurons
    
        for output_index in range(self.output_dimension):

            hessian_matrices += ("\n\n"+str(output_index+1)+"-th outpu"+
            "t neuron:\n\n")

            # Gets the hessian matrix, and separates only the rows and
            # columns linked to variables in the quotient space

            hessian_matrix = hessian_info[output_index][:,
            :self.quotient_space_dimension,
            :self.quotient_space_dimension]

            hessian_matrices += ("Hessian matrix with respect to varia"+
            "bles in the quotient space:\n"+str(hessian_matrix.numpy()))

            # Verifies if the eigenvalues were evaluated

            if return_eigenvalues:

                # Gets the eigenvalues

                eigenvalues = tf.linalg.eigvalsh(hessian_matrix)

                hessian_matrices += "\n\nEigenvalues:\n"+str(
                eigenvalues.numpy())

        print("\nThere follow the hessian matrices with respect to the"+
        " variables in the quotient space:\n"+str(hessian_matrices)+
        "\n\nRemember that the model should be convex when the modulat"+
        "ing function is\nnon-zero. The given modulating function is:"+
        "\n'"+str(self.modulating_function)+"'\n")

# Runs all tests

if __name__=="__main__":

    # Instantiates the class with the methods to be tested

    class_of_tests = TestSVDArchitecture()

    # Creates a list of methods (using their names) that are not to be
    # tested

    reserved_methods = []

    # Calls the function to run all the necessary tests

    run_class_of_tests(class_of_tests, reserved_methods=reserved_methods,
    sort_methods_alphabetically=False)