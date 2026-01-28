# Routine to store training tools, i.e. optimization tools

import tensorflow as tf

import numpy as np

from tqdm import tqdm

from collections import OrderedDict

import time

from copy import deepcopy

from scipy.optimize import minimize

from scipy import stats

from ..tool_box import loss_tools

from ..tool_box import parameters_tools

from ...PythonicUtilities import path_tools

from ...PythonicUtilities import string_tools

# Defines a class to optimize the model's parameters

class ModelTraining:

    def __init__(self, model, training_inputArray, training_trueArray,
    loss_metric=tf.keras.losses.MeanAbsoluteError(), optimizer=
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=
    True), n_iterations=1000, gradient_tolerance=1E-3, 
    float_type=tf.float32, verbose_deltaIterations=100, verbose=False,
    save_model_file="trained_model.keras", parent_path="get current pa"+
    "th"):
        
        # Retrieves the model and the optimization parameters

        self.model = model

        self.loss_metric = loss_metric

        self.optimizer = optimizer

        self.n_iterations = n_iterations

        self.gradient_tolerance = gradient_tolerance

        self.verbose_deltaIterations = verbose_deltaIterations

        self.verbose = verbose

        # Saves the parent path where to save the model

        if parent_path=="get current path":

            # Gets as default the path where this class was instantia-
            # ted

            parent_path = path_tools.get_parent_path_of_file(
            function_calls_to_retrocede=2)

        # Unites the parent path to the model file name, but takes out
        # the termination and forces it to be .keras

        self.save_model_file = path_tools.verify_path(parent_path, 
        path_tools.take_outFileNameTermination(save_model_file)+".kera"+
        "s")

        # Transforms the data to TensorFlow tensors

        self.training_input = tf.constant(training_inputArray, dtype=
        float_type)

        self.training_trueValues = tf.constant(training_trueArray, dtype
        =float_type)

    # Defines a method to evaluate the loss function

    def loss_function(self):

        # Gets the model response

        y_training = self.model(self.training_input)

        # Gets the loss value

        return self.loss_metric(self.training_trueValues, y_training)
    
    # Defines a method for the train step. Uses the tf.function decora-
    # tor to create a map for compilation, hence, speeding it up

    @tf.function

    def train_step(self):

        with tf.GradientTape() as tape:

            loss = self.loss_function()

        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, 
        self.model.trainable_variables))

        return loss, tf.linalg.global_norm(gradients)
    
    # Defines a method for the optimization loop. But names it call so
    # that it is called as soon as the class is created

    def __call__(self):

        # Starts counting time

        start_time = time.time()

        initial_loss = self.loss_function().numpy()

        # Iterates through the optimization loop

        max_digits = len(str(self.n_iterations))

        for i in range(self.n_iterations):

            loss_value, gradient_norm = self.train_step()

            if gradient_norm.numpy()<=self.gradient_tolerance:

                print("The gradient's norm has reached the value of "+
                str(gradient_norm.numpy())+", which is less than the t"+
                "hreshold of "+str(self.gradient_tolerance)+". Thus, s"+
                "tops the optimization procedure at iteration "+str(i)+
                "\n")

                break
        
            # Prints the loss every 10 iterations

            if self.verbose:
            
                if i%self.verbose_deltaIterations==0:

                    print("Iteration "+integer_toString(i, max_digits)+
                    ": loss="+format(loss_value.numpy(), '.5e')+", gra"+
                    "dient norm: "+format(gradient_norm.numpy(), '.5e'))

        # Evaluates the elapsed time

        elapsed_time = time.time()-start_time

        if self.verbose:

            # Gets the final loss function

            final_loss = self.loss_function().numpy()

            print("\n#################################################"+
            "#######################\n#                        Trainin"+
            "g process's log                        #\n###############"+
            "#########################################################"+
            "\n")

            print("Initial loss.: "+format(initial_loss, '.5e'))

            print("Final loss...: "+format(final_loss, '.5e'))

            print("Training time: "+str(elapsed_time)+" seconds.\n")

        # Saves the model automatically and returns it as well

        self.model.save(self.save_model_file)

        return self.model, elapsed_time

# Defines a class to optimize the model's parameters using scipy optimi-
# zers and custom loss functions

class ModelCustomTraining:

    def __init__(self, model, training_inputArray, training_trueArray,
    loss_metric, optimizer="CG", n_iterations=1000, gradient_tolerance=
    1E-3, float_type=None, verbose_deltaIterations=100, verbose=False, 
    save_model_file="trained_model.keras", parent_path="get current pa"+
    "th"):
        
        """
        Class for training a model whose trainable parameters (weights
        and biases are) used as a flatten vector to be trained using a
        scipy framework. Tensorflow is used to evaluate derivatives and
        the loss function only"""
        
        # Retrieves the model and the optimization parameters

        self.model = model

        self.optimizer = optimizer

        self.n_iterations = n_iterations

        self.gradient_tolerance = gradient_tolerance

        self.verbose_deltaIterations = verbose_deltaIterations

        self.verbose = verbose

        self.loss_metric = loss_metric

        # Saves the parent path where to save the model

        if parent_path=="get current path":

            # Gets as default the path where this class was instantia-
            # ted

            self.parent_path = path_tools.get_parent_path_of_file(
            function_calls_to_retrocede=2)

        # Unites the parent path to the model file name, but takes out
        # the termination and forces it to be .keras

        self.save_model_file = path_tools.verify_path(self.parent_path, 
        path_tools.take_outFileNameTermination(save_model_file)+".kera"+
        "s")

        print("Saves the model at:\n"+str(self.save_model_file))

        # Gets the float type of the model trainable parameters

        model_parameters_dtype = self.model.trainable_variables[0].dtype

        # If no float type has been prescribed, selects the type of the
        # trainable parameters as master type

        if float_type is None:

            self.float_type = model_parameters_dtype

        elif model_parameters_dtype!=float_type:

            # If the model trainable parameters and the float type are 
            # not the same, raises an error

            raise TypeError("The 'float_type' asked for the 'ModelCust"+
            "omTraining', "+str(float_type)+", is different to that of"+
            " the trainable parameters of the given model, "+str(
            model_parameters_dtype))
        
        else:

            self.float_type = float_type

        # Transforms the data to TensorFlow tensors

        if hasattr(training_inputArray, "dtype"):

            if training_inputArray.dtype!=float_type:

                raise TypeError("The type of the training input array,"+
                " "+str(training_inputArray.dtype)+", is different to "+
                "the float type required, "+str(float_type))

        self.training_input = tf.constant(training_inputArray, dtype=
        self.float_type)

        if hasattr(training_trueArray, "dtype"):

            if training_trueArray.dtype!=self.float_type:

                raise TypeError("The type of the training input array,"+
                " "+str(training_trueArray.dtype)+", is different to t"+
                "he float type required, "+str(self.float_type))

        self.training_trueValues = tf.constant(training_trueArray, dtype
        =self.float_type)

        # Gets a variable to inform if the model is convex to its input,
        # and saves the regularizing function

        regularizing_function = "smooth absolute value"

        if hasattr(model, "input_convex_model"):

            self.input_convex_model = model.input_convex_model

            if self.input_convex_model is not None:

                if self.input_convex_model!="fully" and (
                self.input_convex_model!="partially"):
                    
                    raise NameError("'input_convex_model' is '"+str(
                    self.input_convex_model)+"', whereas it can be eit"+
                    "her None, 'fully', or 'partially'")
                
            # Gets the regularizing function

            if hasattr(model, "regularizing_function"):
                
                regularizing_function = model.regularizing_function

            else:

                raise AttributeError("The model is required to be inpu"+
                "t convex or partially input convex, but no regularizi"+
                "ng function has been provided for regularizing weight"+
                " matrices")
                
        else:

            self.input_convex_model = None

        # Construct a class to give the loss function, its gradient, and
        # the instructions for parameters (weights and biases) flattening
        # and reconstruction

        self.loss_class, self.model_parameters = loss_tools.build_loss_gradient_varying_model_parameters(
        self.model, self.loss_metric, self.training_input, 
        model_true_values=self.training_trueValues, convex_input_model=
        self.input_convex_model, regularizing_function=
        regularizing_function, verbose=verbose)

        # Gets the number of output neurons. Tries to get first via a 
        # custom attribute

        if hasattr(self.model, "output_dimension"):

            self.n_outputs = self.model.output_dimension

        else:
        
            self.n_outputs = self.model.trainable_variables[-1].shape[0]

    # Defines a method to evaluate the hessian of each output neuron of
    # the model

    def get_hessian_outputs_model(self, eigenvalues=False):

        # Initializes a list of hessian matrices

        hessian_matrices = []

        # Sets input as a variable

        model_input = tf.Variable(self.training_input)

        # Iterates through the output neurons

        for i in range(self.n_outputs):

            # Differentiates twice

            with tf.GradientTape() as tape2:

                with tf.GradientTape() as tape1:

                    tape1.watch(model_input)

                    tape2.watch(model_input)

                    # Gets the model output

                    y_output = self.model(model_input)[:,i]

                gradient = tape1.gradient(y_output, model_input)

            # Gets the hessian in the format (n_samples, n_input, n_sam-
            # ples, n_input)

            full_hessian = tape2.jacobian(gradient, model_input)

            # Reordinates it into the format (n_samples, n_input, n_in-
            # put)

            per_sample_hessian = tf.stack([full_hessian[b,:,b,:] for (b
            ) in range(full_hessian.shape[0])], axis=0)

            # Appends the per sample hessian to the list and their ei-
            # genvalues

            if eigenvalues:

                hessian_matrices.append([per_sample_hessian, 
                tf.linalg.eigvalsh(per_sample_hessian)])

            else:

                hessian_matrices.append(per_sample_hessian)

        # Returns the list of hessian matrices

        return hessian_matrices

    # Defines a method to evaluate the loss function

    def loss_function(self):

        # Gets the loss value and returns it

        return self.loss_class.evaluate_scalar_function(
        self.model_parameters)
    
    # Defines a function to evaluate the loss function on other sets of
    # data

    def loss_unseen_data(self, true_data, unseen_data, output_as_numpy=
    False):

        """
        Function to evaluate the loss function over unseen data after 
        the model has already been trained. 'true_data' is the tensor
        with true values, whereas 'unseen_data' is the tensor with input
        samples for the corresponding true values"""

        # Verifies if the data is a TensorFlow tensor

        if not hasattr(true_data, "dtype"):

            true_data = tf.constant(true_data, dtype=self.float_type)

        if not hasattr(unseen_data, "dtype"):

            unseen_data = tf.constant(unseen_data, dtype=self.float_type)

        # Gets the model and evaluates the response

        y_model = self.model(unseen_data)

        # Gets the loss

        loss_value = self.loss_metric(true_data, y_model)

        if output_as_numpy:

            return loss_value.numpy()
        
        return loss_value
    
    # Defines a method for training, it assembles the optimization pro-
    # blem and runs it

    def set_training(self, n_max_iterations):

        # Sets the minimization problem using the minimize class from 
        # scipy

        minimization_problem = minimize(
        self.loss_class.evaluate_scalar_function, self.model_parameters,
        method=self.optimizer, jac=self.loss_class, tol=
        self.gradient_tolerance, options={"maxiter": n_max_iterations})

        # Updates the model parameters

        self.model_parameters = minimization_problem.x
    
    # Defines a method for the optimization loop. But names it call so
    # that it is called as soon as the class is created

    def __call__(self):

        # Starts counting time

        start_time = time.time()

        initial_loss = self.loss_class.evaluate_scalar_function(
        self.model_parameters).numpy()

        # Iterates through the optimization loop

        max_digits = len(str(self.n_iterations))

        # Verifies if the verbose flag is True

        if self.verbose:

            # Gets the number of iterations for each step, where at the
            # end the convergence information will be printed

            optimization_iteration_groups = int(np.ceil(
            self.n_iterations/self.verbose_deltaIterations))

            for i in range(optimization_iteration_groups):

                print("\n\nStarts the "+str(i)+" groups of optimizatio"+
                "n iterations\n")

                # Calls the optimization procedure

                self.set_training(self.verbose_deltaIterations)

                # Gets the loss function value and the gradient

                loss_value = self.loss_class.evaluate_scalar_function(
                self.model_parameters)

                gradient_value = tf.norm(self.loss_class(
                self.model_parameters)).numpy()

                if gradient_value<=self.gradient_tolerance:

                    print("The gradient's norm has reached the value o"+
                    "f "+str(gradient_value)+", which is less than the"+
                    " threshold of "+str(self.gradient_tolerance)+". T"+
                    "hus, stops the optimization procedure at iteratio"+
                    "n group "+str(i)+"\n")

                    break

                print("Iteration group "+integer_toString(i, max_digits)
                +": loss="+format(loss_value.numpy(), '.5e')+", gradie"+
                "nt norm: "+format(gradient_value, '.5e'))

        # Otherwise, just call the training function

        else:

            self.set_training(self.n_iterations)

        # Evaluates the elapsed time

        self.elapsed_time = time.time()-start_time

        if self.verbose:

            # Gets the final loss function

            final_loss = self.loss_class.evaluate_scalar_function(
            self.model_parameters).numpy()

            print("\n#################################################"+
            "#######################\n#                        Trainin"+
            "g process's log                        #\n###############"+
            "#########################################################"+
            "\n")

            print("Initial loss.: "+format(initial_loss, '.5e'))

            print("Final loss...: "+format(final_loss, '.5e'))

            print("Training time: "+str(self.elapsed_time)+" seconds.\n")

        # Gets the trained parameters and reassigns them to the model

        if self.input_convex_model:

            self.model = parameters_tools.update_model_parameters(
            self.model, self.model_parameters, regularizing_function=
            self.loss_class.regularizing_function)

            # Saves the model automatically and returns it as well

            self.model.save(self.save_model_file)

            return self.model
        
        else:
            
            # Does not regularize the model parameters

            self.model = parameters_tools.update_model_parameters(
            self.model, self.model_parameters)

            # Saves the model automatically and returns it as well

            self.model.save(self.save_model_file)

            return self.model
        
    # Defines a function to perform a Monte Carlo training, i.e. train-
    # ing multiple times and generating multiple models

    def monte_carlo_training(self, n_realizations=50, 
    best_models_rank_size=None, show_reinitialization_distance=False):
        
        print("\n#####################################################"+
        "###################\n#                   Initializes Monte Ca"+
        "rlo training                   #\n###########################"+
        "#############################################\n")
        
        # If no number of best models to be ranked, rank all of the rea-
        # lizations

        realizations_notice = ""

        if best_models_rank_size is None:

            best_models_rank_size = n_realizations

        elif best_models_rank_size>n_realizations:

            realizations_notice = ("The number of realizations is smal"+
            "ler than that of the number of models\nto be ranked and s"+
            "aved. Thus, makes the number of realizations equal to\nth"+
            "e number of models to be saved, "+str(
            best_models_rank_size)+"\n")

            print(realizations_notice)

            n_realizations = best_models_rank_size

        print("Number of realizations:                     "+str(
        n_realizations))

        print("Number of best models to be saved:          "+str(
        best_models_rank_size))

        # Initializes a dictionary of models with their respective loss
        # function value at training as value to the key
        
        models_ranking_dict = OrderedDict()

        for i in range(best_models_rank_size):

            models_ranking_dict[i] = np.inf 

        # Initializes a variable to store the elapsed time for training
        # each model at each realization

        elapsed_time_list = []

        # Iterates through the realizations

        for i in tqdm(range(n_realizations), desc="Training realizatio"+
        "ns"):

            # Reinitializes the model parameters using the same initia-
            # lizers that were assigned when the model was first created

            self.model = parameters_tools.reinitialize_model_parameters(
            self.model)

            # If the value of the distance between the reinitialization
            # of the parameters if to be show

            if show_reinitialization_distance:

                old_parameters = deepcopy(self.model_parameters)

            # Updates the flat vector of model parameters

            self.model_parameters = parameters_tools.model_parameters_to_flat_tensor_and_shapes(
            self.model)[0]

            if show_reinitialization_distance:

                print("\nThe norm of the difference of the previous se"+
                "t of parameters to the new set is: "+str(tf.norm(
                old_parameters-self.model_parameters).numpy()))

            # Initializes the saving of the model as training 

            self.save_model_file = (self.parent_path+"//model_during_t"+
            "raining.keras")

            # Calls the training

            self.__call__()

            # Updates the list of elapsed time for training this model

            elapsed_time_list.append(self.elapsed_time)

            # Gets the loss function

            training_loss = self.loss_unseen_data(
            self.training_trueValues, self.training_input, 
            output_as_numpy=True)

            # Iterates through the best ranking models to check if the
            # current one outperforms any of them

            for model_number in models_ranking_dict.keys():

                # Gets the model loss and compares it to the loss value
                # of the current model

                model_loss = models_ranking_dict[model_number]

                if model_loss>training_loss:

                    # Deletes the worst performing model, which is the
                    # corresponding one to the last key

                    path_tools.delete_file(str(list(
                    models_ranking_dict.keys())[-1])+"_best_model.kera"+
                    "s", parent_path=self.parent_path, 
                    ignore_non_existing_file=True)

                    # Move every other model below this one one step 
                    # downwards

                    for j in range(list(models_ranking_dict.keys())[-1], 
                    model_number, -1):
                        
                        # Gets the value of the previous model and allo-
                        # cates it to the next

                        models_ranking_dict[j] = deepcopy(
                        models_ranking_dict[j-1])

                        # And renames the model file

                        path_tools.rename_file(str(j)+"_best_model.k"+
                        "eras", str(j+1)+"_best_model.keras", 
                        parent_path=self.parent_path, saving_function=
                        self.model.save)
                    
                    # Finally adds the current loss

                    models_ranking_dict[model_number] = deepcopy(
                    training_loss)

                    # Saves this model

                    path_tools.rename_file(self.save_model_file, str(
                    model_number+1)+"_best_model.keras", 
                    saving_function=self.model.save)

                    # Breaks the loop

                    break

        # Evaluates the mean time for training a model and the standard
        # deviation

        elapsed_time_list = np.array(elapsed_time_list)

        mean_elapsed_time = np.mean(elapsed_time_list)

        standard_deviation_elapsed_time = np.std(elapsed_time_list)

        # Loads the best model

        print("\n#####################################################"+
        "###################\n#                Final log of the Monte "+
        "Carlo training                 #\n###########################"+
        "#############################################\n")

        print(realizations_notice)

        print("The best fitting models follow below. The number of the"+
        " model and its\ncorresponding loss function at the training s"+
        "et of data\n")

        print("The mean time for training a model was: "+str(
        mean_elapsed_time))

        print("The standard deviation was: "+str(
        standard_deviation_elapsed_time))

        confidence_radius = (stats.t.ppf(0.5*(1+0.95), len(
        elapsed_time_list)-1)*(standard_deviation_elapsed_time/np.sqrt(
        len(elapsed_time_list)-2)))

        print("With a confidence of 95%, the time for training a model"+
        " lies in the in-\nterval "+str(mean_elapsed_time-
        confidence_radius)+" < t < "+str(mean_elapsed_time+
        confidence_radius)+" seconds\n")

        for model_number, model_loss in models_ranking_dict.items():

            print("model "+str(model_number+1)+" has a loss at the tra"+
            "ining set of:       "+str(
            string_tools.float_to_scientific_notation(model_loss)))

        print("")

        print("Loads the best model")

        self.model = tf.keras.models.load_model(self.parent_path+"//1_"+
        "best_model.keras")

########################################################################
#                               Utilities                              #
########################################################################

# Defines a function to convert an integer to a string with the same
# number of characters as the maximum number of digits. If there are 
# less digits than the maximum number of digits, fill in with blank spa-
# ces

def integer_toString(number, maximum_digits):

    # Converts to string

    number_string = str(number)

    # Adds the blank spaces if necesary

    for i in range(maximum_digits-len(number_string)):

        number_string = " "+number_string

    return number_string