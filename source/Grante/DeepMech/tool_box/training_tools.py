# Routine to store training tools, i.e. optimization tools

import tensorflow as tf

import numpy as np

import time

from copy import deepcopy

from scipy.optimize import minimize

from ..tool_box import loss_tools

from ..tool_box import parameters_tools

# Defines a class to optimize the model's parameters

class ModelTraining:

    def __init__(self, model, training_inputArray, training_trueArray,
    loss_metric=tf.keras.losses.MeanAbsoluteError(), optimizer=
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=
    True), n_iterations=1000, gradient_tolerance=1E-3, 
    float_type=tf.float32, verbose_deltaIterations=100, verbose=False):
        
        # Retrieves the model and the optimization parameters

        self.model = model

        self.loss_metric = loss_metric

        self.optimizer = optimizer

        self.n_iterations = n_iterations

        self.gradient_tolerance = gradient_tolerance

        self.verbose_deltaIterations = verbose_deltaIterations

        self.verbose = verbose

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

        # Returns the trained model

        return self.model, elapsed_time

# Defines a class to optimize the model's parameters using scipy optimi-
# zers and custom loss functions

class ModelCustomTraining:

    def __init__(self, model, training_inputArray, training_trueArray,
    loss_metric, optimizer="CG", n_iterations=1000, gradient_tolerance=
    1E-3, float_type=None, verbose_deltaIterations=100, 
    convex_input_model=False, verbose=False, regularizing_function="sm"+
    "ooth absolute value"):
        
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

        # Gets the float type of the model trainable parameters

        model_parameters_dtype = self.model.trainable_variables[0].dtype

        # If no float type has been prescribed, selects the type of the
        # trainable parameters as master type

        if float_type is None:

            float_type = model_parameters_dtype

        elif model_parameters_dtype!=float_type:

            # If the model trainable parameters and the float type are 
            # not the same, raises an error

            raise TypeError("The 'float_type' asked for the 'ModelCust"+
            "omTraining', "+str(float_type)+", is different to that of"+
            " the trainable parameters of the given model, "+str(
            model_parameters_dtype))

        # Transforms the data to TensorFlow tensors

        if hasattr(training_inputArray, "dtype"):

            if training_inputArray.dtype!=float_type:

                raise TypeError("The type of the training input array,"+
                " "+str(training_inputArray.dtype)+", is different to "+
                "the float type required, "+str(float_type))

        self.training_input = tf.constant(training_inputArray, dtype=
        float_type)

        if hasattr(training_trueArray, "dtype"):

            if training_trueArray.dtype!=float_type:

                raise TypeError("The type of the training input array,"+
                " "+str(training_trueArray.dtype)+", is different to t"+
                "he float type required, "+str(float_type))

        self.training_trueValues = tf.constant(training_trueArray, dtype
        =float_type)

        # Gets a variable to inform if the model is convex to its input
        # and saves the regularizing function for the model parameters

        self.convex_input_model = convex_input_model

        # Construct a class to give the loss function, its gradient, and
        # the instructions for parameters (weights and biases) flattening
        # and reconstruction

        self.loss_class, self.model_parameters = loss_tools.build_loss_gradient_varying_model_parameters(
        self.model, self.loss_metric, self.training_input, 
        model_true_values=self.training_trueValues, convex_input_model=
        self.convex_input_model, regularizing_function=
        regularizing_function)

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

        # Gets the model and evaluates the response

        y_model = self.model(unseen_data)

        # Gets the loss

        loss_value = self.loss_metric(true_data, y_model)

        """y_model2 = parameters_tools.model_output_given_trainable_parameters(
        unseen_data, self.model, self.model_parameters, 
        self.loss_class.shapes_trainable_parameters, 
        regularizing_function=None)

        y_model3 = parameters_tools.model_output_given_trainable_parameters(
        unseen_data, self.model, self.model_parameters, 
        self.loss_class.shapes_trainable_parameters, 
        regularizing_function=self.loss_class.regularizing_function)

        loss_value2 = self.loss_metric(true_data, y_model2)

        loss_value3 = self.loss_metric(true_data, y_model3)

        print(loss_value.numpy(), loss_value2.numpy(), loss_value3.numpy())"""

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

        elapsed_time = time.time()-start_time

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

            print("Training time: "+str(elapsed_time)+" seconds.\n")

        # Gets the trained parameters and reassigns them to the model

        if self.convex_input_model:

            self.model = parameters_tools.update_model_parameters(
            self.model, self.model_parameters, regularizing_function=
            self.loss_class.regularizing_function)

            return self.model
        
        else:
            
            # Does not regularize the model parameters

            self.model = parameters_tools.update_model_parameters(
            self.model, self.model_parameters)

            return self.model

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