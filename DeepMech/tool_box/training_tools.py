# Routine to store training tools, i.e. optimization tools

import tensorflow as tf

import time

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