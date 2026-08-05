# Routine to test custom gradients

import tensorflow as tf

from ....PythonicUtilities.testing_tools import run_class_of_tests

# Defines a function to test the inclusion of custom gradients into 
# tensorflow functions

class TestCustomGradients:

    def __init__(self):

        self.verbose = True

        self.parameters_dtype = "float32"

        # Creates three variables

        self.x_variable = tf.Variable(2.0, dtype=tf.as_dtype(
        self.parameters_dtype))

        self.y_variable = tf.Variable(1.0, dtype=tf.as_dtype(
        self.parameters_dtype))

        self.z_variable = tf.Variable(3.0, dtype=tf.as_dtype(
        self.parameters_dtype))

    # Defines a function to test the evalaution of a function with two
    # variables with custom gradient defined only for one

    def test_partial_custom_gradient(self):

        print("\n#####################################################"+
        "###################\n#                    Tests partial custo"+
        "m gradient                     #\n###########################"+
        "#############################################\n")

        # Defines a function to compute the function f(x,y,z) = (x^2)*
        # exp(y)*z

        @tf.custom_gradient
        def test_function(x, y, z):

            # Evaluates the forward pass

            forward_result = x*x*tf.math.exp(y)*z

            # Evaluates the gradient with respect to x and y only

            def gradient(upstream_gradient):

                return (upstream_gradient*2*x*tf.math.exp(y)*z, 
                upstream_gradient*x*x*tf.math.exp(y)*z, None)

            # Returns the forward result and a function for the gradients 

            return forward_result, gradient

        # Tests the forward pass

        forward_pass_result = test_function(self.x_variable, 
        self.y_variable, self.z_variable)

        print("The forward pass for x="+str(self.x_variable.numpy())+
        ", y="+str(self.y_variable.numpy())+", and z="+str(
        self.z_variable.numpy())+" is: "+str(forward_pass_result.numpy()
        )+"\n")

        # Computes the gradient using automatic differentiation

        with tf.GradientTape() as tape:

            forward_pass_result = test_function(self.x_variable, 
            self.y_variable, self.z_variable)

        gradient_x, gradient_y, gradient_z = tape.gradient(
        forward_pass_result, [self.x_variable, self.y_variable, 
        self.z_variable])

        print("The gradient computed using automatic differentiation i"+
        "s:\ndf/dx = "+str(gradient_x.numpy())+"\ndf/dy = "+str(
        gradient_y.numpy())+"\ndf/dz = "+str(gradient_z)+"\n")

# Runs all tests

if __name__=="__main__":

    # Instantiates the class with the methods to be tested

    class_of_tests = TestCustomGradients()

    # Creates a list of methods (using their names) that are not to be
    # tested

    reserved_methods = []

    # Calls the function to run all the necessary tests

    run_class_of_tests(class_of_tests, reserved_methods=reserved_methods,
    sort_methods_alphabetically=False)