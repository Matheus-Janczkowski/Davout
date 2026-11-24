# Routine to evaluate the principal directions of variation of a scalar-
# valued function of vector argument using a quadratic approximation

import numpy as np

import tensorflow as tf

from ....PythonicUtilities.path_tools import get_parent_path_of_file, take_outFileNameTermination

from ....GraphUtilities.plotting_tools import plane_plot

from ....DeepMech.tool_box.loss_assembler_classes import LpNormError

# Defines a class to store the quadratic approximation and the optimiza-
# tion process relative to the fitting procedure

class QuadraticApproximationOfScalarFunction:

    def __init__(self, training_input_data, training_output_data, dtype=
    "float32", optimizer=None, n_iterations=1000, gradient_tolerance=
    1E-3, gradient_maximum_component_tolerance=1E-5, learning_rate=1E-2,
    results_path=get_parent_path_of_file(path_bits_to_be_excluded=2),
    loss_function=None, plot_loss_history=True, 
    accumulate_data_samples_to_preexisting_samples=False):

        self.results_path = results_path

        self.accumulate_data_samples_to_preexisting_samples = (
        accumulate_data_samples_to_preexisting_samples)

        # Sets the given data into the class

        self.dtype = tf.as_dtype(dtype)

        # Loads the training arrays of data into the class

        self.update_data_arrays(training_input_data, 
        training_output_data)

        # Gets the dimension of the input space

        self.input_dimensionality = self.training_input_data.shape[1]

        # Counts the number of degrees of freedom of the upper triangular
        # factor of the symmetric coefficient matrix

        self.n_dofs_L_matrix = int(0.5*self.input_dimensionality*(
        self.input_dimensionality+1))

        # Gets the tensor of indices to build the [input_dimensionality,
        # input_dimensionality] coefficient matrix from a flat tensor of
        # its DOFs. The result is a tensor [n_dofs_L_matrix, 2]

        self.indices_flat_to_square = tf.where(tf.linalg.band_part(
        tf.ones((self.input_dimensionality, self.input_dimensionality), 
        dtype=self.dtype), 0, -1))

        # Computes the first trial of the L factor of the symmetric coef-
        # ficient matrix

        L_trial = tf.eye(self.input_dimensionality, 
        self.input_dimensionality, dtype=self.dtype)

        # The first trial of the centroid vector and of the scalar coef-
        # ficients are zero. Thus, initializes the parameters as a zero
        # variable with just the L trial different than zero

        self.trainable_flat_parameters = tf.concat([tf.gather_nd(
        L_trial, self.indices_flat_to_square), tf.zeros((
        self.input_dimensionality+1,), dtype=self.dtype)], axis=0)

        self.trainable_flat_parameters = tf.Variable(
        self.trainable_flat_parameters)

        # Sets the optimizer

        if optimizer is None:

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=
            learning_rate)

        else:

            self.optimizer = optimizer

        self.n_iterations = n_iterations

        self.gradient_tolerance = gradient_tolerance 

        self.gradient_maximum_component_tolerance = (
        gradient_maximum_component_tolerance)

        # Sets the loss function

        if loss_function is None:

            # Sets the mean quadratic error as per default

            self.loss_function = self.mean_quadratic_error

        # Otherwise, uses what has been given

        else:

            self.loss_function = loss_function

        # Sets the flag to plot the loss history or not

        self.plot_loss_history = plot_loss_history

    # Defines the call method as a call of the approximation with the
    # trained coefficients

    def __call__(self, input_tensor):

        return self.evaluate_approximation(input_tensor, 
        self.L_coefficient, self.centroid_vector, 
        self.scalar_coefficient)

    # Defines a function to update the data arrays

    def update_data_arrays(self, training_input_data, 
    training_output_data):

        # Verifies if the training data is a string, which means that it
        # is a path. Does the same for the output data

        if isinstance(training_input_data, str):

            training_input_data = np.load(take_outFileNameTermination(
            training_input_data)+".npy")

        if isinstance(training_output_data, str):

            training_output_data = np.load(take_outFileNameTermination(
            training_output_data)+".npy")

        # Checks if the given data samples are to be appended to the 
        # samples that were already within the current input and output
        # data arrays owned by this class

        if self.accumulate_data_samples_to_preexisting_samples and (
        hasattr(self, 'training_input_data')):

            training_input_data = np.concatenate((
            self.training_input_data.numpy(),training_input_data), axis=
            0)

            training_output_data = np.concatenate((
            self.training_output_data.numpy(),training_output_data), 
            axis=0)

        # Converts all data to tensorflow tensors

        self.training_input_data = tf.convert_to_tensor(
        training_input_data, dtype=self.dtype)

        self.training_output_data = tf.convert_to_tensor(
        training_output_data, dtype=self.dtype)

        # Checks that the true output data is a vector
    
        if len(self.training_output_data.shape)!=1:

            raise IndexError("'training_output_data' in 'QuadraticAppr"+
            "oximationOfScalarFunction' must have a shape as (n_sample"+
            "s). Currently, its shape is "+str(
            self.training_output_data.shape))

        # Checks that the training input data and the output data have 
        # the same number of samples

        if self.training_output_data.shape[0]!=(
        self.training_input_data.shape[0]):

            raise IndexError("'training_output_data' has "+str(
            training_output_data.shape[0])+" samples; whereas 'trainin"+
            "g_input_data' has "+str(training_input_data.shape[0])+" s"+
            "amples. They must have the same number of samples")

        # Gets the number of samples

        self.n_training_samples = self.training_input_data.shape[0]

    # Defines a function to build the quadratic approximation. The input
    # tensor is [n_samples, input_dimensionality]; the L_coefficient,
    # [input_dimensionality, input_dimensionality] (upper triangular ma-
    # trix); the centroid vector, [input_dimensionality]

    def evaluate_approximation(self, input_tensor, L_coefficient, 
    centroid_vector, scalar_coefficient):

        # Shifts the input tensor by the centroid vector

        shifted_input_tensor = input_tensor-centroid_vector

        # Computes the multiplication of the shifted input tensor by the
        # upper triangular factor 

        upper_triangular_multiplication = tf.einsum('ij,sj->si', 
        L_coefficient, shifted_input_tensor)

        # Gets the result as a [n_samples] tensor

        return (tf.einsum('si,si->s', upper_triangular_multiplication, 
        upper_triangular_multiplication)+scalar_coefficient)

    # Defines a function to the mean quadratic error

    def mean_quadratic_error(self, y_true, y_approximation):

        # Evaluates the error

        return tf.reduce_mean(tf.square(y_approximation-y_true))

    # Defines a function to compute the loss function between the qua-
    # dratic approximation and the true data

    def evaluate_loss(self, L_coefficient, centroid_vector, 
    scalar_coefficient):

        # Evaluates the approximation

        y_approximation = self.evaluate_approximation(
        self.training_input_data, L_coefficient, centroid_vector, 
        scalar_coefficient)

        # Evaluates the error

        return self.loss_function(self.training_output_data, 
        y_approximation)

    # Defines a function to compute the gradient given a flat tensor of
    # parameters and the loss function

    @tf.function
    def training_step(self, parameters_flat_tensor):

        # Recreates the L coefficient tensor 

        L_coefficient = tf.scatter_nd(self.indices_flat_to_square,
        parameters_flat_tensor[:self.n_dofs_L_matrix], shape=(
        self.input_dimensionality, self.input_dimensionality))

        # Recreates the centroid vector

        centroid_vector = parameters_flat_tensor[self.n_dofs_L_matrix:(
        self.n_dofs_L_matrix+self.input_dimensionality)]

        # Retrieves the scalar coefficient

        scalar_coefficient = parameters_flat_tensor[-1]

        # Computes the gradient tape

        with tf.GradientTape() as tape:

            # Watches only the reconstructed tensors

            tape.watch([L_coefficient, centroid_vector, 
            scalar_coefficient])

            # Computes the loss function

            loss = self.evaluate_loss(L_coefficient, centroid_vector, 
            scalar_coefficient)

        # Computes the gradients

        dloss_dL, dloss_dcentroid, dloss_dscalar = tape.gradient(loss, [
        L_coefficient, centroid_vector, scalar_coefficient])

        # Concatenates the gradient into a flat tensor

        flat_gradient = tf.concat([tf.gather_nd(dloss_dL,
        self.indices_flat_to_square), dloss_dcentroid, tf.reshape(
        dloss_dscalar, (1,))], axis=0)

        # Applies the gradient vector to the optimizer

        self.optimizer.apply_gradients([(flat_gradient, 
        self.trainable_flat_parameters)])

        return loss, flat_gradient

    # Defines a function to fit the parameters using an optimization 
    # problem

    def fit_parameters(self):

        # Initializes a vector of the loss

        self.loss_history = []

        gradient_norm = None

        gradient_maximum_component = None

        # Sets a flag to tell if the iteration process converged

        converged_optimization = False

        # Iterates over the iterations of the optimization procedure

        for i in range(self.n_iterations):

            # Computes the loss and the gradient

            loss, gradient = self.training_step(
            self.trainable_flat_parameters)

            # Appends the loss history

            self.loss_history.append(loss.numpy())

            # Checks the gradient tolerances

            gradient_norm = tf.linalg.norm(gradient).numpy()

            gradient_maximum_component = tf.reduce_max(tf.abs(gradient)
            ).numpy()

            if gradient_norm<self.gradient_tolerance or (
            gradient_maximum_component<(
            self.gradient_maximum_component_tolerance)):

                converged_optimization = i+1

                print("\nThe parameters of the quadratic approximation"+
                " were fit in "+str(i+1)+" iterations. The gradient no"+
                "rm is "+str(gradient_norm)+"; whereas the maximum com"+
                "ponent of the gradient is "+str(
                gradient_maximum_component))

                break

        # If the maximum number of iterations was used

        if converged_optimization==False:
        
            print("\nThe parameters of the quadratic approximation wer"+
            "e fit using the maximum number of iterations, "+str(
            self.n_iterations)+".\nThe gradient norm is "+str(
            gradient_norm)+"; whereas the maximum omponent of the grad"+
            "ient is "+str(gradient_maximum_component)+"\nNeither tole"+
            "rances were met")

            # Updates thes converged optimization to the maximum number 
            # of iterations

            converged_optimization = self.n_iterations

        # Plots the loss history

        if self.plot_loss_history:

            plane_plot("quadratic_approximation_loss_history.png", 
            x_data=np.arange(start=1, stop=converged_optimization+1), 
            y_data=self.loss_history, parent_path=self.results_path)

        print("\nThe initial loss was "+str(self.loss_history[0])+"\nT"+
        "he final loss was "+str(self.loss_history[-1])+"\n"+str(
        self.n_training_samples)+" samples were used for training")

        # Unpacks the fit trainable parameters into the individual ten-
        # sors

        self.L_coefficient = tf.scatter_nd(self.indices_flat_to_square,
        self.trainable_flat_parameters[:self.n_dofs_L_matrix], shape=(
        self.input_dimensionality, self.input_dimensionality))

        # Recreates the centroid vector

        self.centroid_vector = self.trainable_flat_parameters[
        self.n_dofs_L_matrix:(self.n_dofs_L_matrix+
        self.input_dimensionality)]

        # Retrieves the scalar coefficient

        self.scalar_coefficient = self.trainable_flat_parameters[-1]

    # Defines a function to get the eigenvectors of the coefficient ma-
    # trix

    def get_eigenvectors_of_coefficient_matrix(self):

        # Gets the eigenvalues and eigenvectors of the coefficient ma-
        # trix given by the Cholesky factorization

        self.eigenvalues, self.eigenvectors = tf.linalg.eigh(tf.einsum(
        'ik,jk->ij', self.L_coefficient, self.L_coefficient))

        # Normalizes the eigenvectors

        self.eigenvectors, _ = tf.linalg.normalize(self.eigenvectors, 
        axis=0)

        print("\nThe eigenvalues and eigenvectors are:")

        for i in range(self.input_dimensionality):

            print(format(self.eigenvalues[i].numpy(), '.5e')+" and "+
            str(self.eigenvectors[:,i].numpy().tolist()))

        print("\nThe centroid vector is: "+str(
        self.centroid_vector.numpy())+"\nand the scalar coefficient: "+
        str(self.scalar_coefficient.numpy()))

# Testing block

if __name__=="__main__":

    # Gets the file numerical identifier

    identifier = 12

    # Loads the data

    parent_path = get_parent_path_of_file()+"//lagrange_results"

    training_input = np.load(parent_path+"//boundary_points_p_2_rotati"+
    "on_"+str(identifier)+".npy")

    training_output = np.load(parent_path+"//boundary_only_strain_ener"+
    "gy_p_2_rotation_"+str(identifier)+".npy")

    # Sets the loss function

    loss_function = LpNormError(p=12, use_stable_implementation=True)

    # Instantiates the class of the quadratic approximation

    strain_energy_quadratic_class = QuadraticApproximationOfScalarFunction(
    training_input, training_output, results_path=parent_path,
    loss_function=loss_function)

    # Fits the trainable parameters

    strain_energy_quadratic_class.fit_parameters()

    # Gets the eigenvalues and eigenvectors of the coefficient matrix

    strain_energy_quadratic_class.get_eigenvectors_of_coefficient_matrix()

    # Sets a level of energy that must be got in each principal direction

    energy_level = 2.0

    # Gets an array of the coefficients mu that will be multiplied by 
    # each eigenvector to get the principal directions of the mapping
    # matrix of the superellipsoidal sampling technique

    energy_level = tf.constant(energy_level, dtype=
    strain_energy_quadratic_class.dtype)

    mapping_mtrix_eigenvalues = tf.sqrt(tf.abs(energy_level-
    strain_energy_quadratic_class.scalar_coefficient)/
    strain_energy_quadratic_class.eigenvalues)

    print("\nThe eigenvalues of the mapping matrix are:\n"+str(
    mapping_mtrix_eigenvalues.numpy().tolist()))

    # Prints everything as code directly

    for i in range(strain_energy_quadratic_class.input_dimensionality):

        print("\neigenvector_"+str(i+1)+" = np.array("+str(
        strain_energy_quadratic_class.eigenvectors[:,i].numpy().tolist()
        )+")\n\neigenvalue_"+str(i+1)+" = "+str(
        mapping_mtrix_eigenvalues.numpy().tolist()[i]))

    print("\ncentroid_position = [np.array("+str(
    strain_energy_quadratic_class.centroid_vector.numpy().tolist())+")]")