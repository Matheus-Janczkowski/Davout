# Routine to implement and test the automatic finder of the mapping ma-
# trix

import numpy as np

import tensorflow as tf

from .strain_energy_on_superellipsoidal_domain import IVDSimulationClass

from .quadratic_approximator import QuadraticApproximationOfScalarFunction

from ....PythonicUtilities.path_tools import get_parent_path_of_file

from ....DeepMech.tool_box.loss_assembler_classes import LpNormError

from .disc_HGO import simulation_disc

# Defines a class that finds the mapping matrix for a dataset as the da-
# taset is built

class FindSuperellipsoidalParameterization:

    def __init__(self, energy_level, simulation_class, 
    n_sampling_domain_updates, results_path, initial_mapping_matrix, 
    initial_centroid_vector, loss_function=None, dtype="float32",
    plot_loss_history=False):

        self.simulation_class = simulation_class

        self.n_sampling_domain_updates = n_sampling_domain_updates

        self.results_path = results_path

        self.loss_function = loss_function

        self.dtype = dtype

        self.plot_loss_history = plot_loss_history

        # Saves the desired energy level as a tensorflow constant

        self.energy_level = tf.constant(energy_level, dtype=tf.as_dtype(
        dtype))

        # Initializes the mapping matrix and the centroid vector with 
        # the given values

        self.mapping_matrix = initial_mapping_matrix

        self.centroid_vector = initial_centroid_vector

        # Gets the base file names of the simulation class

        self.input_parameters_base_file_name = (
        simulation_class.input_parameters_base_file_name)

        self.output_parameters_base_file_name = (
        simulation_class.output_parameters_base_file_name)

        self.output_visualization_base_file_name = (
        simulation_class.output_visualization_base_file_name)

        # Initializes the variable that stores an instance of the class 
        # for the quadratic approximation as None

        self.quadratic_approximation_class = None

        # Initializes a history of the mapping matrix, another for the
        # centroid vector, and finally one for the Frobenius norm betwe-
        # en consecutive mapping matrices

        self.mapping_matrix_history = [initial_mapping_matrix]

        self.centroid_vector_history = [initial_centroid_vector]

        self.frobenius_norm_consecutive_mapping_matrices = []

    # Defines the method that creates data points and fits the parameters
    # of the superellipsoidal sampling domain automatically

    def __call__(self):

        # Iterates over the iterations for updating the sampling domain

        for update_iteration_index in range(
        self.n_sampling_domain_updates):

            # Updates the names of the files

            boundary_input_parameters_file_name = (
            self.input_parameters_base_file_name+"_"+str(
            update_iteration_index+1))
            
            boundary_values_file_name = (
            self.output_parameters_base_file_name+"_"+str(
            update_iteration_index+1))

            boundary_values_xdmf_file_name = (
            self.output_visualization_base_file_name+"_"+str(
            update_iteration_index+1))

            # Calls the simulation

            input_file_names_list, output_file_names_list = self.simulation_class(
            self.mapping_matrix, self.centroid_vector, 
            boundary_input_parameters_file_name, 
            boundary_values_file_name, boundary_values_xdmf_file_name)

            # If the class for the quadratic approximation of the output
            # given the input samples is None, instantiates the class

            if self.quadratic_approximation_class is None:

                # Takes only the first file name of each list

                self.quadratic_approximation_class = QuadraticApproximationOfScalarFunction(
                input_file_names_list[0], output_file_names_list[0], 
                results_path=self.results_path, loss_function=
                self.loss_function, dtype=self.dtype, plot_loss_history=
                self.plot_loss_history, 
                accumulate_data_samples_to_preexisting_samples=True)

            # Otherwise, updates the training data

            else:

                # Takes only the first file name of each list

                self.quadratic_approximation_class.update_data_arrays(
                input_file_names_list[0], output_file_names_list[0])

            # Fits the parameters. Thus, finds the mapping matrix and
            # the centroid vector

            self.quadratic_approximation_class.fit_parameters()

            # Updates the mapping matrix and the centroid vector

            self.get_mapping_matrix_and_centroid()

            # Updates the history lists

            self.frobenius_norm_consecutive_mapping_matrices.append(
            np.linalg.norm(self.mapping_matrix_history[-1]
            -self.mapping_matrix))

            self.mapping_matrix_history.append(self.mapping_matrix)

            self.centroid_vector_history.append(self.centroid_vector)

            np.save(self.results_path+"//history_frobenius_norm_of_con"+
            "secutive_mapping_matrices.npy", np.asarray(
            self.frobenius_norm_consecutive_mapping_matrices))

            np.save(self.results_path+"//history_mapping_matrices.npy", 
            np.asarray(self.mapping_matrix_history))

            np.save(self.results_path+"//history_centroid_vectors.npy", 
            np.asarray(self.centroid_vector_history))

            # Prints the result

            print("\n#################################################"+
            "######################\n#        Superellipsoidal paramet"+
            "ers were successfully updated         #\n################"+
            "########################################################")

            mapping_matrix_string = ""

            for row_index in range(self.mapping_matrix.shape[0]):

                mapping_matrix_string += "\n"+str(self.mapping_matrix[
                row_index,:].tolist())

            print("\nFinishes computing the "+str(update_iteration_index
            +1)+" iteration to update the superellipsoidal parameteriz"+
            "ation\nThe new mapping matrix is:"+mapping_matrix_string)

            print("\nThe centroid vector is:\n"+str(
            self.centroid_vector.tolist()))

    # Defines a method to get the mapping matrix and the centroid vector
    # from the eigenvectors evaluated during the fitting scheme of the
    # quadratic approximation

    def get_mapping_matrix_and_centroid(self):

        # Gets the eigenvectors and eigenvalues of the corresponding
        # mapping matrix

        self.quadratic_approximation_class.get_eigenvectors_of_coefficient_matrix()

        # Evaluates the eigenvalues of the mapping matrix so that the
        # desired energy level is reached

        mapping_matrix_eigenvalues = tf.sqrt(tf.abs(self.energy_level-
        self.quadratic_approximation_class.scalar_coefficient)/
        self.quadratic_approximation_class.eigenvalues)

        # Assembles the mapping matrix using the spectral decomposition

        self.mapping_matrix = tf.einsum('ik,k,jk->ij', 
        self.quadratic_approximation_class.eigenvectors, 
        mapping_matrix_eigenvalues,
        self.quadratic_approximation_class.eigenvectors).numpy()

        # Simply gets the centroid

        self.centroid_vector = (
        self.quadratic_approximation_class.centroid_vector.numpy())

# Testing block

if __name__=="__main__":

    ####################################################################
    #                   Intervertebral disc settings                   #
    ####################################################################

    # Sets the parent path and the mesh file name

    parent_path = get_parent_path_of_file()

    # Sets the mesh file name
    
    IVD_mesh_file_name = parent_path+"//intervertebral_disc_mesh"

    # Assembles the names of the result files

    displacement_file_name = parent_path+"//displacement_test"

    pressure_file_name = parent_path+"//pressure_test"

    volume_ratio_file_name = parent_path+"//volume_ratio_test"

    strain_energy_file_name = parent_path+"//strain_energy_test"

    forces_and_moments_file_name = (parent_path+"//forces_and_moments_"+
    "test")

    # Options for linear solver: "cg", "gmres", "minres", "mumps"

    linear_solver = "mumps"

    # c for HGO

    c_HGO = 0.195E6

    # kappa value

    kappa = 0.07

    # Bulk modulus 

    bulk_modulus = 15.0E6

    # c2 = c01 for Mooney Rivlin

    c2_mooney = 0.03E6

    # c1 = c10 for Mooney Rivlin 

    c1_mooney = 0.16E6

    # Files of the material properties

    axial_direction_file_name  = parent_path+"//axial_direction_000"

    circumferential_direction_file_name = (parent_path+"//circumferent"+
    "ial_direction_000")

    k1_file_name = parent_path+"//k1_0"

    k2_file_name = parent_path+"//k2_0"

    gamma_file_name = parent_path+"//gamma_0"

    ####################################################################
    #                   Ellipsoidal domain settings                    #
    ####################################################################

    # Initializes a variable with the lambda function to compute the IVD
    # simulation as a function of the translation or of the axle vectors
    # only

    IVD_lambda_function = None

    # Initializes a variable for the mapping matrix and another for the
    # centroid of the ellipsoidal domain

    mapping_matrix = None
    
    centroid_position = None

    # Sets the number of points on the surface of the ellipsoidal domain

    n_points_on_surface = 40

    # Sets a flag that tells if the simulations will be controlled by
    # translation or by rotation, or if it is time to generate the snap-
    # shots

    flag_activity = "rotation" 

    file_name_additive = None

    # Sets a list of p exponents to test with a set of ellipsoids. Then,
    # sets the mapping matrix, and the centroid of the ellipsoidal do-
    # main, too

    p_exponent_list = [2]

    # If this set of simulation is to be controlled by translation only

    if flag_activity=="translation":

        # Critical translations:
        # x: [-, 3.1] mm
        # y: [-3.7, 3.7] mm
        # z: [-2.1, 1.5] mm

        mapping_matrix = np.array([[0.0031, 0.0, 0.0], [0.0, 0.0034, 0.0], 
        [0.0, 0.0, 0.0016]])

        centroid_position = np.array([0.0, 0.0, -0.0004])

        # Sets the axle vector to no rotation at all

        axle_vector = [0.0, 0.0, 0.0]

        # Updates the file name additive 

        file_name_additive = "_translation_"

        # Sets a lambda function for the disc to be a function of the 
        # translation vector only

        IVD_lambda_function = lambda translation: simulation_disc(
        linear_solver, c_HGO, kappa, bulk_modulus, c2_mooney, c1_mooney, 
        IVD_mesh_file_name, k1_file_name, k2_file_name, gamma_file_name, 
        axial_direction_file_name, circumferential_direction_file_name, 
        translation, axle_vector, displacement_file_name, 
        pressure_file_name, volume_ratio_file_name, 
        strain_energy_file_name, forces_and_moments_file_name)

    # Otherwise, if it is a rotation-controlled simulation

    elif flag_activity=="rotation":

        # Critical rotations:
        # x: [-15, 15] degrees
        # y: [-10, 10] degrees
        # z: [-10, 10] degrees

        # Sets the eigenvectors of the mapping matrix and their eigen-
        # values

        eigenvector_1 = np.array([1.0, 0.0, 0.0])

        eigenvalue_1 = 9.5

        eigenvector_2 = np.array([0.0, 1.0, 1.0])

        eigenvalue_2 = 11.0

        eigenvector_3 = np.array([0.0, -1.0, 1.0])

        eigenvalue_3 = 10.0

        # Normalizes the eigenvectors

        eigenvector_1 = (1.0/np.linalg.norm(eigenvector_1))*eigenvector_1

        eigenvector_2 = (1.0/np.linalg.norm(eigenvector_2))*eigenvector_2

        eigenvector_3 = (1.0/np.linalg.norm(eigenvector_3))*eigenvector_3

        # Assembles the mapping matrix using the outer product

        mapping_matrix = (eigenvalue_1*np.einsum('i,j->ij', 
        eigenvector_1, eigenvector_1))+(eigenvalue_2*np.einsum('i,j->ij', 
        eigenvector_2, eigenvector_2))+(eigenvalue_3*np.einsum('i,j->ij', 
        eigenvector_3, eigenvector_3))

        # Sets the centroid of the ellipsoidal domain

        centroid_position = np.array([4.7, 0.0, 0.0])

        # Sets the translation vector to zero

        translation = [0.0, 0.0, 0.0]

        # Updates the file name additive ,

        file_name_additive = "_rotation_"

        # Sets a lambda function for the disc to be a function of the 
        # axle vector only

        IVD_lambda_function = lambda axle_vector: simulation_disc(
        linear_solver, c_HGO, kappa, bulk_modulus, c2_mooney, c1_mooney, 
        IVD_mesh_file_name, k1_file_name, k2_file_name, gamma_file_name, 
        axial_direction_file_name, circumferential_direction_file_name, 
        translation, axle_vector, displacement_file_name, 
        pressure_file_name, volume_ratio_file_name, 
        strain_energy_file_name, forces_and_moments_file_name)

    else:

        raise ValueError("'flag_activity' can be 'translation', 'rotat"+
        "ion'. Currently, it is '"+str(flag_activity))

    # Calls the function to get the ellipsoidal 3D domains

    if flag_activity!="snapshot":

        # Sets the energy level to approximate the boundary of the su-
        # perellipsoidal sampling domain

        energy_level = 2.0

        # Sets the number of times to update the sampling domain

        n_sampling_domain_updates = 5

        # Sets the loss function as the Lp norm

        loss_function = LpNormError(p=12, use_stable_implementation=True)

        # Instantiates the class to simulate the IVD

        IVD_class = IVDSimulationClass(p_exponent_list, 
        get_parent_path_of_file(), n_points_on_surface, 
        IVD_lambda_function, strain_energy_file_name, verbose=False)

        # Instantiates the class to find the parameterization of the su-
        # perellipsoidal domain

        find_parameterization_class = FindSuperellipsoidalParameterization(
        energy_level, IVD_class, n_sampling_domain_updates, parent_path, 
        mapping_matrix, centroid_position, loss_function)

        # Calls the procedure to generate the dataset and update the pa-
        # rameterization of the sampling domain

        find_parameterization_class()