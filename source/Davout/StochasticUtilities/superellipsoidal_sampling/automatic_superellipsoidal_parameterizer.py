# Routine to implement and test the automatic finder of the mapping ma-
# trix

import numpy as np

import tensorflow as tf

from .quadratic_approximator import QuadraticApproximationOfScalarFunction

from ...PythonicUtilities.path_tools import get_parent_path_of_file

from ...DeepMech.tool_box.loss_assembler_classes import LpNormError

from ...GraphUtilities.plotting_tools import plane_plot

# Defines a class that finds the mapping matrix for a dataset as the da-
# taset is built

class FindSuperellipsoidalParameterization:

    def __init__(self, energy_level, simulation_class, 
    n_sampling_domain_updates, results_path, initial_mapping_matrix, 
    initial_centroid_vector, loss_function=None, dtype="float32",
    plot_results=False, verbose=True, n_warm_up_samples=None, 
    n_samples_per_update_cycle=1, null_eigenvalue_tolerance=1E-5,
    shrink_coefficient=0.95, check_inverse_condition_number=False,
    shifted_approximation_flag=False):

        self.simulation_class = simulation_class

        self.n_sampling_domain_updates = n_sampling_domain_updates

        self.results_path = results_path

        self.loss_function = loss_function

        self.dtype = dtype

        self.plot_results = plot_results

        self.verbose = verbose

        self.n_samples_per_update_cycle = n_samples_per_update_cycle

        self.null_eigenvalue_tolerance = null_eigenvalue_tolerance

        self.shrink_coefficient = shrink_coefficient

        self.check_inverse_condition_number = (
        check_inverse_condition_number)

        self.shifted_approximation_flag = shifted_approximation_flag

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

        # If the number of samples to be used in the warm-up cycle of 
        # updating the superellipsoidal parameterization, makes that 
        # number equal to the number of degrees of freedom in the qua-
        # dratic approximation

        if n_warm_up_samples is None:

            input_dimension = initial_mapping_matrix.shape[0]

            self.n_warm_up_samples = int((0.5*input_dimension*(
            input_dimension+1))+input_dimension+1)

        else:

            self.n_warm_up_samples = n_warm_up_samples

    # Defines the method that creates data points and fits the parameters
    # of the superellipsoidal sampling domain automatically

    def __call__(self):

        # Iterates over the iterations for updating the sampling domain

        for update_iteration_index in range(
        self.n_sampling_domain_updates):

            # If this is the first iteration, uses the warm-up number of
            # samples

            if update_iteration_index==0:

                self.simulation_class.n_points_on_surface = (
                self.n_warm_up_samples)

            # Otherwise, uses the number of samples of an ordinary upda-
            # te cycle

            else:

                self.simulation_class.n_points_on_surface = (
                self.n_samples_per_update_cycle)

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

            # If there are converged simulations

            flag_converged_simulation = len(input_file_names_list)>0

            if flag_converged_simulation:

                # If the class for the quadratic approximation of the 
                # output given the input samples is None, instantiates 
                # the class

                if self.quadratic_approximation_class is None:

                    # Takes only the first file name of each list

                    self.quadratic_approximation_class = QuadraticApproximationOfScalarFunction(
                    input_file_names_list[0], output_file_names_list[0], 
                    results_path=self.results_path, loss_function=
                    self.loss_function, dtype=self.dtype, 
                    plot_loss_history=self.plot_results, 
                    accumulate_data_samples_to_preexisting_samples=True,
                    check_inverse_condition_number=
                    self.check_inverse_condition_number,
                    shifted_approximation_flag=
                    self.shifted_approximation_flag)

                # Otherwise, updates the training data

                else:

                    # Takes only the first file name of each list

                    self.quadratic_approximation_class.update_data_arrays(
                    input_file_names_list[0], output_file_names_list[0])

                # Fits the parameters. Thus, finds the mapping matrix 
                # and the centroid vector

                self.quadratic_approximation_class.fit_parameters()

            # Updates the mapping matrix and the centroid vector

            self.get_mapping_matrix_and_centroid(
            flag_converged_simulation)

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

            # Plots the Frobenius norm of the difference between 
            # consecutive mapping matrices

            if self.plot_results:

                plane_plot(self.results_path+"//frobenius_norm_of_"+
                "consecutive_mapping_matrices.png", x_data=np.arange(
                len(self.frobenius_norm_consecutive_mapping_matrices
                )), y_data=np.asarray(
                self.frobenius_norm_consecutive_mapping_matrices))

            # Prints the result

            print("\n#################################################"+
            "######################\n#        Superellipsoidal paramet"+
            "ers were successfully updated         #\n################"+
            "########################################################")

            if self.verbose:

                mapping_matrix_string = "\n["

                for row_index in range(self.mapping_matrix.shape[0]):

                    final_character = ",\n"

                    if row_index==(self.mapping_matrix.shape[0]-1):

                        final_character = "]"

                    mapping_matrix_string += str(self.mapping_matrix[
                    row_index,:].tolist())+final_character

                print("\nFinishes computing the "+str(
                update_iteration_index+1)+"-th iteration to update the"+
                " superellipsoidal parameterization\nThe new mapping m"+
                "atrix is:"+mapping_matrix_string)

                print("\nThe centroid vector is:\n"+str(
                self.centroid_vector.tolist())+"\n\n"+str(
                self.quadratic_approximation_class.n_training_samples)+
                " converged samples were used to fit these set of mapp"+
                "ing matrix and centroid vector")

    # Defines a method to get the mapping matrix and the centroid vector
    # from the eigenvectors evaluated during the fitting scheme of the
    # quadratic approximation

    def get_mapping_matrix_and_centroid(self, flag_converged_simulation):

        # If the flag was not converged, simply multiplies the mapping 
        # matrix by a coefficient equal or less than one to isotropical-
        # ly shrink the sampling domain

        if not flag_converged_simulation:

            self.mapping_matrix = (self.shrink_coefficient*
            self.mapping_matrix)

            # Uses the return statement to stop the update here

            return None

        # Gets the eigenvectors and eigenvalues of the corresponding
        # mapping matrix

        self.quadratic_approximation_class.get_eigenvectors_of_coefficient_matrix()

        # Gets the minimum eigenvalue and compares to the tolerance

        minimum_eigenvalue = tf.reduce_min(
        self.quadratic_approximation_class.eigenvalues)

        # Checks that the energy level is higher than the scalar coeffi-
        # cient of the quadratic approximation

        energy_difference = (self.energy_level-
        self.quadratic_approximation_class.scalar_coefficient)

        minimum_energy_difference = tf.reduce_min(energy_difference)

        # If it is less than the tolerance, does not update the mapping
        # matrix. If the energy difference is negative, the mapping ma-
        # trix cannot be updated, because the fitted quadratic approxi-
        # mation is above the desired energy level

        if (minimum_eigenvalue.numpy()>self.null_eigenvalue_tolerance
        ) and minimum_energy_difference.numpy(
        )>self.null_eigenvalue_tolerance:

            # Evaluates the eigenvalues of the mapping matrix so that 
            # the desired energy level is reached

            mapping_matrix_eigenvalues = tf.sqrt(energy_difference/
            self.quadratic_approximation_class.eigenvalues)

            if self.verbose:

                print("\nThe eigenvalues of the mapping matrix are:\n"+
                str(mapping_matrix_eigenvalues.numpy().tolist()))

            # Assembles the mapping matrix using the spectral decomposi-
            # tion

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
        strain_energy_file_name, forces_and_moments_file_name,
        results_path=parent_path)

    # Otherwise, if it is a rotation-controlled simulation

    elif flag_activity=="rotation":

        # Critical rotations:
        # x: [-15, 15] degrees
        # y: [-10, 10] degrees
        # z: [-10, 10] degrees

        # Sets the eigenvectors of the mapping matrix and their eigen-
        # values

        eigenvector_1 = np.array([1.0, 0.0, 0.0])

        eigenvalue_1 = 15.0

        eigenvector_2 = np.array([0.0, 1.0, 0.0])

        eigenvalue_2 = 10.0

        eigenvector_3 = np.array([0.0, 0.0, 1.0])

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

        centroid_position = np.array([0.0, 0.0, 0.0])

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
        strain_energy_file_name, forces_and_moments_file_name,
        results_path=parent_path)

    else:

        raise ValueError("'flag_activity' can be 'translation', 'rotat"+
        "ion'. Currently, it is '"+str(flag_activity))

    # Calls the function to get the ellipsoidal 3D domains

    if flag_activity!="snapshot":

        # Sets the energy level to approximate the boundary of the su-
        # perellipsoidal sampling domain

        energy_level = 2.0

        # Sets the number of times to update the sampling domain

        n_sampling_domain_updates = 21

        # Sets the loss function as the Lp norm

        loss_function = LpNormError(p=2, use_stable_implementation=True)

        loss_function = None

        # Instantiates the class to simulate the IVD

        IVD_class = IVDSimulationClass(p_exponent_list, 
        get_parent_path_of_file(), n_points_on_surface, 
        IVD_lambda_function, strain_energy_file_name, verbose=False,
        get_points_onto_3D_meshed_ellipsoid=False, 
        include_interior_samples=True)

        # Instantiates the class to find the parameterization of the su-
        # perellipsoidal domain

        find_parameterization_class = FindSuperellipsoidalParameterization(
        energy_level, IVD_class, n_sampling_domain_updates, parent_path, 
        mapping_matrix, centroid_position, loss_function, 
        n_warm_up_samples=2, check_inverse_condition_number=True)

        # Calls the procedure to generate the dataset and update the pa-
        # rameterization of the sampling domain

        find_parameterization_class()