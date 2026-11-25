# Routine to evaluate the strain energy stored by the intervertebral disc
# on the surface of the superellipsoidal sampling topology as a function
# of the exponent p of the Lp norm

import numpy as np

from time import time

import traceback

from .....Davout.CuboidGmsh.aa_tests.RVEs.transfinite.spherical_RVE import sphere_geometry_RVE

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.PythonicUtilities.file_handling_tools import txt_toList

from .....Davout.MultiMech.tool_box.mesh_handling_tools import get_boundary_dofs_to_physical_group, read_mshMesh

from .....Davout.MultiMech.tool_box.functional_tools import construct_monolithicFunctionSpace

from .....Davout.MultiMech.tool_box.read_write_tools import write_field_to_xdmf

from .....Davout.GraphUtilities.paraview_tools import frozen_snapshots

from .....Davout.StochasticUtilities.superellipsoidal_sampling.automatic_superellipsoidal_parameterizer import FindSuperellipsoidalParameterization

from .....Davout.StochasticUtilities.tool_box.sampling_tools import get_random_point_on_elipsoid_surface

from .....Davout.DeepMech.tool_box.loss_assembler_classes import LpNormError

from .disc_HGO import simulation_disc

# Defines a class to simulate the intervertebral disc for points on the
# boundary of a ellipsoidal domin

class IVDSimulationClass:

    def __init__(self, p_exponent, parent_path, n_points_on_surface,
    simulation_function, output_result_file, verbose=True, 
    get_points_onto_3D_meshed_ellipsoid=False, include_interior_samples=
    False):

        # If multiple exponents p are to be evaluated

        if isinstance(p_exponent, list):

            self.p_exponent = p_exponent

        # Otherwise, saves it as a list with a single element

        else:

            self.p_exponent = [p_exponent]

        # Saves some parameters that will be fixed

        self.parent_path = parent_path

        self.n_points_on_surface = n_points_on_surface

        self.verbose = verbose

        self.include_interior_samples = include_interior_samples

        # Defines the bases for the names of the files with the input 
        # values to the operator (values of external parameters, rota-
        # tions or translations in this case); and with the output va-
        # lues, strain energy. Bases because these names will be repli-
        # cated for different sets of simulations

        self.input_parameters_base_file_name = (parent_path+"//boundar"+
        "y_points_rotation")

        self.output_parameters_base_file_name = (parent_path+"//bounda"+
        "ry_strain_energy_rotation")

        # Defines a base name also for the xdmf file with the 
        # visualization result

        self.output_visualization_base_file_name = (parent_path+"//str"+
        "ain_energy_on_boundary_rotation_visualization_copy")

        # Saves the function that computes the actual simulation and the
        # name of the file that stores the result with the output value

        self.simulation_function = simulation_function
        
        self.output_result_file = output_result_file

        # If the sample points are to be mapped onto a meshed 3D ellip-
        # soid, uses the corresponding method

        if get_points_onto_3D_meshed_ellipsoid:

            self.sampling_and_running_method = (
            self.get_strain_energy_onto_3D_meshed_ellipsoid)

        # Otherwise, uses conventional sampling

        else:

            self.sampling_and_running_method = (
            self.get_strain_energy_onto_ellipsoid)

    # Defines a function to get the strain energy evaluated onto multi-
    # ple ellipsoids

    def __call__(self, mapping_matrix, centroid_position, 
    boundary_input_parameters_file_name, boundary_values_file_name, 
    boundary_values_xdmf_file_name):

        # Initializes a list with the input file names and another for
        # the output files

        input_file_names = []

        output_file_names = []

        # Iterates over the list of exponents p

        for p_exponent in self.p_exponent:

            # Adds the p exponent indication to the file names

            current_boundary_values_file_name = (
            boundary_values_file_name+"_p_"+str(p_exponent))
    
            current_boundary_input_parameters_file_name = (
            boundary_input_parameters_file_name+"_p_"+str(p_exponent))
            
            current_boundary_values_xdmf_file_name = (
            boundary_values_xdmf_file_name+"_p_"+str(p_exponent))

            # Calls the function to get the strain energy evaluated on 
            # the ellipsoidal domain

            flag_success = self.sampling_and_running_method(p_exponent, 
            mapping_matrix, centroid_position, 
            current_boundary_input_parameters_file_name, 
            current_boundary_values_file_name, 
            current_boundary_values_xdmf_file_name, verbose=self.verbose)

            # Updates the lists of file names if the simulation was suc-
            # cessful

            if flag_success:
                
                input_file_names.append(
                current_boundary_input_parameters_file_name)

                output_file_names.append(current_boundary_values_file_name)

        # Returns the lists of files names

        return input_file_names, output_file_names

    # Defines a function to generate the superellipsoidal domain in the
    # n-dimensional space without using any mesh. Each point of the sur-
    # face of the ellipsoid is a load vector. The strain energy stored 
    # in the IVD is evaluated for all points generated through 
    # StochasticTools

    def get_strain_energy_onto_ellipsoid(self, p_exponent, 
    mapping_matrix, centroid_position, 
    boundary_input_parameters_file_name, boundary_values_file_name, 
    boundary_values_xdmf_file_name, verbose=True):

        # Gets the eigenvectors and eigenvalues of the mapping matrix

        (mapping_matrix_eigenvalues, mapping_matrix_eigenvectors
        ) = np.linalg.eigh(mapping_matrix)

        # Gets the points on the surface as an array [n_surface_points,
        # input_dimension]

        sample_points = get_random_point_on_elipsoid_surface(None,
        p_norm_value=p_exponent, number_of_samples=
        self.n_points_on_surface, mapping_matrix_rotation_matrix=
        mapping_matrix_eigenvectors, mapping_matrix_eigenvalues=
        mapping_matrix_eigenvalues, centroid_vector=centroid_position,
        return_as_list=False, keep_samples_dimension=True)

        # Initializes an array of unsuccessful translation vectors

        unsuccessful_samples = []

        # Initializes the average simulation time

        average_simulation_time = 0.0

        # Initializes a list of successful boundary points

        boundary_points = []

        # Initializes a list to save only the strain energy values real-
        # ly on the boundary

        boundary_only_energy_values = []

        # Initializes the flag that tells if there were successful simu-
        # lations

        flag_success = False

        # Iterates over the boundary DOFs

        for simulation_index in range(sample_points.shape[0]):

            # Gets the sample point as the row of the array of sample 
            # points

            sample_point = sample_points[simulation_index,:].tolist()

            try:

                # Uses this coordinate as the translation vector for the 
                # IVD

                initial_time = time()

                self.simulation_function(sample_point)

                # Updates the time average

                average_simulation_time += time()-initial_time

                # If the pseudotime steps are also to be included as 
                # samples

                if self.include_interior_samples:

                    # Reads the file with the normal to plane pseudo-
                    # vector. Since the individual components of the ro-
                    # tation pseudovector were provided to 
                    # simulation_disc, they are stored in the 
                    # normal_to_plane_spin_vector inside 
                    # SurfaceTranslationAndRotation

                    pseudovector_samples = np.load(self.parent_path+
                    "//normal_to_plane_pseudovector_history.npy")

                    # Converts them from radians to degrees and appends
                    # as list

                    pseudovector_samples = ((180.0/np.pi)*
                    pseudovector_samples)

                    boundary_points.extend(pseudovector_samples.tolist())

                # Otherwise, includes only the last point

                else:

                    # Updates the list of successful boundary points

                    boundary_points.append(sample_point)

                # Updates the flag that tells if there was at least a 
                # successful simulation

                flag_success = True

            except Exception as error:

                # Updates the list of unsuccessful samples

                unsuccessful_samples.append(sample_point)

                np.save(self.parent_path+"//unsuccessful_samples_p_"+
                str(p_exponent)+".npy", np.array(unsuccessful_samples))

                # Prints the traceback on the terminal

                traceback.print_exc()

                continue

            # Reads the txt with the strain energy as a function of time

            strain_energy_list = txt_toList(self.output_result_file)

            # If the pseudotime steps are also to be included as samples

            if self.include_interior_samples:

                # Appends all values of the strain energy
                
                for strain_energy_pseudotime in strain_energy_list:

                    boundary_only_energy_values.append(
                    strain_energy_pseudotime[1]) 

            # Otherwise, just saves the last value of the strain energy

            else:

                # Appends the strain energy to the list that captures 
                # values on the boundary only

                boundary_only_energy_values.append(strain_energy_list[-1
                ][1])

            np.save(boundary_values_file_name+".npy", np.array(
            boundary_only_energy_values))

            # Saves the successful boundary points

            np.save(boundary_input_parameters_file_name+".npy", np.array(
            boundary_points))

            print("\n#################################################"+
            "#######################\n#         Information on assessm"+
            "ent of the ellipsoid boundary          #\n###############"+
            "#########################################################"+
            "\n")

            print("Finished the "+str(simulation_index+1)+"-th simulat"+
            "ion\n"+str(len(unsuccessful_samples))+" simulations out o"+
            "f "+str(sample_points.shape[0])+" were unsuccessful\nThe "+
            "average time per simulation is "+str(
            average_simulation_time/max(1.0, simulation_index+1-len(
            unsuccessful_samples)))+"\n")

        # Shows the unsuccessful samples

        np.save(self.parent_path+"//unsuccessful_samples_p_"+str(
        p_exponent)+".npy", np.array(unsuccessful_samples))

        unsuccessful_samples_string = "\n"

        for unsuccessful_translation in unsuccessful_samples:

            unsuccessful_samples_string += "\n"+str(
            unsuccessful_translation)

        print("The unsuccessful samples are:"+
        unsuccessful_samples_string)

        return flag_success

    # Defines a function to generate the superellipsoidal domain in 3D 
    # space. Each point of the surface of the ellipsoid is a load vector. 
    # The strain energy stored in the IVD is evaluated and plotted onto 
    # the surface of the ellipsoid using a fenics field

    def get_strain_energy_onto_3D_meshed_ellipsoid(self, p_exponent, 
    mapping_matrix, centroid_position, 
    boundary_input_parameters_file_name, boundary_values_file_name, 
    boundary_values_xdmf_file_name, verbose=True):

        # Sets the file name for the ellipsoidal mesh

        ellipsoidal_mesh_file_name = (self.parent_path+"//ellipsoid_p_"+
        str(p_exponent))

        # Gets the number of transfinite divisions based on the number 
        # of points on the surface of the whole ellipsoid. Each facet of
        # the ellipsoid will have the same number of transfinite divi-
        # sions

        n_transfinite = int(np.round(np.sqrt((self.n_points_on_surface-2
        )/24.0)))+1

        # Builds the mesh of the ellipsoidal domain

        sphere_geometry_RVE(1.0, 0.5, ellipsoidal_mesh_file_name, 
        n_points_per_edge=100, transfinite_radial=n_transfinite, 
        transfinite_x=n_transfinite, transfinite_y=n_transfinite, 
        transfinite_z=n_transfinite, mapping_matrix=mapping_matrix, 
        p_exponent=p_exponent, centroid_position=centroid_position, 
        verbose=verbose)

        # Reads the created mesh

        mesh_data_class = read_mshMesh(ellipsoidal_mesh_file_name)

        # Creates the functional data class for the strain energy field

        functional_data_class = construct_monolithicFunctionSpace({
        "strain energy": {"field type": "scalar", "interpolation funct"+
        "ion": "CG", "polynomial degree": 1}}, mesh_data_class, verbose=
        True)

        # Gets an array of indices of the DOFs in the boundary physical 
        # group and another array with coordinates of all DOFs in the 
        # mesh as [n_dofs, 3]

        _, dofs_coordinates, boundary_dofs = get_boundary_dofs_to_physical_group(
        mesh_data_class, functional_data_class, field_name="strain ene"+
        "rgy", get_dofs_coordinates_too=True, get_higher_order_dofs_too=
        True, get_array_with_all_dofs_of_the_boundary=True)

        # Saves the information about the DOFs

        np.save(self.parent_path+"//dofs_coordinates_p_"+str(p_exponent
        )+".npy", dofs_coordinates)

        np.save(self.parent_path+"//boundary_dofs_p_"+str(p_exponent
        )+".npy", boundary_dofs)

        # Initializes a numpy array to store the strain energy evaluated 
        # for each point on the surface of the ellipsoidal domain

        boundary_strain_energy = np.zeros(dofs_coordinates.shape[0])

        # Initializes an array of unsuccessful translation vectors

        unsuccessful_samples = []

        # Initializes the average simulation time

        average_simulation_time = 0.0

        # Initializes a list of points on the boundary

        boundary_points = []

        # Initializes a list to save only the strain energy values real-
        # ly on the boundary

        boundary_only_energy_values = []

        # Iterates over the boundary DOFs

        for simulation_index, boundary_dof in enumerate(boundary_dofs):

            # Gets the coordinates of this boundary DOF

            DOF_coordinate = dofs_coordinates[boundary_dof,:].tolist()

            try:

                # Uses this coordinate as the translation vector for the 
                # IVD

                initial_time = time()

                self.simulation_function(DOF_coordinate)

                # Updates the time average

                average_simulation_time += time()-initial_time

                # Updates the list of successful boundary points

                boundary_points.append(dofs_coordinates[boundary_dof,:
                ].tolist())

            except Exception as error:

                # Updates the list of unsuccessful samples

                unsuccessful_samples.append(DOF_coordinate)

                np.save(self.parent_path+"//unsuccessful_samples_p_"+
                str(p_exponent)+".npy", np.array(unsuccessful_samples))

                # Prints the traceback on the terminal

                traceback.print_exc()

                continue

            # Reads the txt with the strain energy as a function of time

            strain_energy_list = txt_toList(self.output_result_file)

            # Gets the last value and updates it into the array of strain 
            # energy evaluated at the boundary of the superellipsoidal 
            # domain

            boundary_strain_energy[boundary_dof] = strain_energy_list[-1
            ][1]

            # Appends the strain energy to the list that captures values 
            # on the boundary only

            boundary_only_energy_values.append(strain_energy_list[-1][1])

            # Saves the strain energy values

            np.save(self.parent_path+"//boundary_strain_energy_includi"+
            "ng_interior_dofs_p_"+str(p_exponent)+".npy", 
            boundary_strain_energy)

            np.save(boundary_values_file_name+".npy", np.array(
            boundary_only_energy_values))

            # Saves the successful boundary points

            np.save(boundary_input_parameters_file_name+".npy", np.array(
            boundary_points))

            print("\n#################################################"+
            "#######################\n#         Information on assessm"+
            "ent of the ellipsoid boundary          #\n###############"+
            "#########################################################"+
            "\n")

            print("Finished the "+str(simulation_index+1)+"-th simulat"+
            "ion\n"+str(len(unsuccessful_samples))+" simulations out o"+
            "f "+str(boundary_dofs.shape[0])+" were unsuccessful\nThe "+
            "average time per simulation is "+str(
            average_simulation_time/max(1.0, simulation_index+1-len(
            unsuccessful_samples)))+"\n")

        # Shows the unsuccessful samples

        np.save(self.parent_path+"//unsuccessful_samples_p_"+str(
        p_exponent)+".npy", np.array(unsuccessful_samples))

        unsuccessful_samples_string = "\n"

        for unsuccessful_translation in unsuccessful_samples:

            unsuccessful_samples_string += "\n"+str(
            unsuccessful_translation)

        print("The unsuccessful samples are:"+
        unsuccessful_samples_string)

        # Updates the vector of parameters of the strain energy field

        functional_data_class.solution_fields["strain energy"].vector(
        ).set_local(boundary_strain_energy)

        functional_data_class.solution_fields["strain energy"].vector(
        ).apply("insert")

        # Saves the strain energy field to a xdmf file

        write_field_to_xdmf(functional_data_class, explicit_file_name=
        boundary_values_xdmf_file_name, close_file=False, 
        visualization_copy=True)

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

        centroid_position = [np.array([0.0, 0.0, -0.0004]), 
        np.array([0.0, 0.0, -0.0003])]

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

        centroid_position = [np.array([4.7, 0.0, 0.0])]

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

    elif flag_activity=="snapshot":

        results_path = parent_path+"//lagrange_results"

        frozen_snapshots([results_path+"//strain_energy_on_boundary_p_"+
        "2_translation_1_visualization_copy", results_path+"//strain_e"+
        "nergy_on_boundary_p_4_translation_1_visualization_copy"],
        "strain energy", representation_type="Surface With Edges", 
        axes_color="black", zoom_factor=1.0, resolution_ratio=10,
        extract_surface=True, clip=True, background_color="WhiteBackground", 
        display_reference_configuration=False, transparent_background=
        True, legend_bar_font_color="black", set_camera_interactively=
        False, output_imageFileName="strain_energy_translation.png", 
        output_path=parent_path, read_camera_settings_dictionary=True, 
        clip_marker_size=0.0001, clip_marker_color="black", edges_color=
        ["white", "black"], edge_thickness=10.0, clip_type="Box",
        clip_box_origin=[0.0, 0.0, -10.0], clip_box_lengths_vector=
        [10.0, -10.0, 15.0])

    else:

        raise ValueError("'flag_activity' can be 'translation', 'rotat"+
        "ion', or 'snapshot'. Currently, it is '"+str(flag_activity))

    # Calls the function to get the ellipsoidal 3D domains

    if flag_activity!="snapshot":

        # Sets a flag to plot the ellipsoidal domain and another to get
        # the mapping matrix and the centroid automatically

        flag_plot_ellipsoidal_domain = False 

        flag_get_parameters_automatically = True

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
        mapping_matrix, centroid_position[0], loss_function, 
        n_warm_up_samples=2, check_inverse_condition_number=True)

        # Calls the procedure to generate the dataset and update the pa-
        # rameterization of the sampling domain

        if flag_get_parameters_automatically:

            find_parameterization_class()

        # If the ellipoidal domain is to be checked and plotted

        if flag_plot_ellipsoidal_domain:

            # Instantiates the class to simulate the IVD again

            IVD_class = IVDSimulationClass(p_exponent_list, 
            get_parent_path_of_file(), n_points_on_surface, 
            IVD_lambda_function, strain_energy_file_name, verbose=False)

            for centroid_index, individual_centroid in enumerate(
            centroid_position):

                IVD_class(mapping_matrix, individual_centroid, 
                IVD_class.input_parameters_base_file_name+
                file_name_additive+str(centroid_index+1), 
                IVD_class.output_parameters_base_file_name+
                file_name_additive+str(centroid_index+1), 
                IVD_class.output_visualization_base_file_name+
                file_name_additive+str(centroid_index+1))