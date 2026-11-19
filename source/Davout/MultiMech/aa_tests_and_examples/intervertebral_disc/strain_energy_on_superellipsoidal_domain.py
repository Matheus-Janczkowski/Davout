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

from .disc_HGO import simulation_disc

# Defines a function to get the strain energy evaluated onto multiple 
# ellipsoids

def get_strain_energy_onto_multiple_ellipsoids(p_exponent_list, 
mapping_matrix, centroid_position, n_points_on_surface, parent_path, 
IVD_lambda_function, strain_energy_file_name, verbose=True, 
file_name_additive=""):

    # Iterates over the list of exponents p

    for p_exponent in p_exponent_list:

        # Calls the function to get the strain energy evaluated on the
        # ellipsoidal domain

        get_strain_energy_onto_ellipsoid(p_exponent, mapping_matrix, 
        centroid_position, n_points_on_surface, parent_path, 
        IVD_lambda_function, strain_energy_file_name, verbose=verbose,
        file_name_additive=file_name_additive)

# Defines a function to generate the superellipsoidal domain in 3D space.
# Each point of the surface of the ellipsoid is a load vector. The strain
# energy stored in the IVD is evaluated and plotted onto the surface of
# the ellipsoid using a fenics field

def get_strain_energy_onto_ellipsoid(p_exponent, mapping_matrix, 
centroid_position, n_points_on_surface, parent_path, IVD_lambda_function, 
strain_energy_file_name, verbose=True, file_name_additive=""):

    # Sets the file name for the ellipsoidal mesh

    ellipsoidal_mesh_file_name = parent_path+"//ellipsoid_p_"+str(
    p_exponent)

    # Gets the number of transfinite divisions based on the number of
    # points on the surface of the whole ellipsoid. Each facet of the
    # ellipsoid will have the same number of transfinite divisions

    n_transfinite = int(np.round(np.sqrt((n_points_on_surface-2)/24.0))
    )+1

    # Builds the mesh of the ellipsoidal domain

    sphere_geometry_RVE(1.0, 0.5, ellipsoidal_mesh_file_name, 
    n_points_per_edge=100, transfinite_radial=n_transfinite, 
    transfinite_x=n_transfinite, transfinite_y=n_transfinite, 
    transfinite_z=n_transfinite, mapping_matrix=mapping_matrix, 
    p_exponent=p_exponent, centroid_position=centroid_position, verbose=
    verbose)

    # Reads the created mesh

    mesh_data_class = read_mshMesh(ellipsoidal_mesh_file_name)

    # Creates the functional data class for the strain energy field

    functional_data_class = construct_monolithicFunctionSpace({
    "strain energy": {"field type": "scalar", "interpolation function": 
    "CG", "polynomial degree": 1}}, mesh_data_class, verbose=True)

    # Gets an array of indices of the DOFs in the boundary physical 
    # group and another array with coordinates of all DOFs in the mesh 
    # as [n_dofs, 3]

    _, dofs_coordinates, boundary_dofs = get_boundary_dofs_to_physical_group(
    mesh_data_class, functional_data_class, field_name="strain energy",
    get_dofs_coordinates_too=True, get_higher_order_dofs_too=True, 
    get_array_with_all_dofs_of_the_boundary=True)

    # Saves the information about the DOFs

    np.save(parent_path+"//dofs_coordinates_p_"+str(p_exponent
    )+file_name_additive+".npy", dofs_coordinates)

    np.save(parent_path+"//boundary_dofs_p_"+str(p_exponent
    )+file_name_additive+".npy", boundary_dofs)

    # Initializes a numpy array to store the strain energy evaluated for
    # each point on the surface of the ellipsoidal domain

    boundary_strain_energy = np.zeros(dofs_coordinates.shape[0])

    # Initializes an array of unsuccessful translation vectors

    unsuccessful_translations = []

    # Initializes the average simulation time

    average_simulation_time = 0.0

    # Iterates over the boundary DOFs

    for simulation_index, boundary_dof in enumerate(boundary_dofs):

        # Gets the coordinates of this boundary DOF

        DOF_coordinate = dofs_coordinates[boundary_dof,:].tolist()

        try:

            # Uses this coordinate as the translation vector for the IVD

            initial_time = time()

            IVD_lambda_function(DOF_coordinate)

            # Updates the time average

            average_simulation_time += time()-initial_time

        except Exception as error:

            # Updates the list of unsuccessful translation vectors

            unsuccessful_translations.append(DOF_coordinate)

            np.save(get_parent_path_of_file()+"//unsuccessful_translat"+
            "ions_p_"+str(p_exponent)+file_name_additive+".npy", 
            np.array(unsuccessful_translations))

            # Prints the traceback on the terminal

            traceback.print_exc()

            continue

        # Reads the txt with the strain energy as a function of time

        strain_energy_list = txt_toList(strain_energy_file_name)

        # Gets the last value and updates it into the array of strain e-
        # nergy evaluated at the boundary of the superellipsoidal domain

        boundary_strain_energy[boundary_dof] = strain_energy_list[-1][1]

        # Saves the strain energy values

        np.save(get_parent_path_of_file()+"//boundary_strain_energy_p_"+
        str(p_exponent)+file_name_additive+".npy", 
        boundary_strain_energy)

        print("\n#####################################################"+
        "###################\n#         Information on assessment of t"+
        "he ellipsoid boundary          #\n###########################"+
        "#############################################\n")

        print("Finished the "+str(simulation_index+1)+"-th simulation\n"+
        str(len(unsuccessful_translations))+" simulations out of "+str(
        boundary_dofs.shape[0])+" were unsuccessful\nThe average time "+
        "per simulation is "+str(average_simulation_time/max(1.0,
        simulation_index+1-len(unsuccessful_translations)))+"\n")

    # Shows the unsuccessful translations

    unsuccessful_translations_string = "\n"

    for unsuccessful_translation in unsuccessful_translations:

        unsuccessful_translations_string += "\n"+str(
        unsuccessful_translation)

    print("The unsuccessful translations are:"+
    unsuccessful_translations_string)

    # Updates the vector of parameters of the strain energy field

    functional_data_class.solution_fields["strain energy"].vector(
    ).set_local(boundary_strain_energy)

    functional_data_class.solution_fields["strain energy"].vector(
    ).apply("insert")

    # Saves the strain energy field to a xdmf file

    write_field_to_xdmf(functional_data_class, explicit_file_name=
    get_parent_path_of_file()+"//strain_energy_on_boundary_p_"+str(
    p_exponent)+file_name_additive+".xdmf", close_file=False, 
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
    # translation or by rotation

    flag_load_translation = False 

    file_name_additive = None

    # Sets a list of p exponents to test with a set of ellipsoids. Then,
    # sets the mapping matrix, and the centroid of the ellipsoidal do-
    # main, too

    p_exponent_list = [2]

    # If this set of simulation is to be controlled by translation only

    if flag_load_translation:

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
        strain_energy_file_name, forces_and_moments_file_name)

    # Otherwise, if it is a rotation-controlled simulation

    else:

        # Critical rotations:
        # x: [-15, 15] degrees
        # y: [-10, 10] degrees
        # z: [-10, 10] degrees

        mapping_matrix = np.array([[15.0, 0.0, 0.0], [0.0, 10.0, 0.0], 
        [0.0, 0.0, 10.0]])

        centroid_position = [np.array([0.0, 0.0, 0.0])]

        # Sets the translation vector to zero

        translation = [0.0, 0.0, 0.0]

        # Updates the file name additive 

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

    # Calls the function to get the ellipsoidal 3D domains

    for centroid_index, individual_centroid in enumerate(centroid_position):

        get_strain_energy_onto_multiple_ellipsoids(p_exponent_list, 
        mapping_matrix, individual_centroid, n_points_on_surface, parent_path, 
        IVD_lambda_function, strain_energy_file_name, verbose=False,
        file_name_additive=file_name_additive+str(centroid_index+1))