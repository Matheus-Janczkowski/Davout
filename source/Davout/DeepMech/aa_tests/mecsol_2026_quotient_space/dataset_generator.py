# Routine to give a material parameter as a field

from .....Davout.MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from .....Davout.MultiMech.tool_box.numerical_tools import rotation_tensorEulerRodrigues

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.PythonicUtilities.file_handling_tools import list_toTxt

from .....Davout.StochasticUtilities.tool_box.sampling_tools import get_random_point_on_elipsoid_surface

from .....Davout.DeepMech.aa_tests.mecsol_2026_quotient_space.box_with_imposed_displacement_gradient import solve_BVP

from .....Davout.DeepMech.aa_tests.mecsol_2026_quotient_space.young_modulus_field_generator import generate_field

import numpy as np

import traceback

from time import time

# Defines a function to generate the dataset of a cube

def generate_dataset(n_samples, limits_list, p_norm_list, results_path,
displacement_file_name, young_modulus_file, mesh_file_name, cube_size,
influence_radius, damping_factor, radius_power, mesh_data_class, 
lagrange_multiplier_file_name, save_snapshot_displacement=False):
    
    # Initializes the list of samples

    samples_list = []

    # Gets the names of the variables 

    variables_names = []

    for p_norm, limits_dict in zip(p_norm_list, limits_list):

        new_variables_names = [key for key in limits_dict.keys()]

        variables_names.extend(new_variables_names)

        variables_limits = [limits_dict[name] for name in (
        new_variables_names)]

        # Generates the list of samples

        samples_list.append(get_random_point_on_elipsoid_surface(
        variables_limits, p_norm_value=p_norm, number_of_samples=
        n_samples, return_as_list=False))

    # Stacks the samples array into a single array. Then, converts it to
    # a list

    samples_list = np.hstack(samples_list).tolist()

    # Creates a dictionary of variable names and their column in the da-
    # ta matrix to guarantee ordering consistency (dictionaries do not
    # necessarily preserve order)

    variables_indices = {name: i for i, name in enumerate(
    variables_names)}

    # Adds a header and saves the list of samples in a txt file

    samples_list.insert(0, variables_names)

    list_toTxt(samples_list, "dataset", parent_path=
    get_parent_path_of_file())

    # Initializes a list of successful simulations and another for the
    # corresponding data

    succesful_data = []

    succesful_displacement = []

    # Initializes a counter for failed simulations

    n_failed_simulations = 0

    # Initializes a list of the success of each simulation

    completed_simulations = []

    # Sets the initial time 

    initial_t = time()

    # Gets the number of characteres in the number of samples

    n_algorisms = len(str(n_samples))

    # Iterates through the samples to evaluate the simulation

    for i in range(n_samples):

        # Gets the data for this sample

        sample_data = samples_list[i+1]

        # Gets the simulation number

        simulation_number = str(i+1)

        # If the number of algorisms is different, adds trailing zeros

        if len(simulation_number)!=n_algorisms:

            for algorism in range(n_algorisms-len(simulation_number)):

                simulation_number = "0"+simulation_number

        # Generates the field of Young modulus

        generate_field(cube_size, sample_data[variables_indices["peak E"
        ]], sample_data[variables_indices["base E"]], influence_radius,
        damping_factor, radius_power, mesh_data_class, 
        get_parent_path_of_file(), file_name=young_modulus_file)

        # Gets the displacement gradient as a list from the polar decom-
        # position

        displacement_gradient = get_displacement_gradient(sample_data[
        variables_indices["stretch 1"]], sample_data[variables_indices[
        "stretch 2"]], sample_data[variables_indices["stretch 3"]], 
        sample_data[variables_indices["stretch rotation 1"]], 
        sample_data[variables_indices["stretch rotation 2"]], 
        sample_data[variables_indices["stretch rotation 3"]], 
        sample_data[variables_indices["polar rotation 1"]], sample_data[
        variables_indices["polar rotation 2"]], sample_data[
        variables_indices["polar rotation 3"]])

        try:

            # Calls the solution scheme

            solve_BVP(results_path, displacement_file_name+"_"+
            simulation_number, get_parent_path_of_file()+"//"+
            young_modulus_file, get_parent_path_of_file()+"//"+
            mesh_file_name, displacement_gradient, 
            lagrange_multiplier_file_name+"_"+simulation_number, 
            save_snapshot=save_snapshot_displacement)

            # If the simulation was succesful, reads the displacement 
            # field and updates it to the succesful data. Discards the
            # first column because it contains the time values

            succesful_displacement.extend(np.load(results_path+"//"+
            displacement_file_name+"_"+simulation_number+".npy")[:,1:].tolist())

            # Adds the Young modulus data and the displacement gradient

            successful_sample_data = [sample_data[variables_indices["p"+
            "eak E"]], sample_data[variables_indices["base E"]]]

            successful_sample_data.extend(displacement_gradient[0])

            successful_sample_data.extend(displacement_gradient[1])

            successful_sample_data.extend(displacement_gradient[2])

            succesful_data.append(successful_sample_data)

            # Updates the list of completed simulations

            completed_simulations.append([i+1, True])

            print("The shape of the displacement data is: "+str(
            np.array(succesful_displacement).shape)+"\nand the shape o"+
            "f the input data is: "+str(np.array(succesful_data).shape)+
            "\n")

            # Saves the lists into binary and txt files

            np.save(results_path+"//00_succesful_displacement_matrix.n"+
            "py", np.array(succesful_displacement))

            np.save(results_path+"//00_successful_data_matrix.npy", 
            np.array(succesful_data))

            list_toTxt(completed_simulations, "00_completed_simulations", 
            parent_path=results_path)

        except Exception:

            print("The "+str(i+1)+"-th simulation failed. The error me"+
            "ssage is:\n"+str(traceback.format_exc()))

            # Updates the counter of failed simulations

            n_failed_simulations += 1

            # Updates the list of completed simulations

            completed_simulations.append([i+1, False])

            # Saves this list into a txt file

            list_toTxt(completed_simulations, "00_completed_simulations", 
            parent_path=results_path)

        print("\n"+str(n_failed_simulations)+" simulations out of "+
        "a total of "+str(n_samples)+" failed\n"+str(i+1-
        n_failed_simulations)+" have been successful so far\n")

        # Updates the time 

        t_final = time()

        print("#######################################################"+
        "#################\n##########################################"+
        "##############################\n"+str(t_final-initial_t)+" se"+
        "conds have elapsed since the generation of this dataset start"+
        "ed.\nThis corresponds to an average of "+str((t_final-
        initial_t)/(i+1))+" seconds per simulation\n##################"+
        "######################################################\n#####"+
        "#############################################################"+
        "######\n")

# Defines a function to take a vector of eigenvalues and a rotation axial
# vector to generate a positive definite stretch tensor. This function 
# also takes another rotation axial vector to generate the polar decom-
# position. Then, results the displacement gradient by subtracting the
# identity tensor

def get_displacement_gradient(stretch_1, stretch_2, stretch_3, 
stretch_rotation_1, stretch_rotation_2, stretch_rotation_3, 
polar_rotation_1, polar_rotation_2, polar_rotation_3):
    
    # Gets the rotation tensor of the spectral decomposition of the
    # stretch tensor

    stretch_rotation_tensor = rotation_tensorEulerRodrigues([
    stretch_rotation_1, stretch_rotation_2, stretch_rotation_3],
    return_numpy_array=True)

    # Gets the rotation tensor for the polar decomposition of the defor-
    # mation tensor

    polar_rotation_tensor = rotation_tensorEulerRodrigues([
    polar_rotation_1, polar_rotation_2, polar_rotation_3], 
    return_numpy_array=True)

    # Creates the stretch tensor in matrix format using the spectral de-
    # composition

    stretch_tensor = stretch_rotation_tensor@(np.array([[stretch_1, 
    0.0, 0.0], [0.0, stretch_2, 0.0], [0.0, 0.0, stretch_3]])@
    stretch_rotation_tensor.T)

    # Creates the deformation gradient from the polar decomposition

    deformation_gradient = polar_rotation_tensor@stretch_tensor

    # Subtracts the identity to return the displacement gradient. Then,
    # returns it as a list

    displacement_gradient = (deformation_gradient-np.eye(3)).tolist()

    print("\n\nThe eigenvalues of the stretch tensor are:\n"+str(
    np.sqrt(np.linalg.eigvals(deformation_gradient.T@deformation_gradient
    )))+"\nThe eigenvalues of the right Cauchy-Green tensor are:\n"+str(
    np.linalg.eigvals(deformation_gradient.T@deformation_gradient))+
    "\nThe given stretches are:\n"+
    str([stretch_1, stretch_2, stretch_3])+"\nThe given displacement g"+
    "radient is:\n"+str(displacement_gradient)+"\n\n")

    #float(a)

    return displacement_gradient

# Testing block

if __name__=="__main__":

    # Sets the number of samples, stress limits, and the limits of the
    # modulus

    n_samples = 10000

    limits_stretch_1 = [0.85, 2.5]

    limits_stretch_2 = [0.85, 2.5]

    limits_stretch_3 = [0.85, 2.5]

    limits_rotation_stretch_1 = [-180.0, 180.0]

    limits_rotation_stretch_2 = [-180.0, 180.0]

    limits_rotation_stretch_3 = [-180.0, 180.0]

    limits_polar_rotation_1 = [-180.0, 180.0]

    limits_polar_rotation_2 = [-180.0, 180.0]

    limits_polar_rotation_3 = [-180.0, 180.0]

    limits_base_E = [12E6, 15E6]

    limits_maximum_E = [60E6, 65E6]

    limits_list = [{"stretch 1": limits_stretch_1, "stretch 2": 
    limits_stretch_2, "stretch 3": limits_stretch_3}, {"stretch rotati"+
    "on 1": limits_rotation_stretch_1, "stretch rotation 2": 
    limits_rotation_stretch_2, "stretch rotation 3": 
    limits_rotation_stretch_3, "polar rotation 1": 
    limits_polar_rotation_1, "polar rotation 2": limits_polar_rotation_2,
    "polar rotation 3": limits_polar_rotation_3, "base E": limits_base_E, 
    "peak E": limits_maximum_E}]

    # Sets the p of the L-p norm for the ellipsoid sampling as a list.
    # One norm is for the stretch eigenvalues and the other for every-
    # thing else

    p_norm_list = [2, 16]

    # Creates a mesh for the field

    cube_size = 0.5

    n_divisions = 10

    E_extremum = 10E6

    E_base = 1E6

    influence_radius = 0.4*cube_size

    damping_factor = 10.0

    radius_power = 6

    bias_factor = 2.0

    # Sets the paths and files names

    results_path = get_parent_path_of_file()+"//results"

    displacement_file_name = "displacement_young_modulus_field"

    lagrange_multiplier_file_name = "lagrange_multiplier_young_modulus"

    young_modulus_file = "young_modulus_field"

    mesh_file_name = "box_mesh_mecsol"

    # Constructs the mesh

    mesh_data_class = read_mshMesh({"length x": cube_size, "length y": 
    cube_size, "length z": cube_size, "number of divisions in x": 
    n_divisions, "number of divisions in y": n_divisions, "number of d"+
    "ivisions in z": n_divisions, "verbose": False, "mesh file name": 
    mesh_file_name, "mesh file directory": get_parent_path_of_file(), 
    "bias x": ["Bump", bias_factor], "bias y": ["Bump", bias_factor], 
    "bias z": ["Bump", bias_factor]})

    # Constructs the field

    generate_dataset(n_samples, limits_list, p_norm_list, results_path,
    displacement_file_name, young_modulus_file, mesh_file_name, cube_size,
    influence_radius, damping_factor, radius_power, mesh_data_class,
    lagrange_multiplier_file_name, save_snapshot_displacement=False)