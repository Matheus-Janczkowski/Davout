# Routine to give a material parameter as a field

from .....Davout.MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.PythonicUtilities.stochastic_tools import get_random_point_on_elipsoid_surface

from .....Davout.PythonicUtilities.file_handling_tools import list_toTxt

from .....Davout.DeepMech.aa_tests.mecsol_2026_quotient_space.box_with_young_modulus_field import solve_BVP

from .....Davout.DeepMech.aa_tests.mecsol_2026_quotient_space.young_modulus_field_generator import generate_field

import numpy as np

from time import time

# Defines a function to generate the dataset of a cube

def generate_dataset(n_samples, limits_dict, p_norm, results_path,
displacement_file_name, young_modulus_file, mesh_file_name, cube_size,
influence_radius, damping_factor, radius_power, mesh_data_class, 
save_snapshot_displacement=False):

    # Gets the names of the variables

    variables_names = [key for key in limits_dict.keys()]

    variables_limits = [limits_dict[name] for name in variables_names]

    # Creates a dictionary of variable names and their column in the da-
    # ta matrix to guarantee ordering consistency (dictionaries do not
    # necessarily preserve order)

    variables_indices = {name: i for i, name in enumerate(
    variables_names)}

    # Generates the list of samples

    samples_list = get_random_point_on_elipsoid_surface(variables_limits, 
    p_norm_value=p_norm, number_of_samples=n_samples, return_as_list=
    True)

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

    # Iterates through the samples to evaluate the simulation

    for i in range(n_samples):

        # Gets the data for this sample

        sample_data = samples_list[i+1]

        # Generates the field of Young modulus

        generate_field(cube_size, sample_data[variables_indices["peak E"
        ]], sample_data[variables_indices["base E"]], influence_radius,
        damping_factor, radius_power, mesh_data_class, 
        get_parent_path_of_file(), file_name=young_modulus_file)

        try:

            # Calls the solution scheme

            solve_BVP(results_path, displacement_file_name, 
            get_parent_path_of_file()+"//"+young_modulus_file, 
            get_parent_path_of_file()+"//"+mesh_file_name, [[sample_data[
            variables_indices["P11"]], sample_data[variables_indices[
            "P12"]], sample_data[variables_indices["P13"]]], [
            sample_data[variables_indices["P21"]], sample_data[
            variables_indices["P22"]], sample_data[variables_indices[
            "P23"]]], [sample_data[variables_indices["P31"]], 
            sample_data[variables_indices["P32"]], sample_data[
            variables_indices["P33"]]]], save_snapshot=
            save_snapshot_displacement)

            # If the simulation was succesful, reads the displacement 
            # field and updates it to the succesful data

            succesful_displacement.append(np.load(results_path+"//"+
            displacement_file_name+".npy").tolist())

            succesful_data.append(sample_data)

            # Updates the list of completed simulations

            completed_simulations.append([i+1, True])

            # Saves the lists into txt files

            list_toTxt(succesful_displacement, "succesful_displacement"+
            "_matrix", parent_path=results_path)

            list_toTxt(succesful_data, "successful_data_matrix", 
            parent_path=results_path)

            list_toTxt(completed_simulations, "completed_simulations", 
            parent_path=results_path)

        except Exception as e:

            print("The "+str(i+1)+"-th simulation failed. The error me"+
            "ssage is:\n"+str(e))

            # Updates the counter of failed simulations

            n_failed_simulations += 1

            # Updates the list of completed simulations

            completed_simulations.append([i+1, False])

            # Saves this list into a txt file

            list_toTxt(completed_simulations, "completed_simulations", 
            parent_path=results_path)

        print("\n"+str(n_failed_simulations)+" simulations out of "+
        "a total of "+str(n_samples)+" failed\n")

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

# Testing block

if __name__=="__main__":

    # Sets the number of samples, stress limits, and the limits of the
    # modulus

    n_samples = 3

    limits_U11 = [-10E-1, 10E-1]

    limits_U12 = [-10E-2, 10E-2]

    limits_U13 = [-10E-2, 10E-2]

    limits_U22 = [-10E-1, 10E-1]

    limits_U23 = [-10E-2, 10E-2]

    limits_U33 = [-10E-1, 10E-1]

    limits_sigma11 = [-10E-1, 10E-1]

    limits_sigma12 = [-10E-2, 10E-2]

    limits_sigma13 = [-10E-2, 10E-2]

    limits_sigma22 = [-10E-1, 10E-1]

    limits_sigma23 = [-10E-2, 10E-2]

    limits_sigma33 = [-10E-1, 10E-1]

    limits_base_E = [12E6, 15E6]

    limits_maximum_E = [60E6, 65E6]

    limits_dict = {"U11": limits_U11, "U12": limits_U12, "U13": 
    limits_U13, "U22": limits_U22, "U23": limits_U23, "U33": limits_U33, 
    "base E": limits_base_E, "peak E": limits_maximum_E}

    # Sets the p of the L-p norm for the ellipsoid sampling

    p_norm = 16

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

    generate_dataset(n_samples, limits_dict, p_norm, results_path,
    displacement_file_name, young_modulus_file, mesh_file_name, cube_size,
    influence_radius, damping_factor, radius_power, mesh_data_class,
    save_snapshot_displacement=True)