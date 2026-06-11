# Routine to give a material parameter as a field

from ....MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from ....PythonicUtilities.path_tools import get_parent_path_of_file

from ....PythonicUtilities.stochastic_tools import get_random_point_on_elipsoid_surface

from ....PythonicUtilities.file_handling_tools import list_toTxt

# Defines a function to generate the dataset of a cube

def generate_dataset(n_samples, limits_dict, p_norm):

    # Gets the names of the variables

    variables_names = [key for key in limits_dict.keys()]

    variables_limits = [limits_dict[name] for name in variables_names]

    # Generates the list of samples

    samples_list = get_random_point_on_elipsoid_surface(variables_limits, 
    p_norm_value=p_norm, number_of_samples=n_samples)

    # Adds a header and saves the list of samples in a txt file

    samples_list.insert(0, variables_names)

    list_toTxt(samples_list, "dataset", parent_path=
    get_parent_path_of_file())

# Testing block

if __name__=="__main__":

    # Sets the number of samples, stress limits, and the limits of the
    # modulus

    n_samples = 500

    limits_P11 = [-10E6, 10E6]

    limits_P12 = [-10E6, 10E6]

    limits_P13 = [-10E6, 10E6]

    limits_P21 = [-10E6, 10E6]

    limits_P22 = [-10E6, 10E6]

    limits_P23 = [-10E6, 10E6]

    limits_P31 = [-10E6, 10E6]

    limits_P32 = [-10E6, 10E6]

    limits_P33 = [-10E6, 10E6]

    limits_base_E = [12E6, 15E6]

    limits_maximum_E = [60E6, 65E6]

    limits_dict = {"P11": limits_P11, "P12": limits_P12, "P13": 
    limits_P13, "P21": limits_P21, "P22": limits_P22, "P23": limits_P23, 
    "P31": limits_P31, "P32": limits_P32, "P33": limits_P33, "base E": 
    limits_base_E, "peak E": limits_maximum_E}

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

    # Constructs the mesh

    mesh_data_class = read_mshMesh({"length x": cube_size, "length y": 
    cube_size, "length z": cube_size, "number of divisions in x": 
    n_divisions, "number of divisions in y": n_divisions, "number of d"+
    "ivisions in z": n_divisions, "verbose": False, "mesh file name": 
    "box_mesh_mecsol", "mesh file directory": get_parent_path_of_file(), 
    "bias x": ["Bump", bias_factor], "bias y": ["Bump", bias_factor], 
    "bias z": ["Bump", bias_factor]})

    # Constructs the field

    generate_dataset(n_samples, limits_dict, p_norm)