# Routine to give a material parameter as a field

from .....Grante.MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from .....Grante.MultiMech.tool_box.expressions_tools import interpolate_scalar_function

from .....Grante.MultiMech.tool_box.read_write_tools import write_field_to_xdmf

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

# Creates a mesh for the field

H = 1.0

W = 0.2

mesh_data_class = read_mshMesh({"length x": 0.3, "length y": W, "len"+
"gth z": H, "number of divisions in x": 5, "number of divisions in y": 
5, "number of divisions in z": 25, "verbose": False, "mesh file name": 
"box_mesh", "mesh file directory": get_parent_path_of_file()})

# Creates a python function for the field

def E_function(position_vector):

    # Gets the coordinates

    x, y, z = position_vector

    # Returns the value of the Young modulus linearly varying across z

    return (1E6)+((y/W)*3E6)

# Interpolates this field into a finite element space

E_field, functional_data_class = interpolate_scalar_function(E_function, 
{"E": {"field type": "scalar", "interpolation function": "CG", "polyno"+
"mial degree":1}}, name="E", mesh_data_class=mesh_data_class)

# Saves this field into a xdmf file

write_field_to_xdmf(functional_data_class, directory_path=
get_parent_path_of_file(), visualization_copy=True, field_type="scalar",
interpolation_function="CG", polynomial_degree=1)

write_field_to_xdmf({"monolithic solution": E_field, "mesh file":
mesh_data_class.mesh_file}, visualization_copy=True, explicit_file_name=
get_parent_path_of_file()+"//E_from_dict", field_type="scalar", 
interpolation_function="CG", polynomial_degree=1)