# Routine to give a material parameter as a field

from ....MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from ....MultiMech.tool_box.expressions_tools import interpolate_scalar_function

from ....MultiMech.tool_box.read_write_tools import write_field_to_xdmf

from ....PythonicUtilities.path_tools import get_parent_path_of_file

from ....GraphUtilities.paraview_tools import frozen_snapshots

import numpy as np

# Defines a function to generate a field of Young modulus

def generate_field(cube_size, E_extremum, E_base, influence_radius, 
damping_factor, radius_power, mesh_data_class, parent_path, 
save_snapshot=False):

    # Creates a python function for the field

    def E_function(position_vector):

        # Gets the coordinates

        x, y, z = position_vector

        # Gets the radius from the center

        r = np.sqrt(((x-(0.5*cube_size))**2)+((y-(0.5*cube_size))**2)+((z-(
        0.5*cube_size))**2))

        # Gets the exponential interpolation

        return E_base+((E_extremum-E_base)*np.exp(-damping_factor*((r/
        influence_radius)**radius_power)))

    # Interpolates this field into a finite element space

    E_field, functional_data_class = interpolate_scalar_function(
    E_function, {"E": {"field type": "scalar", "interpolation function": 
    "CG", "polynomial degree": 1}}, name="E", mesh_data_class=
    mesh_data_class)

    # Saves this field into a xdmf file

    write_field_to_xdmf(functional_data_class, directory_path=
    parent_path, visualization_copy=True, field_type="scalar",
    interpolation_function="CG", polynomial_degree=1)

    write_field_to_xdmf({"monolithic solution": E_field, "mesh file":
    mesh_data_class.mesh_file}, visualization_copy=True, 
    explicit_file_name=parent_path+"//E_from_dict", field_type="scalar", 
    interpolation_function="CG", polynomial_degree=1)

    # Saves a snapshot of the solution using the automatization of ParaView

    if save_snapshot:

        frozen_snapshots("E_from_dict_visualization_copy.xdmf", "E", 
        input_path=parent_path, 
        camera_position=[1.352147972653376, 0.8655841453703876, 0.820538687418374],
        camera_focal_point=[0.1091743049381585, 0.210624339846976, 0.17486914104493073],
        camera_up_direction=[-0.37508506619984744, -0.18385590773777907, 0.9085748171195223],
        camera_parallel_scale=0.40019526483955303,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.7160471545405493, 0.15249999999999997],
        legend_bar_length=0.7000000000000003,
        size_in_pixels={'aspect ratio': 0.7117437722419929, 'pixels in width': 562},
        axes_color=[0.0, 0.0, 0.0], get_attributes_render=False, 
        output_imageFileName="young_modulus_field.png", 
        resolution_ratio=5, clip=True, clip_plane_origin=[0.5*cube_size, 
        0.5*cube_size, 0.5*cube_size], clip_plane_normal_vector=[1.0, 
        2.0, 0.0], representation_type="Surface With Edges", 
        set_camera_interactively=False)

# Testing block

if __name__=="__main__":

    # Sets the parent path to the files

    parent_path = get_parent_path_of_file()

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

    generate_field(cube_size, E_extremum, E_base, influence_radius, 
    damping_factor, radius_power, mesh_data_class, parent_path, 
    save_snapshot=True)