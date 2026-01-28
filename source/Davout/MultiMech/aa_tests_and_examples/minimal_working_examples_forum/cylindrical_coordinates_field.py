# Routine to turn an expression in cylindrical coordinates into a field
# projected onto a finite element space 

from dolfin import *

import numpy as np

from .....Grante.MultiMech.tool_box import mesh_handling_tools

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Grante.MultiMech.tool_box.expressions_tools import interpolate_scalar_function

from .....Grante.MultiMech.tool_box.expressions_tools import interpolate_tensor_function

# Gets the path to the mesh

mesh_path = (get_parent_path_of_file(path_bits_to_be_excluded=2)+"//te"+
"st_meshes//malha")

# Reads the mesh

mesh_data_class = mesh_handling_tools.read_mshMesh(mesh_path)

# Creates the functions spaces

W = FunctionSpace(mesh_data_class.mesh, "CG", 1)

V = VectorFunctionSpace(mesh_data_class.mesh, "CG", 1)

# Tests a sine function

def sine(x):

    return np.sin(x[0]*x[1]*1.0)

def vector_sine(x, component):

    return np.array([np.sin(x[0]), np.sin(x[1]), np.sin(x[2])])[component]

    # Evalutes the maximum radius

u_interpolation = interpolate_scalar_function(sine, W)

v_interpolation = interpolate_tensor_function(vector_sine, V)

file = XDMFFile(get_parent_path_of_file()+"//sine_interpolation.xdmf")

file.write(u_interpolation)

file = XDMFFile(get_parent_path_of_file()+"//sine_vector_interpolation.xdmf")

file.write(v_interpolation)