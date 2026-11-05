# Routine to experiment with quadrature elements to allow for implemen-
# tations concerned with algorithmic procedures in the quadrature points

from dolfin import *

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from .....Davout.MultiMech.constitutive_models.hyperelasticity.isotropic_hyperelasticity import NeoHookean

from .....Davout.MultiMech.tool_box import functional_tools

from .....Davout.MultiMech.tool_box import variational_tools

from .....Davout.MultiMech.tool_box.read_write_tools import write_field_to_xdmf

# Creates a box mesh using gmsh

mesh_data_class = read_mshMesh({"length x": 0.3, "length y": 0.2, "len"+
"gth z": 1.0, "number of divisions in x": 5, "number of divisions in y":
5, "number of divisions in z": 25, "verbose": False, "mesh file name": 
"box_mesh", "mesh file directory": get_parent_path_of_file()})

# Neumann boundary conditions

maximum_load = 5E5

# Sets the constitutive model

E = 1E6

poisson = 0.3

constitutive_models = {"volume 1": NeoHookean({"E": E, "nu": poisson})}

# Sets the function space for the displacement field

functional_data_class = functional_tools.construct_monolithicFunctionSpace(
{"Displacement": {"field type": "vector", "interpolation function": "C"+
"G", "polynomial degree": 2}, "Pressure": {"field type": "scalar", "in"+
"terpolation function": "CG", "polynomial degree": 1}}, mesh_data_class,
verbose=True)

# Sets the function space for a vector-valued internal variable and ano-
# ther scalar-valued internal variable. They are both found in quadra-
# ture points alone. Note that 'polynomial degree' stands for quadrature
# degree in the case of quadrature finite elements

quadrature_functional_data_class = functional_tools.construct_monolithicFunctionSpace(
{"vector internal variable": {"field type": "vector", "interpolation f"+
"unction": "Quadrature", "polynomial degree": 2, "dimension": 3}, 
"scalar internal variable": {"field type": "scalar", "interpolation fu"+
"nction": "Quadrature", "polynomial degree": 2}}, mesh_data_class,
verbose=True)