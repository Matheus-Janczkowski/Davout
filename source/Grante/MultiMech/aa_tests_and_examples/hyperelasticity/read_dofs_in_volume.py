# Routine to give a material parameter as a field

from dolfin import *

from .....Grante.MultiMech.tool_box.mesh_handling_tools import read_mshMesh, find_dofs_in_volume

from .....Grante.MultiMech.tool_box.expressions_tools import interpolate_scalar_function

from .....Grante.MultiMech.tool_box.read_write_tools import read_field_from_xdmf, write_field_to_xdmf

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

# Creates a mesh for the field

H = 1.0

W = 0.2

L = 0.3

mesh_data_class = read_mshMesh({"length x": L, "length y": W, "length "+
"z": H, "number of divisions in x": 2, "number of divisions in y": 3, 
"number of divisions in z": 5, "verbose": False, "mesh file name": "bo"+
"x_mesh", "mesh file directory": get_parent_path_of_file(), "number of"+
" subdomains in z direction": 3})

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

# Reads the Young modulus field

E_field, functional_data_class = read_field_from_xdmf("e.xdmf", "box_m"+
"esh", {"E":{"field type": "scalar", "interpolation function": "CG", 
"polynomial degree":1}}, directory_path=get_parent_path_of_file(),
return_functional_data_class=True)

########################################################################
#                          DOFs from physical                          #
########################################################################

# Gets the domain DOFs related to the physical group 

DOFs_volume_1 = find_dofs_in_volume(mesh_data_class, 
functional_data_class, physical_group_name="volume 1")

print("There are "+str(len(DOFs_volume_1))+" DOFs in volume 1 using th"+
"e physical group name.\nThey are: "+str(DOFs_volume_1))

########################################################################
#                       DOFs from region function                      #
########################################################################

# Defines a function to return true if a point is within that region

def region_function(x, y, z):

    if (x>=0.0 and x<=L) and (y>=0.0 and y<=W) and (z>=0.0 and z<=H):

        return True 
    
    else:

        return False
    
DOFs_volume_1 = find_dofs_in_volume(mesh_data_class, 
functional_data_class, region_function=region_function)

print("\n\nThere are "+str(len(DOFs_volume_1))+" DOFs in volume 1 usin"+
"g a region function.\nThey are: "+str(DOFs_volume_1))

########################################################################
#                           Consistency test                           #
########################################################################

# To test if the right DOFs were picked, nullifies all other DOFs and 
# plots the result

for dof in range(len(E_field.vector()[:])):

    if not (dof in DOFs_volume_1):

        E_field.vector()[dof] = 0.0

# Updates the monolithic solution and saves the field again

functional_data_class.monolithic_solution = E_field

write_field_to_xdmf(functional_data_class, directory_path=
get_parent_path_of_file(), visualization_copy=True, field_type="scalar",
interpolation_function="CG", polynomial_degree=1, explicit_file_name=
get_parent_path_of_file()+"//E_with_volume_1_dofs")

# Prints on the screen the values of the requisite DOFs from the vector 
# of parameters

print(E_field.vector()[DOFs_volume_1])