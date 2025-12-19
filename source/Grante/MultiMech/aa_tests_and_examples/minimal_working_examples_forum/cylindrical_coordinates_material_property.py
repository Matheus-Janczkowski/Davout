# Routine to turn an expression in cylindrical coordinates into a field
# projected onto a finite element space 

import numpy as np

from .....Grante.MultiMech.tool_box import mesh_handling_tools

from .....Grante.MultiMech.tool_box.expressions_tools import interpolate_scalar_function

from .....Grante.MultiMech.tool_box.read_write_tools import write_field_to_xdmf

from .....Grante.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Grante.PythonicUtilities.coordinate_systems_tools import cartesian_to_cylindrical_coordinates

from .....Grante.PythonicUtilities.interpolation_tools import spline_1D_interpolation

# Defines the parametric curves for the circumferential variation of the
# material parameter using splines. The x points are the angles in a cy-
# lindrical coordinate system, and the y points are the property value at
# those angles

k_superior_parametric_curve = spline_1D_interpolation(x_points=[0.0, 
0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.25*np.pi, 1.5*np.pi, (1.7*
np.pi), 2*np.pi], y_points=[5.0, 5.5, 6.0, 6.5, 7.0, 6.5, 6.0, 5.5, 5.0])

k_inferior_parametric_curve = spline_1D_interpolation(x_points=[0.0, 
0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.25*np.pi, 1.5*np.pi, (1.7*
np.pi), 2*np.pi], y_points=[2.0, 2.5, 3.0, 3.5, 4.0, 3.5, 3.0, 2.5, 2.0])

# Gets the path to the mesh

mesh_path = (get_parent_path_of_file(path_bits_to_be_excluded=2)+"//te"+
"st_meshes//intervertebral_disc_mesh")

# Reads the mesh

mesh_data_class = mesh_handling_tools.read_mshMesh(mesh_path)

# Gets the nodes on the outer and inner lateral surfaces

tolerance_maximum = 5E-2

tolerance_minimum = 5E-2

outer_lateral_nodes = mesh_handling_tools.find_nodesOnSurface(
mesh_data_class, "lateral external", return_coordinates=True)

inner_lateral_nodes = mesh_handling_tools.find_nodesOnSurface(
mesh_data_class, "lateral internal", return_coordinates=True)

# Gets the maximum and minimum values of z

minimum_z_outer = min(outer_lateral_nodes[:,2])

maximum_z_outer = max(outer_lateral_nodes[:,2])

minimum_z_inner = min(inner_lateral_nodes[:,2])

maximum_z_inner = max(inner_lateral_nodes[:,2])

# Gets a normalization of the z values to be able to compare with the 
# angles during nearest neighbors

norm_factor_outer = (2*np.pi)/(maximum_z_outer-minimum_z_outer)

norm_factor_inner = (2*np.pi)/(maximum_z_inner-minimum_z_inner)

# Transforms the nodes on the lateral surfaces to cylindrical coordina-
# tes, and separates the radius from the angle and the z value

radius_outer = np.zeros(len(outer_lateral_nodes))

theta_z_outer = np.zeros((len(outer_lateral_nodes), 2))

radius_inner = np.zeros(len(inner_lateral_nodes))

theta_z_inner = np.zeros((len(inner_lateral_nodes), 2))

for i in range(len(outer_lateral_nodes)):

    x, y, z = outer_lateral_nodes[i]

    # Gets the cylindrical coordinates

    theta, r, z = cartesian_to_cylindrical_coordinates(x, y, z)

    # Updates the arrays of cylindrical coordinates

    radius_outer[i] = r 

    theta_z_outer[i,0] = theta 

    # Normalizes the 

    theta_z_outer[i,1] = z*norm_factor_outer

for i in range(len(inner_lateral_nodes)):

    x, y, z = inner_lateral_nodes[i]

    # Gets the cylindrical coordinates

    theta, r, z = cartesian_to_cylindrical_coordinates(x, y, z)

    # Updates the arrays of cylindrical coordinates

    radius_inner[i] = r 

    theta_z_inner[i,0] = theta 

    theta_z_inner[i,1] = z*norm_factor_inner

# Defines a material property function in cylindrical coordinates

def k_material(x_vector, current_physical_group=None):

    # Gets the coordinates

    x, y, z = x_vector

    # Gets the cylindrical coordinates

    theta, r, z = cartesian_to_cylindrical_coordinates(x, y, z)

    # Finds the points in this 

    # Gets the two points on the outer surface that are the closest to
    # these values of theta

    distances_outer_surface = np.linalg.norm(theta_z_outer-np.array([
    theta, z*norm_factor_outer]), axis=1)

    indexes_outer = np.argsort(distances_outer_surface)[:2]

    theta_1_outer = theta_z_outer[indexes_outer[0],0]

    theta_2_outer = theta_z_outer[indexes_outer[1],0]

    weight_outer = 0.0

    # If the angles are not the same

    if theta_2_outer!=theta_1_outer:

        weight_outer = ((theta_2_outer-theta)/(theta_2_outer-
        theta_1_outer))

    # Otherwise, normalizes by the z coordinate

    else:

        theta_1_outer = theta_z_outer[indexes_outer[0],1]

        theta_2_outer = theta_z_outer[indexes_outer[1],1]

        weight_outer = ((theta_2_outer-z*norm_factor_outer)/(
        theta_2_outer-theta_1_outer))

    closest_2_outer = radius_outer[indexes_outer]

    # Gets the two closest point on the inner surface

    distances_inner_surface = np.linalg.norm(theta_z_inner-np.array([
    theta, z*norm_factor_inner]), axis=1)

    indexes_inner = np.argsort(distances_inner_surface)[:2]

    theta_1_inner = theta_z_inner[indexes_inner[0],0]

    theta_2_inner = theta_z_inner[indexes_inner[1],0]

    weight_inner = 0.0

    # If the angles are not the same

    if theta_2_inner!=theta_1_inner:

        weight_inner = ((theta_2_inner-theta)/(theta_2_inner-
        theta_1_inner))

    # Otherwise, normalizes by the z coordinate

    else:

        theta_1_inner = theta_z_inner[indexes_inner[0],1]

        theta_2_inner = theta_z_inner[indexes_inner[1],1]

        weight_inner = ((theta_2_inner-z*norm_factor_inner)/(
        theta_2_inner-theta_1_inner))

    closest_2_inner = radius_inner[indexes_inner]

    # Linearly interpolates the 3 three points to get the extreme radii

    maximum_radius = ((closest_2_outer[0]*weight_outer)+(
    closest_2_outer[1]*(1-weight_outer)))

    minimum_radius = ((closest_2_inner[0]*weight_inner)+(
    closest_2_inner[1]*(1-weight_inner)))

    # Evaluates the limits of the k parameter in the current angle

    k_s = k_superior_parametric_curve(theta)

    k_i = k_inferior_parametric_curve(theta)

    # Verifies the limit radii

    if r>maximum_radius+tolerance_maximum:

        if current_physical_group=="annulus":

            raise NameError("The radius of the node is larger than the"+
            " corresponding maximum radius of the annulus region. Asse"+
            "rt tolerance or discretization")

        return 0.0
    
    elif r<minimum_radius-tolerance_minimum:

        if current_physical_group=="annulus":

            raise NameError("The radius of the node is smaller than th"+
            "e corresponding minimum radius of the annulus region. Ass"+
            "ert tolerance or discretization")

        return 0.0

    # Interpolates linearly across the radial direction

    k_value = ((k_s*((r-minimum_radius)/(maximum_radius-minimum_radius))
    )+(k_i*((maximum_radius-r)/(maximum_radius-minimum_radius))))

    return k_value

# Interpolates and gets the functional data class

u_interpolation, functional_data_class = interpolate_scalar_function(
k_material, {"k property": {"field type": "scalar", "interpolation fun"+
"ction": "CG", "polynomial degree": 1}}, mesh_data_class=mesh_data_class)

# Writes the xdmf file. Additional care is taken to secure it can be 
# load back into a fenics function later

write_field_to_xdmf(functional_data_class, directory_path=
get_parent_path_of_file(), visualization_copy=True)