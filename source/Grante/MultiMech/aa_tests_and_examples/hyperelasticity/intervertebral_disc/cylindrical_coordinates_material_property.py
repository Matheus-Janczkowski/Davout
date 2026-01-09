# Routine to turn an expression in cylindrical coordinates into a field
# projected onto a finite element space 

from dolfin import *

import numpy as np

from ......Grante.MultiMech.tool_box import mesh_handling_tools

from ......Grante.MultiMech.tool_box.expressions_tools import interpolate_scalar_function

from ......Grante.MultiMech.tool_box.read_write_tools import write_field_to_xdmf

from ......Grante.PythonicUtilities.coordinate_systems_tools import cartesian_to_cylindrical_coordinates

from ......Grante.PythonicUtilities.interpolation_tools import spline_1D_interpolation

# Defines the parametric curves for the circumferential variation of the
# material parameter using splines. The x points are the angles in a cy-
# lindrical coordinate system, and the y points are the property value at
# those angles

def properties_disc_interpolation(y_points_sup, y_points_inf, name_property, parent_path, field_file_name):

    k_superior_parametric_curve = spline_1D_interpolation(x_points=[0.0, 
    0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.25*np.pi, 1.5*np.pi, (1.75* 
    np.pi), 2*np.pi], y_points= y_points_sup)

    k_inferior_parametric_curve = spline_1D_interpolation(x_points=[0.0, 
    0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi, 1.25*np.pi, 1.5*np.pi, (1.75*
    np.pi), 2*np.pi], y_points=y_points_inf)

    # Gets the path to the mesh

    mesh_path = parent_path + "/intervertebral_disc_mesh" 

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
    k_material, {str(name_property): {"field type": "scalar", "interpolation fun"+
    "ction": "CG", "polynomial degree": 1}}, mesh_data_class=mesh_data_class)

    # Writes the xdmf file. Additional care is taken to secure it can be 
    # load back into a fenics function later. The flag 'visualization_
    # copy' is True to save a .xdmf copy of the field using the conven-
    # tional write method, this copy will not crash inside paraview. But,
    # it is meant for visualization only, and can be deleted later

    write_field_to_xdmf(functional_data_class, visualization_copy=True, 
    explicit_file_name=parent_path+"//"+field_file_name, field_type="scalar",
    interpolation_function="CG", polynomial_degree=1)

def sup_inf_parameters(base_value_parameter, percentage_circumferential_variation,
percentage_radial_variation, field_name, parent_path, file_name):

    """base_value_parameter: indicates the base value relating to the position A-1 where 1 is the most external part
       percentage_circumferential_variation: percentage value thats increments the circumferential position (A-B)
       percentage_radial_variation: percentage base value (layer 1-2 - most external part)"""

    # Starts the lists that will be filled

    sup_parameter = []
    inf_parameter = []

    # Defines a list containing the percentages incremets 

    percentage_circumferential_list = []

    # It is necessary only the first and the last layer
    # of the annulus fibrosus, because the middle will
    # be interpolated

    percentage_radial_to_layer5 = percentage_radial_variation*4

    # Defines a loop to get the percentages in each 
    # circumferential position

    for i in range(4):

        percentage_circumferential_list.append(percentage_circumferential_variation*(i+1))    

    # Defines the list values of the interior layer
    #  of the disc

    ABCDE_layer1 = []

    ABCDE_layer1.append(base_value_parameter)

    # Defines a loop to put all the new calculated 
    # values in the list 

    for j in percentage_circumferential_list:

        ABCDE_layer1.append(base_value_parameter+(base_value_parameter*j))
    
    ABCDE_layer5 = []

    # Defines a loop to put the values in respect 
    # of the internal (inferior) layer

    for k in ABCDE_layer1:

        ABCDE_layer5.append(k+(k*percentage_radial_to_layer5))

    # Put the calculated values in order, according
    # to the preset angle sequence of the disc 
    
    sup_parameter.append(ABCDE_layer1[2])
    sup_parameter.append(ABCDE_layer1[3])
    sup_parameter.append(ABCDE_layer1[4])
    sup_parameter.append(ABCDE_layer1[3])
    sup_parameter.append(ABCDE_layer1[2])
    sup_parameter.append(ABCDE_layer1[1])
    sup_parameter.append(ABCDE_layer1[0])
    sup_parameter.append(ABCDE_layer1[1])
    sup_parameter.append(ABCDE_layer1[2])

    inf_parameter.append(ABCDE_layer5[2])
    inf_parameter.append(ABCDE_layer5[3])
    inf_parameter.append(ABCDE_layer5[4])
    inf_parameter.append(ABCDE_layer5[3])
    inf_parameter.append(ABCDE_layer5[2])
    inf_parameter.append(ABCDE_layer5[1])
    inf_parameter.append(ABCDE_layer5[0])
    inf_parameter.append(ABCDE_layer5[1])
    inf_parameter.append(ABCDE_layer5[2])

    print(file_name)
    print(parent_path)

    properties_disc_interpolation(sup_parameter, inf_parameter,
    field_name, parent_path, file_name)

    return sup_parameter, inf_parameter


if __name__=="__main__":

    #parent_path = "/home/grante_maximus/rafael_prado_ic/TCC_rafael_prado/IntervertebralDiscANN/material_properties_interpolation"
 
    # source.Grante.MultiMech.aa_tests_and_examples.hyperelasticity.intervertebral_disc.cylindrical_coordinates_material_property

    parent_path = "source/Grante/MultiMech/aa_tests_and_examples/hyperelasticity/intervertebral_disc"

    k1_sup, k1_inf = sup_inf_parameters(2.88, -0.13, -0.15, "k1", parent_path, "k1_000")

    k2_sup, k2_inf = sup_inf_parameters(61.86, -0.19, -0.03, "k2", parent_path, "k2_000")

    gamma_sup, gamma_inf = sup_inf_parameters(36.73, 0.07, 0.03, "gamma", parent_path, "gamma_000")