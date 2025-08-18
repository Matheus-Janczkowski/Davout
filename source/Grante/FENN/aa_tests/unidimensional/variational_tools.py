# Routine to integrate the stiffness matrix of a 1D elasticity problem

import numpy as np

from scipy import sparse

########################################################################
#                           Stiffness matrix                           #
########################################################################

# Defines a function to integrate the stiffness matrix over a Gauss point

def integrate_K_on_Gauss_point(ksi, helmholtz_potential_derivative, 
interpolation_function_gradient, local_displacement_vector, jacobian):
    
    # Evaluates the derivative of the interpolation functions

    B_bar = interpolation_function_gradient(ksi)*(1/jacobian)

    # Evaluates the deformation gradient

    F = np.matmul(np.transpose(B_bar), local_displacement_vector)[0]+1.0

    # Evaluates the right Cauchy-Green strain tensor

    C = F*F 

    # Evaluates the derivative of the Helmholtz potential with respect 
    # to the right Cauchy-Green strain tensor, both the first and the 
    # second derivatives

    dPsi, ddPsi = helmholtz_potential_derivative(C)

    # Computes the derivative of the first Piola-Kirchhoff with respect
    # to the deformation gradient

    C_bar = (4*ddPsi*C)+(2*dPsi)

    # Evaluates the integrand on this Gauss point and returns it

    return np.matmul(B_bar, C_bar*np.transpose(B_bar))

# Defines a function to assemble the stiffness matrix

def assemble_stiffness_matrix(helmholtz_potential_derivatives, 
cross_section_areas, current_displacement_vector, initial_element, 
final_element, elements_constitutive_relations, elements_areas, 
nodes_coordinates, constrained_nodes):

    # Evaluates the number of DOFs

    n_dofs = final_element-initial_element+2

    # Initializes the matrix

    K = sparse.lil_matrix((n_dofs, n_dofs))

    # If the current displacement vector is None, initializes it

    if current_displacement_vector is None:

        current_displacement_vector = np.zeros(n_dofs)

    # Iterates through the elements

    for i in range(initial_element-1, final_element):

        # Evaluates the element length

        element_length = nodes_coordinates[i+1]-nodes_coordinates[i]

        # Gets the element constitutive relation

        elements_constitutive_relation = elements_constitutive_relations[
        i]

        # Gets the element's cross section area

        element_area = elements_areas[i]

        # Gets the local displacement vector

        dof1 = i+0

        dof2 = i+1

        local_displacement_vector = current_displacement_vector[dof1:(
        dof2+1)]

        # Evaluates the integrand

        K_local = (element_length*cross_section_areas[element_area]*
        integrate_K_on_Gauss_point(0.0, helmholtz_potential_derivatives[
        elements_constitutive_relation], lambda ksi: np.array([[-0.5], [
        0.5]]), local_displacement_vector, 0.5*element_length))

        """print(K_local)

        print("")"""

        # Sums this local stiffness matrix to the global one

        if not (dof1 in constrained_nodes):

            K[dof1, dof1] += K_local[0,0]

        else:

            K[dof1, dof1] = 1.0

        if (not (dof1 in constrained_nodes)) and (not (dof2 in (
        constrained_nodes))):

            K[dof1, dof2] += K_local[0,1]

            K[dof2, dof1] += K_local[1,0]

        if not (dof2 in constrained_nodes):

            K[dof2, dof2] += K_local[1,1]

        else:

            K[dof2, dof2] = 1.0

    return K

########################################################################
#                               Residual                               #
########################################################################

# Defines a function to integrate the residual vector over a Gauss point

def integrate_R_on_Gauss_point(ksi, helmholtz_potential_derivative, 
interpolation_function_gradient, interpolation_function,
local_displacement_vector, element_body_force, jacobian):
    
    # Evaluates the interpolation functions

    N_bar = interpolation_function(ksi)
    
    # Evaluates the derivative of the interpolation functions

    B_bar = interpolation_function_gradient(ksi)*(1/jacobian)

    # Evaluates the deformation gradient

    F = np.matmul(np.transpose(B_bar), local_displacement_vector)[0]+1.0

    # Evaluates the right Cauchy-Green strain tensor

    C = F*F 

    # Evaluates the derivative of the Helmholtz potential with respect 
    # to the right Cauchy-Green strain tensor, both the first and the 
    # second derivatives

    dPsi, ddPsi = helmholtz_potential_derivative(C)

    # Computes the first Piola-Kirchhoff 

    P = 2*F*dPsi

    #print("P=", P)

    # Evaluates the integrand on this Gauss point and returns it

    return (B_bar*P)-(N_bar*element_body_force)

# Defines a function to assemble the residue vector

def assemble_residue_vector(helmholtz_potential_derivatives, 
cross_section_areas, current_displacement_vector, initial_element, 
final_element, elements_constitutive_relations, elements_areas, 
nodes_coordinates, constrained_nodes, elements_body_forces,
body_forces, tractions, load_step, n_steps):
    
    # Evaluates the load step

    load_delta = 1.0

    if n_steps>1:

        load_delta = (load_step/(n_steps-1))

    # Evaluates the number of DOFs

    n_dofs = final_element-initial_element+2

    # Initializes the residue vector

    R = np.zeros(n_dofs)

    # Iterates through the elements

    for i in range(initial_element-1, final_element):

        # Evaluates the element length

        element_length = nodes_coordinates[i+1]-nodes_coordinates[i]

        # Gets the element constitutive relation

        elements_constitutive_relation = elements_constitutive_relations[
        i]

        # Gets the element's cross section area

        element_area = elements_areas[i]

        # Gets the element's body force

        element_body_force = elements_body_forces[i]

        # Gets the local displacement vector

        dof1 = i+0

        dof2 = i+1

        local_displacement_vector = current_displacement_vector[dof1:(
        dof2+1)]

        # Evaluates the integrand

        R_local = (element_length*cross_section_areas[element_area]*
        integrate_R_on_Gauss_point(0.0, helmholtz_potential_derivatives[
        elements_constitutive_relation], lambda ksi: np.array([[-0.5], [
        0.5]]), lambda ksi: np.array([[0.5*(1-ksi)], [0.5*(1+ksi)]]),
        local_displacement_vector, body_forces[element_body_force]*
        load_delta, 0.5*element_length))

        #print(R_local)

        # Sums this local stiffness matrix to the global one

        if not (dof1 in constrained_nodes):

            R[dof1] += R_local[0]

        if not (dof2 in constrained_nodes):

            R[dof2] += R_local[1]

    #print("R=", R)

    # Adds the tractions to the residual

    for traction in tractions:

        # Gets the node and the traction value

        node, T = traction

        # Updates the residual vector

        R[node] -= T*load_delta

    #print("R=", R)

    return R