# Routine to control the finite element analysis of a hyperelastic bar

import numpy as np

from scipy import sparse

from ....MultiMech.tool_box import plotting_tools

from . import variational_tools

from . import hyperelastic_1D_energies as hyperelastic_energy_densities

# Defines a function to set the data for the FEA of a hyperelastic bar

def FEA_bar():

    # Sets the bar length

    bar_length = 10.0

    # Sets the possible cross section areas

    cross_section_areas = [1.0, 1.0]

    # Sets the elements areas

    n_elements = 5

    elements_areas = [0 for i in range(int(np.floor(n_elements*0.5)))]

    elements_areas.extend([1 for i in range(int(np.floor(n_elements*0.5)
    ), n_elements)])

    # Sets the possible hyperelastic energy potentials

    E1 = 1E6

    E2 = 10E6

    helmholtz_potential_derivatives = [lambda C: hyperelastic_energy_densities.neo_hookean(
    C, E1), lambda C: hyperelastic_energy_densities.neo_hookean(C, E2)]

    # Sets the elements constitutive relations

    elements_constitutive_relations = [0 for i in range(int(np.floor(
    n_elements*0.5)))]

    elements_constitutive_relations.extend([1 for i in range(int(
    np.floor(n_elements*0.5)), n_elements)])

    # Sets the nodes coordinates

    nodes_coordinates = np.linspace(0.0, bar_length, n_elements+1)

    # Sets the nodes to be constrained

    constrained_nodes = [nodes_coordinates[0]]

    # Sets the body forces

    body_forces = [0.0]

    # Sets the elements body forces

    elements_body_forces = [0 for i in range(n_elements)]

    # Sets the tractions directly as lists of node and corresponding 
    # traction

    tractions = [[len(nodes_coordinates)-1, 2E6]]

    # Sets the number of load steps

    load_steps = 1

    # Sets the maximum number Newton-Raphson iterations

    max_newton_iterations = 15

    # Sets the convergence criterion

    convergence_criterion = 1E-4

    # Calls the Newton-Raphson procedure

    newton_raphson(load_steps, max_newton_iterations, 
    convergence_criterion, helmholtz_potential_derivatives, 
    cross_section_areas, elements_constitutive_relations, elements_areas, 
    nodes_coordinates, constrained_nodes, elements_body_forces, 
    body_forces, tractions)

# Defines a function to perform the Newton-Raphson scheme 

def newton_raphson(load_steps, max_newton_iterations, 
convergence_criterion, helmholtz_potential_derivatives, 
cross_section_areas, elements_constitutive_relations, elements_areas, 
nodes_coordinates, constrained_nodes, elements_body_forces, body_forces, 
tractions, initial_element=1, final_element=None, 
current_displacement_vector=None):
    
    if final_element is None:

        final_element = len(elements_constitutive_relations)

    if current_displacement_vector is None:

        n_dofs = final_element-initial_element+2

        current_displacement_vector = np.zeros(n_dofs)

    # Iterates through the load steps

    for i in range(load_steps):

        print("\nSets load step "+str(i+1)+"\n")

        # Iterates through the Newton-Raphson steps

        flag_convergence = False

        for j in range(max_newton_iterations):

            # Assembles the residual vector

            R = variational_tools.assemble_residue_vector(
            helmholtz_potential_derivatives, cross_section_areas, 
            current_displacement_vector, initial_element, final_element, 
            elements_constitutive_relations, elements_areas, 
            nodes_coordinates, constrained_nodes, elements_body_forces,
            body_forces, tractions, i, load_steps)

            residual_norm = np.linalg.norm(R)

            print("Initializes Newton iteration "+str(j+1)+" with resi"+
            "dual norm equals to "+str(format(residual_norm, ".5e")))

            # If the norm of the residual vector is less than the con-
            # vergence criterion, stops

            if residual_norm<=convergence_criterion:

                print("\nNewton-Raphson scheme converged at iteration "+
                str(j+1))

                flag_convergence = True

                break

            # Evaluates the stiffness matrix

            K = variational_tools.assemble_stiffness_matrix(
            helmholtz_potential_derivatives, cross_section_areas, 
            current_displacement_vector, initial_element, final_element, 
            elements_constitutive_relations, elements_areas, 
            nodes_coordinates, constrained_nodes)

            # Gets the incremental step

            lu_K = sparse.linalg.splu(sparse.csc_matrix(K))

            Delta_u = lu_K.solve(-R)

            current_displacement_vector += Delta_u

            #print("Delta_u=", Delta_u)

            #print("u=", current_displacement_vector, "\n")

        if not flag_convergence:

            raise InterruptedError("The Newton-Raphson scheme did not "+
            "converge")
        
        print("Converged u=", current_displacement_vector)

        # Plots the configuration

        plot_configurations(nodes_coordinates, 
        current_displacement_vector, title="deformed_configuration_loa"+
        "d_step_"+str(i+1))
        
# Defines a function to plot the deformed configuration against the re-
# ferential one

def plot_configurations(nodes_coordinates, displacement_vector, title=
"deformed_configuration"):

    # Assembles the y coordinates

    y_data = [[2.0 for i in range(len(nodes_coordinates))], [1.0 for (i
    ) in range(len(nodes_coordinates))]]

    # Assembles the x coordinates of the deformed configuration

    x_deformed = []

    for i in range(len(nodes_coordinates)):

        x_deformed.append(nodes_coordinates[i]+displacement_vector[i])

    # Assembles the x coordinates for both curves

    x_data = [nodes_coordinates, x_deformed]

    # Plots

    plotting_tools.plane_plot(title, x_data=x_data, y_data=y_data, 
    highlight_points=True, color_map="coolwarm", label=["Referential",
    "Deformed, $u\\left(X=L\\right)="+str(round(displacement_vector[-1], 
    ndigits=3))+"$"])

# Runs this code only if this file is explicitely run, not just imported

if __name__=="__main__":

    FEA_bar()