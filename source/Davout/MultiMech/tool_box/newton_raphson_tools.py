# Routine to store methods to perform custom Newton-Raphson procedures

from dolfin import *

import numpy as np

from ...MultiMech.tool_box.functional_tools import set_nonlinearProblem

from ...PythonicUtilities.dictionary_tools import verify_obligatory_and_optional_keys

# Defines a class to perform the classical Newton-Raphson procedure

class NewtonRaphson:

    def __init__(self, n_load_steps, solver=None, variational_form=None,
    functional_data_class=None, boundary_conditions_list=None,
    solver_parameters=None, variational_form_update_function=None,
    post_processing_function=None, use_custom_newton=False,
    variational_form_for_differentiation=None):

        self.n_load_steps = n_load_steps

        self.variational_form_update_function = (
        variational_form_update_function)

        self.post_processing_function = post_processing_function

        # If no solver parameters were given, just put default values

        if solver_parameters is None:

            solver_parameters = {"newton_relative_tolerance": 1E-4, "n"+
            "ewton_absolute_tolerance": 1E-4, "newton_maximum_iteratio"+
            "ns": 25, "linear_solver": "mumps"}

        # Verifies if the solver was not previously built

        if (solver is None) and (not use_custom_newton):

            # Saves the flag for custom solver as false

            self.use_custom_newton = False

            # Checks if the functional data class, variational form, and
            # the list of boundary conditions was given

            if (variational_form is None) or (functional_data_class is (
            None)) or (boundary_conditions_list is None):

                raise ValueError("'solver' in 'NewtonRaphson' was prov"+
                "ided as None, but so was 'variational_form', 'functio"+
                "nal_data_class', or 'boundary_conditions_list'. In ca"+
                "se 'solver' is not provided, the other three have to "+
                "be")

            # Builds the solver

            self.solver = set_nonlinearProblem(variational_form, 
            functional_data_class, boundary_conditions_list, 
            solver_parameters=solver_parameters,
            special_residual_form_for_consistent_derivation=
            variational_form_for_differentiation)

            self.solver_method = self.solver.solve

        # If a custom Newton-Raphson algorithm is to be preferred

        elif use_custom_newton:

            # Saves the flag for custom solver as True

            self.use_custom_newton = True

            # Checks if the functional data class, variational form, and
            # the list of boundary conditions was given

            if (variational_form is None) or (functional_data_class is (
            None)) or (boundary_conditions_list is None):

                raise ValueError("'solver' in 'NewtonRaphson' was prov"+
                "ided as None, but so was 'variational_form', 'functio"+
                "nal_data_class', or 'boundary_conditions_list'. In ca"+
                "se 'solver' is not provided or 'use_custom_newton' is"+
                " True, the other three have to be")

            # Checks if the solver parameters have the necessary keys

            verify_obligatory_and_optional_keys(solver_parameters, ["n"+
            "ewton_relative_tolerance", "newton_absolute_tolerance",
            "newton_maximum_iterations", "linear_solver"], [], "solver"+
            "_parameters", "NewtonRaphson", check_not_expected_keys=
            False)

            self.solver_parameters = solver_parameters

            self.boundary_conditions_list = boundary_conditions_list

            self.functional_data_class = functional_data_class

            self.variational_form = variational_form

            # If there is a special residual form for the evaluation of
            # the jacobian of the residual, such in cases of dissipative
            # phenomena

            if variational_form_for_differentiation is not None:

                # Computes the Gateaux derivative of the variational 
                # form

                self.gateaux_derivative = derivative(
                variational_form_for_differentiation, 
                functional_data_class.monolithic_solution, 
                functional_data_class.trial_functions)

            # Otherwise, simply differentiates the same variational form

            else:

                # Computes the Gateaux derivative of the variational 
                # form

                self.gateaux_derivative = derivative(variational_form, 
                functional_data_class.monolithic_solution, 
                functional_data_class.trial_functions)

            # Sets the method that solves the system of equations 

            self.solver_method = self.custom_newton_solver

        # Otherwise, just saves the solver object

        else:

            # Saves the flag for custom solver as false

            self.use_custom_newton = False

            self.solver = solver

            self.solver_method = self.solver.solve

    def __call__(self):

        # Iterates over the load steps

        for i in range(self.n_load_steps):

            print("\n#################################################"+
            "########################\n#                       Starts "+
            "the "+str(i+1)+"-th load step                       #\n##"+
            "#########################################################"+
            "##############\n")

            # Evaluates the current pseudo time

            if self.n_load_steps>1:

                self.pseudo_time = (i/(self.n_load_steps-1))

            else:

                self.pseudo_time = 0.0

            # Update the variational form

            if (self.variational_form_update_function is not None) and (
            not self.use_custom_newton):

                # The second argument is a flag that informs that this
                # is an iteration outside of the Newton-Raphson procedure

                self.variational_form_update_function(self.pseudo_time,
                flag_newton_raphson_iteration=False)

            # Calls the solver

            self.solver_method()

            # Calls the function that carries out post-processing

            if self.post_processing_function is not None:

                self.post_processing_function(self.pseudo_time, i)

        # Informs which solver was used

        if self.use_custom_newton:

            print("\nThe custom Newton-Raphson implementation was used")

        else:

            print("\nThe custom Newton-Raphson implementation was not "+
            "used. Instead, the FEniCS NonlinearVariationalProblem was"+
            " deployed")

    # Defines a method to compute the Newton-Raphson iterations for a
    # single load-step

    def custom_newton_solver(self):

        # Creates a function for the temporary field used for updating
        # the desired field

        delta_u = Function(
        self.functional_data_class.monolithic_function_space)

        initial_residual_norm = None

        # Creates a list of homogeneous boundary conditions for the so-
        # lution field, since the boundary conditions will be applied to
        # the incremental correction

        self.homogeneous_boundary_conditions = []

        for bc in self.boundary_conditions_list:

            # Takes the object of the Dirichlet boundary condition and
            # homogenize it

            homogeneous_bc = DirichletBC(bc)

            homogeneous_bc.homogenize()

            self.homogeneous_boundary_conditions.append(homogeneous_bc)

        # Iterates over the Newton-Raphson iterations

        for i in range(self.solver_parameters["newton_maximum_iteratio"+
        "ns"]):

            # Updates the constitutive internal variables. The same 
            # function is used to update the variational form and even-
            # tual algorithmic tangent modulus. Boundary conditions and
            # other "externally" controlled stuff will not be modified
            # here since the pseudo time will be the same

            if self.variational_form_update_function is not None:

                # The second argument is a flag that informs that this
                # is a Newton-Raphson iteration

                self.variational_form_update_function(self.pseudo_time,
                flag_newton_raphson_iteration=True)

            # Assembles the residual and Jacobian

            residual_vector = assemble(self.variational_form)

            residual_vector *= -1.0

            tangent_matrix = assemble(self.gateaux_derivative)

            # Applies boundary conditions

            for bc in self.homogeneous_boundary_conditions:

                bc.apply(tangent_matrix, residual_vector)

            # Evaluates the residual vector's norm

            residual_norm = residual_vector.norm("l2")
            
            if not np.isfinite(residual_norm):

                raise RuntimeError("Newton-Raphson iteration diverged,"+
                " since NaN or Inf number were detected in the residua"+
                "l.")

            # Evaluates the relative residual norm

            if initial_residual_norm is None:

                initial_residual_norm = max(residual_norm, 1E-30)

            relative_residual = residual_norm/initial_residual_norm

            # Prints convergence information

            print("Newton iteration "+str(i)+": r (abs) = "+str(format(
            residual_norm, '.3e'))+" (tol = "+str(format(
            self.solver_parameters["newton_absolute_tolerance"], '.3e'))
            +") r (rel) = "+str(format(relative_residual, '.3e'))+" (t"+
            "ol = "+str(format(self.solver_parameters["newton_relative"+
            "_tolerance"], '.3e'))+")")

            # Checks convergence

            if residual_norm<self.solver_parameters["newton_absolute_t"+
            "olerance"] or relative_residual<self.solver_parameters["n"+
            "ewton_relative_tolerance"]:
                
                return i, True

            # Solves the Newton-Raphson system

            delta_u.vector().zero()

            solve(tangent_matrix, delta_u.vector(), residual_vector, 
            self.solver_parameters["linear_solver"])

            # Updates the solution vector

            self.functional_data_class.monolithic_solution.vector(
            ).axpy(1.0, delta_u.vector())

        raise ValueError("The custom Newton-Raphson solver tried to mi"+
        "nimize the residual for "+str(self.solver_parameters["newton_"+
        "maximum_iterations"])+" iterations to no avail")