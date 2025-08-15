# Routine to store the multiscale methods for micropolar physics

from dolfin import *

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import variational_tools

from ..tool_box import pseudotime_stepping_tools as newton_raphson_tools

from ..tool_box import programming_tools

from ..tool_box import functional_tools

from ..tool_box import multiscale_boundary_conditions_tools as multiscale_BCsTools

# Defines a function to model a hyperelastic problem with a displacement
# only in the microscale. It uses the macro quantities read from txt fi-
# les

@programming_tools.optional_argumentsInitializer({'neumann_loads': 
lambda: [], 'dirichlet_loads': lambda: [], 'solution_name': lambda: []})

def hyperelastic_microscale(displacement_multiscaleBC, 
macro_displacementFileName, macro_gradDisplacementFileName, 
constitutive_model, post_processes, mesh_fileName, solver_parameters, 
polynomial_degree=2, solution_name=None, verbose=False, 
fluctuation_field=False):

    ####################################################################
    #                               Mesh                               #
    ####################################################################

    # Reads the mesh and constructs some fenics objects using the xdmf 
    # file

    mesh_dataClass = mesh_tools.read_mshMesh(mesh_fileName, verbose=
    verbose)

    ####################################################################
    #                          Function space                          #
    ####################################################################

    # Assembles a dictionary of finite elements for the one and primal 
    # field: displacement. The field has a key and the corresponding va-
    # lue is another dictionary, which has keys for necessary informa-
    # tion to create finite elements

    elements_dictionary = {"Displacement": {"field type": "vector", "i"+
    "nterpolation function": "Lagrange", "polynomial degree": 
    polynomial_degree}}

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Builds a dictionary of multiscale boundary conditions. Each key is
    # the name of the field to be constrained, whereas the value is a
    # dictionary with two mandatory keys: "boundary condition" that sta-
    # tes the type of boundary condition to be applied and has the name
    # of one of the classes defined in multiscale_classes.py; "macro in-
    # formation", that is a dictionary of the macro file paths

    multiscale_BCsDict = dict()

    multiscale_BCsDict["Displacement"] = {"boundary condition": 
    displacement_multiscaleBC, "macro information": {"macro field file":
    macro_displacementFileName, "macro field gradient file":
    macro_gradDisplacementFileName}}

    # Calls up the function to automatically select the boundary condi-
    # tions

    (bilinear_form, linear_form, boundary_conditions, 
    macro_quantitiesClasses, fields_namesDict, solution_fields, 
    variation_fields, trial_functions, monolithic_solution, 
    mixed_element, volume_inverse, fields_corrections) = multiscale_BCsTools.select_multiscaleBoundaryConditions(
    multiscale_BCsDict, elements_dictionary, mesh_dataClass, 
    fluctuation_field=fluctuation_field)

    ####################################################################
    #                         Variational forms                        #
    ####################################################################

    # Constructs the variational form for the inner work

    internal_VarForm = variational_tools.hyperelastic_internalWorkFirstPiola(
    "Displacement", solution_fields, variation_fields, 
    constitutive_model, mesh_dataClass)

    ####################################################################
    #              Problem and solver parameters setting               #
    ####################################################################

    # Assembles the residual and the nonlinear problem object. Sets the
    # solver parameters too

    residual_form = ((volume_inverse*internal_VarForm)+bilinear_form-
    linear_form)

    solver = functional_tools.set_nonlinearProblem(residual_form, 
    monolithic_solution, trial_functions, boundary_conditions, 
    solver_parameters=solver_parameters)

    ####################################################################
    #                 Solution and pseudotime stepping                 #
    ####################################################################

    # Iterates through the pseudotime stepping algortihm 

    if len(solution_name)==0:

        for field_name in fields_namesDict:

            solution_name.append([field_name, "Microscale"])

    # The hyperelastic microscale problem is, in nature, a single-field
    # problem. But, if minimally constrained boundary condition is used,
    # the problems gains one field for each Lagrange multiplier. Thus,
    # the solution of the problem is given to an algorithms that can 
    # handle multiple fields. 
    # If the solution, however, does not have multiple fields, the fol-
    # lowing function will automatically reconnect with the single-field
    # framework

    newton_raphson_tools.newton_raphsonMultipleFields(solver, 
    monolithic_solution, fields_namesDict, mesh_dataClass, 
    constitutive_model, post_processesList=post_processes, solution_name=
    solution_name, macro_quantitiesClasses=macro_quantitiesClasses, 
    fields_corrections=fields_corrections)