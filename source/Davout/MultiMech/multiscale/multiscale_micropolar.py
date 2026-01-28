# Routine to store the multiscale methods for micropolar physics

from dolfin import *

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import variational_tools

from ..tool_box import functional_tools

from ..tool_box import pseudotime_stepping_tools as newton_raphson_tools

from ...PythonicUtilities import programming_tools

from ..tool_box import multiscale_boundary_conditions_tools as multiscale_BCsTools

# Defines a function to model a hyperelastic problem with a displacement
# and a microrotation fields only in the microscale. It uses the macro
# quantities read from txt files

@programming_tools.optional_argumentsInitializer({'neumann_loads': 
lambda: [], 'dirichlet_loads': lambda: [], 'solution_name': lambda: []})

def micropolar_microscale(displacement_multiscaleBC, 
microrotation_multiscaleBC, macro_displacementFileName, 
macro_gradDisplacementFileName, macro_microrotationFileName, 
macro_gradMicrorotationFileName, constitutive_model, post_processes, 
mesh_fileName, solver_parameters, polynomial_degreeDisplacement=2, 
polynomial_degreeMicrorotation=2, solution_name=None, verbose=False, 
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

    # Assembles a dictionary of finite elements for the two primal 
    # fields: displacement and microrotation. Each field has a key and
    # the corresponding value is another dictionary, which has keys for
    # necessary information to create finite elements

    elements_dictionary = {"Displacement": {"field type": "vector", "i"+
    "nterpolation function": "CG", "polynomial degree": 
    polynomial_degreeDisplacement}, "Microrotation": {"field type": "v"+
    "ector", "interpolation function": "CG", "polynomial degree": 
    polynomial_degreeMicrorotation}}

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

    multiscale_BCsDict["Microrotation"] = {"boundary condition": 
    microrotation_multiscaleBC, "macro information": {"macro field fil"+
    "e": macro_microrotationFileName, "macro field gradient file":
    macro_gradMicrorotationFileName}}

    # Calls up the function to automatically select the boundary condi-
    # tions

    (bilinear_form, linear_form, boundary_conditions, 
    macro_quantitiesClasses, fields_namesDict, solution_fields, 
    variation_fields, trial_functions, monolithic_solution, 
    mixed_element, volume_inverse, fields_corrections
    ) = multiscale_BCsTools.select_multiscaleBoundaryConditions(
    multiscale_BCsDict, elements_dictionary, mesh_dataClass, 
    fluctuation_field=fluctuation_field)

    ####################################################################
    #                         Variational forms                        #
    ####################################################################

    # Constructs the variational form for the inner work

    internal_VarForm = variational_tools.hyperelastic_micropolarInternalWorkFirstPiola(
    "Displacement", "Microrotation", solution_fields, variation_fields, 
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

    # The micropolar microscale problem is by definition a mixed pro-
    # blem (displacement and microrotation fields). Thus, there is no 
    # need to test whether the solution has multiple fields in it

    newton_raphson_tools.newton_raphsonMultipleFields(solver, 
    monolithic_solution, fields_namesDict, mesh_dataClass, 
    constitutive_model, post_processesList=post_processes, solution_name=
    solution_name, macro_quantitiesClasses=macro_quantitiesClasses, 
    fields_corrections=fields_corrections)