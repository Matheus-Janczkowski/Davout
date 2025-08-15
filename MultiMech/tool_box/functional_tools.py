# Routine to store functions to help with functional analysis tasks and
# programmatic tools

from dolfin import *

import numpy as np

from abc import ABC, abstractmethod

from petsc4py import PETSc

from ..tool_box import file_handling_tools as file_tools

from ..tool_box import numerical_tools

from ..tool_box import programming_tools

from ..tool_box import boundary_conditions_tools as bc_tools

from ..tool_box import dirichlet_load_tools as dirichlet_tools

########################################################################
#                     Boundary conditions selector                     #
########################################################################

# Defines a function to get from a dictionary of dictionaries the boun-
# dary conditions. The key is the physical group, whereas the value is a
# dictionary of information to construct the boundary condition using o-
# ne of the methods below

@programming_tools.optional_argumentsInitializer({'boundary_conditions':
lambda: [], 'dirichlet_loads': lambda: []})

def construct_DirichletBCs(boundary_conditionsDict, fields_namesDict,
monolithic_functionSpace, mesh_dataClass, boundary_conditions=None,
dirichlet_loads=None):

    # Initializes a dictionary of functions that generate boundary con-
    # ditions

    bcs_functionsDict = programming_tools.dispatch_functions([], 
    bc_tools)[1]

    # Initializes a dictionary of functions that generate special cases
    # of fancier boundary conditions

    complex_bcsFunctionsDict = programming_tools.dispatch_functions([], 
    dirichlet_tools)[1]

    # Initializes a dictionary with common arguments to the boundary 
    # condition generation functions

    method_arguments = {"field_functionSpace": monolithic_functionSpace, 
    "mesh_dataClass": mesh_dataClass, "fields_namesDict": 
    fields_namesDict, "complex_bcsFunctionsDict": 
    complex_bcsFunctionsDict}

    # Iterates through the physical groups

    for physical_group, bc_dictionary in boundary_conditionsDict.items():

        # Verifies if the bc_dictionary is in fact a list, thus, adding 
        # multiple boundary conditions to a single physical group

        if isinstance(bc_dictionary, list):

            # Iterates through the boundary conditions

            for nested_bcDict in bc_dictionary:
                
                # Checks if this component is a dictionary

                if isinstance(nested_bcDict, dict):

                    # Verifies if it has the key 'BC case'

                    if not ("BC case" in nested_bcDict):

                        raise KeyError("The dictionary of information "+
                        "for constructing boundary conditions does not"+
                        " have the key 'BC case'. It must have this ke"+
                        "y to find the appropriate method to generate "+
                        "the Dirichlet boundary condition")
                    
                    # Gets the BC case

                    BC_case = nested_bcDict["BC case"]

                    # Gets the user-given data

                    user_data = dict()

                    for key, value in nested_bcDict.items():

                        if key!="BC case":

                            user_data[key] = value

                    # Adds the physical_group to the dictionary of argu-
                    # emnts, and also the current list of boundary con-
                    # ditions

                    method_arguments["boundary_physicalGroups"] = physical_group

                    method_arguments["boundary_conditions"] = boundary_conditions

                    # Adds the list of classes that control the applica-
                    # tion of displacements along the time steps

                    method_arguments["dirichtlet_loads"] = dirichlet_loads

                    # Gets the boundary conditions list and (if necessa-
                    # ry) the loading parameter

                    result = programming_tools.dispatch_functions(
                    BC_case, None, fixed_inputVariablesDict=
                    method_arguments, second_sourceFixedArguments=
                    user_data, methods_functionsDict=bcs_functionsDict, 
                    return_list=True, return_singleFunction=True,
                    all_argumentsFixed=True)[0]()

                    # Verifies if the result is a tuple

                    if isinstance(result, tuple):

                        boundary_conditions = result[0]

                        dirichlet_loads = result[1]

                    else:

                        boundary_conditions = result

                # Verifies if it is a DirichletBC instance

                elif isinstance(nested_bcDict, DirichletBC):

                    # Appends to the list of boundary conditions

                    boundary_conditions.append(nested_bcDict)

                else:

                    raise TypeError("The information to construct a bo"+
                    "undary condition must be a dictionary or a direct"+
                    " instance of DirichletBC fenics class. Whereas th"+
                    "e given value is "+str(nested_bcDict))

        # If the bc_dictioanry is not a list, updates the boundary con-
        # dition list directly

        else:
            
            # Checks if this component is a dictionary

            if isinstance(bc_dictionary, dict):

                # Verifies if it has the key 'BC case'

                if not ("BC case" in bc_dictionary):

                    raise KeyError("The dictionary of information for "+
                    "constructing boundary conditions does not have th"+
                    "e key 'BC case'. It must have this key to find th"+
                    "e appropriate method to generate the Dirichlet bo"+
                    "undary condition")
                
                # Gets the BC case

                BC_case = bc_dictionary["BC case"]

                # Gets the user-given data

                user_data = dict()

                for key, value in bc_dictionary.items():

                    if key!="BC case":

                        user_data[key] = value

                # Adds the physical_group to the dictionary of arguments,
                # and also the current list of boundary conditions

                method_arguments["boundary_physicalGroups"] = physical_group

                method_arguments["boundary_conditions"] = boundary_conditions

                # Adds the list of classes that control the applica-
                # tion of displacements along the time steps

                method_arguments["dirichtlet_loads"] = dirichlet_loads

                # Gets the boundary conditions list and (if necessa-
                # ry) the loading parameter

                result = programming_tools.dispatch_functions(
                BC_case, None, fixed_inputVariablesDict=
                method_arguments, second_sourceFixedArguments=
                user_data, methods_functionsDict=bcs_functionsDict, 
                return_list=True, return_singleFunction=True,
                all_argumentsFixed=True)[0]()

                # Verifies if the result is a tuple

                if isinstance(result, tuple):

                    boundary_conditions = result[0]

                    dirichlet_loads = result[1]

                else:

                    boundary_conditions = result

            # Verifies if it is a DirichletBC instance

            elif isinstance(bc_dictionary, DirichletBC):

                # Appends to the list of boundary conditions

                boundary_conditions.append(bc_dictionary)

            else:

                raise TypeError("The information to construct a bounda"+
                "ry condition must be a dictionary or a direct instanc"+
                "e of DirichletBC fenics class. Whereas the given valu"+
                "e is "+str(bc_dictionary))
    
    # Returns the boundary conditions' list and the list of controlling
    # Dirichlet boundary conditions loads

    return boundary_conditions, dirichlet_loads

########################################################################
#                            Solver setting                            #
########################################################################

# Defines a function to set the nonlinear problem from the residual and
# the Gateaux derivative

def set_nonlinearProblem(residual_form, monolithic_solution, 
trial_functions, boundary_conditions, solver_parameters=None):
    
    # Evaluates the Gateaux derivative of the residue form
    
    residual_derivative = derivative(residual_form, monolithic_solution, 
    trial_functions)

    # Sets the nonlinear variational problem

    Res = NonlinearVariationalProblem(residual_form, monolithic_solution, 
    boundary_conditions, J=residual_derivative)

    # Sets the solver to this problem

    """solver = NonlinearVariationalSolver(Res)

    if not (solver_parameters is None):

        solver = set_solverParameters(solver, solver_parameters)

    return solver"""

    return create_solverClass(Res, solver_parameters)

# Defines a function to create a class of NonlinearVariationalSolver 
# with extra features taken from the solver_parameters dictionary

def create_solverClass(non_linearProblem, solver_parameters):

    if solver_parameters is None:

        return NonlinearVariationalSolver(non_linearProblem)
        
    # Verifies if solver_parameters is not a dictionary

    elif not isinstance(solver_parameters, dict):

        raise TypeError("solver_parameters is not a dictionary, thus i"+
        "t cannot be used to set the parameters of the solver")
    
    # Gets the admissible keys for the custom solver

    custom_parameters = give_customSolverParametersKeys()

    # Selects the parameters that are not native to fenics implementation

    non_nativeParameters = dict()

    for name, value in solver_parameters.items():

        if name in custom_parameters:

            # Adds this parameters

            non_nativeParameters[name] = value

    # If there is no non native parameters, creates the solver

    if len(non_nativeParameters.keys())==0:

        return set_solverParameters(NonlinearVariationalSolver(
        non_linearProblem), solver_parameters)
    
    raise NotImplementedError("The custom solver has not been implemented yet")

    # Eliminates the keys that are for the custom solver

    for name in non_nativeParameters.keys():

        solver_parameters.pop(name)

    # Defines the class of the custom solver

    class CustomSolver:

        def __init__(self, problem):
            
            # Receives the NonlinearVariationalProblem and sets some de-
            # fault parameters

            self.problem = problem

            self.parameters = {"nonlinear_solver": "newton", "newton_s"+
            "olver": dict()}

            self.parameters["newton_solver"] = {"maximum_iterations": 25, 
            "relative_tolerance": 1e-8, "absolute_tolerance": 1e-8, "r"+
            "eport": True, "error_on_nonconvergence": True, "linear_so"+
            "lver": "lu"}

        def solve(self):
            
            # Gets the solution, the residual,the Gateaux derivative, 
            # and boundary conditions

            u = self.problem.u

            F_form = self.problem.F

            J_form = self.problem.J

            bcs = self.problem.bcs

            if J_form is None:

                du = TrialFunction(u.function_space())

                J_form = derivative(F_form, u, du)

            # Initializes the residual vector and the solution update
            # step

            residual_vector = None

            du = Function(u.function_space())

            # Gets some parameters

            max_iter = self.parameters["newton_solver"]["maximum_iterations"]

            a_tol = self.parameters["newton_solver"]["absolute_tolerance"]

            r_tol = self.parameters["newton_solver"]["relative_tolerance"]

            report = self.parameters["newton_solver"]["report"]

            error_on_nonconvergence = self.parameters["newton_solver"]["error_on_nonconvergence"]

            linear_solver = self.parameters["newton_solver"]["linear_solver"]

            converged = False

            initial_residual = None

            for k in range(1, max_iter+1):

                # Assembles residual

                residual_vector = assemble(F_form)

                for bc in bcs:

                    bc.apply(residual_vector, u.vector())

                res_norm = residual_vector.norm("l2")

                if k==1:

                    initial_residual = res_norm*1.0

                if report:

                    print(f"  Newton iteration {k}: r (norm) = {res_norm:.6e}")

                if not np.isfinite(res_norm):

                    raise RuntimeError("Newton solver diverged: residu"+
                    "al is NaN or Inf")
                
                elif res_norm>non_nativeParameters["maximum_residual"]:

                    raise RuntimeError("Newton solver diverged: residu"+
                    "al is "+str(res_norm)+", which is larger than the"+
                    " tolerance of "+str(non_nativeParameters["maximum"+
                    "_residual"]))

                if res_norm < a_tol:

                    converged = True

                    break

                elif (res_norm/initial_residual)<r_tol:

                    converged = True

                    break

                # Assembles Jacobian

                J = assemble(J_form)

                for bc in bcs:

                    bc.apply(J)

                # Solves for update

                solve(J, du.vector(), residual_vector, linear_solver)

                # Update solution

                u.vector().axpy(-1.0, du.vector())

                u.vector().apply("insert")

            if not converged:

                msg = f"Newton solver did not converge after {max_iter} iterations."

                if error_on_nonconvergence:

                    raise RuntimeError(msg)
                
                else:

                    print(msg)

            elif report:

                print(f"  Newton solver converged in {k} iterations.\n")
        
    # Instantiates the new class and sets the parameters

    return set_solverParameters(CustomSolver(non_linearProblem), 
    solver_parameters)

# Defines a function to give the parameters for the upgrade of the Newton
# implementation native to fenics

def give_customSolverParametersKeys():

    return ["maximum_residual"]

# Defines a function to update solver parameters

def set_solverParameters(solver, solver_parameters):

    # Sets a list of implemented solver parameters

    admissible_keys = ["nonlinear_solver", "linear_solver", "newton_re"+
    "lative_tolerance", "newton_absolute_tolerance", "newton_maximum_i"+
    "terations", "preconditioner", "krylov_absolute_tolerance", "krylo"+
    "v_relative_tolerance", "krylov_maximum_iterations", "krylov_monit"+
    "or_convergence", "petsc_options"]

    # Gets the keys of the solver parameters dictionary

    parameter_types = solver_parameters.keys()

    # Iterates the keys of the solver parameters to verify if any of 
    # them is not admissible

    for key in parameter_types:

        if not (key in admissible_keys):

            raise NameError("The key "+str(key)+" is not an admissible"+
            " key to set solver parameters. Check the admissible keys:"+
            "\n"+str(admissible_keys)+"\nand the parameters for the cu"+
            "stom solver:\n"+str(give_customSolverParametersKeys()))
        
    # Sets the solver parameters

    if "nonlinear_solver" in parameter_types:

        solver.parameters["nonlinear_solver"] = solver_parameters["non"+
        "linear_solver"]

    else:

        solver.parameters["nonlinear_solver"] = "newton"

    if "linear_solver" in parameter_types:

        solver.parameters["newton_solver"]["linear_solver"] = (
        solver_parameters["linear_solver"])

    if "newton_relative_tolerance" in parameter_types:

        solver.parameters["newton_solver"]["relative_tolerance"] = (
        solver_parameters["newton_relative_tolerance"])

    if "newton_absolute_tolerance" in parameter_types:

        solver.parameters["newton_solver"]["absolute_tolerance"] = (
        solver_parameters["newton_absolute_tolerance"])

    if "newton_maximum_iterations" in parameter_types:

        solver.parameters["newton_solver"]["maximum_iterations"] = (
        solver_parameters["newton_maximum_iterations"])

    if "preconditioner" in parameter_types:

        solver.parameters["newton_solver"]["preconditioner"] = (
        solver_parameters["preconditioner"])

    if "krylov_absolute_tolerance" in parameter_types:

        solver.parameters['newton_solver']['krylov_solver']['absolute_'+
        'tolerance'] = solver_parameters["krylov_absolute_tolerance"]

    if "krylov_relative_tolerance" in parameter_types:

        solver.parameters['newton_solver']['krylov_solver']['relative_'+
        'tolerance'] = solver_parameters["krylov_relative_tolerance"]

    if "krylov_maximum_iterations" in parameter_types:

        solver.parameters['newton_solver']['krylov_solver']['maximum_i'+
        'terations'] = solver_parameters["krylov_maximum_iterations"]

    if "krylov_monitor_convergence" in parameter_types:

        solver.parameters['newton_solver']['krylov_solver']['monitor_c'+
        'onvergence'] = solver_parameters["krylov_monitor_convergence"]

    if "petsc_options" in parameter_types:

        opts = PETSc.Options()

        # If the snes solver is set, prescribe the tolerances differently

        if "nonlinear_solver" in solver_parameters:
            
            if solver_parameters["nonlinear_solver"]=="snes":

                if "newton_relative_tolerance" in parameter_types:

                    opts["snes_rtol"] = solver_parameters["newton_rela"+
                    "tive_tolerance"]

                if "newton_absolute_tolerance" in parameter_types:

                    opts["snes_atol"] = solver_parameters["newton_abso"+
                    "lute_tolerance"]

                if "newton_maximum_iterations" in parameter_types:

                    opts["snes_max_it"] = solver_parameters["newton_ma"+
                    "ximum_iterations"]

                if "linear_solver" in parameter_types:

                    opts["pc_factor_mat_solver_type"] = solver_parameters[
                    "linear_solver"]

                if "krylov_absolute_tolerance" in parameter_types:

                    opts["ksp_atol"] = solver_parameters["krylov_absol"+
                    "ute_tolerance"]

                if "krylov_relative_tolerance" in parameter_types:

                    opts["ksp_rtol"] = solver_parameters["krylov_relat"+
                    "ive_tolerance"]

                if "krylov_maximum_iterations" in parameter_types:

                    opts["ksp_max_it"] = solver_parameters["krylov_max"+
                    "imum_iterations"]

        if solver_parameters["petsc_options"]:

            # Defines options to check and avoid NaN/Inf in the solution

            opts["snes_check_jacobian_domain_error"] = None

            opts["snes_check_function_norm"] = ""  

            opts["snes_check_function_value"] = ""

            opts["snes_check_jacobian_norm"] = ""

            opts["snes_check_jacobian_largest"] = ""

    # Returns the updated solver

    return solver

########################################################################
#                       Finite element creation                        #
########################################################################

# Defines a function to select a field from a list of fields

def select_fields(split_fieldsList, required_fieldsNames, 
fields_namesDict):
    
    # Tests the solution is a list

    if not isinstance(split_fieldsList, list):

        # If not, simply gives the field

        return split_fieldsList

    # Initializes a list of fields that will be used

    retrieved_fields = []

    # Iterates through the required fields' names

    for name in required_fieldsNames:

        # Verifies if this name is in the dictionary of fields' names 
        # that came from the generation of the finite elements

        if not (name in fields_namesDict):
                
            raise NameError("The field '"+str(name)+"' is required in "+
            "this post-process, but it is not in the list of names of "+
            "fields, that is: "+str(fields_namesDict.keys()))
        
        # Appends the field by its name

        retrieved_fields.append(split_fieldsList[fields_namesDict[name]])

    # Returns the list of fields

    if len(retrieved_fields)==1:

        return retrieved_fields[0]

    return retrieved_fields

# Defines a function to construct a function space given a dictionary of
# instructions to create finite elements

def construct_monolithicFunctionSpace(elements_dictionary, 
mesh_dataClass, verbose=False):
    
    # Transforms the dictionary of instructions into real finite elements
    # and get the names of the fields

    elements_dictionary, fields_names = construct_elementsDictionary(
    elements_dictionary, mesh_dataClass)

    # Constructs the mixed element

    mixed_element = 0

    if len(fields_names)>1:

        mixed_element = MixedElement([elements_dictionary[field_name
        ] for field_name in (fields_names)])

    else:

        mixed_element = elements_dictionary[fields_names[0]]

    # Constructs the monolithic function space using the finite element
    # or the set of finite elements, which were previously created

    monolithic_functionSpace = FunctionSpace(mesh_dataClass.mesh, 
    mixed_element)

    # Creates the function for the updated solution, i.e. the vector of 
    # parameters. Then, splits into the individual fields. As the mono-
    # lithic function space has an specific order of fields, retrieves 
    # the individual fields following this order. Does the same for the
    # variations

    trial_functions = TrialFunction(monolithic_functionSpace)

    monolithic_solution = Function(monolithic_functionSpace)

    solution_functions = []

    variation_functions = []

    if len(fields_names)>1:

        solution_functions = split(monolithic_solution)

        variation_functions = split(TestFunction(monolithic_functionSpace
        ))

    else:

        solution_functions = [monolithic_solution]

        variation_functions = [TestFunction(monolithic_functionSpace)]

    # Organizes them into dictionaries

    solution_fields = dict()

    variation_fields = dict()

    fields_namesDict = dict()

    for i in range(len(fields_names)):

        field_name = fields_names[i]

        # Retrives the individual field and its variation

        solution_fields[field_name] = solution_functions[i]

        variation_fields[field_name] = variation_functions[i]

        fields_namesDict[field_name] = i

    if verbose:

        print("Finishes creating the function spaces\n")

    return (monolithic_functionSpace, monolithic_solution, fields_names, 
    solution_fields, variation_fields, trial_functions, fields_namesDict)

# Defines a function to construct a dictionary of elements from a dic-
# tionary of finite elements' instructions

def construct_elementsDictionary(elements_dictionary, mesh_dataClass):

    # Sets a list of allowed interpolation functions

    allowed_interpolationFunction = ["CG", "DG", "Lagrange"]

    # Initializes a list of names of the fields

    fields_names = []

    # Iterates through the dictionary of elements
    
    for field_name, element_dictionary in elements_dictionary.items():

        # Verifies if this is already a fenics element. Tests by veri-
        # fying if has a cell attribute

        try:

            cell = element_dictionary.cell()

            # Updates the list of fields' names

            fields_names.append(field_name)

        # If not, it must be a dictionary with finite element's informa-
        # tion

        except:

            # Verifies if the element is in fact a function space

            if isinstance(element_dictionary, FunctionSpace):

                # Gets the element from the function space

                elements_dictionary[field_name] = (
                element_dictionary.ufl_element())

            # Otherwise, creates the element

            else:

                if not isinstance(element_dictionary, dict):

                    raise TypeError("The dictionary to construct the f"+
                    "inite element of the field '"+str(field_name)+"' "
                    "is not a dictionary. Check out: "+str(
                    element_dictionary))
                
                # Gets the polynomial degree. Uses second degree polyno-
                # mial as default

                polynomial_degree = 2

                if "polynomial degree" in element_dictionary:

                    polynomial_degree = element_dictionary["polynomial"+
                    " degree"]
                
                # Gets the interpolation function

                interpolation_function = ""

                if "interpolation function" in element_dictionary:

                    interpolation_function = element_dictionary["inter"+
                    "polation function"]

                    # Verifies if this interpolation function is one of 
                    # the admissible ones

                    if not (interpolation_function in (
                    allowed_interpolationFunction)):
                        
                        raise NameError("The interpolation function '"+
                        str(interpolation_function)+"' is not one of t"+
                        "he admissible functions. Check the list ahead"+
                        ": "+str(allowed_interpolationFunction))
                    
                elif "shape function" in element_dictionary:

                    interpolation_function = element_dictionary["shape"+
                    " function"]

                    # Verifies if this interpolation function is one of 
                    # the admissible ones

                    if not (interpolation_function in (
                    allowed_interpolationFunction)):
                        
                        raise NameError("The interpolation function '"+
                        str(interpolation_function)+"' is not one of t"+
                        "he admissible functions. Check the list ahead"+
                        ": "+str(allowed_interpolationFunction))
                    
                else:

                    # Uses continuous Galerkin as default

                    interpolation_function = "CG"

                # Gets the element type (scalar, vector, tensor...)

                if "field type" in element_dictionary:

                    element_type = element_dictionary["field type"]

                    # If the field is a scalar-valued field

                    if element_type=="scalar":

                        # Creates the element

                        elements_dictionary[field_name] = FiniteElement(
                        interpolation_function, 
                        mesh_dataClass.mesh.ufl_cell(), polynomial_degree)

                    # If the field is a vector-valued field

                    elif element_type=="vector":

                        # Creates the element

                        elements_dictionary[field_name] = VectorElement(
                        interpolation_function, 
                        mesh_dataClass.mesh.ufl_cell(), polynomial_degree)

                    # If the field is a tensor-valued field

                    elif element_type=="tensor":

                        # Creates the element

                        elements_dictionary[field_name] = TensorElement(
                        interpolation_function, 
                        mesh_dataClass.mesh.ufl_cell(), polynomial_degree)

                    else:

                        raise NameError("There is no '"+str(element_type
                        )+"' field implemented. A field can be either "+
                        "'scalar', 'vector', or 'tensor' for the eleme"+
                        "nt to be dynamically generated")
                    
                else:

                    raise KeyError("There is no key 'field type' in th"+
                    "e dictionary to create finite elements.")
                
            # Updates the list of fields' names

            fields_names.append(field_name)
        
    # Returns the finite elements dictionary

    return elements_dictionary, fields_names

########################################################################
#                        Time stepping classes                         #
########################################################################

# Defines an abstract class as template for the time-stepping algorithms

class TimeSteppingClasses(ABC):

    @abstractmethod

    # The following methods have the pass argument only because they 
    # will be defined in the child classes

    def update(self, time_value):

        pass

# Defines a class to store macroquantities as constants, then, it is up-
# dated during the time-stepping algorithm. This class retrieves the da-
# ta from txt files, convert them to lists and, then, uses them to as-
# sign to FEniCS constants

class MacroQuantitiesInTime(TimeSteppingClasses):

    def __init__(self, dictionary_VariablesToFiles, time_tolerance=1E-5):

        # Stores the dictionary of variables

        self.variables = dictionary_VariablesToFiles

        # Initializes a list of time keys. All the macro quantities must
        # have the same time points

        self.time_keys = []

        # Iterates through the dictionary of file names to read them in-
        # to lists

        for variable, file_name in self.variables.items():

            # Converts the lists into dictionaries

            self.variables[variable] = (file_tools.list_toDict(
            file_tools.txt_toList(file_name)))

            # Updates the time keys

            if len(self.time_keys)==0:

                self.time_keys = list(self.variables[variable].keys())

            # Otherwise, checks if the time keys are equal given a tole-
            # rance

            else:

                test_timeKeysConsistency(self.time_keys, list(
                self.variables[variable].keys()), time_tolerance=
                time_tolerance, variable_name=variable)

            # Initializes the variable in the class data structure to u-
            # se it after as class.variable. Uses FEniCS constant, so it
            # might be just assigned later to spare compilation time

            setattr(self, variable, Constant(self.variables[variable][
            self.time_keys[0]]))

    # Defines a function to update the macro quantitites given the cur-
    # rent time value

    def update(self, time_value):
        
        # Iterates through the variables

        for variable, variable_dataDict in self.variables.items():

            # Gets the keys of the variable data dictionary

            data_keys = list(variable_dataDict.keys())

            # Gets the closest key to the given time value

            time_key = numerical_tools.closest_numberInList(time_value,
            data_keys)

            # Gets the data value for this time step to the variable 
            # contained in this class with this variable's name. Uses 
            # the assign method, since this variables has already been
            # created as a fenics Constant

            getattr(self, variable).assign(Constant(variable_dataDict[
            time_key]))

            # Deletes this key-data pair from the data dictionary

            variable_dataDict.pop(time_key)

# Defines a function to test if the time keys are the same comparing the
# standard with a given set

def test_timeKeysConsistency(standard_timeKeys, compared_timeKeys, 
variable_name="NoGivenName", time_tolerance=1E-5):

    # Checks if it has the same number of points as the first
    # one

    if len(compared_timeKeys)!=len(standard_timeKeys):

        raise ValueError("All the macro quantities must have the same "+
        "number of time points. The first macro quantity had "+str(len(
        standard_timeKeys))+" time points, whereas quantity "+str(
        variable_name)+" has "+str(len(compared_timeKeys)+" time points"))
    
    # If they have the same number of time points, checks if
    # they are the same points

    for i in range(len(standard_timeKeys)):

        if (abs(standard_timeKeys[i]-compared_timeKeys[i])>
        time_tolerance):
            
            raise ValueError("The time points must be equal for all ma"+
            "cro quantities. The "+str(i)+"-th point of the first quan"+
            "tity is "+str(standard_timeKeys[i])+" whereas the same po"+
            "int in the "+str(variable_name)+" macro quantity is "+str(
            compared_timeKeys[i]))

# Defines a function to transform a key to ascii pattern

def convert_stringToASCII(string_toConvert):

    # Initializes the converted string

    converted_string = ""

    # Iterates through the characters

    for character in string_toConvert:

        # Checks if it is a space

        if character==" " or character==' ':

            # Substitutes the blank space for an underline

            converted_string += "_"

        # Checks if it is ascii. If the character encoding is larger
        # than 128, just ignore it, for it's not ascii

        elif ord(character)<128:

            converted_string += character

    # If the converted string is empty, returns a basic string

    if len(converted_string)==0:

        return "foo_variable"

    # Returns the string

    return converted_string

########################################################################
#     Creation and pre-evaluation of discontinuous function spaces     #
########################################################################

# Defines a function to create a R^3 -> R discontinuous function space
# and populates it with values using physical groups information. The 
# keys are the physical groups tags and the values are the values of the
# property to be interpolated using a function space

def physical_groupToDGSpace(physical_dictionary, mesh, mesh_function):

    # Creates a discontinuous Galerkin function space

    DG_functionSpace = FunctionSpace(mesh, "DG", 0)

    # Creates the function with parameters vector

    DG_function = Function(DG_functionSpace)

    # Iterates through the elements of the mesh

    for cell in cells(mesh):

        # Gets the physical group tag (numerical) to which this element 
        # belongs

        physical_groupTag = mesh_function[cell]

        # Iterates through the keys (physical groups numerical tags) and
        # the values of the physical dictionary

        #flag_test = True

        try:

            DG_function.vector()[cell.index()] = physical_dictionary[
            physical_groupTag]
            
        except:

            raise ValueError("The dictionary of properties "+str(
            physical_dictionary)+" does not contain the physical group"+
            " tag "+str(physical_groupTag))
            
    # Returns the function

    return DG_function