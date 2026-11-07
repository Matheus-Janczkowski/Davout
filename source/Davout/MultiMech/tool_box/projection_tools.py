# Routine to store methods for projections to and between finite-element
# spaces

from dolfin import *

import numpy as np

from ...PythonicUtilities import programming_tools, dictionary_tools

from ..tool_box.variational_tools import verify_physicalGroups

# Defines a class to perform local projections from ufl expressions to 
# quadrature finite-element spaces and back to expressions

class ExpressionProjectionOntoQuadratureSpace:

    def __init__(self, field_name, functional_data_class):

        # Retrieves the function space, the dictionary of fields names,
        # and the integration measure object

        monolithic_function_space = (
        functional_data_class.monolithic_function_space)

        fields_names_dict = functional_data_class.fields_names_dict

        self.dx_object = functional_data_class.mesh_data_class.dx

        # Checks if the field name is within the dictionary of names

        if not (field_name in fields_names_dict):

            fields_names_string = dictionary_tools.list_keys(
            fields_names_dict, None, simply_list_keys=True)

            raise ValueError("'field_name' equal to '"+str(field_name
            )+"' was asked to project such field in a quadrature space"+
            ", but that field is not available. Check the available fi"+
            "elds' names:\n"+fields_names_string)

        # Initializes the function space of that field

        field_function_space = None

        # Checks if the functional data class has only one field

        if len(fields_names_dict)==1:

            # Gets the function space directly from the provided monoli-
            # thic function space. Collapses it to create a standalone
            # version of that function space

            field_function_space = monolithic_function_space.collapse()

        # Otherwise, if there are multiple fields

        else:

            # Retrieves the function space using the index provided by
            # the dictionary of fields. Collapses it to create a stand-
            # alone version of that function space

            field_function_space = monolithic_function_space.sub(
            fields_names_dict[field_name]).collapse()

        # Defines the bilinear form of the local projection problem
        
        projection_trial = TrialFunction(field_function_space)

        self.project_test = TestFunction(field_function_space)

        self.projection_bilinear_form = inner(projection_trial, 
        self.project_test)*self.dx_object

        # Creates the output function for the projected expression

        self.expression_function = Function(field_function_space)

        # Gets the tuple of dimensions of this field. It will be () if
        # scalar; (dim) if vector; (dim, dim, ..., dim) if tensor

        tensor_dimension = functional_data_class.fields_tensor_dimensions_dict[
        field_name]

        # Checks if tensor dimension is an integer

        if isinstance(tensor_dimension, int):

            # Converts it to a tuple of a single number

            tensor_dimension = (tensor_dimension,)

        # Gets the number of components in the field

        number_of_components = 1

        if len(tensor_dimension)>0:

            number_of_components = int(np.prod(tensor_dimension))

        # Gets the number of DOFs in each finite element of the quadra-
        # ture space

        number_of_element_dofs = len(field_function_space.dofmap(
        ).cell_dofs(0))

        # Calculates the number of quadrature points in each finite el-
        # ement

        number_of_quadrature_points_per_element = (
        number_of_element_dofs//number_of_components)

        # Gets the total number of quadrature points in the mesh

        number_of_elements = (
        functional_data_class.mesh_data_class.mesh.num_cells())

        self.number_of_quadrature_points = (number_of_elements*
        number_of_quadrature_points_per_element)

        # Assembles the shape of the numpy array that will convey the
        # values of this field for each quadrature point

        self.numpy_array_shape = [self.number_of_quadrature_points]

        self.numpy_array_shape.extend(list(tensor_dimension))

    # Defines a function to project an ufl expression onto a quadrature
    # space

    def project_expression_onto_quadrature_space(self, expression):

        # Constructs the linear form of the projection

        projection_linear_form = inner(expression, self.project_test
        )*self.dx_object

        # Constructs and factorizes the local solver

        self.projection_solver = LocalSolver(
        self.projection_bilinear_form, projection_linear_form)

        self.projection_solver.factorize()

        # Projects the expression to the quadrature function space

        self.projection_solver.solve_local_rhs(self.expression_function)

        # Retrieves the numerical values

        expression_values = self.expression_function.vector().get_local()

        expression_values = expression_values.reshape(
        self.numpy_array_shape)

        return expression_values

    # Defines a function to convert the quadrature finite-element inter-
    # polation to numpy arrays [n_quadrature_points, tensor_dimension]

    def convert_quadrature_values_to_numpy_array(self, 
    function_in_quadrature_space):

        # Gets the values of the function defined on a quadrature space
        # at each quadrature point

        function_values = (function_in_quadrature_space.vector()
        .get_local())

        # Reshapes these values to a numpy array according to the dimen-
        # sions automatically retrieved from the functional data class

        function_values_array = function_values.reshape(
        self.numpy_array_shape)

        return function_values_array

    # Defines a function to update the quadrature points of a quadrature
    # interpolation with values coming from a numpy array 
    # [n_quadrature_points, tensor_dimension]

    def update_quadrature_values_with_numpy_array(self, 
    function_in_quadrature_space, function_values_array):

        # Sets the numpy array as values for the quadrature interpolation

        function_in_quadrature_space.vector().set_local(
        function_values_array.ravel())

        function_in_quadrature_space.vector().apply("insert")

# Defines a function to project a field that is piecewise continuous in-
# to a finite element space. The field is given by a list of lists, whe-
# re each sublist is a pair of the function to be projected (i.e. the 
# value of the field) and the tag associated with its particular domain
# region (i.e. the physical group number or name)

@programming_tools.optional_argumentsInitializer({'solution_names': 
lambda: ["field", "field"]})

def project_piecewiseField(field_list, dx, V, physical_groupsList,
physical_groupsNamesToTags, solution_names=None, verbose=True):
    
    # Creates the projected field

    projected_fieldTrial = TrialFunction(V)

    v = TestFunction(V)

    projected_field = Function(V)

    projected_field.rename(*solution_names)

    # Assembles and solves the variational form

    bilinear_form = inner(projected_fieldTrial, v)*dx

    # Initializes the linear form

    linear_form = 0.0

    # Iterates through the pairs of function to be projected and physi-
    # cal group tag

    for projected_function, subdomain in field_list:

        if subdomain=="":

            if verbose:

                print("Projects over entire domain because subdomain w"+
                "as defined as "+str(subdomain))

            # Updates the linear form

            linear_form += (inner(projected_function, v)*dx)

        else:

            if verbose:

                print("Projects over the subdomain "+str(subdomain))

            # Checks the subdomain for strings

            subdomain = verify_physicalGroups(subdomain,
            physical_groupsList, physical_groupsNamesToTags=
            physical_groupsNamesToTags)

            # Updates the linear form

            linear_form += (inner(projected_function, v)*dx(subdomain))

    # Solves the algebraic system to get the FEM parameters

    solve(bilinear_form==linear_form, projected_field)

    # Returns the projected field

    return projected_field

# Defines a function to project a field defined on the boundary only

def project_overBoundary(field_list, ds, V, physical_groupsList,
physical_groupsNamesToTags, solution_names=None, verbose=True,
verify_physical_groups=True):
    
    # Creates the projected field

    projected_fieldTrial = TrialFunction(V)

    v = TestFunction(V)

    projected_field = Function(V)

    if not (solution_names is None):

        projected_field.rename(*solution_names)

    # Assembles and solves the variational form

    bilinear_form = inner(projected_fieldTrial, v)*ds

    # Initializes the linear form

    linear_form = 0.0

    # Iterates through the pairs of function to be projected and physi-
    # cal group tag

    for projected_function, subdomain in field_list:

        if subdomain=="":

            if verbose:

                print("Projects over entire domain because subdomain w"+
                "as defined as "+str(subdomain))

            # Updates the linear form

            linear_form += (inner(projected_function, v)*ds)

        else:

            if verbose:

                print("Projects over the subdomain "+str(subdomain))

            # Checks the subdomain for strings

            if verify_physical_groups:

                subdomain = verify_physicalGroups(subdomain,
                physical_groupsList, physical_groupsNamesToTags=
                physical_groupsNamesToTags)

            # Updates the linear form

            linear_form += (inner(projected_function, v)*ds(subdomain))

    # Assembles the linear and the bilinear forms. Uses flags keep_dia-
    # gonal and ident_zeros because the bilinear form is mostly zero, 
    # since the field is zero within the domain

    A = assemble(bilinear_form, keep_diagonal=True)

    A.ident_zeros()

    L = assemble(linear_form)

    # Solves the algebraic system to get the FEM parameters

    solve(A, projected_field.vector(), L)

    # Returns the projected field

    return projected_field

# Defines a function to project a field over a region of the domain

def projection_overRegion(field, V, dx, subdomain, physical_groupsList,
physical_groupsNamesToTags):

    # Checks the subdomain for strings

    subdomain = verify_physicalGroups(subdomain, physical_groupsList, 
    physical_groupsNamesToTags=physical_groupsNamesToTags)

    # Creates the projected field

    projected_fieldTrial = TrialFunction(V)

    v = TestFunction(V)

    projected_field = Function(V)

    # Assembles and solves the variational form

    bilinear_form = inner(projected_fieldTrial, v)*dx
                     
    linear_form = (inner(field, v)*dx(subdomain))

    solve(bilinear_form==linear_form, projected_field)

    # Returns the projected field

    return projected_field