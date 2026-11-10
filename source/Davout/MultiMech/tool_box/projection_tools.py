# Routine to store methods for projections to and between finite-element
# spaces

from dolfin import *

import numpy as np

from ...PythonicUtilities import programming_tools, dictionary_tools

from ..tool_box.variational_tools import verify_physicalGroups

import warnings

from ffc.quadrature.deprecation import (
QuadratureRepresentationDeprecationWarning)

warnings.simplefilter("ignore",
QuadratureRepresentationDeprecationWarning)

# Defines a class to instantiate for multiple fields at once the class 
# ExpressionProjectionOntoQuadratureSpace

class MultipleFieldsProjectionOntoQuadratureSpace:

    def __init__(self, field_names_list, functional_data_class, dtype=
    np.int32, point_absolute_tolerance=1E-6, point_relative_tolerance=
    1E-8):

        # Verifies if field name is a string

        if isinstance(field_names_list, str):

            # Puts it into a list

            field_names_list = [field_names_list]

        # Checks if it is not a list

        elif not isinstance(field_names_list, list):

            raise TypeError("'field_names_list' at 'MultipleFieldsProj"+
            "ectionOntoQuadratureSpace' must be a string (for a single"+
            " field) or a list. Currently, it is: "+str(field_names_list
            ))

        # Initializes a dictionary of instances of the class for projec-
        # tions onto quadrature spaces

        self.classes_dictionary = {}

        # Iterates over the fields

        for field_name in field_names_list:

            # Instantiates the class for projection

            self.classes_dictionary[field_name] = ExpressionProjectionOntoQuadratureSpace(
            field_name, functional_data_class, dtype=dtype, 
            point_absolute_tolerance=point_absolute_tolerance,
            point_relative_tolerance=point_relative_tolerance,
            expression_function=functional_data_class.solution_fields[
            field_name])

    # Defines a function to project an ufl expression onto a quadrature
    # space

    def project_expression_onto_quadrature_space(self, field_name, 
    expression):

        # Checks if field name is in the dictionary of fields names

        if not (field_name in self.classes_dictionary):

            available_names = dictionary_tools.list_keys(
            self.classes_dictionary, None, simply_list_keys=True)

            raise KeyError("'field_name' in 'project_expression_onto_q"+
            "uadrature_space' is not an available field name. Check th"+
            "e available names:\n"+str(available_names))

        # Returns that function defined in the class of this field
        
        return self.classes_dictionary[field_name
        ].project_expression_onto_quadrature_space(expression)

    # Defines a function to convert the quadrature finite-element inter-
    # polation to numpy arrays [n_quadrature_points, tensor_dimension]

    def convert_quadrature_values_to_numpy_array(self, field_name,
    function_in_quadrature_space):

        # Checks if field name is in the dictionary of fields names

        if not (field_name in self.classes_dictionary):

            available_names = dictionary_tools.list_keys(
            self.classes_dictionary, None, simply_list_keys=True)

            raise KeyError("'field_name' in 'convert_quadrature_values"+
            "_to_numpy_array' is not an available field name. Check th"+
            "e available names:\n"+str(available_names))

        # Returns that function defined in the class of this field
        
        return self.classes_dictionary[field_name
        ].convert_quadrature_values_to_numpy_array(
        function_in_quadrature_space)

    # Defines a function to update the quadrature points of a quadrature
    # interpolation with values coming from a numpy array 
    # [n_quadrature_points, tensor_dimension]

    def update_quadrature_values_with_numpy_array(self, field_name,
    function_in_quadrature_space, function_values_array):

        # Checks if field name is in the dictionary of fields names

        if not (field_name in self.classes_dictionary):

            available_names = dictionary_tools.list_keys(
            self.classes_dictionary, None, simply_list_keys=True)

            raise KeyError("'field_name' in 'update_quadrature_values_"+
            "with_numpy_array' is not an available field name. Check t"+
            "he available names:\n"+str(available_names))

        # Runs that function defined in the class of this field
        
        self.classes_dictionary[field_name
        ].update_quadrature_values_with_numpy_array(
        function_in_quadrature_space, function_values_array)

# Defines a class to perform local projections from ufl expressions to 
# quadrature finite-element spaces and back to expressions

class ExpressionProjectionOntoQuadratureSpace:

    def __init__(self, field_name, functional_data_class, dtype=np.int32,
    point_absolute_tolerance=1E-6, point_relative_tolerance=1E-8,
    expression_function=None):

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

        print("\nTreats "+str(field_name))

        # Sets the form compilation to quadrature

        self.form_compiler_parameters = {"representation": "quadrature"}

        # Defines the bilinear form of the local projection problem
        
        projection_trial = TrialFunction(field_function_space)

        self.project_test = TestFunction(field_function_space)

        self.projection_bilinear_form = inner(projection_trial, 
        self.project_test)*self.dx_object

        print("\nAssembles the bilinear form for the projection of fie"+
        "ld '"+str(field_name)+"', which was defined in a quadrature s"+
        "pace")

        self.assembled_bilinear_form = assemble(
        self.projection_bilinear_form, form_compiler_parameters=
        self.form_compiler_parameters)

        # Creates the output function for the projected expression

        if expression_function is not None:

            self.expression_function = expression_function

        # Otherwise, creates it new

        else:

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

        map_of_DOFs = field_function_space.dofmap()

        # Gets the coordinates of each quadrature DOF and reshapes them
        # to a [n_quadrature_DOFs, space_dimension]

        space_dimension = (
        functional_data_class.mesh_data_class.mesh.geometry().dim())

        quadrature_DOFs_coordinates = (
        field_function_space.tabulate_dof_coordinates())

        quadrature_DOFs_coordinates = quadrature_DOFs_coordinates.reshape(
        (-1, space_dimension))

        # Gets the total number of quadrature points in the mesh

        number_of_elements = (
        functional_data_class.mesh_data_class.mesh.num_cells())

        self.number_of_quadrature_points = int(
        quadrature_DOFs_coordinates.shape[0]/number_of_components)

        # Calculates the number of quadrature points in each finite el-
        # ement

        number_of_quadrature_points_per_element = int(
        self.number_of_quadrature_points/number_of_elements)

        if np.mod(self.number_of_quadrature_points,number_of_elements
        )!=0:

            raise IndexError("The number of finite elements in the qua"+
            "drature space of the field '"+str(field_name)+"' is "+str(
            number_of_elements)+", whereas the number of quadrature po"+
            "ints is "+str(self.number_of_quadrature_points)+". Thus, "+
            "there is no integer number of quadrature points per eleme"+
            "nt. Check the mesh or the quadrature space")

        # Constructs a numpy array of integer indices to get the quadra-
        # ture values from and to the fields in correct order

        self.numpy_array_indices = np.empty((
        self.number_of_quadrature_points, *tensor_dimension), dtype=
        dtype)

        print("\nStarts building the array of indices for the quadratu"+
        "re field '"+str(field_name)+"'")

        # Iterates over the finite elements of the quadrature mesh

        for element_index in range(number_of_elements):

            # Gets the indices of the DOFs (quadrature points) of this
            # element

            element_DOFs = map_of_DOFs.cell_dofs(element_index)

            # Retrieves the coordinates of the quadrature DOFs within
            # this element, and reshapes them to an array [
            # n_quadrature_DOFs_per_element, space_dimension]

            element_quadrature_DOFs_coordinates = (
            quadrature_DOFs_coordinates[element_DOFs,:])

            # Initializes a boolean array of quadrature DOFs that have 
            # already been found

            found_quadrature_DOFs = np.zeros(
            element_quadrature_DOFs_coordinates.shape[0], dtype=bool)

            # Initializes a list that will hold the DOFs indices for
            # each quadrature point

            DOFs_indices_per_quadrature_point = []

            # Iterates through the quadrature DOFs of the element

            for quadrature_dof_index in range(
            element_quadrature_DOFs_coordinates.shape[0]):

                # If this index was already found, skips this iteration

                if found_quadrature_DOFs[quadrature_dof_index]:

                    continue

                # Gets the coordinates of this quadrature DOF

                quadrature_DOF_coordinates = (
                element_quadrature_DOFs_coordinates[quadrature_dof_index])

                # Gets the DOFs indices of this element that are close 
                # to that coordinates

                quadrature_point_dofs_indices = np.where(np.all(
                np.isclose(element_quadrature_DOFs_coordinates, 
                quadrature_DOF_coordinates, atol=
                point_absolute_tolerance, rtol=point_relative_tolerance
                ), axis=1))[0]

                # Checks if the number of DOFs per quadrature points was
                # found

                if quadrature_point_dofs_indices.shape[0]!=(
                number_of_components):

                    raise IndexError(str(number_of_components)+" DOFs "+
                    "are required for each quadrature point in the qua"+
                    "drature space of the field '"+str(field_name)+"'."+
                    " In contrast, for the quadrature point at "+str(
                    quadrature_DOF_coordinates)+" "+str(
                    quadrature_point_dofs_indices.shape[0])+" DOFs wer"+
                    "e found. Check the tolerances at 'ExpressionProje"+
                    "ctionOntoQuadratureSpace'")

                # Appends these DOFs to the list of DOFs per quadrature
                # point

                DOFs_indices_per_quadrature_point.append(element_DOFs[
                quadrature_point_dofs_indices])

                # Updates the list of found DOFs

                found_quadrature_DOFs[quadrature_point_dofs_indices
                ] = True

            # Checks if the number of quadrature points found is the 
            # specified number of quadrature points per element

            if len(DOFs_indices_per_quadrature_point
            )!=number_of_quadrature_points_per_element:

                raise IndexError("In some quadrature element of the fi"+
                "eld '"+str(field_name)+"', "+str(len(
                DOFs_indices_per_quadrature_point))+" quadrature point"+
                "s were found, but "+str(
                number_of_quadrature_points_per_element)+" were specif"+
                "ied, check the tolerances supplied to 'ExpressionProj"+
                "ectionOntoQuadratureSpace'")

            # Reshapes the indices according to the tensor dimension

            indices_per_quadrature_point = np.asarray(
            DOFs_indices_per_quadrature_point, dtype=dtype).reshape((
            number_of_quadrature_points_per_element, *tensor_dimension))

            # Updates the DOFs of this element into the array of indices

            self.numpy_array_indices[int(element_index*
            number_of_quadrature_points_per_element):int((element_index+
            1)*number_of_quadrature_points_per_element)
            ] = indices_per_quadrature_point

        print("\Finishes building the array of indices for the quadrat"+
        "ure field '"+str(field_name)+"'")

    # Defines a function to project an ufl expression onto a quadrature
    # space

    def project_expression_onto_quadrature_space(self, expression):

        # Constructs the linear form of the projection

        projection_linear_form = inner(expression, self.project_test
        )*self.dx_object

        # Asselmbles the linear form

        assembled_linear_form = assemble(projection_linear_form, 
        form_compiler_parameters=self.form_compiler_parameters)

        # Solves the system of linear algebraic equations of the projec-
        # tion

        solve(self.assembled_bilinear_form, 
        self.expression_function.vector(), assembled_linear_form)

        # Retrieves the numerical values

        expression_values = self.expression_function.vector().get_local()

        # Returns the values reorganized into an array 
        # [n_quadrature_points, tensor_dimension]

        return expression_values[self.numpy_array_indices]

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

        function_values_array = function_values[self.numpy_array_indices]

        return function_values_array

    # Defines a function to update the quadrature points of a quadrature
    # interpolation with values coming from a numpy array 
    # [n_quadrature_points, tensor_dimension]

    def update_quadrature_values_with_numpy_array(self, 
    function_in_quadrature_space, function_values_array):

        # Sets the numpy array as values for the quadrature interpola-
        # tion

        function_vector = function_in_quadrature_space.vector(
        ).get_local()

        # Uses the indices of the DOFs per quadrature point in reverse 
        # order

        function_vector[self.numpy_array_indices] = function_values_array

        function_in_quadrature_space.vector().set_local(function_vector)

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