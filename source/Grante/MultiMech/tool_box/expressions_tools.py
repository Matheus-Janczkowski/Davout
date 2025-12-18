# Routine to store user expressions

from dolfin import *

import numpy as np

from ...PythonicUtilities.function_tools import get_functions_arguments

from ...MultiMech.tool_box.mesh_handling_tools import get_domain_dofs_to_physical_group

from ...MultiMech.tool_box.functional_tools import construct_monolithicFunctionSpace

# Defines a function to interpolate a python function into a finite ele-
# ment space, the function must return a scalar

def interpolate_scalar_function(scalar_function, function_space, name=
None, mesh_data_class=None):

    # Verifies the number of arguments, it must be only one: the position
    # vector in the mesh

    number_of_arguments = get_functions_arguments(scalar_function, 
    number_of_arguments_only=True, positional_arguments_only=True)

    if number_of_arguments!=1:

        raise ValueError("A scalar function is to be interpolated onto"+
        " a finite element space, but it has "+str(number_of_arguments)+
        " arguments, whereas it should have just 1, the position vector")
    
    # Verifies if the function space is a dictionary, which means it is
    # the set of instructions to construct the function space

    return_functional_data = False

    if isinstance(function_space, dict):

        if mesh_data_class is None:

            raise ValueError("'function_space' provided to 'interpolat"+
            "e_scalar_function' is a dictionary, thus this dictionary "+
            "must be a set of instructions to construct the function s"+
            "pace. But no mesh_data_class was provided")
        
        # Creates the function space

        function_data_class = construct_monolithicFunctionSpace(
        function_space, mesh_data_class, all_data_must_be_provided=
        False)

        function_space = function_data_class.monolithic_function_space

        return_functional_data = True
    
    # Verifies if the element has Lagrangian interpolation functions

    if not (function_space.ufl_element().family()=="Lagrange"):

        raise TypeError("The function space is not lagrangian, but "+str(
        function_space.ufl_element().family())+". Thus the expression "+
        "cannot be interpolated. You must choose a 'CG', 'Lagrange', o"+
        "r 'P' finite element family")
    
    # Gets the coordinates of the DOFs

    dofs_coordinates = function_space.tabulate_dof_coordinates()

    # Gets a dictionary of the DOFs to physical group if the mesh data
    # class is given

    dofs_dictionary = {}

    scalar_arguments = {}

    if mesh_data_class is not None:

        dofs_dictionary = get_domain_dofs_to_physical_group(
        mesh_data_class, function_space)

        # Gets the arguments of the scalar function

        scalar_arguments = get_functions_arguments(scalar_function)

    # Gets the values of the function in the nodes

    nodes_values = np.zeros(len(dofs_coordinates))

    if "current_physical_group" in scalar_arguments:

        for dof in range(len(dofs_coordinates)):

            current_physical_group = None

            for physical_name, physical_dofs in dofs_dictionary.items():

                if dof in physical_dofs:

                    current_physical_group = physical_name

            nodes_values[dof] = scalar_function(dofs_coordinates[dof],
            current_physical_group=current_physical_group)

    else:

        for dof in range(len(dofs_coordinates)):

            nodes_values[dof] = scalar_function(dofs_coordinates[dof])

    # Creates a Function element over the finite element space

    function_object = Function(function_space)

    # Updates the vector of parameters

    function_object.vector()[:] = nodes_values

    # Renames it if the name is not None

    if name is not None:

        function_object.rename(name, "DNS")

    # Returns the function object

    if return_functional_data:

        # Updates the function object

        function_data_class.monolithic_solution = function_object

        return function_object, function_data_class

    return function_object

# Defines a function to interpolate a vector-valued or tensor-valued 
# function on a function space. The function must return a numpy array 
# and it must receive as argument a position vector and the number of 
# the component to be evaluated

def interpolate_tensor_function(vector_function, function_space, name=
None, mesh_data_class=None):

    # Verifies the number of arguments, it must be only one: the position
    # vector in the mesh

    number_of_arguments = get_functions_arguments(vector_function, 
    number_of_arguments_only=True)

    if number_of_arguments!=2:

        raise ValueError("A vector-valued function is to be interpolat"+
        "ed onto a finite element space, but it has "+str(
        number_of_arguments)+" arguments, whereas it should have just "+
        "2, the position vector and the local component to be evaluate"+
        "d. Example: ([x,y,z], 2) => the second component of a tensor "+
        "will be evaluated at x, y, z")
    
    # Verifies if the function space is a dictionary, which means it is
    # the set of instructions to construct the function space

    return_functional_data = False

    if isinstance(function_space, dict):

        if mesh_data_class is None:

            raise ValueError("'function_space' provided to 'interpolat"+
            "e_scalar_function' is a dictionary, thus this dictionary "+
            "must be a set of instructions to construct the function s"+
            "pace. But no mesh_data_class was provided")
        
        # Creates the function space

        function_data_class = construct_monolithicFunctionSpace(
        function_space, mesh_data_class, all_data_must_be_provided=
        False)

        function_space = function_data_class.monolithic_function_space

        return_functional_data = True
    
    # Verifies if the element has Lagrangian interpolation functions

    if not (function_space.ufl_element().family()=="Lagrange"):

        raise TypeError("The function space is not lagrangian, but "+str(
        function_space.ufl_element().family())+". Thus the expression "+
        "cannot be interpolated. You must choose a 'CG', 'Lagrange', o"+
        "r 'P' finite element family")
    
    # Gets the coordinates of the DOFs

    dofs_coordinates = function_space.tabulate_dof_coordinates()

    # Gets the dimension of the field from the function space

    field_number_of_components = int(np.prod(function_space.ufl_element(
    ).value_shape()))

    # Gets the values of the function in the DOFs

    DOFs_values = np.zeros(function_space.dim())

    for dof in range(function_space.dim()):

        # Gets the coordinates of the DOF

        dof_coordinate = dofs_coordinates[dof]

        # Evaluates the function and updates the corresponding local 
        # number of the DOF only

        DOFs_values[dof] = vector_function(dof_coordinate, dof % (
        field_number_of_components))

    # Creates a Function element over the finite element space

    function_object = Function(function_space)

    # Updates the vector of parameters

    function_object.vector()[:] = DOFs_values

    # Renames it if the name is not None

    if name is not None:

        function_object.rename(name, "DNS")

    # Returns the function object

    if return_functional_data:

        # Updates the function object

        function_data_class.monolithic_solution = function_object

        return function_object, function_data_class

    return function_object