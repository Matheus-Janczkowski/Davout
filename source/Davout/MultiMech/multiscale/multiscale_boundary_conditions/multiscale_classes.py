# Routine to store classes for different multiscale boundary conditions

from dolfin import *

from abc import ABC, abstractmethod

import numpy as np

from ...tool_box import functional_tools

from ...tool_box import boundary_conditions_tools as BC_tools

from ...multiscale.multiscale_boundary_conditions import multiscale_expressions

from ....PythonicUtilities import programming_tools

########################################################################
#                Boundary conditions's classes templates               #
########################################################################

# Defines a template for the boundary conditions' classes

class BCsClassTemplate(ABC):

    @abstractmethod

    # The following methods have the pass argument only because they 
    # will be defined in the child classes. But they need to be created
    # in the child classes

    def update(self, bilinear_form, linear_form, trial_functionsDict,
    test_functionsDict, boundary_conditions, monolithic_functionSpace):

        pass

########################################################################
#                      First order homogenization                      #
########################################################################

# Defines a class for the minimally-constrained boundary condition using 
# first order homogenization

class MinimallyConstrainedFirstOrderBC(BCsClassTemplate):

    def __init__(self,  constrained_fieldName, fields_names, 
    elements_dictionary, mesh_dataClass, macro_quantitiesFilesDict, 
    macro_quantitiesClasses, volume_inverse, fluctuation_field,
    centroid_coordinates, fixed_nodeSubdomain, fixed_nodeExpression):
        
        # Gets the name of the field to be minimally constrained and its
        # gradient. But removes non ASCII characters and blank spaces so 
        # this names can be used as variables' names

        self.constrained_fieldName = functional_tools.convert_stringToASCII(
        constrained_fieldName)

        self.constrained_gradientFieldName = (constrained_fieldName+"_"+
        "gradient")
        
        # Evaluates the volume of the microscale

        if volume_inverse is None:

            self.volume_inverse = (1.0/assemble(1.0*mesh_dataClass.dx))

        else:

            self.volume_inverse = volume_inverse
        
        # Stores the macro quantities that will be used as boundary con-
        # ditions

        self.macro_quantitiesClasses = macro_quantitiesClasses

        # Swaps the keys of the files dictionary to keep the names of 
        # the field and its gradient as variables in the macro quanti-
        # ties class. Then, updates the list of macro quantities' clas-
        # ses

        self.macro_quantitiesClasses.append(
        functional_tools.MacroQuantitiesInTime(
        programming_tools.change_dictionaryKeys(
        macro_quantitiesFilesDict, [["macro field file", 
        self.constrained_fieldName], ["macro field gradient file",
        self.constrained_gradientFieldName]])))

        # Gets the number of dimensions of the primal field

        n_dimsPrimalField = len(elements_dictionary[
        self.constrained_fieldName].value_shape())

        # Constructs the fields of the Lagrange multipliers to cons-
        # train the field and its gradient

        if n_dimsPrimalField==0:

            self.lagrange_fieldElement = FiniteElement("Real", 
            mesh_dataClass.mesh.ufl_cell(), 0)

            self.lagrange_gradFieldElement = VectorElement("Real", 
            mesh_dataClass.mesh.ufl_cell(), 0)

        elif n_dimsPrimalField==1:

            self.lagrange_fieldElement = VectorElement("Real", 
            mesh_dataClass.mesh.ufl_cell(), 0)

            self.lagrange_gradFieldElement = TensorElement("Real", 
            mesh_dataClass.mesh.ufl_cell(), 0)

        else:

            # Gets the dimension of the space

            space_dimension = mesh_dataClass.mesh.topology().dim()

            # Constructs the tensor element

            dimensions_list = [space_dimension for tensor_index in range(
            n_dimsPrimalField)]

            self.lagrange_fieldElement = TensorElement("Real", 
            mesh_dataClass.mesh.ufl_cell(), 0, tuple(dimensions_list))

            # Adds oen tensorial index due to the gradient

            self.lagrange_gradFieldElement = TensorElement("Real", 
            mesh_dataClass.mesh.ufl_cell(), 0, tuple(
            dimensions_list.append(space_dimension)))
        
        # Gives names for the Lagrange multiplier fiels and adds them to
        # the list of fields' names. These names will be used later to 
        # pick up the trial and test functions relative to these Lagran-
        # ge fields

        self.lagrange_fieldName = (self.constrained_fieldName+"_lagran"+
        "ge_multiplier")

        self.lagrange_gradFieldName = (self.constrained_gradientFieldName
        +"_lagrange_multiplier")

        self.fields_names = fields_names

        self.fields_names.extend([self.lagrange_fieldName, 
        self.lagrange_gradFieldName])
        
        # Adds the elements of the Lagrange multipliers

        self.elements_dictionary = elements_dictionary

        self.elements_dictionary[self.lagrange_fieldName] = (
        self.lagrange_fieldElement)

        self.elements_dictionary[self.lagrange_gradFieldName] = (
        self.lagrange_gradFieldElement)

        # Verifies if fluctuation field flag is true, which means that 
        # the field in the microscale boundary value problem is, in 
        # fact, the fluctuation. In either case, constructs the boundary
        # condition's expression and the field correction for the fluc-
        # tuation

        self.field_expression, self.field_correction, self.centroid = (
        multiscale_expressions.construct_fieldCorrections(
        fluctuation_field, self.volume_inverse, mesh_dataClass, 
        self.macro_quantitiesClasses, self.constrained_fieldName, 
        self.constrained_gradientFieldName, elements_dictionary, 
        centroid_coordinates))

        # Just saves a node to be constrained. This node is necessary 
        # just for periodic BCs

        self.fixed_nodeSubdomain = fixed_nodeSubdomain
            
        self.fixed_nodeExpression = fixed_nodeExpression
        
    # Defines a function to in fact construct the boundary condition

    def update(self, bilinear_form, linear_form, boundary_conditions,
    trial_functionsDict, test_functionsDict, boundary_conditionIndex,
    mesh_dataClass, monolithic_functionSpace):

        # Gets the trial and test functions

        primal_fieldTrial = trial_functionsDict[
        self.constrained_fieldName]

        primal_fieldTest = test_functionsDict[self.constrained_fieldName]

        lagrange_fieldTrial = trial_functionsDict[self.lagrange_fieldName]

        lagrange_fieldTest = test_functionsDict[self.lagrange_fieldName]

        lagrange_gradFieldTrial = trial_functionsDict[
        self.lagrange_gradFieldName]

        lagrange_gradFieldTest = test_functionsDict[
        self.lagrange_gradFieldName]

        # Adds the variational bits to the bilinear form

        # Adds the work done by the constraint on the field

        bilinear_form += (self.volume_inverse*((inner(lagrange_fieldTest,
        getattr(self.macro_quantitiesClasses[boundary_conditionIndex], 
        self.constrained_fieldName))*mesh_dataClass.dx)-(inner(
        lagrange_fieldTest, primal_fieldTrial)*mesh_dataClass.dx)))

        # Adds the work done by the constraint on the gradient of the 
        # field

        bilinear_form += (self.volume_inverse*((inner(
        lagrange_gradFieldTest, getattr(self.macro_quantitiesClasses[
        boundary_conditionIndex], self.constrained_gradientFieldName))*
        mesh_dataClass.dx)-(inner(lagrange_gradFieldTest, grad(
        primal_fieldTrial))*mesh_dataClass.dx)))

        # Adds the work done by the lagrange multipliers

        linear_form += (self.volume_inverse*((inner(lagrange_fieldTrial, 
        primal_fieldTest)*mesh_dataClass.dx)+(inner(grad(primal_fieldTest
        ), lagrange_gradFieldTrial)*mesh_dataClass.dx)))

        # Returns the boundary conditions and variational forms

        return bilinear_form, linear_form, boundary_conditions
    
# Defines a class for the linear boundary condition considering first 
# order homogenization

class LinearFirstOrderBC(BCsClassTemplate):

    def __init__(self,  constrained_fieldName, fields_names, 
    elements_dictionary, mesh_dataClass, macro_quantitiesFilesDict, 
    macro_quantitiesClasses, volume_inverse, fluctuation_field, 
    centroid_coordinates, fixed_nodeSubdomain, fixed_nodeExpression):
        
        # Gets the name of the field to be minimally constrained and its
        # gradient. But removes non ASCII characters and blank spaces so 
        # this names can be used as variables' names

        self.constrained_fieldName = functional_tools.convert_stringToASCII(
        constrained_fieldName)

        self.constrained_gradientFieldName = (constrained_fieldName+"_"+
        "gradient")
        
        # Evaluates the volume of the microscale

        if volume_inverse is None:

            self.volume_inverse = (1.0/assemble(1.0*mesh_dataClass.dx))

        else:

            self.volume_inverse = volume_inverse
        
        # Stores the macro quantities that will be used as boundary con-
        # ditions

        self.macro_quantitiesClasses = macro_quantitiesClasses

        # Swaps the keys of the files dictionary to keep the names of 
        # the field and its gradient as variables in the macro quanti-
        # ties class. Then, updates the list of macro quantities' clas-
        # ses

        self.macro_quantitiesClasses.append(
        functional_tools.MacroQuantitiesInTime(
        programming_tools.change_dictionaryKeys(
        macro_quantitiesFilesDict, [["macro field file", 
        self.constrained_fieldName], ["macro field gradient file",
        self.constrained_gradientFieldName]])))

        # Finds the index of this field to be constrained

        self.field_index = 0

        for i in range(len(fields_names)):

            if fields_names[i]==self.constrained_fieldName:

                self.field_index = i 

                break

        # Saves the atributes that will be retrieved again for latter
        # classes
        
        self.elements_dictionary = elements_dictionary

        self.fields_names = fields_names

        # Verifies if fluctuation field flag is true, which means that 
        # the field in the microscale boundary value problem is, in 
        # fact, the fluctuation. In either case, constructs the boundary
        # condition's expression and the field correction for the fluc-
        # tuation

        self.field_expression, self.field_correction, self.centroid  = (
        multiscale_expressions.construct_fieldCorrections(
        fluctuation_field, self.volume_inverse, mesh_dataClass, 
        self.macro_quantitiesClasses, self.constrained_fieldName, 
        self.constrained_gradientFieldName, elements_dictionary,
        centroid_coordinates))

        # Just saves a node to be constrained. This node is necessary 
        # just for periodic BCs

        self.fixed_nodeSubdomain = fixed_nodeSubdomain
            
        self.fixed_nodeExpression = fixed_nodeExpression
        
    # Defines a function to in fact construct the boundary condition

    def update(self, bilinear_form, linear_form, boundary_conditions,
    trial_functionsDict, test_functionsDict, boundary_conditionIndex, 
    mesh_dataClass, monolithic_functionSpace):
        
        # Adds the zero fluctuation boundary condtion. Checks first if
        # the formulation has multiple fields

        if monolithic_functionSpace.ufl_element().family()=="Mixed":

            boundary_conditions.append(DirichletBC(
            monolithic_functionSpace.sub(self.field_index), 
            self.field_expression, "on_boundary"))

        else:

            boundary_conditions.append(DirichletBC(
            monolithic_functionSpace, self.field_expression, "on_bound"+
            "ary"))

        # Returns the boundary conditions and variational forms

        return bilinear_form, linear_form, boundary_conditions
    
# Defines a class for the periodic boundary condition considering first 
# order homogenization

class PeriodicFirstOrderBC(BCsClassTemplate):

    def __init__(self,  constrained_fieldName, fields_names, 
    elements_dictionary, mesh_dataClass, macro_quantitiesFilesDict, 
    macro_quantitiesClasses, volume_inverse, fluctuation_field, 
    centroid_coordinates, fixed_nodeSubdomain, fixed_nodeExpression):
        
        # Gets the name of the field to be minimally constrained and its
        # gradient. But removes non ASCII characters and blank spaces so 
        # this names can be used as variables' names

        self.constrained_fieldName = functional_tools.convert_stringToASCII(
        constrained_fieldName)

        self.constrained_gradientFieldName = (constrained_fieldName+"_"+
        "gradient")
        
        # Evaluates the volume of the microscale

        if volume_inverse is None:

            self.volume_inverse = (1.0/assemble(1.0*mesh_dataClass.dx))

        else:

            self.volume_inverse = volume_inverse
        
        # Stores the macro quantities that will be used as boundary con-
        # ditions

        self.macro_quantitiesClasses = macro_quantitiesClasses

        # Swaps the keys of the files dictionary to keep the names of 
        # the field and its gradient as variables in the macro quanti-
        # ties class. Then, updates the list of macro quantities' clas-
        # ses

        self.macro_quantitiesClasses.append(
        functional_tools.MacroQuantitiesInTime(
        programming_tools.change_dictionaryKeys(
        macro_quantitiesFilesDict, [["macro field file", 
        self.constrained_fieldName], ["macro field gradient file",
        self.constrained_gradientFieldName]])))

        # Finds the index of this field to be constrained

        self.field_index = 0

        for i in range(len(fields_names)):

            if fields_names[i]==self.constrained_fieldName:

                self.field_index = i 

                break

        # Saves the atributes that will be retrieved again for latter
        # classes
        
        self.elements_dictionary = elements_dictionary

        self.fields_names = fields_names

        # Verifies if fluctuation field flag is true, which means that 
        # the field in the microscale boundary value problem is, in 
        # fact, the fluctuation. In either case, constructs the boundary
        # condition's expression and the field correction for the fluc-
        # tuation

        self.field_expression, self.field_correction, self.centroid = (
        multiscale_expressions.construct_fieldCorrections(
        fluctuation_field, self.volume_inverse, mesh_dataClass, 
        self.macro_quantitiesClasses, self.constrained_fieldName, 
        self.constrained_gradientFieldName, elements_dictionary,
        centroid_coordinates))

        # Makes a subdomain class to capture the region where to apply
        # the boundary condition to fixed node. Finds the node that is
        # closest to the centroid

        if (fixed_nodeSubdomain is None) or (fixed_nodeExpression is None
        ):

            self.fixed_nodeSubdomain = BC_tools.generate_nodeSubdomain(
            self.centroid, mesh_dataClass)

            # Gets the expression for the boundary condition

            self.fixed_nodeExpression = Constant(np.zeros(
            self.elements_dictionary[self.constrained_fieldName
            ].value_shape()))

        else:

            self.fixed_nodeSubdomain = fixed_nodeSubdomain
            
            self.fixed_nodeExpression = fixed_nodeExpression
        
    # Defines a function to in fact construct the boundary condition

    def update(self, bilinear_form, linear_form, boundary_conditions,
    trial_functionsDict, test_functionsDict, boundary_conditionIndex, 
    mesh_dataClass, monolithic_functionSpace):
        
        # Checks first if the formulation has multiple fields

        if monolithic_functionSpace.ufl_element().family()=="Mixed":

            boundary_conditions.append(DirichletBC(
            monolithic_functionSpace.sub(self.field_index), 
            self.fixed_nodeExpression, self.fixed_nodeSubdomain, method=
            "pointwise"))

        else:

            boundary_conditions.append(DirichletBC(
            monolithic_functionSpace, self.fixed_nodeExpression, 
            self.fixed_nodeSubdomain, method="pointwise"))

        # Returns the boundary conditions and variational forms

        return bilinear_form, linear_form, boundary_conditions