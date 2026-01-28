# Routine to store user expressions for multiscale boundary conditions

from dolfin import *

########################################################################
#                   Boundary conditions' expressions                   #
########################################################################

# Defines a class for an expression for a scalar linear field with res-
# pect to space coordinates

class LinearScalarFieldExpression(UserExpression):

    def __init__(self, mean_field, gradient_field, x_centroid, 
    y_centroid, z_centroid, **kwargs):

        super().__init__(**kwargs)

        # Initializes the values

        self.mean_field = mean_field

        self.gradient_field = gradient_field

        self.x_centroid = x_centroid

        self.y_centroid = y_centroid

        self.z_centroid = z_centroid

    # Defines a function to evaluate the expression

    def eval(self, value, x):

        # Gets the position vector shifting the coordinates vector by 
        # the centroid

        y = [x[0]-self.x_centroid, x[1]-self.y_centroid, (x[2]-
        self.z_centroid)]

        # Checks if the mean field is a list or else, if it is a Cons-
        # tant. If it is a Constant, gets the numerical values

        if isinstance(self.mean_field, list):

            mean_field = self.mean_field

        else:

            mean_field = [a for a in self.mean_field.values()]

        # Checks if the gradient of field is a list or else, if it is a
        # Constant. If it is a Constant, gets the numerical values

        if isinstance(self.gradient_field, list):

            gradient_field = self.gradient_field

        else:

            gradient_field = self.gradient_field.values()

        # Updates the values of the expression to the linear profile 
        # that is required

        value = (mean_field[0]+(gradient_field[0]*y[0])+(
        gradient_field[1]*y[1])+(gradient_field[2]*y[2]))

    def value_shape(self):

        return ()

# Defines a class for an expression for a vector linear field with res-
# pect to space coordinates

class LinearVectorFieldExpression(UserExpression):

    def __init__(self, mean_field, gradient_field, x_centroid, 
    y_centroid, z_centroid, **kwargs):

        super().__init__(**kwargs)

        # Initializes the values

        self.mean_field = mean_field

        self.gradient_field = gradient_field

        self.x_centroid = x_centroid
        
        self.y_centroid = y_centroid
        
        self.z_centroid = z_centroid

    # Defines a function to evaluate the expression

    def eval(self, value, x):

        # Gets the position vector shifting the coordinates vector by 
        # the centroid

        y = [x[0]-self.x_centroid, x[1]-self.y_centroid, (x[2]-
        self.z_centroid)]

        # Checks if the mean field is a list or else, if it is a Cons-
        # tant. If it is a Constant, gets the numerical values

        if isinstance(self.mean_field, list):

            mean_field = self.mean_field

        else:

            mean_field = [a for a in self.mean_field.values()]

        # Checks if the gradient of field is a list or else, if it is a
        # Constant. If it is a Constant, gets the numerical values

        if isinstance(self.gradient_field, list):

            gradient_field = self.gradient_field

        else:

            A = self.gradient_field.values()

            gradient_field = [[A[0], A[1], A[2]], [A[3], A[4], A[5]], [
            A[6], A[7], A[8]]]

        # Updates the values of the expression to the linear profile 
        # that is required

        value[0] = (mean_field[0]+(gradient_field[0][0]*y[0])+
        (gradient_field[0][1]*y[1])+(gradient_field[0][2]*y[2]
        ))

        value[1] = (mean_field[1]+(gradient_field[1][0]*y[0])+
        (gradient_field[1][1]*y[1])+(gradient_field[1][2]*y[2]
        ))

        value[2] = (mean_field[2]+(gradient_field[2][0]*y[0])+
        (gradient_field[2][1]*y[1])+(gradient_field[2][2]*y[2]
        ))

    def value_shape(self):

        return (3,)

########################################################################
#                  Boundary conditions' domain classes                 #
########################################################################

# Defines a class for a function space to have a domain constrained as
# periodic. The domain must be a cube (the dimensions don't have to be
# the same) aligned with the X, Y, and Z axes

class PeriodicCubicBoundary(SubDomain):

    def __init__(self, x_centroid, y_centroid, z_centroid, 
    mesh_dataClass, tolerance=None):

        super().__init__()

        # Initializes the values

        self.x_centroid = x_centroid
        
        self.y_centroid = y_centroid
        
        self.z_centroid = z_centroid

        # Gets the cube dimensions. To this end, gets the extreme values
        # of each coordinate

        mesh_coordinates = mesh_dataClass.mesh.coordinates()

        x_min = mesh_coordinates[:,0].min()

        x_max = mesh_coordinates[:,0].max()

        y_min = mesh_coordinates[:,1].min()

        y_max = mesh_coordinates[:,1].max()

        z_min = mesh_coordinates[:,2].min()

        z_max = mesh_coordinates[:,2].max()

        self.length_x = x_max-x_min

        self.length_y = y_max-y_min

        self.length_z = z_max-z_min

        self.semi_lengthX = 0.5*self.length_x

        self.semi_lengthY = 0.5*self.length_y

        self.semi_lengthZ = 0.5*self.length_z

        # If no tolerance is given, sets it as a thousandth of the smal-
        # lest length

        if tolerance is None:

            self.tolerance = abs(min([self.length_x, self.length_y,
            self.length_z])*0.001)

        else:

            self.tolerance = tolerance

        # Checks if the mesh is in fact a cube by comparing the volume
        # with the analytical volume

        numerical_volume = float(assemble(1.0*mesh_dataClass.dx))
        
        analytical_volume = self.length_x*self.length_y*self.length_z

        if not near(numerical_volume, analytical_volume, self.tolerance):

            raise TypeError("The numerically evaluated volume of the m"+
            "esh, "+str(numerical_volume)+", is not equal to the analy"+
            "tical volume, "+str(analytical_volume)+", thus, the mesh "+
            "is probably not a cube and periodic boundary condition ca"+
            "nnot be used")

    # Redefines the inside method to give all True for points on the 
    # master facets, which are conveined here as x=-0.5*length_x, y=-0.5
    # *length_y, and z=-0.5*length_z planes. The slave facets are falsi-
    # fied because they are dependent on the master ones

    def inside(self, x, on_boundary):

        flag_xFacet = near(self.x_centroid-self.semi_lengthX, x[0], 
        self.tolerance)

        flag_yFacet = near(self.y_centroid-self.semi_lengthY, x[1], 
        self.tolerance)

        flag_zFacet = near(self.z_centroid-self.semi_lengthZ, x[2], 
        self.tolerance)

        return bool ((flag_xFacet or flag_yFacet or flag_zFacet) and
        on_boundary)

    # Defines a method to map the field from one facet to the opposite

    def map(self, x, y):

        y[0] = x[0] - self.length_x if near(x[0], self.x_centroid+
        self.semi_lengthX, self.tolerance) else x[0]

        y[1] = x[1] - self.length_y if near(x[1], self.y_centroid+
        self.semi_lengthY, self.tolerance) else x[1]
        
        y[2] = x[2] - self.length_z if near(x[2], self.z_centroid+
        self.semi_lengthZ, self.tolerance) else x[2]

########################################################################
#                            Field updating                            #
########################################################################

# Defines a function to construct the updates of a fluctuation field gi-
# ven a first order approximation

def construct_fieldCorrections(fluctuation_field, volume_inverse, 
mesh_dataClass, macro_quantitiesClasses, constrained_fieldName, 
constrained_gradientFieldName, elements_dictionary, centroid_coordinates
):

    # Gets the number of dimensions of the primal field

    n_dimsPrimalField = len(elements_dictionary[constrained_fieldName
    ].value_shape())
    
    # Finds the centroid of the mesh

    if centroid_coordinates is None:

        position_vector = mesh_dataClass.x

        centroid_coordinates = [float(volume_inverse*assemble(
        position_vector[0]*mesh_dataClass.dx)), float(volume_inverse*
        assemble(position_vector[1]*mesh_dataClass.dx)), float(
        volume_inverse*assemble(position_vector[2]*mesh_dataClass.dx))]

    # Tests if the primal field is the fluctuation field
        
    if fluctuation_field:

        # Saves the centroid as a constant vector

        centroid_vector = Constant(centroid_coordinates)

        # Constructs the expressions for the field on the boundary con-
        # sidering zero fluctuations on the boundary

        if n_dimsPrimalField==0:

            field_expression = Constant(0.0)

            # Constructs a tensorial expression for the updating of the 
            # solution field inside the variational form; constructs an 
            # expression to update the solution field for post-proces-
            # sing; and constructs the function space to interpolate the 
            # correcting expression into

            return (field_expression, [(getattr(macro_quantitiesClasses[
            -1], constrained_fieldName)+dot(getattr(
            macro_quantitiesClasses[-1], constrained_gradientFieldName),
            (mesh_dataClass.x-centroid_vector))), 
            LinearScalarFieldExpression(getattr(macro_quantitiesClasses[
            -1], constrained_fieldName), getattr(macro_quantitiesClasses[
            -1], constrained_gradientFieldName), *centroid_coordinates), 
            FunctionSpace(mesh_dataClass.mesh, elements_dictionary[
            constrained_fieldName])], centroid_coordinates)

        elif n_dimsPrimalField==1:

            field_expression = Constant((0.0, 0.0, 0.0))

            # Constructs a tensorial expression for the updating of the 
            # solution field inside the variational form; constructs an 
            # expression to update the solution field for post-proces-
            # sing; and constructs the function space to interpolate the 
            # correcting expression into

            return (field_expression, [(getattr(macro_quantitiesClasses[
            -1], constrained_fieldName)+dot(getattr(
            macro_quantitiesClasses[-1], constrained_gradientFieldName),
            (mesh_dataClass.x-centroid_vector))), 
            LinearVectorFieldExpression(getattr(macro_quantitiesClasses[
            -1], constrained_fieldName), getattr(macro_quantitiesClasses[
            -1], constrained_gradientFieldName), *centroid_coordinates), 
            FunctionSpace(mesh_dataClass.mesh, elements_dictionary[
            constrained_fieldName])], centroid_coordinates)

        else:

            raise ValueError("There is not possible yet to use the linear "+
            "multiscale boundary condition for fields with dimensionality "+
            "larger than a 1 (a vector). The given dimensionality is "+str(
            n_dimsPrimalField))
        
    else:

        # Constructs the expressions for the field on the boundary con-
        # sidering zero fluctuations on the boundary

        if n_dimsPrimalField==0:

            return (LinearScalarFieldExpression(getattr(
            macro_quantitiesClasses[-1], constrained_fieldName),getattr(
            macro_quantitiesClasses[-1], constrained_gradientFieldName), 
            *centroid_coordinates), None, centroid_coordinates)

        elif n_dimsPrimalField==1:

            return (LinearVectorFieldExpression(getattr(
            macro_quantitiesClasses[-1], constrained_fieldName),getattr(
            macro_quantitiesClasses[-1], constrained_gradientFieldName), 
            *centroid_coordinates), None, centroid_coordinates)

        else:

            raise ValueError("There is not possible yet to use the"+
            " linear multiscale boundary condition for fields with"+
            " dimensionality larger than a 1 (a vector). The given"+
            " dimensionality is "+str(n_dimsPrimalField))