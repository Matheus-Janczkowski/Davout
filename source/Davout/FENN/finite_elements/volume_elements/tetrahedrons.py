# Routine to store classes of tetrahedron finite elements. Each class is
# made for a type of finite element

import tensorflow as tf

from ...tool_box.tensorflow_utilities import convert_object_to_tensor

from ...tool_box.math_tools import jacobian_3D_element

# Defines a class to store the tetrahedron element 

class Tetrahedron:

    # Creates a dictionary with the types of elements created by this 
    # class. The keys are GMSH element type, and the values are other 
    # dictionaries with necessary information

    stored_elements = {11: {"polynomial degree": 2, "number of nodes": 
    10, "name": "tetrahedron of 10 nodes", "indices of the gmsh conn"+
    "ectivity": [1, 2, 3, 0, 5, 8, 7, 4, 9, 6], "suitable boundary e"+
    "lement type tag": 9}}

    def __init__(self, node_coordinates, dofs_per_element, 
    polynomial_degree=2, quadrature_degree=2, dtype=tf.float32, 
    integer_dtype=tf.int32):
        
        # Saves the numerical type

        self.dtype = dtype

        self.integer_dtype = integer_dtype

        # Saves the number of elements

        self.number_elements = len(node_coordinates)

        # Ensures node coordinates and dofs per element are tensors with 
        # the given type

        node_coordinates = convert_object_to_tensor(node_coordinates,
        self.dtype)

        # The dofs per element is a tensor [n_elements, n_nodes, 
        # n_dofs_per_node]

        self.dofs_per_element = convert_object_to_tensor(
        dofs_per_element, self.integer_dtype)
        
        # Evaluates the Gauss points and their corresponding weights

        self.get_quadrature_points(quadrature_degree)

        # Evaluates the u quantity

        self.u = 1.0-self.r-self.s-self.t

        # Precomputes the shape functions in the Gauss points

        if polynomial_degree==2:

            self.make_quadratic_shape_functions()

        else:

            raise ValueError("'polynomial_degree' was given as "+str(
            polynomial_degree)+". However, only "+str(polynomial_degree
            )+" is currently implemented")
        
        # Precomputes the shape functions and their first order deriva-
        # tives in the original finite element coordinates. This is car-
        # ried out for all elements

        self.evaluate_shape_function_and_derivatives(node_coordinates)

    # Defines a function to store the quadrature points and the corres-
    # ponding weights with respect to the quadrature degree

    def get_quadrature_points(self, quadrature_degree):

        # Selects a quadrature degree of 1, thus, there is a single point

        if quadrature_degree==1:

            self.r = tf.constant([0.25], dtype=self.dtype)

            self.s = tf.constant([0.25], dtype=self.dtype)

            self.t = tf.constant([0.25], dtype=self.dtype)

            self.weights = tf.constant([1.0/6.0], dtype=self.dtype)

        # Selects a quadrature degree of 2, thus, there are four points

        elif quadrature_degree==2:

            self.r = tf.constant([0.1381966011250105, 0.5854101966249685, 
            0.1381966011250105, 0.1381966011250105], dtype=self.dtype)

            self.s = tf.constant([0.1381966011250105, 0.1381966011250105, 
            0.5854101966249685, 0.1381966011250105], dtype=self.dtype)

            self.t = tf.constant([0.1381966011250105, 0.1381966011250105, 
            0.1381966011250105, 0.5854101966249685], dtype=self.dtype)

            self.weights = tf.constant([1.0/24.0, 1.0/24.0, 1.0/24.0,
            1.0/24.0], dtype=self.dtype)

        # Selects a quadrature degree of 3, thus, there are five points

        elif quadrature_degree==3:

            self.r = tf.constant([0.25, 0.5, 1.0/6.0, 1.0/6.0, 1.0/6.0], 
            dtype=self.dtype)

            self.s = tf.constant([0.25, 1.0/6.0, 0.5, 1.0/6.0, 1.0/6.0], 
            dtype=self.dtype)

            self.t = tf.constant([0.25, 1.0/6.0, 1.0/6.0, 0.5, 1.0/6.0], 
            dtype=self.dtype)

            self.weights = tf.constant([-2.0/15.0, 3.0/40.0, 3.0/40.0, 
            3.0/40.0, 3.0/40.0], dtype=self.dtype)

        else:

            raise ValueError("A quadrature degree of "+str(
            quadrature_degree)+" was asked to created tetrahedral fini"+
            " elements. But the only available degrees are 1 (one poin"+
            "t), 2 (four point), 3 (five points).")

        # Saves the number of quadrature points

        self.number_quadrature_points = self.weights.shape[0]

    # Defines a function to calculate quadratic shape functions for a 10-
    # node tetrahedron

    def make_quadratic_shape_functions(self):

        # All shape functions ahead will have added a new dimension to 
        # allow for concatenation and further 
        
        ################################################################
        #                        Shape functions                       #
        ################################################################

        # First node: r = 1, s = 0, t = 0

        N_1 = (self.r*((2*self.r)-1.0))[..., tf.newaxis]

        # Second node: r = 0, s = 1, t = 0

        N_2 = (self.s*((2*self.s)-1.0))[..., tf.newaxis]

        # Third node: r = 0, s = 0, t = 1

        N_3 = (self.t*((2*self.t)-1.0))[..., tf.newaxis]

        # Fourth node: r = 0, s = 0, t = 0

        N_4 = (self.u*((2*self.u)-1.0))[..., tf.newaxis]

        # Fifth node: r = 0.5, s = 0.5, t = 0

        N_5 = (4*self.r*self.s)[..., tf.newaxis]

        # Sixth node: r = 0, s = 0.5, t = 0.5

        N_6 = (4*self.s*self.t)[..., tf.newaxis]

        # Seventh node: r = 0, s = 0, t = 0.5

        N_7 = (4*self.t*self.u)[..., tf.newaxis]

        # Eigth node: r = 0.5, s = 0, t = 0

        N_8 = (4*self.r*self.u)[..., tf.newaxis]

        # Nineth node: r = 0.5, s = 0, t = 0.5

        N_9 = (4*self.r*self.t)[..., tf.newaxis]

        # Tenth node: r = 0, s = 0.5, t = 0

        N_10 = (4*self.s*self.u)[..., tf.newaxis]

        # Concatenates all shape function into a single tensor

        self.shape_functions_tensor = tf.concat((N_1, N_2, N_3, N_4, N_5, 
        N_6, N_7, N_8, N_9, N_10), axis=-1)
        
        ################################################################
        #                  Shape functions derivatives                 #
        ################################################################

        # Computes expressions that are needed for the evaluation of the
        # derivatives of the shape function with respect to the natural
        # coordinates

        dN1_dr = ((4*self.r)-1.0)[..., tf.newaxis]

        dN2_ds = ((4*self.s)-1.0)[..., tf.newaxis]

        dN3_dt = ((4*self.t)-1.0)[..., tf.newaxis]

        dN4_dr = (1.0-(4*self.u))[..., tf.newaxis]

        dN5_dr = (4*self.s)[..., tf.newaxis]

        dN5_ds = (4*self.r)[..., tf.newaxis]

        dN6_ds = (4*self.t)[..., tf.newaxis]

        quadruple_u = (4*self.u)[..., tf.newaxis]

        null_vector = tf.zeros_like(N_1)

        # Computes the actual concatenated array of derivatives. Each 
        # array encompasses the derivatives of the ten shape functions
        # with respect to a single natural coordinate

        dN_dr = tf.concat([dN1_dr, null_vector, null_vector, dN4_dr, 
        dN5_dr, null_vector, -dN6_ds, quadruple_u-dN5_ds, dN6_ds, -dN5_dr
        ], axis=-1)

        dN_ds = tf.concat([null_vector, dN2_ds, null_vector, dN4_dr, 
        dN5_ds, dN6_ds, -dN6_ds, -dN5_ds, null_vector, quadruple_u-dN5_dr
        ], axis=-1)

        dN_dt = tf.concat([null_vector, null_vector, dN3_dt, dN4_dr,
        null_vector, dN5_dr, quadruple_u-dN6_ds, -dN5_ds, dN5_ds, -dN5_dr
        ], axis=-1)

        # Compacts them into a single array

        self.natural_derivatives_N = tf.stack([dN_dr, dN_ds, dN_dt], 
        axis=-1)

    # Defines a function to return the shape functions evaluated at the
    # original coordinates of the finite element

    def evaluate_shape_function_and_derivatives(self, nodes_coordinates):

        """Computes the shape functions and their derivatives in the 
        original system of coordinates of the finite elements. Computes
        jacobians to perform the mapping of the derivatives"""

        # Gets the x, y, and z coordinates of the nodes. Adds the new a-
        # xis in the middle to compatibilize it with the dimension of 
        # quadrature points. It is important to note that the nodes here
        # denote the midpoints too. Just like in the book The Finite El-
        # ement Method by Hughes

        x = nodes_coordinates[..., 0]

        y = nodes_coordinates[..., 1]

        z = nodes_coordinates[..., 2]

        # Gets the jacobian determinant and its inverse

        det_J, J_inv = jacobian_3D_element(self.natural_derivatives_N, x, 
        y, z)

        # The jacobian inverse is a tensor of [elements, quadrature 
        # points, original coordinates, natural coordinates]. Whereas the
        # natural derivatives are a tensor of [quadrature points, nodes,
        # natural coordinates]. Thus, the derivatives of the shape func-
        # tions in the original coordinates are a tensor of [elements,
        # quadrature points, nodes, original coordinates]

        self.shape_functions_derivatives = tf.einsum('eqxr,qnr->eqnx', 
        J_inv, self.natural_derivatives_N)

        # Multiplies the quadrature weights by the determinant of the 
        # jacobian transformation to have the correct integration measure

        self.dx = tf.einsum('eq,q->eq', det_J, self.weights)

    # Defines a function to recover the DOFs of the current field using 
    # an indices tensor [n_elements, n_nodes, n_physical_dimensions]

    def get_field_dofs(self, field_vector):

        # Gathers the field to get a tensor with the same dimensions as
        # the indices tensor

        return tf.gather(field_vector, self.dofs_per_element)