# Routine to store classes of tetrahedron finite elements. Each class is
# made for a type of finite element

import tensorflow as tf

from ...tool_box.tensorflow_utilities import convert_object_to_tensor

from ...tool_box.math_tools import jacobian_2D_element

# Defines a class to store the triangle element

class Triangle:

    # Creates a dictionary with the types of elements created by this 
    # class. The keys are GMSH element type, and the values are other 
    # dictionaries with necessary information

    stored_elements = {9: {"polynomial degree": 2, "number of nodes": 6, 
    "name": "triangle of 6 nodes", "indices of the gmsh connectivity": [
    1, 2, 0, 4, 5, 3]}}

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

        # Verifies if this is a triangle embedded in a 3D space by coun-
        # ting the number of coordinates for each node

        if node_coordinates.shape[2]>2:

            self.triangle_in_3D_space = True 

        else:

            self.triangle_in_3D_space = False

        # The dofs per element is a tensor [n_elements, n_nodes, 
        # n_dofs_per_node]

        self.dofs_per_element = convert_object_to_tensor(
        dofs_per_element, self.integer_dtype)
        
        # Evaluates the Gauss points and their corresponding weights

        self.get_quadrature_points(quadrature_degree)

        # Evaluates the u quantity

        self.u = 1.0-self.r-self.s

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

            self.r = tf.constant([1.0/3.0], dtype=self.dtype)

            self.s = tf.constant([1.0/3.0], dtype=self.dtype)

            self.weights = tf.constant([1.0/2.0], dtype=self.dtype)

        # Selects a quadrature degree of 2, thus, there are three points

        elif quadrature_degree==2:

            self.r = tf.constant([0.5, 0.5, 0.0], dtype=self.dtype)

            self.s = tf.constant([0.5, 0.0, 0.5], dtype=self.dtype)

            self.weights = tf.constant([1.0/6.0, 1.0/6.0, 1.0/6.0], 
            dtype=self.dtype)

        # Selects a quadrature degree of 3, thus, there are four points

        elif quadrature_degree==3:

            self.r = tf.constant([1.0/3.0, 0.6, 0.2, 0.2], dtype=
            self.dtype)

            self.s = tf.constant([1.0/3.0, 0.2, 0.6, 0.2], dtype=
            self.dtype)

            self.weights = tf.constant([-27.0/96.0, 25.0/96.0, 25.0/96.0, 
            25.0/96.0], dtype=self.dtype)

        else:

            raise ValueError("A quadrature degree of "+str(
            quadrature_degree)+" was asked to created triangle fini el"+
            "ements. But the only available degrees are 1 (one point),"+
            " 2 (three point), 3 (four points).")

        # Saves the number of quadrature points

        self.number_quadrature_points = self.weights.shape[0]

    # Defines a function to calculate quadratic shape functions for a 6-
    # node triangle

    def make_quadratic_shape_functions(self):

        # All shape functions ahead will have added a new dimension to 
        # allow for concatenation and further 
        
        ################################################################
        #                        Shape functions                       #
        ################################################################

        # First node: r = 1, s = 0
        N_1 = (self.r*((2*self.r)-1.0))[..., tf.newaxis]

        # Second node: r = 0, s = 1

        N_2 = (self.s*((2*self.s)-1.0))[..., tf.newaxis]

        # Third node: r = 0, s = 0

        N_3 = (self.u*((2*self.u)-1.0))[..., tf.newaxis]

        # Fourth node: r = 0.5, s = 0.5

        N_4 = (4*self.r*self.s)[..., tf.newaxis]

        # Fifth node: r = 0, s = 0.5

        N_5 = (4*self.s*self.u)[..., tf.newaxis]

        # Sixth node: r = 0.5, s = 0

        N_6 = (4*self.r*self.u)[..., tf.newaxis]

        # Concatenates all shape function into a single tensor

        self.shape_functions_tensor = tf.concat((N_1, N_2, N_3, N_4, N_5, 
        N_6), axis=-1)
        
        ################################################################
        #                  Shape functions derivatives                 #
        ################################################################

        # Computes expressions that are needed for the evaluation of the
        # derivatives of the shape function with respect to the natural
        # coordinates

        dN1_dr = ((4*self.r)-1.0)[..., tf.newaxis]

        dN2_ds = ((4*self.s)-1.0)[..., tf.newaxis]

        dN3_dr = (1.0-(4*self.u))[..., tf.newaxis]

        dN4_dr = (4*self.s)[..., tf.newaxis]

        dN4_ds = (4*self.r)[..., tf.newaxis]

        quadruple_u = (4*self.u)[..., tf.newaxis]

        null_vector = tf.zeros_like(N_1)

        # Computes the actual concatenated array of derivatives. Each 
        # array encompasses the derivatives of the ten shape functions
        # with respect to a single natural coordinate

        dN_dr = tf.concat([dN1_dr, null_vector, dN3_dr, dN4_dr, -dN4_dr, 
        quadruple_u-dN4_ds], axis=-1)

        dN_ds = tf.concat([null_vector, dN2_ds, dN3_dr, dN4_ds, 
        quadruple_u-dN4_dr, -dN4_ds], axis=-1)

        # Compacts them into a single array

        self.natural_derivatives_N = tf.stack([dN_dr, dN_ds], axis=-1)

    # Defines a function to return the shape functions evaluated at the
    # original coordinates of the finite element

    def evaluate_shape_function_and_derivatives(self, nodes_coordinates):

        """Computes the shape functions and their derivatives in the 
        original system of coordinates of the finite elements. Computes
        jacobians to perform the mapping of the derivatives"""

        # Gets the x, and y coordinates of the nodes. It is important to 
        # note that the nodes here denote the midpoints too. Just like 
        # in the book The Finite Element Method by Hughes

        x = nodes_coordinates[..., 0]

        y = nodes_coordinates[..., 1]

        # If it is a triangle embedded in a 3D space, the tridimensional
        # coordinates must be projected onto a 2D coordinate system lo-
        # cal to the element

        if self.triangle_in_3D_space:

            # Gets the z coordinate

            z = nodes_coordinates[..., 2]

            # Translates every node by the third node

            x -= x[:,2:3]

            y -= y[:,2:3]

            z -= z[:,2:3]

            # Stacks the coordinates as three vectors

            vectors = tf.stack([x, y, z], axis=-1)

            # Retrieves the first two vectors, that are embedded in the
            # plane of the element

            v_1 = vectors[:,0,:]

            v_2 = vectors[:,1,:]

            # Evaluates the outward pointing vector using the cross pro-
            # duct

            e_3 = tf.linalg.cross(v_1, v_2)

            # Normalizes e1 and e3

            eps = tf.keras.backend.epsilon()

            e_1 = v_1/(tf.linalg.norm(v_1, axis=-1, keepdims=True)+eps)

            e_3 = e_3/(tf.linalg.norm(e_3, axis=-1, keepdims=True)+eps)

            # Now, computes e2 again to keep all of them orthogonal to
            # each other

            e_2 = tf.linalg.cross(e_3, e_1)

            # Projects all vector onto the new e1 and e2 vectors, to the
            # the coordinates on the local system of each element

            x = tf.einsum('eni,ei->en', vectors, e_1)

            y = tf.einsum('eni,ei->en', vectors, e_2)

        # Gets the jacobian determinant and its inverse

        det_J, J_inv = jacobian_2D_element(self.natural_derivatives_N, x, 
        y)

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