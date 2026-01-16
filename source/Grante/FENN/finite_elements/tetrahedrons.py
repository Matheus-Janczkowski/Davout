# Routine to store classes of tetrahedron finite elements. Each class is
# made for a type of finite element

import tensorflow as tf

from ..tool_box.math_tools import jacobian_3D_element

# Defines a class to store the tetrahedron element with quadratic shape
# functions and 10 nodes

class Tetrahedron:

    def __init__(self):
        
        pass 

    # Defines a function to return the 10 quadratic shape functions

    def get_shape_functions(self, r, s, t, nodes_coordinates):

        # All shape functions ahead will have added a new dimension to 
        # allow for concatenation and further 
        
        ################################################################
        #                        Shape functions                       #
        ################################################################

        # First node: r = 1, s = 0, t = 0

        N_1 = (r*((2*r)-1.0))[..., tf.newaxis]

        # Second node: r = 0, s = 1, t = 0

        N_2 = (s*((2*s)-1.0))[..., tf.newaxis]

        # Third node: r = 0, s = 0, t = 1

        N_3 = (t*((2*t)-1.0))[..., tf.newaxis]

        # Evaluates the u quantity

        u = 1.0-r-s-t

        # Fourth node: r = 0, s = 0, t = 0

        N_4 = (u*((2*u)-1.0))[..., tf.newaxis]

        # Fifth node: r = 0.5, s = 0.5, t = 0

        N_5 = (4*r*s)[..., tf.newaxis]

        # Sixth node: r = 0, s = 0.5, t = 0.5

        N_6 = (4*s*t)[..., tf.newaxis]

        # Seventh node: r = 0, s = 0, t = 0.5

        N_7 = (4*t*u)[..., tf.newaxis]

        # Eigth node: r = 0.5, s = 0, t = 0

        N_8 = (4*r*u)[..., tf.newaxis]

        # Nineth node: r = 0.5, s = 0, t = 0.5

        N_9 = (4*r*t)[..., tf.newaxis]

        # Tenth node: r = 0, s = 0.5, t = 0

        N_10 = (4*s*u)[..., tf.newaxis]

        # Concatenates all shape function into a single tensor

        shape_functions_tensor = tf.concat((N_1, N_2, N_3, N_4, N_5, N_6, 
        N_7, N_8, N_9, N_10), axis=-1)
        
        ################################################################
        #                  Shape functions derivatives                 #
        ################################################################

        # Computes expressions that are needed for the evaluation of the
        # derivatives of the shape function with respect to the natural
        # coordinates

        dN1_dr = ((4*r)-1.0)[..., tf.newaxis]

        dN2_ds = ((4*s)-1.0)[..., tf.newaxis]

        dN3_dt = ((4*t)-1.0)[..., tf.newaxis]

        dN4_dr = (1.0-(4*u))[..., tf.newaxis]

        dN5_dr = (4*s)[..., tf.newaxis]

        dN5_ds = (4*r)[..., tf.newaxis]

        dN6_ds = (4*t)[..., tf.newaxis]

        quadruple_u = (4*u)[..., tf.newaxis]

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

        natural_derivatives_N = tf.stack([dN_dr, dN_ds, dN_dt], axis=-1)

        # Gets the x, y, and z coordinates of the nodes. Adds the new a-
        # xis in the middle to compatibilize it with the dimension of 
        # quadrature points. It is important to note that the nodes here
        # denote the midpoints too. Just like in the book The Finite El-
        # ement Method by Hughes

        x = nodes_coordinates[..., 0]

        y = nodes_coordinates[..., 1]

        z = nodes_coordinates[..., 2]

        # Gets the jacobian determinant and its inverse

        det_J, J_inv = jacobian_3D_element(natural_derivatives_N, x, y, 
        z)

        # The jacobian inverse is a tensor of [elements, quadrature 
        # points, original coordinates, natural coordinates]. Whereas the
        # natural derivatives are a tensor of [quadrature points, nodes,
        # natural coordinates]. Thus, the derivatives of the shape func-
        # tions in the original coordinates are a tensor of [elements,
        # quadrature points, nodes, original coordinates]

        shape_functions_derivatives = tf.einsum('eqxr,qnr->eqnx', J_inv, 
        natural_derivatives_N)

        # Gets the derivatives of the 

        return shape_functions_tensor, shape_functions_derivatives, det_J