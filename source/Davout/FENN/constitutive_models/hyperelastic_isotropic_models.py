# Routine to store classes of isotropic hyperelastic constitutive models.
# A class is defined for each constitutive model

import tensorflow as tf

from ..tool_box.math_tools import get_inverse

# Defines a class for a Neo Hookean hyperelastic model

class NeoHookean:

    def __init__(self, material_properties, mesh_data_class):
        
        # Gets the material parameters

        E = material_properties["E"]

        nu = material_properties["nu"]

        # Evaluates the Lamé parameters

        self.mu = tf.constant(E/(2*(1+nu)), dtype=mesh_data_class.dtype)

        self.lmbda = tf.constant((nu*E)/((1+nu)*(1-2*nu)), dtype=
        mesh_data_class.dtype)

        # Initializes the identity tensor attribute that the code will
        # automatically fill it later

        self.identity_tensor = None

    # Defines a function to evaluate the free energy density

    @tf.function
    def strain_energy(self, F):

        # Evaluates the right Cauchy-Green strain tensor
        
        C = tf.matmul(F, F, transpose_a=True)

        # Evaluates its invariants

        I1_C = tf.linalg.trace(C)

        J  = tf.linalg.det(F)

        ln_J = tf.math.log(J)

        # Calculates the Helmholtz potential

        return ((0.5*self.mu*(I1_C-3))-(self.mu*ln_J)+((0.5*self.lmbda)*(
        ln_J**2)))
    
    # Defines a function to get the first Piola-Kirchhoff stress tensor

    @tf.function
    def first_piola_kirchhoff(self, F):

        # Computes the transpose of the inverse of the deformation gra-
        # dient. Transposes only the two last indices

        F_inv_transposed = get_inverse(tf.transpose(F, perm=[0, 1, 3, 2
        ]), self.identity_tensor)

        # Computes the jacobian

        J = tf.linalg.det(F)

        # Evaluates the analytical expression for the first Piola-
        # Kirchhoff stress tensor as a tensor [n_elements, 
        # n_quadrature_points, 3, 3]

        return ((self.mu*F)+tf.einsum('eq,eqij->eqij', ((self.lmbda*
        tf.math.log(J))-self.mu), F_inv_transposed))