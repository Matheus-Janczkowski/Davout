# Routine to store classes of isotropic hyperelastic constitutive models.
# A class is defined for each constitutive model

import tensorflow as tf

# Defines a class for a Neo Hookean hyperelastic model

class NeoHookean:

    def __init__(self, material_properties, dtype=tf.float32):
        
        # Gets the material parameters

        E = material_properties["E"]

        nu = material_properties["nu"]

        # Evaluates the Lamé parameters

        self.mu = tf.constant(E/(2*(1+nu)), dtype=dtype)

        self.lmbda = tf.constant((nu*E)/((1+nu)*(1-2*nu)), dtype=dtype)

    # Defines a function to evaluate the free energy density

    @tf.function
    def strain_energy(self, F):

        # Evaluates the right Cauchy-Green strain tensor

        #C = tf.einsum('...iK,...jK->...ij', F, F)
        C = tf.matmul(F, F, transpose_a=True)

        # Evaluates its invariants

        I1_C = tf.linalg.trace(C)

        J  = tf.linalg.det(F)

        ln_J = tf.math.log(J)

        # Calculates the Helmholtz potential

        return (0.5*self.mu*(I1_C-3))-(self.mu*ln_J)+((0.5*self.lmbda)*(ln_J**2))