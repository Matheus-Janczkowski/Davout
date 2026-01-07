# Routine to store classes of isotropic hyperelastic constitutive models.
# A class is defined for each constitutive model

import tensorflow as tf

# Defines a class for a Neo Hookean hyperelastic model

class NeoHookean:

    def __init__(self, material_parameters):
        
        # Gets the material parameters

        E = self.material_properties["E"]

        nu = self.material_properties["nu"]

        # Evaluates the Lamé parameters

        self.mu = E/(2*(1+nu))

        self.lmbda = (nu*E)/((1+nu)*(1-2*nu))

    # Defines a function to evaluate the free energy density

    @tf.function
    def psi_neo_hookean(self, F):

        # Evaluates the right Cauchy-Green strain tensor

        C = tf.einsum('...iK,...jK->...ij', F, F)

        # Evaluates its invariants

        I1_C = tf.linalg.trace(C)

        J  = tf.linalg.det(F)

        ln_J = tf.math.log(J)

        # Calculates the Helmholtz potential

        return (0.5*self.mu*(I1_C-3))-(self.mu*ln_J)+((0.5*self.lmbda)*(ln_J**2))