# Routine to store classes of isotropic hyperelastic constitutive models.
# A class is defined for each constitutive model

import tensorflow as tf

from ..tool_box.math_tools import get_inverse

# Defines a class for a Neo Hookean hyperelastic model

class NeoHookean:

    def __init__(self, material_properties, mesh_data_class):

        # Verifies if material properties are a list, which indicates 
        # different material properties along different realizations of
        # the BVP

        if isinstance(material_properties, list):

            # Initializes a list for each Lamé parameter, to store their
            # values across realizations

            self.mu = []

            self.lmbda = []

            # Iterates through the dictionaries in the material proper-
            # ties list

            for material_dict in material_properties:

                # Gets the material parameters

                E = material_dict["E"]

                nu = material_dict["nu"]

                # Evaluates the Lamé parameters

                self.mu.append(E/(2*(1+nu)))

                self.lmbda.append((nu*E)/((1+nu)*(1-2*nu)))

            # Gets the number of realizations

            self.n_material_realizations = len(self.mu)

            # Converts the Lamé parameters to tensors [n_realizations,
            # 1, 1]

            self.mu = tf.reshape(tf.constant(self.mu, dtype=
            mesh_data_class.dtype), [self.n_material_realizations, 1, 1])

            self.lmbda = tf.reshape(tf.constant(self.lmbda, dtype=
            mesh_data_class.dtype), [self.n_material_realizations, 1, 1])

        # Otherwise, if it is a dictionary

        elif isinstance(material_properties, dict):
        
            # Gets the material parameters

            E = material_properties["E"]

            nu = material_properties["nu"]

            # Evaluates the Lamé parameters

            self.mu = E/(2*(1+nu))

            self.lmbda = (nu*E)/((1+nu)*(1-2*nu))

            # Gets the number of realizations

            self.n_material_realizations = 0

            # Converts the Lamé parameters to tensors [0]

            self.mu = tf.constant(self.mu, dtype=mesh_data_class.dtype)

            self.lmbda = tf.constant(self.lmbda, dtype=
            mesh_data_class.dtype)

        else:

            raise TypeError("'material_properties' in 'NeoHookean' con"+
            "stitutive model class must be a list for realizations wit"+
            "h different material parameters or a dictionary, for a si"+
            "ngle value of material parameters across the realizations"+
            " of the BVP")

        # Initializes the identity tensor attribute that the code will
        # automatically fill it later

        self.identity_tensor = None

    # Defines a function to evaluate the free energy density. F, the de-
    # formation energy, is a tensor [n_realizations, n_elements, n_qua-
    # drature_points, 3, 3]

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

        F_inv_transposed = get_inverse(tf.transpose(F, perm=[0, 1, 2, 4, 
        3]), self.identity_tensor)

        # Computes the jacobian

        J = tf.linalg.det(F)

        # Evaluates the analytical expression for the first Piola-
        # Kirchhoff stress tensor as a tensor [n_realizations, n_ele-
        # ments, n_quadrature_points, 3, 3]

        #return ((self.mu*F)+tf.einsum('peq,peqij->peqij', ((self.lmbda*
        #tf.math.log(J))-self.mu), F_inv_transposed))

        # Adds two dimensions at the end to make this tensor [n_realiza-
        # tions, n_elements, n_quadrature_points, 1, 1]

        scalar_coefficient = tf.expand_dims(tf.expand_dims((self.lmbda*
        tf.math.log(J))-self.mu, axis=-1), axis=-1)

        return (self.mu*F)+(scalar_coefficient*F_inv_transposed)