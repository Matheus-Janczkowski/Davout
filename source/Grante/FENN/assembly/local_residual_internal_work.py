# Routine to store methods to calculate the local (individual finite el-
# ement level) residual vector due to internal forces

import tensorflow as tf 

# Defines a class to get the constitutive model dictionary and transform
# into a compiled evaluation of the strain energy and of the first Piola-
# Kirchhoff stress tensor

class CompiledFirstPiolaKirchhoff:

    def __init__(self, constitutive_models_dict, 
    elements_in_region_dictionary):
        
        # Initializes the list of strain energy functions and the list
        # of elements owned to each region

        self.elements_assigned_to_models = []

        self.energy_functions_list = []

        # Iterates through the dictionary of constitutive models

        for physical_group, constitutive_class in (
        constitutive_models_dict.items()):

            # Adds a tensor with a tensor containing the indices of the
            # elements that belong to this region. Uses the physical 
            # group as key in the dictionary of elements assigned to each
            # region

            self.elements_assigned_to_models.append(
            elements_in_region_dictionary[physical_group])

            # Adds the energy function

            self.energy_functions_list.append(
            constitutive_class.strain_energy)

        # Gets the number of materials

        self.n_materials = len(self.energy_functions_list)

    # Defines a function to assemble the total strain energy

    @tf.function
    def assemble_total_strain_energy(self, F):

        return tf.add_n([self.energy_functions_list[i](tf.gather(F, 
        self.elements_assigned_to_models[i])) for i in range(
        self.n_materials)])
    
    # Defines a function to compute the first Piola-Kichhoff stress ten-
    # sor

    @tf.function
    def first_piola_kirchhoff(self, F):

        # Computes the derivative of the strain energy with respect to 
        # the deformation gradient

        with tf.GradientTape() as tape:

            tape.watch(F)

            psi = self.assemble_total_strain_energy(F)

        return tape.gradient(psi, F)