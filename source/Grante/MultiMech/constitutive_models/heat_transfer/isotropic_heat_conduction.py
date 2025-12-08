# Routine to store isotropic heat conduction models

from abc import ABC, abstractmethod

from dolfin import *

import ufl_legacy as ufl

from ...tool_box import constitutive_tools

# Defines an abstract class to force all classes ahead to have the same
# methods. To enforce it, the abstract method is used before the methods.
# The material model classes have two methods in the case of heat con-
# duction: the Gibbs potential (conduction energy)); the referential heat
# flux.

class HeatMaterialModel(ABC):

    @abstractmethod

    # The following methods have the pass argument only because they 
    # will be defined in the child classes

    def heat_energy(self, temperature_gradient):

        pass

    def referential_heat_flux(self, temperature):

        pass

# Defines a class to evaluate the strain energy and the respective
# heat flux in the reference configuration

class Fourier(HeatMaterialModel):

    def __init__(self, material_properties):

        # Sets the names of the fields that are necessary to compute 
        # this model

        self.required_fieldsNames = ["Displacement"]

        # Checks the keys of the dictionary of material parameters

        constitutive_tools.check_materialDictionary(material_properties, 
        ["k"])

        self.k = Constant(material_properties["k"])

        # Evaluates the conductivity tensor

        I = Identity(3)

        self.kappa = self.k*I

    # Defines a function to evaluate the conduction energy

    def heat_energy(self, temperature_gradient):

        # Gets the energy

        return -0.5*dot(self.kappa*temperature_gradient, 
        temperature_gradient)
    
    # Defines a function to evaluate the referential heat flux

    def referential_heat_flux(self, temperature):

        Grad_T = grad(temperature)

        Grad_T = variable(Grad_T)

        energy = self.heat_energy(Grad_T)

        #heat_flux = diff(energy, Grad_T)

        heat_flux = -self.kappa*grad(temperature)

        return heat_flux