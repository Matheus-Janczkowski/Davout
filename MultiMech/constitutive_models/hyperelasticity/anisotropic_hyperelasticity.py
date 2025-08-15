# Routine to store anisotropic constitutive models
# 
# The list of implemented anisotropic hyperelastic models is:
# 
# 1. unconstrained Holzapfel-Gasser-Ogden

from abc import ABC, abstractmethod

import numpy as np

from dolfin import *

import ufl_legacy as ufl

from ...tool_box import tensor_tools

from ...tool_box import constitutive_tools

# Defines an abstract class to force all classes ahead to have the same
# methods. To enforce it, the abstract method is used before the methods.
# The material model classes have four methods in the case of hyperelas-
# ticity: the Helmholtz potential (strain energy); the second Piola-
# Kirchhoff stress tensor; the first Piola-Kirchhoff stress tensor; and
# the Cauchy stress tensor.

class HyperelasticMaterialModel(ABC):

    @abstractmethod

    # The following methods have the pass argument only because they 
    # will be defined in the child classes

    def strain_energy(self, strain_tensor):

        pass

    def first_piolaStress(self, displacement):

        pass

    def second_piolaStress(self, displacement):

        pass

    def cauchy_stress(self, displacement):

        pass

# Defines a function to implement the strain energy and the stress ten-
# sor for the Holzapfel-Gasser-Ogden unconstrained material model. This
# implementation was done as in 
# https://help.febio.org/docs/FEBioUser-3-6/UM36-4.1.4.11.html, where it
# is called unconstrained Holzapfel-Gasser-Ogden

class Holzapfel_Gasser_Ogden_Unconstrained(HyperelasticMaterialModel):

    # Initializes the properties

    def __init__(self, material_properties):

        # Sets the names of the fields that are necessary to compute 
        # this model

        self.required_fieldsNames = ["Displacement"]

        # Checks the keys of the dictionary of material parameters

        constitutive_tools.check_materialDictionary(material_properties, 
        ["c", "k1", "k2", "gamma", "kappa", "k", "local system of coor"+
        "dinates: a direction", "local system of coordinates: d direct"+
        "ion"])
        
        self.c = Constant(material_properties["c"])

        self.k1 = Constant(material_properties["k1"])

        self.k2 = Constant(material_properties["k2"])

        # Converts the angle from degrees to radians

        self.gamma = Constant(material_properties["gamma"]*(ufl.pi/180.0
        ))

        self.kappa = Constant(material_properties["kappa"])

        self.k = Constant(material_properties["k"])

        vector_e1 = material_properties["local system of coordinates: "+
        "a direction"]

        vector_e2 = material_properties["local system of coordinates: "+
        "d direction"]

        vector_e3 = ufl.cross(vector_e1, vector_e2)

        # Normalizes these vectors

        norm_e1 = Constant(tensor_tools.L2_normVector(vector_e1))

        norm_e2 = Constant(tensor_tools.L2_normVector(vector_e2))

        norm_e3 = Constant(tensor_tools.L2_normVector(vector_e3))

        self.e1 = Constant((1/norm_e1)*vector_e1)

        self.e2 = Constant((1/norm_e2)*vector_e2)

        self.e3 = Constant((1/norm_e3)*vector_e3)

    # Defines the strain energy

    def strain_energy(self, C):

        # Evaluates the invariants of the right Cauhcy-Green tensor
        
        I1_C = ufl.tr(C)

        I2_C = ufl.det(C)

        J = ufl.sqrt(I2_C)

        # Evaluates the two rotation matrices (for + and - gamma). Uses
        # the local e_3 direction as axial vector

        R1 = tensor_tools.rotation_tensorEulerRodrigues(self.gamma*
        self.e3)

        R2 = tensor_tools.rotation_tensorEulerRodrigues(-self.gamma*
        self.e3)

        # Evaluates the two fiber directional vectors

        alpha_1 = R1*self.e1

        alpha_2 = R2*self.e1

        # Evaluates the anisotropic invariants

        I4_alpha_1 = ufl.dot(alpha_1, C*alpha_1)

        I4_alpha_2 = ufl.dot(alpha_2, C*alpha_2)

        # Defines the Macauley bracket operator

        def Macaulay(variable):

            return ufl.conditional(ufl.lt(variable, 0), 0, variable)
        
        # Evaluates the energy parcels
        
        E_alpha_1 = (self.kappa*(I1_C-3))+((1-(3*self.kappa))*(
        I4_alpha_1-1))
        
        E_alpha_2 = (self.kappa*(I1_C-3))+((1-(3*self.kappa))*(
        I4_alpha_2-1))

        # Evaluates the energy relative to the neo-hookean matrix

        energy_matrix = self.c*(I1_C-3) 

        # Evaluates the energy parcels relative to the fibers

        energy_fiber1 = ((self.k1/(2*self.k2))*(ufl.exp(self.k2*((
        Macaulay(E_alpha_1)**2)))-1))

        energy_fiber2 = ((self.k1/(2*self.k2))*(ufl.exp(self.k2*((
        Macaulay(E_alpha_2)**2)))-1))

        # Evaluates the volumetric energy

        ln_J = ufl.ln(J)

        energy_volumetric = ((self.k*((((J**2)-1)*0.5)-ln_J))-(2*self.c*
        ln_J))

        return (energy_matrix+energy_fiber1+energy_fiber2+
        energy_volumetric)
    
    # Defines a function to evaluate the second Piola-Kirchhoff stress 
    # tensor as the derivative of the Helmholtz free energy density po-
    # tential

    def second_piolaStress(self, u):

        S = constitutive_tools.S_fromDPsiDC(self.strain_energy, u)

        return S

    # Defines a function to evaluate the first Piola-Kirchhoff stress 
    # tensor as the result of the proper operation over the second one

    def first_piolaStress(self, u):

        # Evaluates the second Piola-Kirchhoff stress tensor and pulls
        # it back to the first Piola-Kirchhoff stress tensor

        S = self.second_piolaStress(u)
        
        P = constitutive_tools.P_fromS(S, u)

        return P
    
    # Defines a function to evaluate the Cauchy stress tensor as the 
    # push forward of the second Piola-Kirchhoff stress tensor
    
    def cauchy_stress(self, u):

        # Evaluates the second Piola-Kirchhoff stress tensor and pushes 
        # it forward to the deformed configuration

        S = self.second_piolaStress(u)

        sigma = constitutive_tools.push_forwardS(S, u)

        return sigma