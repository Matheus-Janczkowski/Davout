# Routine to store anisotropic constitutive models
# 
# The list of implemented isotropic hyperelastic models is:
# 
# 1. Neo-Hookean

from abc import ABC, abstractmethod

from dolfin import *

import ufl_legacy as ufl

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

    def second_piolaStress(self, displacement):

        pass

    def first_piolaStress(self, displacement):

        pass

    def cauchy_stress(self, displacement):

        pass

# Defines a class to evaluate the strain energy and the respective
# first Piola-Kirchhoff stress tensor for the neo-hookean hyperelastic 
# model from Javier Bonet, the same as in https://help.febio.org/docs/FE
# BioUser-3-6/UM36-4.1.3.17.html. It is called there as unconstrained
# Neo-Hookean material

class Neo_Hookean(HyperelasticMaterialModel):

    def __init__(self, material_properties):

        # Sets the names of the fields that are necessary to compute 
        # this model

        self.required_fieldsNames = ["Displacement"]

        # Checks the keys of the dictionary of material parameters

        constitutive_tools.check_materialDictionary(material_properties, 
        ["E", "nu"])

        self.E = Constant(material_properties["E"])

        self.v = Constant(material_properties["nu"])

        # Evaluates the Lamé parameters

        self.mu = Constant(self.E/(2*(1+self.v)))

        self.lmbda = Constant(self.v*self.E/((1+self.v)*(1-2*self.v)))

    # Defines a function to evaluate the Helmholtz free energy density

    def strain_energy(self, C):

        # Evaluates the right Cauchy-Green strain tensor invariants

        I1_C = ufl.tr(C)

        I2_C = ufl.det(C)

        J = ufl.sqrt(I2_C)
        
        # Evaluates the trace-related part

        psi_1 = (self.mu/2)*(I1_C - 3)

        # Evaluates the jacobian-related part

        ln_J = ufl.ln(J)

        psi_2 = -(self.mu*ln_J)+((self.lmbda*0.5)*((ln_J)**2))

        return psi_1+psi_2
    
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
    
    # Defines a function to get the first elasticity tensor, i.e. dP/dF

    def first_elasticityTensor(self, u):

        # Evaluates the deformation gradient

        I = Identity(3)

        F = variable(grad(u)+I) 

        # Evaluates the right Cauchy-Green strain tensor, C. Makes C a 
        # variable to differentiate the Helmholtz potential with respect 
        # to C

        C = (F.T)*F
        
        C = variable(C)  

        # Evaluates the Helmholtz potential

        W = self.strain_energy(C)

        # Evaluates the second Piola-Kirchhoff stress tensor differenti-
        # ating the potential w.r.t. C

        S = 2*diff(W,C)

        # Gets the first Piola-Kirchhoff stress tensor from the second

        P = F*S

        # Evaluates the first elasticity tensor by differentiating the
        # first Piola-Kirchhoff stress tensor w.r.t. the deformation 
        # gradient

        C_first = diff(P, F)
        
        # Stores the tensors inside the a dictionary so the variational
        # form and the post-processes can distinguish between them

        result = {"first_elasticity_tensor": C_first}

        return result
    
    # Defines a function to get the second elasticity tensor, i.e. dS/dC

    def second_elasticityTensor(self, u):

        # Evaluates the deformation gradient

        I = Identity(3)

        F = grad(u)+I

        # Evaluates the right Cauchy-Green strain tensor, C. Makes C a 
        # variable to differentiate the Helmholtz potential with respect 
        # to C

        C = (F.T)*F
        
        C = variable(C)  

        # Evaluates the Helmholtz potential

        W = self.strain_energy(C)

        # Evaluates the second Piola-Kirchhoff stress tensor differenti-
        # ating the potential w.r.t. C

        S = 2*diff(W,C)

        # Evaluates the second elasticity tensor by differentiating the
        # second Piola-Kirchhoff stress tensor w.r.t. the right Cauchy-
        # Green strain tensor

        C_second = diff(S, C)
        
        # Stores the tensors inside a dictionary so the variational form
        # and the post-processes can distinguish between them

        result = {"second_elasticity_tensor": C_second}

        return result
    
    # Defines a function to get the third elasticity tensor, i.e. 
    # dsigma/db

    def third_elasticityTensor(self, u):

        # Evaluates the deformation gradient

        I = Identity(3)

        F = grad(u)+I

        # Evaluates the left Cauchy-Green strain tensor, b. Makes b a 
        # variable to differentiate the Helmholtz potential with respect 
        # to b

        b = F*(F.T)
        
        b = variable(b)  

        # Evaluates the Helmholtz potential with b, because b and C have
        # the same eigenvalues, thus, the same invariants

        W = self.strain_energy(b)

        # Evaluates the Cauchy stress tensor differentiating the poten-
        # tial w.r.t. b

        J = ufl.sqrt(det(b))

        sigma = (2/J)*b*diff(W,b)

        # Evaluates the third elasticity tensor by differentiating the
        # Cauchy stress tensor w.r.t. the left Cauchy-Green strain ten-
        # sor

        C_third = diff(sigma, b)
        
        # Stores the tensors inside a dictionary so the variational form
        # and the post-processes can distinguish between them

        result = {"third_elasticity_tensor": C_third}

        return result