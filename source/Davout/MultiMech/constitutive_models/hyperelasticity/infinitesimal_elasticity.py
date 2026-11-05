# Routine to store infinitesimal elastic constitutive models
# 
# The list of implemented isotropic models is:
# 
# 1. Linear isotropic elastic model

from abc import ABC, abstractmethod

from dolfin import *

import ufl_legacy as ufl

from ....MultiMech.tool_box import constitutive_tools

# Defines an abstract class to force all classes ahead to have the same
# methods. To enforce it, the abstract method is used before the methods.
# The material model classes have two methods in the case of infinitesi-
# mal elasticity: the Helmholtz potential (strain energy); the Cauchy 
# stress tensor.

class ElasticMaterialModel(ABC):

    @abstractmethod

    # The following methods have the pass argument only because they 
    # will be defined in the child classes

    def check_model(self, code_given_information):

        pass

    def strain_energy(self, strain_tensor):

        pass

    def cauchy_stress(self, displacement):

        pass

# Defines a class to evaluate the strain energy and the respective
# first Cauchy stress tensor for linear isotropic elasticity

class LinearIsotropicElastic(ElasticMaterialModel):

    """
    Linear elastic compressible elastic model
    
    E: Young modulus
    
    nu: Poisson ratio
    """

    def __init__(self, material_properties):

        # Sets the names of the fields that are necessary to compute 
        # this model

        self.required_fieldsNames = ["Displacement"]

        self.material_properties = material_properties

    # Defines a method to check the validity of the user-given informa-
    # tion

    def check_model(self, code_given_information):

        # Checks the keys of the dictionary of material parameters

        self.material_properties = constitutive_tools.check_materialDictionary(
        self.material_properties, ["E", "nu"], code_given_information=
        code_given_information)

        self.E = self.material_properties["E"]

        #print("Maximum value of the Young Modulus: "+str(self.E.vector().max()))

        self.v = self.material_properties["nu"]

        # Evaluates the Lamé parameters

        self.mu = self.E/(2*(1+self.v))

        self.lmbda = self.v*self.E/((1+self.v)*(1-2*self.v))

        # Empties the material properties dictionary

        self.material_properties = None

    # Defines a function to evaluate the Helmholtz free energy density

    def strain_energy(self, epsilon):

        # Evaluates the trace of the infinitesimal strain tensor and the
        # Frobenius norm squared

        I1_epsilon = ufl.tr(epsilon)

        frobenius_norm_sqquared = ufl.tr(epsilon*epsilon)

        return (self.mu*frobenius_norm_sqquared)+(0.5*self.lmbda*(
        I1_epsilon**2))
    
    # Defines a function to evaluate the Cauchy stress tensor directly
    # from the expression
    
    def cauchy_stress(self, u):

        # Evaluates the infinitesimal strain tensor

        epsilon = sym(grad(u))

        # Evaluates the Cauchy stress tensor directly
        
        I = Identity(u.geometric_dimension())

        sigma = (2.0*self.mu*epsilon)+(self.lmbda*ufl.tr(epsilon)*I)

        return sigma
    
    # Defines a function to get the elasticity tensor, i.e.dsigma/db

    def elasticity_tensor(self, u):

        # Evaluates the infinitesimal strain tensor

        epsilon = sym(grad(u))

        epsilon = ufl.variable(epsilon)

        # Evaluates the Cauchy stress tensor directly

        I = Identity(u.geometric_dimension())

        sigma = (2.0*self.mu*epsilon)+(self.lmbda*ufl.tr(epsilon)*I)

        # Evaluates the elasticity tensor by differentiating the Cauchy 
        # stress tensor w.r.t. the infinitesimal strain tensor

        C = diff(sigma, epsilon)
        
        # Stores the tensors inside a dictionary so the variational form
        # and the post-processes can distinguish between them

        result = {"elasticity_tensor": C}

        return result