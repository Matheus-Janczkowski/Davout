# Routine to store anisotropic constitutive models
# 
# The list of implemented isotropic hyperelastic models is:
# 
# 1. Neo-Hookean

from abc import ABC, abstractmethod

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

    def strain_energy(self, strain_tensor, curvature_tensor):

        pass

    def second_piolaStress(self, displacement, microrotation):

        pass

    def first_piolaStress(self, displacement, microrotation):

        pass

    def cauchy_stress(self, displacement, microrotation):

        pass

# Defines a class to evaluate the strain energy and the respective
# first Piola-Kirchhoff stress tensor for the neo-hookean hyperelastic 
# micropolar model. This implementation is the same as from Bauer et al-
# Micropolar hyperelasticity: constitutive model, consistent lineariza-
# tion and simulation of 3D scale effects

class Micropolar_Neo_Hookean(HyperelasticMaterialModel):

    def __init__(self, material_properties):

        # Sets the names of the fields that are necessary to compute 
        # this model

        self.required_fieldsNames = ["Displacement", "Microrotation"]

        # Checks the keys of the dictionary of material parameters

        constitutive_tools.check_materialDictionary(material_properties, 
        ["E", "nu", "flag bending", "characteristic length", "alpha",
        "gamma", "N"])

        # Gets the Young modulus and the Poisson ratio

        young_modulus = material_properties["E"]

        poisson_ratio = material_properties["nu"]

        # Gets the flag for bending and the characteristic length

        flag_bending = material_properties["flag bending"]

        characteristic_length = material_properties["characteristic le"+
        "ngth"]

        # Gets the parameters

        self.mu = Constant(young_modulus/(2*(1+poisson_ratio)))

        self.lmbda = Constant((poisson_ratio*young_modulus)/((1+
        poisson_ratio)*(1-(2*poisson_ratio))))

        self.alpha = Constant(material_properties["alpha"])

        self.gamma = Constant(material_properties["gamma"])

        # Selects the beta parameter using the characteristic length

        if flag_bending:

            self.beta = Constant((4*self.mu)*(characteristic_length**2))

        else:

            self.beta = Constant((2*self.mu*(characteristic_length**2))-
            self.gamma)

        # Gets the micropolar number, which varies between 0 and 1. If
        # null, it is Cauchy continuum; if 1, it is couple stress theory

        N_micropolar = material_properties["N"]

        if N_micropolar<0 or N_micropolar>1:

            raise ValueError("The micropolar number must be bound by ["+
            "0,1]. The given N is: "+str(N_micropolar))
        
        elif abs(N_micropolar-1.0)<1E-5:

            self.kappa = Constant(2*self.mu)

        else:

            self.kappa = Constant(2*self.mu*((N_micropolar**2)/(1-(
            N_micropolar**2))))

        # Precomputes the Helmholtz free potential differentiation

        V_transposed = variable(Identity(3))

        k_transposed = variable(Identity(3))

        pre_psi = self.strain_energy(V_transposed.T, k_transposed.T)

        self.dpsi_dVt = diff(pre_psi, V_transposed)

        self.dpsi_dkt = diff(pre_psi, k_transposed)

    # Defines a function to evaluate the Helmholtz free energy density

    def strain_energy(self, V_bar, k_curvatureSpatial):

        # Evaluates the traces

        I1 = tr(V_bar*V_bar.T)

        I2 = tr(V_bar*V_bar)

        I3 = tr(k_curvatureSpatial)

        I4 = tr(k_curvatureSpatial*k_curvatureSpatial)

        I5 = tr(k_curvatureSpatial*k_curvatureSpatial.T)

        # And the determinant

        J = det(V_bar)

        # Evaluates the energy parcels from Bauer et al

        psi_NH = 0.5*self.mu*(I1-3.0)

        ln_J = ufl.ln(J)

        # Bauer

        #psi_vol = ((0.25*self.lmbda*((J**2)-1))-(0.5*self.lmbda*(ln_J))
        #-self.mu*ln_J)

        # Ramezani

        psi_vol = -(self.mu*ln_J)+((self.lmbda*0.5)*((ln_J)**2))

        psi_hat = 0.25*self.kappa*(I1-I2)

        psi_tilde = (0.5*((self.alpha*(I3**2))+(self.beta*I4)+(
        self.gamma*I5)))

        return psi_NH+psi_vol+psi_hat+psi_tilde
    
    # Defines a function to evaluate the second Piola-Kirchhoff stress 
    # tensors using a pull-back operation over the Cauchy stress tensors.
    # The argument is a list of the two fields

    def second_piolaStress(self, fields_list):

        # Retrieves the two first fields

        u, phi, *_ = fields_list

        # Evaluates the Cauchy stress and the couple stress

        result_cauchy = self.cauchy_stress(fields_list)

        sigma = result_cauchy["cauchy"]

        sigma_couple = result_cauchy["couple_cauchy"]

        # Evaluates the second Piola-Kirchhoff stress tensors using the
        # pull back operation

        S = constitutive_tools.S_fromCauchy(sigma, u)

        S_couple = constitutive_tools.S_fromCauchy(sigma_couple, u)

        # Stores the tensors inside the a dictionary so the variational
        # form and the post-processes can distinguish between them

        result = {"second_piola_kirchhoff":S, "couple_second_piola_kir"+
        "chhoff": S_couple}

        return result

    # Defines a function to evaluate the first Piola-Kirchhoff stress 
    # and its couple from the Cauchy stress

    def first_piolaStress(self, fields_list):

        # Retrieves the fields

        u, phi, *_  = fields_list

        # Evaluates the Cauchy stress and the couple stress

        result_cauchy = self.cauchy_stress(fields_list)

        sigma = result_cauchy["cauchy"]

        sigma_couple = result_cauchy["couple_cauchy"]

        # Evaluates the first Piola-Kirchhoff stress tensors using the
        # Piola transformation

        P = constitutive_tools.P_fromCauchy(sigma, u)

        P_couple = constitutive_tools.P_fromCauchy(sigma_couple, u)

        # Stores the tensors inside the a dictionary so the variational
        # form and the post-processes can distinguish between them

        result = {"first_piola_kirchhoff":P, "couple_first_piola_kirch"+
        "hoff": P_couple}

        return result

    # Defines a function to evaluate the Kirchhoff stress and its couple 
    # from the Cauchy stress

    def kirchhoff_stress(self, fields_list):

        # Retrieves the fields

        u, phi, *_  = fields_list

        # Evaluates the Cauchy stress and the couple stress

        result_cauchy = self.cauchy_stress(fields_list)

        sigma = result_cauchy["cauchy"]

        sigma_couple = result_cauchy["couple_cauchy"]

        # Multiplies by the jacobian to get the Kirchhoff stresses

        tau = constitutive_tools.tau_fromCauchy(sigma, u)

        tau_couple = constitutive_tools.tau_fromCauchy(sigma_couple, u)

        # Stores the tensors inside the a dictionary so the variational
        # form and the post-processes can distinguish between them

        result = {"kirchhoff":tau, "couple_kirchhoff": tau_couple}

        return result
    
    # Defines a function to evaluate the Cauchy stress tensor and its 
    # couple tensor using the derivative of the Helmholtz potential with 
    # respect to the V_bar tensor. This function receives the deforma-
    # tion gradient, the rotation tensor of the microrotation field, and 
    # the curvature tensor in the referential configuration
    
    def cauchy_stress(self, fields_list):

        # Retrieves the fields

        u, phi, *_  = fields_list

        # Evaluates the deformation gradient

        I = Identity(3)

        F = grad(u)+I 

        # Evaluates the rotation tensor using phi

        R_bar = tensor_tools.rotation_tensorEulerRodrigues(phi)

        # Evaluates the curvature tensor in the referential configuration

        K_curvatureReferential = constitutive_tools.micropolar_curvatureTensor(
        phi)

        # Evaluates the micropolar stretch and the jacobian
    
        V_bar = F*(R_bar.T)

        # Evaluates the push-forward of the curvature tensor

        k_curvatureSpatial = R_bar*K_curvatureReferential*R_bar.T

        # Evaluates the derivative replacing the already precompiled de-
        # rivative

        evaluate_diffSigma = ufl.replace(self.dpsi_dVt, {Identity(3): 
        V_bar.T})

        sigma = V_bar*evaluate_diffSigma

        # Evaluates the derivative replacing the already precompiled de-
        # rivative

        evaluate_diffCouple = ufl.replace(self.dpsi_dkt, {Identity(3): 
        k_curvatureSpatial.T})

        sigma_couple = V_bar*evaluate_diffCouple

        # Stores the tensors inside the a dictionary so the variational
        # form and the post-processes can distinguish between them

        result = {"cauchy": sigma, "couple_cauchy": sigma_couple}

        return result
    
    # Defines a function to get the first elasticity tensor, i.e. dP/dF

    def first_elasticityTensor(self, fields_list):

        # Retrieves the fields

        u, phi, *_  = fields_list

        # Evaluates the deformation gradient

        I = Identity(3)

        F = variable(grad(u)+I) 

        # Evaluates the rotation tensor using phi

        R_bar = tensor_tools.rotation_tensorEulerRodrigues(phi)

        # Evaluates the micropolar stretch and the jacobian
    
        V_bar = F*(R_bar.T)

        # Evaluates the derivative replacing the already precompiled de-
        # rivative

        evaluate_diffSigma = ufl.replace(self.dpsi_dVt, {Identity(3): 
        V_bar.T})

        sigma = V_bar*evaluate_diffSigma
        
        # Evaluates the determinant of the deformation gradient

        J = ufl.det(F)

        # Uses the Piola transformation

        P = J*sigma*(inv(F).T)

        # Evaluates the first elasticity tensor by differentiating the
        # first Piola-Kirchhoff stress tensor w.r.t. the deformation 
        # gradient

        C_first = diff(P, F)
        
        # Stores the tensors inside the a dictionary so the variational
        # form and the post-processes can distinguish between them

        result = {"first_elasticity_tensor": C_first}

        return result