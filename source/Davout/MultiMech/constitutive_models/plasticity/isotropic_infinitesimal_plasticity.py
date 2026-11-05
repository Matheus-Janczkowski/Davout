# Routine to store infinitesimal elasto-plastic constitutive models
# 
# The list of implemented isotropic models is:
# 
# 1. elasto-plastic isotropic model with exponential hardening and back 
#    stress

from abc import ABC, abstractmethod

from dolfin import *

import ufl_legacy as ufl

from ....MultiMech.tool_box import constitutive_tools

# Defines an abstract class to force all classes ahead to have the same
# methods. To enforce it, the abstract method is used before the methods.
# The material model classes have three methods in the case of infinite-
# simal elasto-plasticity: the Helmholtz potential (strain energy); the 
# Cauchy stress tensor; and the tangent Cauchy stress tensor (linearized
# stress tensor to give the consistent tangent modulus).

class ElastoplasticMaterialModel(ABC):

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
# first Cauchy stress tensor for infinitesimal isotropic elasto-plasti-
# city

class ExponentialHardeningElastoPlasticWithBackStress(
ElastoplasticMaterialModel):

    """
    Isotropic model for elastoplasticity with exponential hardening and
    back stress
    
    E: Young modulus
    
    nu: Poisson ratio

    sigma_y: limit elastic stress

    H: back stress modulus

    K_1: linear contribution to hardening

    K_2: coefficient of the exponential contribution to hardening

    theta: coefficient for the saturation of the exponential contribu-
    tion to hardening
    """

    def __init__(self, material_properties, newton_raphson_tolerance=
    1E-6):

        # Sets the names of the fields that are necessary to compute 
        # this model

        self.required_fieldsNames = ["Displacement"]

        self.material_properties = material_properties

        self.newton_raphson_tolerance = newton_raphson_tolerance

    # Defines a method to check the validity of the user-given informa-
    # tion

    def check_model(self, code_given_information):

        # Checks the keys of the dictionary of material parameters

        self.material_properties = constitutive_tools.check_materialDictionary(
        self.material_properties, ["E", "nu", "sigma_y", "H", "K_1", 
        "K_2", "theta"], code_given_information=code_given_information)

        self.E = self.material_properties["E"]

        self.v = self.material_properties["nu"]

        self.sigma_y = self.material_properties["sigma_y"]

        self.H = self.material_properties["H"]

        self.K_1 = self.material_properties["K_1"]

        self.K_2 = self.material_properties["K_2"]

        self.theta = self.material_properties["theta"]

        # Evaluates the Lamé parameters

        self.mu = self.E/(2*(1+self.v))

        self.lmbda = self.v*self.E/((1+self.v)*(1-2*self.v))

        # Empties the material properties dictionary

        self.material_properties = None

    # Defines a function to evaluate the Helmholtz free energy density
    # as a function of the elastic strain tensor

    def strain_energy(self, elastic_epsilon):

        # Evaluates the trace of the infinitesimal strain tensor and the
        # Frobenius norm squared

        I1_epsilon = ufl.tr(elastic_epsilon)

        frobenius_norm_sqquared = ufl.tr(elastic_epsilon*elastic_epsilon)

        return (self.mu*frobenius_norm_sqquared)+(0.5*self.lmbda*(
        I1_epsilon**2))
    
    # Defines a function to evaluate the Cauchy stress tensor directly
    # from the expression. This function uses th
    
    def cauchy_stress(self, u, internal_variables_dictionary):

        # Retrieves from the dictionary of internal variables the plas-
        # tic strain tensor and the accumulated plastic strain. Both 
        # from the last pseudo-time step

        last_plastic_strain_tensor = internal_variables_dictionary["la"+
        "st_plastic_strain_tensor"]

        last_accumulated_plastic_strain = internal_variables_dictionary[
        "last_accumulated_plastic_strain"]

        # Evaluates the infinitesimal strain tensor and the identity 
        # tensor

        epsilon = sym(grad(u))

        dimension = u.geometric_dimension()

        I = Identity(dimension)

        # Considering the trial plastic strain tensor as the last plas-
        # tic strain, computes the back stress

        trial_back_stress = self.evaluate_back_stress(
        last_plastic_strain_tensor)

        # Estimates the current elastic strain tensor

        trial_current_elastic_strain_tensor = (epsilon-
        last_plastic_strain_tensor)

        # Computes the trial stress tensor

        trial_sigma = ((2.0*self.mu*trial_current_elastic_strain_tensor
        )+(self.lmbda*ufl.tr(trial_current_elastic_strain_tensor)*I)-
        trial_back_stress)

        # Checks whether the yield function is less than zero

        yield_function = self.evaluate_yield_criterion(trial_sigma,
        last_accumulated_plastic_strain, I, dimension)

        return ufl.conditional(yield_function<0.0, (trial_sigma,
        internal_variables_dictionary), self.update_plastic_strain(
        trial_current_elastic_strain_tensor, last_plastic_strain_tensor, 
        last_accumulated_plastic_strain, epsilon, I))
    
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

    ####################################################################
    #                      Plasticity formulation                      #
    ####################################################################

    # Defines a function to compute the limit stress in the yield func-
    # tion and its derivative with respect to the accumulated plastic 
    # strain

    def limit_stress_model(self, accumulated_plastic_strain):

        return (self.sigma_y+(self.K_1*accumulated_plastic_strain)+(
        self.K_2*(1.0-ufl.exp(-self.theta*accumulated_plastic_strain))),
        self.K_1+(self.K_2*self.theta*ufl.exp(-self.theta*
        accumulated_plastic_strain)))

    # Defines a function to compute the back stress

    def evaluate_back_stress(self, plastic_strain_tensor):

        return (2.0/3.0)*self.H*plastic_strain_tensor

    # Defines a function to compute the yield criterion using the von
    # Mises criterion

    def evaluate_yield_criterion(self, stress_tensor, 
    accumulated_plastic_strain, identity_tensor, dimension):

        # Computes the limit stress

        limit_stress, _ = self.limit_stress_model(
        accumulated_plastic_strain)

        # Computes the deviatoric parcel of the stress

        deviatoric_stress = stress_tensor-((ufl.tr(stress_tensor)/
        dimension)*identity_tensor)

        # Computes the von-Mises criterion

        return ufl.sqrt((3.0/2.0)*ufl.inner(deviatoric_stress, 
        deviatoric_stress))-limit_stress

    # Defines a function to update the plastic strain tensor and the ac-
    # cumulated plastic strain

    def update_plastic_strain(self, trial_elastic_strain_tensor, 
    trial_plastic_strain_tensor, accumulated_plastic_strain, 
    strain_tensor, identity):

        # Computes the trial direction of update of the strain tensor

        trial_update_direction = (2.0*self.mu*trial_elastic_strain_tensor
        )-self.evaluate_back_stress(trial_plastic_strain_tensor)

        # Computes the norm of the trial update direction

        norm_trial_update_direction = ufl.sqrt(ufl.inner(
        trial_update_direction, trial_update_direction))

        # Computes the equivalent stress using the trial update direction

        trial_equivalent_stress = ufl.sqrt((3.0/2.0)
        )*norm_trial_update_direction

        # Initializes the plastic update increment

        update_increment = 0.0

        # Iterates over the number of inner Newton-Raphson iterations

        for _ in range(self.maximum_inner_newton_raphson_iterations):

            # Updates the accumulated plastic strain

            accumulated_plastic_strain += update_increment

            # Computes new limit stress and its derivative with respect
            # to the accumulated plastic strain

            sigma_y, K = self.limit_stress_model(
            accumulated_plastic_strain)

            # Computes the yield function as a function of the accumula-
            # ted plastic strain

            phi = (ufl.sqrt(3.0/2.0)*(1.0-((update_increment/
            trial_equivalent_stress)*((3.0*self.mu)+self.H)))*
            norm_trial_update_direction)-sigma_y

            # Checks if the yield function is 0

            if (phi**2)<self.newton_raphson_tolerance:

                break 

            # Otherwise, updates the update increment

            update_increment += phi/((3.0*self.mu)+self.H+K)

        # Updates the plastic strain tensor

        trial_plastic_strain_tensor += (update_increment*ufl.sqrt(3.0/
        2.0)*(trial_update_direction/norm_trial_update_direction))

        # Computes the stress tensor

        elastic_strain_tensor = strain_tensor-trial_plastic_strain_tensor

        stress_tensor = (2.0*self.mu*elastic_strain_tensor)+(self.lmbda*
        ufl.tr(elastic_strain_tensor)*identity)

        # Returns the updated plastic strain tensor and the accumulated
        # plastic strain in a dictionary

        return stress_tensor, {"last_plastic_strain_tensor": 
        trial_plastic_strain_tensor, "last_accumulated_plastic_strain": 
        accumulated_plastic_strain}