# Routine to store infinitesimal elasto-plastic constitutive models
# 
# The list of implemented isotropic models is:
# 
# 1. elasto-plastic isotropic model with exponential hardening and back 
#    stress

from abc import ABC, abstractmethod

from dolfin import *

import numpy as np

import ufl_legacy as ufl

p, q, r, s = ufl.indices(4)

from ....MultiMech.tool_box import constitutive_tools

from ....MultiMech.tool_box.projection_tools import MultipleFieldsProjectionOntoQuadratureSpace

from ....MultiMech.tool_box.functional_tools import construct_monolithicFunctionSpace

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

    def check_model(self, code_given_information_dict):

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
    1E-5, maximum_inner_newton_raphson_iterations=100):

        # Sets the names of the fields that are necessary to compute 
        # this model

        self.required_fieldsNames = ["Displacement"]

        self.material_properties = material_properties

        self.newton_raphson_tolerance = newton_raphson_tolerance

        self.maximum_inner_newton_raphson_iterations = (
        maximum_inner_newton_raphson_iterations)

    # Defines a method to check the validity of the user-given informa-
    # tion

    def check_model(self, code_given_information_dict):

        # Checks the keys of the dictionary of material parameters

        self.material_properties = constitutive_tools.check_materialDictionary(
        self.material_properties, ["E", "nu", "sigma_y", "H", "K_1", 
        "K_2", "theta", "mesh_data_class"], code_given_information=
        code_given_information_dict)

        self.E = self.material_properties["E"]

        self.v = self.material_properties["nu"]

        self.sigma_y = float(self.material_properties["sigma_y"])

        self.H = float(self.material_properties["H"])

        self.K_1 = float(self.material_properties["K_1"])

        self.K_2 = float(self.material_properties["K_2"])

        self.theta = float(self.material_properties["theta"])

        # Evaluates the Lamé parameters

        self.mu = self.E/(2*(1+self.v))

        self.lmbda = self.v*self.E/((1+self.v)*(1-2*self.v))

        self.bulk_modulus = self.lmbda+((2.0/3.0)*self.mu)

        self.numpy_lmbda = float(self.lmbda)

        self.numpy_mu = float(self.mu)

        # Retrieves the mesh data class

        self.mesh_data_class = self.material_properties["mesh_data_cla"+
        "ss"]

        # Empties the material properties dictionary

        self.material_properties = None

        # Sets the function space for a vector-valued internal variable 
        # and another scalar-valued internal variable. They are both 
        # found in quadrature points alone. Note that 'polynomial degree' 
        # stands for quadrature degree in the case of quadrature finite 
        # elements

        self.quadrature_functional_data_class = construct_monolithicFunctionSpace(
        {"accumulated plastic strain": {"field type": "scalar", "interp"+
        "olation function": "Quadrature", "polynomial degree": 2},
        "strain tensor": {"field type": "tensor", "interpolation func"+
        "tion": "Quadrature", "polynomial degree": 2, "tensor order": 2}, 
        "stress tensor": {"field type": "tensor", "interpolation funct"+
        "ion": "Quadrature", "polynomial degree": 2, "tensor order": 2},
        "plastic strain tensor": {"field type": "tensor", "interpolati"+
        "on function": "Quadrature", "polynomial degree": 2, "tensor o"+
        "rder": 2}, 
        "normalized trial update direction": {"field type": "tensor", 
        "interpolation function": "Quadrature", "polynomial degree": 2, 
        "tensor order": 2},
        "update increment": {"field type": "scalar", "interpolation fu"+
        "nction": "Quadrature", "polynomial degree": 2},
        "trial equivalent stress reciprocal": {"field type": "scalar", 
        "interpolation function": "Quadrature", "polynomial degree": 2},
        "hardening tangent modulus": {"field type": "scalar", "interpo"+
        "lation function": "Quadrature", "polynomial degree": 2}}, 
        self.mesh_data_class, verbose=True, use_fenics_split_method=
        False)

        # Instantiates the class that projects expressions from and to 
        # quadrature spaces. This is important for the algorithmic pro-
        # cedure within elastoplasticity 

        self.quadrature_projection_class = MultipleFieldsProjectionOntoQuadratureSpace(
        ["accumulated plastic strain", "strain tensor", "plastic strai"+
        "n tensor", "stress tensor", "normalized trial update direction", 
        "update increment", "trial equivalent stress reciprocal", "har"+
        "dening tangent modulus"], self.quadrature_functional_data_class)

    # Defines a function to evaluate the Helmholtz free energy density
    # as a function of the elastic strain tensor

    def strain_energy(self, elastic_epsilon):

        # Evaluates the trace of the infinitesimal strain tensor and the
        # Frobenius norm squared

        I1_epsilon = ufl.tr(elastic_epsilon)

        frobenius_norm_sqquared = ufl.tr(elastic_epsilon*elastic_epsilon)

        return (self.mu*frobenius_norm_sqquared)+(0.5*self.lmbda*(
        I1_epsilon**2))
    
    # Defines a function to evaluate the Cauchy stress tensor as the ex-
    # pression that was defined in the method for updating the internal
    # variables. The object must be defined there, since it will be 
    # changed accordingly to the build up of the variational form or not

    def cauchy_stress(self, u):

        epsilon = sym(grad(u))
        
        return ufl.as_tensor(self.C_fourth[p,q,r,s]*epsilon[r,s], (p,q)
        )+self.sigma_0
    
    # Defines a function to update the internal variables and to get the 
    # tangent tensor, i.e.dsigma/depsilon

    def update_internal_variables(self, u, 
    flag_newton_raphson_iteration=True, verbose=False):

        if verbose:

            print("\nUpdate internal variables")

        # Retrieves from the class of quadrature spaces the last itera-
        # tion of the plastic strain tensor and the accumulated plastic
        # strain. Convert both to numpy arrays in the format 
        # [n_quadrature_points, 3, 3] (plastic strain tensor) and 
        # [n_quadrature_points] (accumulated plastic strain)

        last_plastic_strain_tensor = self.quadrature_projection_class.convert_quadrature_values_to_numpy_array(
        "plastic strain tensor", 
        self.quadrature_functional_data_class.solution_fields["plastic"+
        " strain tensor"])

        last_accumulated_plastic_strain = self.quadrature_projection_class.convert_quadrature_values_to_numpy_array(
        "accumulated plastic strain", 
        self.quadrature_functional_data_class.solution_fields["accumul"+
        "ated plastic strain"])

        if verbose:

            print("\nThe accumulated plastic strain before the return-"+
            "mapping algorithm lies in the interval ["+str(np.min(
            last_accumulated_plastic_strain))+", "+str(np.max(
            last_accumulated_plastic_strain))+"]")

        # Evaluates the infinitesimal strain tensor and the identity 
        # tensor

        epsilon = sym(grad(u))

        dimension = u.geometric_dimension()

        # Projects the strain tensor onto the quadrature space

        projected_epsilon = self.quadrature_projection_class.project_expression_onto_quadrature_space(
        "strain tensor", epsilon)

        # Assembles the identity tensor in the format 
        # [n_quadrature_points, 3, 3]

        projected_identity = np.broadcast_to(np.eye(dimension, dtype=
        projected_epsilon.dtype), projected_epsilon.shape)

        # Considering the trial plastic strain tensor as the last plas-
        # tic strain, computes the back stress

        trial_back_stress = self.evaluate_back_stress(
        last_plastic_strain_tensor)

        # Estimates the current elastic strain tensor

        trial_current_elastic_strain_tensor = (projected_epsilon-
        last_plastic_strain_tensor)

        # Computes the trial stress tensor

        stress_tensor = ((2.0*self.numpy_mu*trial_current_elastic_strain_tensor
        )+(self.numpy_lmbda*np.trace(trial_current_elastic_strain_tensor, 
        axis1=1, axis2=2)[:, None, None]*projected_identity))

        # Checks whether the yield function is less than zero

        yield_function = self.evaluate_yield_criterion(stress_tensor
        -trial_back_stress, last_accumulated_plastic_strain, 
        projected_identity, dimension, verbose=verbose)

        # Creates a mask for the points that are not within the yield
        # surface

        not_converged_quadrature_points_mask = yield_function>(
        self.newton_raphson_tolerance)
            
        # If some quadrature points are not within the yield surface,
        # calls the predictor-corrector algorithm

        if np.any(not_converged_quadrature_points_mask):

            if verbose:

                print("\nThere are points undergoing plastic deformati"+
                "on")
            
            (last_plastic_strain_tensor, last_accumulated_plastic_strain
            ) = self.update_plastic_strain(
            trial_current_elastic_strain_tensor, 
            last_plastic_strain_tensor, last_accumulated_plastic_strain, 
            projected_epsilon, projected_identity,
            not_converged_quadrature_points_mask,
            flag_newton_raphson_iteration, verbose)[1:3]

        else:

            # Updates the function with the stress

            self.quadrature_projection_class.project_expression_onto_quadrature_space(
            "stress tensor", (2.0*self.mu*epsilon)+(self.lmbda*ufl.tr(
            epsilon)*Identity(dimension)))

        # Gets the functions of the normalized trial update direction 
        # and the update increment as ufl expressions

        normalized_trial_update_direction = self.quadrature_functional_data_class.solution_fields[
        "normalized trial update direction"]

        update_increment = self.quadrature_functional_data_class.solution_fields[
        "update increment"]

        trial_equivalent_stress_reciprocal = self.quadrature_functional_data_class.solution_fields[
        "trial equivalent stress reciprocal"]

        # Computes the projector tensor

        identity = Identity(dimension)

        projector_tensor = ufl.as_tensor((identity[p,r]*identity[q,s
        ])-((1.0/dimension)*identity[p,q]*identity[r,s]), (p,q,r,s))

        # Computes the tensor product of the normalized trial direction

        eta_tensor_eta = ufl.as_tensor((
        normalized_trial_update_direction[p,q]*
        normalized_trial_update_direction[r,s]), (p,q,r,s))

        # Computes the tensor product of the identity tensor with itself

        I_tensor_I = ufl.as_tensor((identity[p,q]*identity[r,s]), (p,q,
        r,s))

        # Evaluates the derivative of the limit stress with respect to
        # the accumulated plastic strain

        sigma_y, K = self.limit_stress_model(last_accumulated_plastic_strain)

        if verbose:

            print("\nThe updated limit stress lies in the interval ["+str(
            np.min(sigma_y))+", "+str(np.max(sigma_y))+"]\n")

        self.quadrature_projection_class.update_quadrature_values_with_numpy_array(
        "hardening tangent modulus", 
        self.quadrature_functional_data_class.solution_fields["hardeni"+
        "ng tangent modulus"], K)

        K = self.quadrature_functional_data_class.solution_fields["har"+
        "dening tangent modulus"]

        # Assembles the fourth-order tangent modulus

        C = (self.bulk_modulus*I_tensor_I)+(2.0*self.mu*(1.0-(((3.0*
        self.mu)+self.H)*(update_increment*
        trial_equivalent_stress_reciprocal)))*projector_tensor)+(2.0*
        self.mu*((3.0*self.mu)+self.H)*((update_increment*
        trial_equivalent_stress_reciprocal)-(1.0/((3.0*self.mu)+self.H+K
        )))*eta_tensor_eta)

        # Computes the coefficient sigma_0 such that
        # sigma = C:(delta_epsilon)+sigma = C:epsilon+(sigma-epsilon_0)
        #       = C:epsilon+sigma_0

        self.C_fourth = C

        self.quadrature_projection_class.project_expression_onto_quadrature_space(
        "strain tensor", epsilon)

        strain_tensor = self.quadrature_functional_data_class.solution_fields[
        "strain tensor"]

        self.stress_tensor = self.quadrature_functional_data_class.solution_fields[
        "stress tensor"]

        self.sigma_0 = self.stress_tensor-ufl.as_tensor((C[p,q,r,s]*
        strain_tensor[r,s]), (p,q))

    ####################################################################
    #                      Plasticity formulation                      #
    ####################################################################

    # Defines a function to compute the limit stress in the yield func-
    # tion and its derivative with respect to the accumulated plastic 
    # strain

    def limit_stress_model(self, accumulated_plastic_strain):

        return (self.sigma_y+(self.K_1*accumulated_plastic_strain)+(
        self.K_2*(1.0-np.exp(-self.theta*accumulated_plastic_strain))),
        self.K_1+(self.K_2*self.theta*np.exp(-self.theta*
        accumulated_plastic_strain)))

    # Defines a function to compute the back stress

    def evaluate_back_stress(self, plastic_strain_tensor):

        return (2.0/3.0)*self.H*plastic_strain_tensor

    # Defines a function to compute the yield criterion using the von
    # Mises criterion

    def evaluate_yield_criterion(self, stress_tensor, 
    accumulated_plastic_strain, identity_tensor, dimension, verbose):

        # Computes the limit stress

        limit_stress, _ = self.limit_stress_model(
        accumulated_plastic_strain)

        # Computes the deviatoric parcel of the stress

        deviatoric_stress = stress_tensor-((np.trace(stress_tensor,
        axis1=1, axis2=2)/dimension)[:, None, None]*identity_tensor)

        von_mises_stress = np.sqrt((3.0/2.0)*np.einsum('ijk,ijk->i', 
        deviatoric_stress, deviatoric_stress))

        if verbose:

            print("\nThe von-Mises stress in the whole mesh lies in th"+
            "e following interval: ["+str(np.min(von_mises_stress))+","+
            " "+str(np.max(von_mises_stress))+"]\nwhereas the limit st"+
            "ress is an element of the interval ["+str(np.min(
            limit_stress))+", "+str(np.max(limit_stress))+"]")

        # Computes the von-Mises criterion

        phi = von_mises_stress-limit_stress

        return phi

    # Defines a function to update the plastic strain tensor and the ac-
    # cumulated plastic strain

    def update_plastic_strain(self, trial_elastic_strain_tensor, 
    trial_plastic_strain_tensor, accumulated_plastic_strain, 
    strain_tensor, identity, not_converged_quadrature_points_mask,
    flag_newton_raphson_iteration, verbose):

        # Computes the deviatoric part of the trial elastic strain tensor

        trial_elastic_deviatoric_strain = trial_elastic_strain_tensor-((
        np.trace(trial_elastic_strain_tensor, axis1=1, axis2=2)
        /identity.shape[2])[:, None, None]*identity)

        # Computes the trial direction of update of the strain tensor

        trial_update_direction = (2.0*self.numpy_mu*
        trial_elastic_deviatoric_strain)-self.evaluate_back_stress(
        trial_plastic_strain_tensor)

        # Computes the norm of the trial update direction

        norm_trial_update_direction = np.sqrt(np.einsum('ijk,ijk->i',
        trial_update_direction, trial_update_direction))

        # Computes the equivalent stress using the trial update direction

        trial_equivalent_stress = np.sqrt((3.0/2.0)
        )*norm_trial_update_direction

        # Computes the reciprocal of the equivalent trial stress, but
        # makes sure the result is zero then the equivalent stress is 
        # zero

        reciprocal_trial_equivalent_stress = np.divide(1.0, 
        trial_equivalent_stress, out=np.zeros_like(
        trial_equivalent_stress, dtype=trial_equivalent_stress.dtype), 
        where=(trial_equivalent_stress!=0.0))

        # Initializes the plastic update increment

        update_increment = np.zeros_like(accumulated_plastic_strain,
        dtype=accumulated_plastic_strain.dtype)

        # Initializes the updated accumulated plastic strain

        updated_accumulated_plastic_strain = (accumulated_plastic_strain
        +0.0)

        # Saves the initial state of the mask for quadrature points that
        # underwent plastic deformation. This mask will be progressively
        # modified as the quadrature points converge during the consti-
        # tutive Newton-Raphson

        plastic_points_mask = not_converged_quadrature_points_mask.copy()

        # Initializes the yield function

        phi = None

        # Iterates over the number of inner Newton-Raphson iterations

        for _ in range(self.maximum_inner_newton_raphson_iterations):

            # Updates the accumulated plastic strain

            updated_accumulated_plastic_strain = (
            accumulated_plastic_strain+update_increment)

            # Computes new limit stress and its derivative with respect
            # to the accumulated plastic strain

            sigma_y, K = self.limit_stress_model(
            updated_accumulated_plastic_strain)

            # Computes the yield function as a function of the accumula-
            # ted plastic strain

            phi = (np.sqrt(3.0/2.0)*(1.0-((update_increment*
            reciprocal_trial_equivalent_stress)*((3.0*self.numpy_mu)+
            self.H)))*norm_trial_update_direction)-sigma_y

            if verbose:

                print("phi during correction lies in the interval ["+str(
                np.min(phi))+", "+str(np.max(phi))+"]")

            # Updates the mask of points that are not within the yield
            # surface yet

            not_converged_quadrature_points_mask &= np.abs(phi
            )>self.newton_raphson_tolerance

            # Checks if the yield function is 0 for all quadrature points

            if not np.any(not_converged_quadrature_points_mask):

                break 

            # Updates only the unconverged quadrature points

            update_increment[not_converged_quadrature_points_mask] += (
            phi[not_converged_quadrature_points_mask]/((3.0*self.numpy_mu)+
            self.H+K[not_converged_quadrature_points_mask]))

        # Checks whether the Newton-Raphson procedure converged for all 
        # quadrature points
        
        if np.any(not_converged_quadrature_points_mask):

            raise RuntimeError("The Newton-Raphson procedure to correc"+
            "t all quadrature points\nundergoing plastic deformation h"+
            "as not converged for all of them. Check the\nrequired tol"+
            "erances. The maximum value of the yield function is "+str(
            np.max(phi))) 

        # Initializes the normalized update direction as a zero array. 
        # Thus, points that are plastified receive no update

        normalized_trial_update_direction = np.zeros_like(
        trial_update_direction)

        # Uses the mask of plastic quadrature points to update only tho-
        # se

        normalized_trial_update_direction[plastic_points_mask] = (
        trial_update_direction[plastic_points_mask]/
        norm_trial_update_direction[plastic_points_mask, np.newaxis, 
        np.newaxis])

        # Updates the plastic strain tensor

        trial_plastic_strain_tensor[plastic_points_mask] += (
        update_increment[plastic_points_mask, np.newaxis, np.newaxis]*
        np.sqrt(3.0/2.0)*normalized_trial_update_direction[
        plastic_points_mask])

        # Computes the stress tensor

        elastic_strain_tensor = strain_tensor-trial_plastic_strain_tensor

        stress_tensor = (2.0*self.numpy_mu*elastic_strain_tensor)+(
        self.numpy_lmbda*np.trace(elastic_strain_tensor, axis1=1, axis2=
        2)[:, None, None]*identity)

        self.evaluate_yield_criterion(stress_tensor-self.evaluate_back_stress(
        trial_plastic_strain_tensor), updated_accumulated_plastic_strain, 
        identity, identity.shape[2], False)

        # Updates the functions of all intermediate quantities

        self.quadrature_projection_class.update_quadrature_values_with_numpy_array(
        "stress tensor", 
        self.quadrature_functional_data_class.solution_fields["stress "+
        "tensor"], stress_tensor)
        
        self.quadrature_projection_class.update_quadrature_values_with_numpy_array(
        "normalized trial update direction", 
        self.quadrature_functional_data_class.solution_fields["normali"+
        "zed trial update direction"], normalized_trial_update_direction)

        self.quadrature_projection_class.update_quadrature_values_with_numpy_array(
        "update increment", 
        self.quadrature_functional_data_class.solution_fields["update "+
        "increment"], update_increment)

        # Saves the reciprocal of the equivalent stress

        self.quadrature_projection_class.update_quadrature_values_with_numpy_array(
        "trial equivalent stress reciprocal", 
        self.quadrature_functional_data_class.solution_fields["trial e"+
        "quivalent stress reciprocal"], 
        reciprocal_trial_equivalent_stress)

        # Updates the functions of plastic strain tensor and of the ac-
        # cumulated plastic strain only if this is a truly updating ite-
        # ration (Newton-Raphson iterations must have the same internal
        # variables)

        if not flag_newton_raphson_iteration:

            self.quadrature_projection_class.update_quadrature_values_with_numpy_array(
            "plastic strain tensor", 
            self.quadrature_functional_data_class.solution_fields["pla"+
            "stic strain tensor"], trial_plastic_strain_tensor)

            self.quadrature_projection_class.update_quadrature_values_with_numpy_array(
            "accumulated plastic strain", 
            self.quadrature_functional_data_class.solution_fields["acc"+
            "umulated plastic strain"], 
            updated_accumulated_plastic_strain)

        # Returns the updated plastic strain tensor and the accumulated
        # plastic strain in a dictionary

        return (stress_tensor, trial_plastic_strain_tensor,
        updated_accumulated_plastic_strain, update_increment, 
        normalized_trial_update_direction, trial_equivalent_stress)