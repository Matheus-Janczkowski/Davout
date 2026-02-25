# Routine to assemble the residual vector of a hyperelastic compressible 
# Cauchy continuum

import tensorflow as tf

from ..assembly.hyperelastic_internal_work import CompressibleInternalWorkReferenceConfiguration

from ..assembly.surface_forces_work import ReferentialTractionWork

from ..assembly.dirichlet_boundary_conditions_enforcer import DirichletBoundaryConditions

from ..tool_box.mesh_info_tools import verify_mesh_realizations

# Defines a class that assembles the residual vector

class CompressibleHyperelasticity:

    def __init__(self, mesh_data_class, constitutive_models_dict, 
    vector_of_parameters=None, traction_dictionary=None, 
    boundary_conditions_dict=None, time=0.0, n_realizations=1, 
    save_vector_of_parameters_in_class=True):
        
        # Verifies if there are realizations of the mesh and gathers im-
        # portant information, such as global number of DOFs and numeri-
        # cal types

        (self.global_number_dofs, self.dtype, self.integer_dtype
        ) = verify_mesh_realizations(mesh_data_class, n_realizations, 
        "Displacement", "CompressibleHyperelasticity")

        # Creates a time object

        self.time = tf.Variable(time, dtype=self.dtype)

        # Initializes the global residual vector as a null variable

        self.global_residual_vector = tf.Variable(tf.zeros([
        n_realizations, self.global_number_dofs], dtype=self.dtype))

        # Initializes the vector of parameters of the FEM approximation
        # of the field using the global number of DOFs

        if vector_of_parameters is None:

            vector_of_parameters = tf.Variable(tf.zeros([n_realizations, 
            self.global_number_dofs], dtype=self.dtype))
        
        # Instantiates the class to enforce Dirichlet boundary conditions

        self.BCs_class = DirichletBoundaryConditions(
        vector_of_parameters, boundary_conditions_dict, mesh_data_class, 
        self.time, apply_BCs_at_initialization=
        save_vector_of_parameters_in_class)
        
        # Instantiates the class to compute the parcel of the residual
        # vector due to the variation of the internal work

        self.internal_work_variation = CompressibleInternalWorkReferenceConfiguration(
        n_realizations, constitutive_models_dict, mesh_data_class)

        # Instantiates the class to compute the parcel of the residual
        # vector due to the variation of the external work made by the
        # surface tractions

        self.traction_work_variation = ReferentialTractionWork(
        n_realizations, traction_dictionary, mesh_data_class, "D"+
        "isplacement")

        # Saves the vector of parameters if needed

        if save_vector_of_parameters_in_class:

            self.vector_of_parameters = vector_of_parameters

    # Defines a function to update and to apply all boundary conditions

    @tf.function
    def apply_all_boundary_conditions(self, vector_of_parameters):

        # Updates all load curves

        self.BCs_class.update_boundary_conditions()

        # Applies all boundary conditions and returns the vector of pa-
        # rameters

        return self.BCs_class.apply_boundary_conditions(
        vector_of_parameters)

    # Defines a function to compute the residual vector

    @tf.function
    def evaluate_residual_vector(self, vector_of_parameters):

        # Nullifies the residual for this evaluation

        self.global_residual_vector.assign(tf.zeros_like(
        self.global_residual_vector))

        # Adds the parcel of the variation of the internal work

        self.internal_work_variation.assemble_residual_vector(
        self.global_residual_vector, vector_of_parameters)

        # Adds the parcel of the variation of the work due to surface 
        # tractions

        self.traction_work_variation.assemble_residual_vector(
        self.global_residual_vector)

        return self.global_residual_vector