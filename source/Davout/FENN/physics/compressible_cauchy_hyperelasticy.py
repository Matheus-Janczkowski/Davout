# Routine to assemble the residual vector of a hyperelastic compressible 
# Cauchy continuum

import tensorflow as tf

from ..assembly.hyperelastic_internal_work import CompressibleInternalWorkReferenceConfiguration

from ..assembly.surface_forces_work import ReferentialTractionWork

from ..assembly.dirichlet_boundary_conditions_enforcer import DirichletBoundaryConditions

# Defines a class that assembles the residual vector

class CompressibleHyperelasticity:

    def __init__(self, mesh_data_class, constitutive_models_dict, 
    vector_of_parameters=None, traction_dictionary=None, 
    boundary_conditions_dict=None, time=0.0, n_realizations=1):
        
        self.global_number_dofs = mesh_data_class.global_number_dofs

        self.dtype = mesh_data_class.dtype

        # Creates a time object

        self.time = tf.Variable(time, dtype=self.dtype)

        # Initializes the global residual vector as a null variable

        self.global_residual_vector = tf.Variable(tf.zeros([
        self.global_number_dofs], dtype=self.dtype))

        # Initializes the vector of parameters of the FEM approximation
        # of the field using the global number of DOFs

        if vector_of_parameters is None:

            vector_of_parameters = tf.Variable(tf.zeros([
            self.global_number_dofs], dtype=self.dtype))

        # Verifies if the domain elements have the field displacement

        if not ("Displacement" in mesh_data_class.domain_elements):

            raise NameError("There is no field named 'Displacement' in"+
            " the mesh. Thus, it is not possible to compute Compressib"+
            "leHyperelasticity")
        
        # Instantiates the class to enforce Dirichlet boundary conditions

        self.BCs_class = DirichletBoundaryConditions(vector_of_parameters, 
        boundary_conditions_dict, mesh_data_class.boundary_elements["D"+
        "isplacement"], mesh_data_class.boundary_physicalGroupsNameToTag, 
        self.time)
        
        # Instantiates the class to compute the parcel of the residual
        # vector due to the variation of the internal work

        self.internal_work_variation = CompressibleInternalWorkReferenceConfiguration(
        vector_of_parameters, constitutive_models_dict, 
        mesh_data_class.domain_elements["Displacement"], 
        mesh_data_class.domain_physicalGroupsNameToTag)

        # Instantiates the class to compute the parcel of the residual
        # vector due to the variation of the external work made by the
        # surface tractions

        self.traction_work_variation = ReferentialTractionWork(
        vector_of_parameters, traction_dictionary, 
        mesh_data_class.boundary_elements["Displacement"], 
        mesh_data_class.boundary_physicalGroupsNameToTag)

    # Defines a function to compute the residual vector

    @tf.function
    def evaluate_residual_vector(self):

        # Nullifies the residual for this evaluation

        self.global_residual_vector.assign(tf.zeros_like(
        self.global_residual_vector))

        # Adds the parcel of the variation of the internal work

        self.internal_work_variation.assemble_residual_vector(
        self.global_residual_vector)

        # Adds the parcel of the variation of the work due to surface 
        # tractions

        self.traction_work_variation.assemble_residual_vector(
        self.global_residual_vector)

        return self.global_residual_vector