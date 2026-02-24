# Routine to assemble the residual vector of a hyperelastic compressible 
# Cauchy continuum

import tensorflow as tf

from ..assembly.hyperelastic_internal_work import CompressibleInternalWorkReferenceConfiguration

from ..assembly.surface_forces_work import ReferentialTractionWork

from ..assembly.dirichlet_boundary_conditions_enforcer import DirichletBoundaryConditions

from ..tool_box.mesh_tools import MshMeshData

# Defines a class that assembles the residual vector

class CompressibleHyperelasticity:

    def __init__(self, mesh_data_class, constitutive_models_dict, 
    vector_of_parameters=None, traction_dictionary=None, 
    boundary_conditions_dict=None, time=0.0, n_realizations=1):
        
        # Verifies if mesh data class is an instance of the class of mesh
        # data

        if isinstance(mesh_data_class, MshMeshData):
        
            self.global_number_dofs = mesh_data_class.global_number_dofs

            self.dtype = mesh_data_class.dtype

            # Verifies if the domain elements have the field displacement

            if not ("Displacement" in mesh_data_class.domain_elements):

                raise NameError("There is no field named 'Displacement"+
                "' in the mesh. Thus, it is not possible to compute Co"+
                "mpressibleHyperelasticity")

        # Otherwise, if it is a list, there will be a mesh for multiple
        # realizations
        
        elif isinstance(mesh_data_class, list):

            # Verifies if the number of meshes is within bounds of the
            # number of given realizations

            if len(mesh_data_class)>n_realizations:

                raise IndexError(str(len(mesh_data_class))+" meshes we"+
                "re provided, but only "+str(n_realizations)+" realiza"+
                "tions were given. The number of meshes must be at mos"+
                "t equal to the number of realizations")

            # Gets the number of DOFs and the type from the first mesh 
        
            self.global_number_dofs = mesh_data_class[0
            ].global_number_dofs

            self.dtype = mesh_data_class[0].dtype

            self.integer_dtype = mesh_data_class[0].integer_dtype

            # Iterates through the other meshes to verify if the number 
            # of DOFs and the float type are the same

            for i in range(len(mesh_data_class)):

                # Verifies the number of global DOFs

                if mesh_data_class[i].global_number_dofs!=(
                self.global_number_dofs):
                    
                    raise ValueError("The "+str(i+1)+"-th mesh has "+
                    str(mesh_data_class[i].global_number_dofs)+" DOFs,"+
                    " whereas the first mesh has "+str(
                    self.global_number_dofs)+" DOFs. All meshes, acros"+
                    "s all realizations must have the same number of D"+
                    "OFs and the same conectivities")
                
                # Verifies the float type

                if mesh_data_class[i].dtype!=self.dtype:
                    
                    raise ValueError("The "+str(i+1)+"-th mesh has dty"+
                    "pe="+str(mesh_data_class[i].dtype)+", whereas the"+
                    " first mesh has dtype="+str(self.dtype)+". All me"+
                    "shes, across all realizations must have the same "+
                    "numerical type dtype")
                
                # Verifies the integer type

                if mesh_data_class[i].integer_dtype!=self.integer_dtype:
                    
                    raise ValueError("The "+str(i+1)+"-th mesh has int"+
                    "eger_dtype="+str(mesh_data_class[i].integer_dtype)+
                    ", whereas the first mesh has integer_dtype="+str(
                    self.integer_dtype)+". All meshes, across all real"+
                    "izations must have the same integer type")
                
                # Verifies if the domain elements have the field displa-
                # cement

                if not ("Displacement" in mesh_data_class[i
                ].domain_elements):

                    raise NameError("There is no field named 'Displace"+
                    "ment' in the "+str(i+1)+"-th mesh. Thus, it is no"+
                    "t possible to compute CompressibleHyperelasticity")
                
        # If it is not a list, throws an error

        else:

            raise TypeError("'mesh_data_class' in 'CompressibleHyperel"+
            "asticity' is neither an instance of the class 'MshMeshDat"+
            "a' neither is it a list")

        # Creates a time object

        self.time = tf.Variable(time, dtype=self.dtype)

        # Initializes the global residual vector as a null variable

        self.global_residual_vector = tf.Variable(tf.zeros([
        n_realizations, self.global_number_dofs], dtype=self.dtype))

        # Initializes the vector of parameters of the FEM approximation
        # of the field using the global number of DOFs

        if vector_of_parameters is None:

            self.vector_of_parameters = tf.Variable(tf.zeros([
            n_realizations, self.global_number_dofs], dtype=self.dtype))
        
        # Instantiates the class to enforce Dirichlet boundary conditions

        self.BCs_class = DirichletBoundaryConditions(
        self.vector_of_parameters, boundary_conditions_dict, 
        mesh_data_class, self.time)
        
        # Instantiates the class to compute the parcel of the residual
        # vector due to the variation of the internal work

        self.internal_work_variation = CompressibleInternalWorkReferenceConfiguration(
        self.vector_of_parameters, constitutive_models_dict, 
        mesh_data_class)

        # Instantiates the class to compute the parcel of the residual
        # vector due to the variation of the external work made by the
        # surface tractions

        self.traction_work_variation = ReferentialTractionWork(
        self.vector_of_parameters, traction_dictionary, mesh_data_class)

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