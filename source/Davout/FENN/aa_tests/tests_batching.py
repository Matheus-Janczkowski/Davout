# Routine to store tests for batching the FEM

import numpy as np

from dolfin import assemble

from ..physics.compressible_cauchy_hyperelasticy import CompressibleHyperelasticity

from ..constitutive_models.hyperelastic_isotropic_models import NeoHookean

from ..tool_box import mesh_tools

from ...MultiMech.tool_box.mesh_handling_tools import create_box_mesh, read_mshMesh, dofs_per_node_finder_class

from ...MultiMech.tool_box import functional_tools, variational_tools

from ...MultiMech.constitutive_models.hyperelasticity.isotropic_hyperelasticity import NeoHookean as NeoHookeanMultiMech

from ...PythonicUtilities.path_tools import get_parent_path_of_file

from ...PythonicUtilities.numpy_tools import get_rows_given_column_values

from ...PythonicUtilities.testing_tools import run_class_of_tests

# Defines a function to compare residual vectors provided by FENN against
# those provided by FEniCS

def compare_residual_vectors(residual_vector, residual_vector_fenics,
assembled_residual, mesh_data_class, corresponding_realizations, 
test_name):
    
    # COnverts the residual vector from tensorflow tensor to numpy array

    residual_vector = residual_vector.numpy()

    n_nonzero_components = 0

    for i in range(residual_vector.shape[1]):

        if np.linalg.norm(residual_vector[:,i])>1E-5:

            n_nonzero_components += 1

        print("FENN: residual_vector[:,"+str(i)+"]="+str(residual_vector[
        :,i])+";            FEniCS: residual_vector["+str(i)+"]="+str(
        residual_vector_fenics[i, corresponding_realizations]), flush=
        True)

    print("\nThe FEniCS residual vector has a length of "+str(len(
    assembled_residual)), flush=True)

    print("The FENN residual vector has a shape of "+str(
    residual_vector.shape)+"\nwhere the first value tells the number t"+
    "ells the number of realizations of the BVP and the second one tel"+
    "ls the number of DOFs", flush=True)

    print("\nThere are "+str(n_nonzero_components)+" non-zero componen"+
    "ts in the residual vector calculated by FENN", flush=True)

    # Reads the quantity of DOFs from the first element of mesh data 
    # class for all meshes must share the same connectivity

    print("\nThe read mesh has "+str(mesh_data_class[0
    ].global_number_dofs)+" degrees of freedom in the '"+str(test_name)+
    "' test", flush=True)

# Defines a function to test the ANN tools methods

class TestANNTools:

    def __init__(self):

        ################################################################
        #                Exterior (variable) parameters                #
        ################################################################

        self.length_x = [0.2, 0.3]
        
        self.length_y = [0.3, 0.4]
        
        self.length_z = [1.0, 1.1]

        # Sets the base loads

        self.base_dirichlet_load = [0.1, 0.2, 0.3]

        self.base_neumann_load = [0.0E5, 1.0E5, 1.5E5]

        self.base_current_time = 1.0

        # Sets the base material properties

        self.base_material_properties = [{"E": 1E6, "nu": 0.4}, {"E": 
        2E6, "nu": 0.4}]

        ################################################################
        #                        Mesh generation                       #
        ################################################################

        self.file_directory = get_parent_path_of_file()

        # Creates a box mesh 

        self.file_name = ["box_"+str(i+1) for i in range(len(
        self.length_x))]
        
        self.n_divisions_x = 2 

        self.n_divisions_y = 2
        
        self.n_divisions_z = 3

        self.quadrature_degree = 2

        self.n_subdomains_z = 2

        # Iterates over the mesh lengths

        for length_x, length_y, length_z, file_name in zip(self.length_x, 
        self.length_y, self.length_z, self.file_name):
            
            # Creates the geometry and the mesh accordingly. However,
            # keeps the same discretization

            create_box_mesh(length_x, length_y, length_z, 
            self.n_divisions_x, self.n_divisions_y, self.n_divisions_z, 
            file_name=file_name, verbose=False, convert_to_xdmf=False, 
            file_directory=self.file_directory, mesh_polinomial_order=2, 
            n_subdomains_z=self.n_subdomains_z)

        # Defines a dictionary of finite element per field

        self.elements_per_field = {"Displacement": {"number of DOFs pe"+
        "r node": 3, "required element type": "tetrahedron of 10 nodes"}}

        # Reads this mesh

        self.mesh_data_class = mesh_tools.read_msh_mesh(self.file_name, 
        self.quadrature_degree, self.elements_per_field, verbose=True)

        # Gets the number of variations per category

        self.n_dirichlet_loads = len(self.base_dirichlet_load)

        self.n_neumann_loads = len(self.base_neumann_load)

        self.n_internal_works = len(self.base_material_properties)

        self.n_meshes = len(self.length_x)

        # Initializes a numpy array for the residual vector across para-
        # meters combinations (realizations). Takes the first element of 
        # mesh data class for all meshes must share the same connectivity

        self.assembled_residual = np.zeros((
        self.mesh_data_class[0].global_number_dofs, int(
        self.n_internal_works*self.n_dirichlet_loads*
        self.n_neumann_loads*self.n_meshes)))

        # Initializes an array telling the combination of indices of each
        # one of the varied parameters

        self.combinations = np.zeros((self.assembled_residual.shape[1],
        4), dtype=int)

        realizations_counter = 0

        # Iterates over the meshes

        mesh_counter = 0

        for length_x, length_y, length_z, file_name in zip(self.length_x, 
        self.length_y, self.length_z, self.file_name):

            ############################################################
            #                          FEniCS                          #
            ############################################################

            # Evaluates the residual vector using FEniCS

            mesh_data_class_fenics = read_mshMesh({"length x": length_x, 
            "length y": length_y, "length z": length_z, "number of div"+
            "isions in x": self.n_divisions_x, "number of divisions in"+
            " y": self.n_divisions_y, "number of divisions in z": 
            self.n_divisions_z, "verbose": False, "mesh file name": "b"+
            "ox_mesh", "mesh file directory": get_parent_path_of_file(), 
            "number of subdomains in z direction": self.n_subdomains_z})

            functional_data_class = functional_tools.construct_monolithicFunctionSpace(
            {"Displacement": {"field type": "vector", "interpolation f"+
            "unction": "CG", "polynomial degree": 2}}, 
            mesh_data_class_fenics)

            # Initializes a list of bcs and of Dirichlet loads

            list_of_bcs = []

            list_of_dirichlet_loads = []

            # Iterates over the list of base Dirichlet loads

            for dirichlet_load in self.base_dirichlet_load:

                bcs, dirichlet_loads = functional_tools.construct_DirichletBCs(
                {"top": {"BC case": "PrescribedDirichletBC", "bc_infor"+
                "mationsDict": {"load_function": "linear", "degrees_of"+
                "FreedomList": 2, "end_point": [1.0, dirichlet_load]}}, 
                "bottom": {"BC case": "FixedSupportDirichletBC"}}, 
                functional_data_class, mesh_data_class_fenics)

                # Updates the load class

                dirichlet_loads[0].update_load(self.base_current_time)

                # Then appends them into the lists

                list_of_bcs.append(bcs)

                list_of_dirichlet_loads.append(dirichlet_loads)

            # Initializes a list of variations of the external work and 
            # of Neumann_loads loads

            list_of_external_work = []

            list_of_neumann_loads = []

            # Iterates over the Neumann loads

            for neumann_load in self.base_neumann_load:

                # Variational form of the exterior work using an uniform 
                # referential traction

                external_work, neumann_loads = variational_tools.traction_work(
                {"top": {"load case": "UniformReferentialTraction", "a"+
                "mplitude_tractionX": 0.0, "amplitude_tractionY": 0.0, 
                "amplitude_tractionZ": neumann_load, "parametric_load_"+
                "curve": "square_root", "t": 0.0, "t_final": 1.0}}, "D"+
                "isplacement", functional_data_class, 
                mesh_data_class_fenics, [])

                # Updates the load class

                neumann_loads[0].update_load(self.base_current_time)

                # Then appends them into the lists

                list_of_external_work.append(external_work)

                list_of_neumann_loads.append(neumann_loads)

            # Constructs a list of instances of the constitutive model 
            # class with different sets of material properties

            list_of_constitutive_model_multimech = [NeoHookeanMultiMech(
            material_properties) for material_properties in (
            self.base_material_properties)]

            # Checks the constitutive models

            [constitutive_model.check_model(None) for (constitutive_model
            ) in list_of_constitutive_model_multimech]

            list_of_internal_work = [variational_tools.hyperelastic_internalWorkFirstPiola(
            "Displacement", functional_data_class, constitutive_mode, 
            mesh_data_class_fenics) for constitutive_mode in (
            list_of_constitutive_model_multimech)]

            ############################################################
            #                        Comparison                        #
            ############################################################

            # Creates a class that finds the DOFs closest to a point

            dofs_finder = dofs_per_node_finder_class(
            functional_data_class)

            # Constructs a list of DOFs per node in FEniCS enumeration 
            # using the node numbering of GMSH

            dofs_fenics_from_gmsh_nodes = np.array([dofs_finder(
            *node_coordinates) for node_coordinates in (
            self.mesh_data_class[mesh_counter].nodes_coordinates)])

            # Iterates over the Dirichlet boundary conditions

            for bc_index, bcs in enumerate(list_of_bcs):

                # Updates the Dirichlet boundary conditions

                for bc in bcs:

                    bc.apply(functional_data_class.solution_fields["Di"+
                    "splacement"].vector())

                # Iterates over the Neumann boundary conditions

                for neumann_index, external_work in enumerate(
                list_of_external_work):

                    # Iterates over the internal work

                    for inner_index, internal_work in enumerate(
                    list_of_internal_work):

                        # Assembles the residual vector and stores it in 
                        # the corresponding realization column

                        self.assembled_residual[:,realizations_counter
                        ] = assemble(internal_work-external_work)

                        # Updates the realization realizations_counter 
                        # and the array of combinations

                        self.combinations[realizations_counter,:
                        ] = np.asarray([mesh_counter, bc_index, 
                        neumann_index, inner_index], dtype=int)

                        realizations_counter += 1

            # Updates the mesh couter

            mesh_counter += 1

        # Creates a numpy array similar to the assembled residual. The
        # only difference is that this array will have the same DOF or-
        # dering as gmsh

        self.residual_vector_fenics = np.zeros_like(
        self.assembled_residual)

        for dof_number, dof_value in enumerate(self.assembled_residual):

            # Gets the DOF enumeration in gmsh

            dof_number_gmsh = np.argwhere(dofs_fenics_from_gmsh_nodes==
            dof_number)[0]

            # Gets the DOF number in the FENN numbering system using the
            # node number. Takes the first mesh for all meshes share the
            # same connectivities

            dof_number_gmsh = self.mesh_data_class[0].dofs_node_dict[
            "Displacement"][dof_number_gmsh[0]][dof_number_gmsh[1]]

            self.residual_vector_fenics[dof_number_gmsh,:] = dof_value

    # Defines a function to test batching material parameters

    def test_batching_material_parameters(self):

        print("\n#####################################################"+
        "###################\n#                  Tests batching materi"+
        "al parameters                  #\n###########################"+
        "#############################################\n", flush=True)

        # Creates a dictionary to tell Dirichlet boundary conditions

        boundary_conditions_dict = {"top": {"BC case": "PrescribedDiri"+
        "chletBC", "load_function": "linear", "degrees_ofFreedomList": 2,
        "end_point": [1.0, self.base_dirichlet_load[0]], "field name": "D"+
        "isplacement"}, "bottom": {"BC case": "FixedSupportDirichletBC", 
        "field name": "Displacement"}}

        # Sets the dictionary of constitutive models

        material_properties = self.base_material_properties

        constitutive_models = dict()

        for subdomain in range(self.n_subdomains_z):

            constitutive_models["volume "+str(subdomain+1)] = NeoHookean(
            material_properties, self.mesh_data_class[0])

        # Sets the dictionary of traction classes

        traction_dictionary = {"top": {"load case": "TractionVectorOnS"+
        "urface", "amplitude_tractionX": 0.0, "amplitude_tractionY": 0.0, 
        "amplitude_tractionZ": self.base_neumann_load[0]}}

        # Instantiates the class to evaluate the residual vector

        residual_class = CompressibleHyperelasticity(self.mesh_data_class[0],
        constitutive_models, traction_dictionary=traction_dictionary, 
        boundary_conditions_dict=boundary_conditions_dict, time=
        self.base_current_time, save_vector_of_parameters_in_class=True,
        n_realizations=2)

        # Gets the vector of parameters and the global residual vector

        vector_of_parameters = residual_class.vector_of_parameters

        global_residual_vector = residual_class.global_residual_vector

        # Evaluates the residual

        residual_vector = residual_class.evaluate_residual_vector(
        vector_of_parameters, global_residual_vector)

        # Finds the indices of the realizations whose Dirichlet and Neu-
        # mann boundary conditions are in the first set position. The 
        # combinations array has its columns related to:
        # 0. Mesh
        # 1. Dirichlet boundary conditions
        # 2. Neumann boundary conditions
        # 3. Material parameters

        _, corresponding_realizations = get_rows_given_column_values(
        self.combinations, [0, 1, 2], [0, 0, 0])

        compare_residual_vectors(residual_vector, 
        self.residual_vector_fenics, self.assembled_residual, 
        self.mesh_data_class, corresponding_realizations, "BATCHED MAT"+
        "ERIAL PARAMETERS")

    # Defines a function to test batching Dirichlet boundary conditions

    def test_batching_dirichlet_boundary_conditions(self):

        print("\n#####################################################"+
        "###################\n#             Tests batching Dirichlet b"+
        "oundary conditions             #\n###########################"+
        "#############################################\n", flush=True)

        # Creates a dictionary to tell Dirichlet boundary conditions

        boundary_conditions_dict = {"top": {"BC case": "PrescribedDiri"+
        "chletBC", "load_function": "linear", "degrees_ofFreedomList": 2,
        "end_point": [1.0, [self.base_dirichlet_load]], "field name": 
        "Displacement"}, "bottom": {"BC case": "FixedSupportDirichletB"+
        "C", "field name": "Displacement"}}

        # Sets the dictionary of constitutive models. Selects the first
        # set of material parameters only

        material_properties = self.base_material_properties[0]

        constitutive_models = dict()

        for subdomain in range(self.n_subdomains_z):

            constitutive_models["volume "+str(subdomain+1)] = NeoHookean(
            material_properties, self.mesh_data_class[0])

        # Sets the dictionary of traction classes

        traction_dictionary = {"top": {"load case": "TractionVectorOnS"+
        "urface", "amplitude_tractionX": 0.0, "amplitude_tractionY": 0.0, 
        "amplitude_tractionZ": self.base_neumann_load[0]}}

        # Instantiates the class to evaluate the residual vector. Se-
        # lects the number of realizations as the number of Dirichlet
        # loads

        residual_class = CompressibleHyperelasticity(self.mesh_data_class[0],
        constitutive_models, traction_dictionary=traction_dictionary, 
        boundary_conditions_dict=boundary_conditions_dict, time=
        self.base_current_time, save_vector_of_parameters_in_class=True,
        n_realizations=len(self.base_dirichlet_load))

        # Gets the vector of parameters and the global residual vector

        vector_of_parameters = residual_class.vector_of_parameters

        global_residual_vector = residual_class.global_residual_vector

        # Evaluates the residual

        residual_vector = residual_class.evaluate_residual_vector(
        vector_of_parameters, global_residual_vector)

        # Finds the indices of the realizations whose Dirichlet and Neu-
        # mann boundary conditions are in the first set position. The 
        # combinations array has its columns related to:
        # 0. Mesh
        # 1. Dirichlet boundary conditions
        # 2. Neumann boundary conditions
        # 3. Material parameters

        _, corresponding_realizations = get_rows_given_column_values(
        self.combinations, [0, 2, 3], [0, 0, 0])

        compare_residual_vectors(residual_vector, 
        self.residual_vector_fenics, self.assembled_residual, 
        self.mesh_data_class, corresponding_realizations, "BATCHED DIR"+
        "ICHLET BCs")

    # Defines a function to test batching Neumann boundary conditions

    def test_batching_neumann_boundary_conditions(self):

        print("\n#####################################################"+
        "###################\n#              Tests batching Neumann bo"+
        "undary conditions              #\n###########################"+
        "#############################################\n", flush=True)

        # Creates a dictionary to tell Dirichlet boundary conditions

        boundary_conditions_dict = {"top": {"BC case": "PrescribedDiri"+
        "chletBC", "load_function": "linear", "degrees_ofFreedomList": 2,
        "end_point": [1.0, self.base_dirichlet_load[0]], "field name": 
        "Displacement"}, "bottom": {"BC case": "FixedSupportDirichletB"+
        "C", "field name": "Displacement"}}

        # Sets the dictionary of constitutive models. Selects the first
        # set of material parameters only

        material_properties = self.base_material_properties[0]

        constitutive_models = dict()

        for subdomain in range(self.n_subdomains_z):

            constitutive_models["volume "+str(subdomain+1)] = NeoHookean(
            material_properties, self.mesh_data_class[0])

        # Sets the dictionary of traction classes

        traction_dictionary = {"top": {"load case": "TractionVectorOnS"+
        "urface", "amplitude_tractionX": np.zeros(len(
        self.base_neumann_load)), "amplitude_tractionY": np.zeros(len(
        self.base_neumann_load)), "amplitude_tractionZ": 
        self.base_neumann_load}}

        # Instantiates the class to evaluate the residual vector. Se-
        # lects the number of realizations as the number of Dirichlet
        # loads

        residual_class = CompressibleHyperelasticity(self.mesh_data_class[0],
        constitutive_models, traction_dictionary=traction_dictionary, 
        boundary_conditions_dict=boundary_conditions_dict, time=
        self.base_current_time, save_vector_of_parameters_in_class=True,
        n_realizations=len(self.base_neumann_load))

        # Gets the vector of parameters and the global residual vector

        vector_of_parameters = residual_class.vector_of_parameters

        global_residual_vector = residual_class.global_residual_vector

        # Evaluates the residual

        residual_vector = residual_class.evaluate_residual_vector(
        vector_of_parameters, global_residual_vector)

        # Finds the indices of the realizations whose Dirichlet and Neu-
        # mann boundary conditions are in the first set position. The 
        # combinations array has its columns related to:
        # 0. Mesh
        # 1. Dirichlet boundary conditions
        # 2. Neumann boundary conditions
        # 3. Material parameters

        _, corresponding_realizations = get_rows_given_column_values(
        self.combinations, [0, 1, 3], [0, 0, 0])

        compare_residual_vectors(residual_vector, 
        self.residual_vector_fenics, self.assembled_residual, 
        self.mesh_data_class, corresponding_realizations, "BATCHED NEU"+
        "MANN BCs")

    # Defines a function to test batching meshes

    def test_batching_meshes(self):

        print("\n#####################################################"+
        "###################\n#  Tests batching meshes with same conne"+
        "ctivity but different geometry #\n###########################"+
        "#############################################\n", flush=True)

        # Creates a dictionary to tell Dirichlet boundary conditions

        boundary_conditions_dict = {"top": {"BC case": "PrescribedDiri"+
        "chletBC", "load_function": "linear", "degrees_ofFreedomList": 2,
        "end_point": [1.0, self.base_dirichlet_load[0]], "field name": 
        "Displacement"}, "bottom": {"BC case": "FixedSupportDirichletB"+
        "C", "field name": "Displacement"}}

        # Sets the dictionary of constitutive models. Selects the first
        # set of material parameters only

        material_properties = self.base_material_properties[0]

        constitutive_models = dict()

        for subdomain in range(self.n_subdomains_z):

            constitutive_models["volume "+str(subdomain+1)] = NeoHookean(
            material_properties, self.mesh_data_class[0])

        # Sets the dictionary of traction classes

        traction_dictionary = {"top": {"load case": "TractionVectorOnS"+
        "urface", "amplitude_tractionX": 0.0, "amplitude_tractionY": 0.0,
        "amplitude_tractionZ": self.base_neumann_load[0]}}

        # Instantiates the class to evaluate the residual vector. Se-
        # lects the number of realizations as the number of Dirichlet
        # loads. Note that the whole mesh data class will be used; not
        # only the first element

        residual_class = CompressibleHyperelasticity(self.mesh_data_class,
        constitutive_models, traction_dictionary=traction_dictionary, 
        boundary_conditions_dict=boundary_conditions_dict, time=
        self.base_current_time, save_vector_of_parameters_in_class=True,
        n_realizations=len(self.mesh_data_class))

        # Gets the vector of parameters and the global residual vector

        vector_of_parameters = residual_class.vector_of_parameters

        global_residual_vector = residual_class.global_residual_vector

        # Evaluates the residual

        residual_vector = residual_class.evaluate_residual_vector(
        vector_of_parameters, global_residual_vector)

        # Finds the indices of the realizations whose Dirichlet and Neu-
        # mann boundary conditions are in the first set position. The 
        # combinations array has its columns related to:
        # 0. Mesh
        # 1. Dirichlet boundary conditions
        # 2. Neumann boundary conditions
        # 3. Material parameters

        _, corresponding_realizations = get_rows_given_column_values(
        self.combinations, [1, 2, 3], [0, 0, 0])

        compare_residual_vectors(residual_vector, 
        self.residual_vector_fenics, self.assembled_residual, 
        self.mesh_data_class, corresponding_realizations, "BATCHED MES"+
        "HES WITH SAME CONNECTIVITIES BUT DIFFERENT GEOMETRY")

    # Defines a function to test batching everything

    def test_batching_everything(self):

        print("\n#####################################################"+
        "###################\n#                       Tests batching E"+
        "VERYTHING                      #\n###########################"+
        "#############################################\n", flush=True)

        # Gets the least number of variations per parameter

        n_realizations = np.min([self.n_dirichlet_loads, self.n_meshes, 
        self.n_internal_works, self.n_neumann_loads])

        # Creates a dictionary to tell Dirichlet boundary conditions

        boundary_conditions_dict = {"top": {"BC case": "PrescribedDiri"+
        "chletBC", "load_function": "linear", "degrees_ofFreedomList": 2,
        "end_point": [1.0, [self.base_dirichlet_load[0:n_realizations]]], 
        "field name": "Displacement"}, "bottom": {"BC case": "FixedSup"+
        "portDirichletBC", "field name": "Displacement"}}

        # Sets the dictionary of constitutive models. Selects the first
        # set of material parameters only

        material_properties = self.base_material_properties[0:
        n_realizations]

        constitutive_models = dict()

        for subdomain in range(self.n_subdomains_z):

            # Takes the first mesh only, since the only use of mesh data
            # class in the constitutive model class is to take the float
            # type

            constitutive_models["volume "+str(subdomain+1)] = NeoHookean(
            material_properties, self.mesh_data_class[0])

        # Sets the dictionary of traction classes

        traction_dictionary = {"top": {"load case": "TractionVectorOnS"+
        "urface", "amplitude_tractionX": np.zeros(n_realizations), "am"+
        "plitude_tractionY": np.zeros(n_realizations), "amplitude_trac"+
        "tionZ": self.base_neumann_load[0:n_realizations]}}

        # Instantiates the class to evaluate the residual vector. Se-
        # lects the number of realizations as the number of Dirichlet
        # loads. Note that the whole mesh data class will be used; not
        # only the first element

        residual_class = CompressibleHyperelasticity(
        self.mesh_data_class[0:n_realizations], constitutive_models, 
        traction_dictionary=traction_dictionary, 
        boundary_conditions_dict=boundary_conditions_dict, time=
        self.base_current_time, save_vector_of_parameters_in_class=True,
        n_realizations=n_realizations)

        # Gets the vector of parameters and the global residual vector

        vector_of_parameters = residual_class.vector_of_parameters

        global_residual_vector = residual_class.global_residual_vector

        # Evaluates the residual

        residual_vector = residual_class.evaluate_residual_vector(
        vector_of_parameters, global_residual_vector)

        # Finds the indices of the realizations whose Dirichlet and Neu-
        # mann boundary conditions are in the first set position. The 
        # combinations array has its columns related to:
        # 0. Mesh
        # 1. Dirichlet boundary conditions
        # 2. Neumann boundary conditions
        # 3. Material parameters

        _, corresponding_realizations1 = get_rows_given_column_values(
        self.combinations, [0, 1, 2, 3], [0, 0, 0, 0])

        _, corresponding_realizations2 = get_rows_given_column_values(
        self.combinations, [0, 1, 2, 3], [1, 1, 1, 1])

        corresponding_realizations = np.concatenate((
        corresponding_realizations1, corresponding_realizations2), axis=0)

        compare_residual_vectors(residual_vector, 
        self.residual_vector_fenics, self.assembled_residual, 
        self.mesh_data_class, corresponding_realizations, "BATCHED EVE"+
        "RYTHING")

# Runs all tests

if __name__=="__main__":

    class_of_tests = TestANNTools()

    run_class_of_tests(class_of_tests)