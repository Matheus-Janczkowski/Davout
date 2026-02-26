# Routine to store tests for batching the FEM

import unittest

import tensorflow as tf

from time import time

import numpy as np

from dolfin import assemble

from ..physics.compressible_cauchy_hyperelasticy import CompressibleHyperelasticity

from ..constitutive_models.hyperelastic_isotropic_models import NeoHookean

from ..tool_box import mesh_tools

from ...MultiMech.tool_box.mesh_handling_tools import create_box_mesh, read_mshMesh, dofs_per_node_finder_class

from ...MultiMech.tool_box import functional_tools, variational_tools

from ...MultiMech.constitutive_models.hyperelasticity.isotropic_hyperelasticity import NeoHookean as NeoHookeanMultiMech

from ...PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        self.file_name = "box"

        self.file_directory = get_parent_path_of_file()

        # Creates a box mesh 

        length_x = 0.2
        
        length_y = 0.3
        
        length_z = 1.0
        
        n_divisions_x = 2 

        n_divisions_y = 2
        
        n_divisions_z = 3

        self.quadrature_degree = 2

        self.n_subdomains_z = 2

        create_box_mesh(length_x, length_y, length_z, n_divisions_x, 
        n_divisions_y, n_divisions_z, file_name=self.file_name, verbose=
        False, convert_to_xdmf=False, file_directory=self.file_directory, 
        mesh_polinomial_order=2, n_subdomains_z=self.n_subdomains_z)

        # Defines a dictionary of finite element per field

        self.elements_per_field = {"Displacement": {"number of DOFs pe"+
        "r node": 3, "required element type": "tetrahedron of 10 nodes"}}

        # Reads this mesh

        self.mesh_data_class = mesh_tools.read_msh_mesh(self.file_name, 
        self.quadrature_degree, self.elements_per_field, verbose=True)

        # Sets the base loads

        self.base_dirichlet_load = [0.1]

        self.base_neumann_load = [0.0E5]

        self.base_current_time = 1.0

        # Sets the base material properties

        self.base_material_properties = [{"E": 1E6, "nu": 0.4}, {"E": 
        2E6, "nu": 0.4}]

        ################################################################
        #                            FEniCS                            #
        ################################################################

        # Evaluates the residual vector using FEniCS

        mesh_data_class_fenics = read_mshMesh({"length x": length_x, "length y": 
        length_y, "length z": length_z, "number of divisions in x": 
        n_divisions_x, "number of divisions in y": n_divisions_y, "num"+
        "ber of divisions in z": n_divisions_z, "verbose": False, "mes"+
        "h file name": "box_mesh", "mesh file directory": 
        get_parent_path_of_file(), "number of subdomains in z direction":
        self.n_subdomains_z})

        functional_data_class = functional_tools.construct_monolithicFunctionSpace(
        {"Displacement": {"field type": "vector", "interpolation funct"+
        "ion": "CG", "polynomial degree": 2}}, mesh_data_class_fenics)

        # Initializes a list of bcs and of Dirichlet loads

        list_of_bcs = []

        list_of_dirichlet_loads = []

        # Iterates over the list of base Dirichlet loads

        for dirichlet_load in self.base_dirichlet_load:

            bcs, dirichlet_loads = functional_tools.construct_DirichletBCs(
            {"top": {"BC case": "PrescribedDirichletBC", "bc_informati"+
            "onsDict": {"load_function": "linear", "degrees_ofFreedomL"+
            "ist": 2, "end_point": [1.0, dirichlet_load]}}, "bottom": {
            "BC case": "FixedSupportDirichletBC"}}, 
            functional_data_class, mesh_data_class_fenics)

            # Updates the load class

            dirichlet_loads[0].update_load(self.base_current_time)

            # Then appends them into the lists

            list_of_bcs.append(bcs)

            list_of_dirichlet_loads.append(dirichlet_loads)

        # Initializes a list of variations of the external work and of 
        # Neumann_loads loads

        list_of_external_work = []

        list_of_neumann_loads = []

        # Iterates over the Neumann loads

        for neumann_load in self.base_neumann_load:

            # Variational form of the exterior work using an uniform re-
            # ferential traction

            external_work, neumann_loads = variational_tools.traction_work(
            {"top": {"load case": "UniformReferentialTraction", "ampli"+
            "tude_tractionX": 0.0, "amplitude_tractionY": 0.0, "amplit"+
            "ude_tractionZ": neumann_load, "parametric_load_curve": "s"+
            "quare_root", "t": 0.0, "t_final": 1.0}}, "Displacement", 
            functional_data_class, mesh_data_class_fenics, [])

            # Updates the load class

            neumann_loads[0].update_load(self.base_current_time)

            # Then appends them into the lists

            list_of_external_work.append(external_work)

            list_of_neumann_loads.append(neumann_loads)

        # Constructs a list of instances of the constitutive model class
        # with different sets of material properties

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

        ################################################################
        #                          Comparison                          #
        ################################################################

        # Creates a class that finds the DOFs closest to a point

        dofs_finder = dofs_per_node_finder_class(functional_data_class)

        # Constructs a list of DOFs per node in FEniCS enumeration using
        # the node numbering of GMSH

        dofs_fenics_from_gmsh_nodes = np.array([dofs_finder(
        *node_coordinates) for node_coordinates in (
        self.mesh_data_class.nodes_coordinates)])

        # Gets the number of variations per category

        n_dirichlet_loads = len(self.base_dirichlet_load)

        n_neumann_loads = len(self.base_neumann_load)

        n_internal_works = len(self.base_material_properties)

        # Initializes a numpy array for the residual vector across para-
        # meters combinations (realizations)

        self.assembled_residual = np.zeros(
        self.mesh_data_class.global_number_dofs, int(n_internal_works*
        n_dirichlet_loads*n_neumann_loads))

        # Initializes an array telling the combination of indices of each
        # one of the varied parameters

        self.combinations = np.zeros((self.assembled_residual.shape[1],
        3), dtype=int)

        counter = 0

        # Iterates over the Dirichlet boundary conditions

        for bc_index, bcs in enumerate(list_of_bcs):

            # Updates the Dirichlet boundary conditions

            for bc in bcs:

                bc.apply(functional_data_class.solution_fields["Displa"+
                "cement"].vector())

            # Iterates over the Neumann boundary conditions

            for neumann_index, external_work in enumerate(
            list_of_external_work):

                # Iterates over the internal work

                for inner_index, internal_work in enumerate(
                list_of_internal_work):

                    # Assembles the residual vector and stores it in the
                    # corresponding realization column

                    self.assembled_residual[:,counter] = assemble(
                    internal_work-external_work)

                    # Updates the realization counter and the array of
                    # combinations

                    self.combinations[counter,:] = np.asarray([bc_index,
                    neumann_index, inner_index], dtype=int)

                    counter += 1

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
            # node number

            dof_number_gmsh = self.mesh_data_class.dofs_node_dict["Dis"+
            "placement"][dof_number_gmsh[0]][dof_number_gmsh[1]]

            self.residual_vector_fenics[dof_number_gmsh,:] = dof_value

    # Defines a function to test batching material parameters

    def test_batching_material_parameters(self):

        print("\n#####################################################"+
        "###################\n#                  Tests batching materi"+
        "al parameters                  #\n###########################"+
        "#############################################\n")

        ################################################################
        #                             FENN                             #
        ################################################################

        print("The read mesh has "+str(self.mesh_data_class.global_number_dofs
        )+" degrees of freedom")

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
            material_properties, self.mesh_data_class)

        # Sets the dictionary of traction classes

        traction_dictionary = {"top": {"load case": "TractionVectorOnS"+
        "urface", "amplitude_tractionX": 0.0, "amplitude_tractionY": 0.0, 
        "amplitude_tractionZ": self.base_neumann_load[0]}}

        # Instantiates the class to evaluate the residual vector

        residual_class = CompressibleHyperelasticity(self.mesh_data_class,
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

        # Plots both to compare

        residual_vector = residual_vector.numpy()

        n_nonzero_components = 0

        for i in range(residual_vector.shape[1]):

            if abs(residual_vector[0,i])>1E-5:

                n_nonzero_components += 1

            print("FENN: residual_vector[:,"+str(i)+"]="+str(
            residual_vector[:,i])+";            FEniCS: residual_vecto"+
            "r["+str(i)+"]="+str(self.residual_vector_fenics[i]))

        print("\nThe FEniCS residual vector has a length of "+str(len(
        self.assembled_residual)))

        print("The FENN residual vector has a shape of "+str(
        residual_vector.shape)+"\nwhere the first value tells the numb"+
        "er tells the number of realizations of the BVP and the second"+
        " one tells the number of DOFs")

        print("\nThere are "+str(n_nonzero_components)+" non-zero comp"+
        "onents in the residual vector calculated by FENN")#"""

# Runs all tests

if __name__=="__main__":

    unittest.main()