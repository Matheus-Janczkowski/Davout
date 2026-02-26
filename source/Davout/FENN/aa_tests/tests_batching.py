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

        pass

    # Defines a function to test batching material parameters

    def test_batching_material_parameters(self):

        print("\n#####################################################"+
        "###################\n#                  Tests batching materi"+
        "al parameters                  #\n###########################"+
        "#############################################\n")

        file_name = "box"

        file_directory = get_parent_path_of_file()

        # Creates a box mesh 

        length_x = 0.2
        
        length_y = 0.3
        
        length_z = 1.0
        
        n_divisions_x = 2 

        n_divisions_y = 2
        
        n_divisions_z = 3

        quadrature_degree = 2

        n_subdomains_z = 2

        create_box_mesh(length_x, length_y, length_z, n_divisions_x, 
        n_divisions_y, n_divisions_z, file_name=file_name, verbose=False, 
        convert_to_xdmf=False, file_directory=file_directory, 
        mesh_polinomial_order=2, n_subdomains_z=n_subdomains_z)

        ################################################################
        #                             FENN                             #
        ################################################################

        # Defines a dictionary of finite element per field

        elements_per_field = {"Displacement": {"number of DOFs per nod"+
        "e": 3, "required element type": "tetrahedron of 10 nodes"}}

        # Reads this mesh

        mesh_data_class = mesh_tools.read_msh_mesh(file_name, 
        quadrature_degree, elements_per_field, verbose=True)

        print("The read mesh has "+str(mesh_data_class.global_number_dofs
        )+" degrees of freedom")

        # Sets the loads

        dirichlet_load = 0.1

        neumann_load = 0.0E5

        current_time = 1.0

        # Creates a dictionary to tell Dirichlet boundary conditions

        boundary_conditions_dict = {"top": {"BC case": "PrescribedDiri"+
        "chletBC", "load_function": "linear", "degrees_ofFreedomList": 2,
        "end_point": [1.0, dirichlet_load], "field name": "Displacement"
        }, "bottom": {"BC case": "FixedSupportDirichletBC", "field nam"+
        "e": "Displacement"}}

        # Sets the dictionary of constitutive models

        material_properties = [{"E": 1E6, "nu": 0.4}, {"E": 1E6, "nu": 0.4}]

        constitutive_models = dict()

        for subdomain in range(n_subdomains_z):

            constitutive_models["volume "+str(subdomain+1)] = NeoHookean(
            material_properties, mesh_data_class)

        # Sets the dictionary of traction classes

        traction_dictionary = {"top": {"load case": "TractionVectorOnS"+
        "urface", "amplitude_tractionX": 0.0, "amplitude_tractionY": 0.0, 
        "amplitude_tractionZ": neumann_load}}

        # Instantiates the class to evaluate the residual vector

        residual_class = CompressibleHyperelasticity(mesh_data_class,
        constitutive_models, traction_dictionary=traction_dictionary, 
        boundary_conditions_dict=boundary_conditions_dict, time=
        current_time, save_vector_of_parameters_in_class=True,
        n_realizations=2)

        # Gets the vector of parameters and the global residual vector

        vector_of_parameters = residual_class.vector_of_parameters

        global_residual_vector = residual_class.global_residual_vector

        # Evaluates the residual

        residual_vector = residual_class.evaluate_residual_vector(
        vector_of_parameters, global_residual_vector)

        """
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
        n_subdomains_z})

        functional_data_class = functional_tools.construct_monolithicFunctionSpace(
        {"Displacement": {"field type": "vector", "interpolation funct"+
        "ion": "CG", "polynomial degree": 2}}, mesh_data_class_fenics)

        # Dirichlet boundary conditions

        bcs, dirichlet_loads = functional_tools.construct_DirichletBCs({
        "top": {"BC case": "PrescribedDirichletBC", "bc_information"+
        "sDict": {"load_function": "linear", "degrees_ofFreedomList": 2,
        "end_point": [1.0, dirichlet_load]}}, "bottom": {"BC case": "F"+
        "ixedSupportDirichletBC"}}, functional_data_class, 
        mesh_data_class_fenics)

        # Variational form of the exterior work using an uniform referential 
        # traction

        external_work, neumann_loads = variational_tools.traction_work({
        "top": {"load case": "UniformReferentialTraction", "amplitude_"+
        "tractionX": 0.0, "amplitude_tractionY": 0.0, "amplitude_tract"+
        "ionZ": neumann_load, "parametric_load_curve": "square_root", 
        "t": 0.0, "t_final": 1.0}}, "Displacement", 
        functional_data_class, mesh_data_class_fenics, [])

        constitutive_model_multimech = NeoHookeanMultiMech(
        material_properties)

        constitutive_model_multimech.check_model(None)

        internal_work = variational_tools.hyperelastic_internalWorkFirstPiola(
        "Displacement", functional_data_class, 
        constitutive_model_multimech, mesh_data_class_fenics)

        # Update the load class

        neumann_loads[0].update_load(current_time)

        dirichlet_loads[0].update_load(current_time)

        # Updates the Dirichlet boundary conditions

        for bc in bcs:

            bc.apply(functional_data_class.solution_fields["Displaceme"+
            "nt"].vector())

        ################################################################
        #                          Comparison                          #
        ################################################################

        # Creates a class that finds the DOFs closest to a point

        dofs_finder = dofs_per_node_finder_class(functional_data_class)

        # Constructs a list of DOFs per node in FEniCS enumeration using
        # the node numbering of GMSH

        dofs_fenics_from_gmsh_nodes = np.array([dofs_finder(
        *node_coordinates) for node_coordinates in (
        mesh_data_class.nodes_coordinates)])

        # Assembles the residual vector and stores as a list

        assembled_residual = assemble(internal_work-external_work)

        residual_vector_fenics = np.zeros(residual_vector.shape[1])

        for dof_number, dof_value in enumerate(assembled_residual):

            # Gets the DOF enumeration in gmsh

            dof_number_gmsh = np.argwhere(dofs_fenics_from_gmsh_nodes==
            dof_number)[0]

            # Gets the DOF number in the FENN numbering system using the
            # node number

            dof_number_gmsh = mesh_data_class.dofs_node_dict["Displace"+
            "ment"][dof_number_gmsh[0]][dof_number_gmsh[1]]

            residual_vector_fenics[dof_number_gmsh] = dof_value

        # Plots both to compare

        residual_vector = residual_vector.numpy()

        n_nonzero_components = 0

        for i in range(residual_vector.shape[1]):

            if abs(residual_vector[0,i])>1E-5:

                n_nonzero_components += 1

            print("FENN: residual_vector["+str(i)+"]="+str(
            residual_vector[0,i])+";            FEniCS: residual_vecto"+
            "r["+str(i)+"]="+str(residual_vector_fenics[i]))

        print("\nThe FEniCS residual vector has a length of "+str(len(
        assembled_residual)))

        print("The FENN residual vector has a shape of "+str(
        residual_vector.shape)+"\nwhere the first value tells the numb"+
        "er tells the number of realizations of the BVP and the second"+
        " one tells the number of DOFs")

        print("\nThere are "+str(n_nonzero_components)+" non-zero comp"+
        "onents in the residual vector calculated by FENN")#"""

# Runs all tests

if __name__=="__main__":

    unittest.main()