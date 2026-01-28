# Routine to store tests for the finite elements

# Routine to store some tests to evaluate convex input neural networks

import unittest

import tensorflow as tf

from ..physics.compressible_cauchy_hyperelasticy import CompressibleHyperelasticity

from ..constitutive_models.hyperelastic_isotropic_models import NeoHookean

from ..tool_box import mesh_tools

from ...MultiMech.tool_box.mesh_handling_tools import create_box_mesh

from ...PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        pass

    # Defines a function to test reading a mesh

    def test_hyperelastic_residual_vector(self):

        print("\n#####################################################"+
        "###################\n#              Tests the evaluation of t"+
        "he residual vector             #\n###########################"+
        "#############################################\n")

        file_name = "box"

        file_directory = get_parent_path_of_file()

        # Creates a box mesh 

        length_x = 0.2
        
        length_y = 0.3
        
        length_z = 1.0
        
        n_divisions_x = 2 

        n_divisions_y = 2
        
        n_divisions_z = 2

        quadrature_degree = 2

        create_box_mesh(length_x, length_y, length_z, n_divisions_x, 
        n_divisions_y, n_divisions_z, file_name=file_name, verbose=False, 
        convert_to_xdmf=False, file_directory=file_directory, 
        mesh_polinomial_order=2)

        # Defines a dictionary of finite element per field

        elements_per_field = {"Displacement": {"number of DOFs per nod"+
        "e": 3, "required element type": "tetrahedron of 10 nodes"}}

        # Reads this mesh

        mesh_data_class = mesh_tools.read_msh_mesh(file_name, 
        quadrature_degree, elements_per_field, verbose=True)

        # Initializes the vector of parameters as the null vector

        vector_of_parameters = tf.Variable(tf.zeros([
        mesh_data_class.global_number_dofs], dtype=mesh_data_class.dtype))

        # Sets the dictionary of constitutive models

        constitutive_models = {"volume 1": NeoHookean({"E": 1E6, "nu": 
        0.4}, mesh_data_class)}

        # Instantiates the class to evaluate the residual vector

        residual_class = CompressibleHyperelasticity(mesh_data_class,
        vector_of_parameters, constitutive_models)

        # Evaluates the residual

        residual_vector = residual_class.evaluate_residual_vector()

        print("\nThe residual vector is:\n"+str(residual_vector)+"\n")

# Runs all tests

if __name__=="__main__":

    unittest.main()