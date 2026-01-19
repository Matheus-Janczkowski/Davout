# Routine to store tests for the finite elements

# Routine to store some tests to evaluate convex input neural networks

import unittest

import numpy as np

from ...finite_elements.tetrahedrons import Tetrahedron

from ...tool_box import mesh_tools

from ....MultiMech.tool_box.mesh_handling_tools import create_box_mesh

from ....PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a function to test the ANN tools methods

class TestANNTools(unittest.TestCase):

    def setUp(self):

        # Defines a list of nodes coordinates for two tetrahedral ele-
        # ments opposing each other

        nodes_coordinates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 
        0.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [
        0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0
        ], [2.0/3.0, 2.0/3.0, 2.0/3.0], [1.0/3.0, 1.0/3.0, 5.0/6.0], [
        1.0/3.0, 5.0/6.0, 1.0/3.0], [5.0/6.0, 1.0/3.0, 1.0/3.0]]

        # Gets a list of nodes coordinates per element

        self.nodes_coordinates_elements = []

        # Appends the nodes of the first element

        self.nodes_coordinates_elements.append([nodes_coordinates[0], 
        nodes_coordinates[1], nodes_coordinates[2], nodes_coordinates[3],
        nodes_coordinates[4], nodes_coordinates[5], nodes_coordinates[6],
        nodes_coordinates[7], nodes_coordinates[8], nodes_coordinates[9]])

        # Appends the nodes of the second element

        self.nodes_coordinates_elements.append([nodes_coordinates[1], 
        nodes_coordinates[0], nodes_coordinates[2], nodes_coordinates[10],
        nodes_coordinates[4], nodes_coordinates[8], nodes_coordinates[11],
        nodes_coordinates[12], nodes_coordinates[5], nodes_coordinates[13]])

    # Defines a function to test reading a mesh

    def test_mesh_reader(self):

        print("\n#####################################################"+
        "###################\n#                       Tests the msh me"+
        "sh reader                      #\n###########################"+
        "#############################################\n")

        file_name = "box"

        file_directory = get_parent_path_of_file()

        # Creates a box mesh 

        length_x = 0.2
        
        length_y = 0.3
        
        length_z = 1.0
        
        n_divisions_x = 2 

        n_divisions_y = 2 
        
        n_divisions_z = 5

        create_box_mesh(length_x, length_y, length_z, n_divisions_x, 
        n_divisions_y, n_divisions_z, file_name=file_name, verbose=False, 
        convert_to_xdmf=False, file_directory=file_directory)

        # Reads this mesh

        mesh_data_class = mesh_tools.read_msh_mesh(file_name)

    # Defines a function to test the instantiation of tetrahedron class

    def test_quadratic_tetrahedron(self):

        print("\n#####################################################"+
        "###################\n#              Tests the second order te"+
        "trahedral element              #\n###########################"+
        "#############################################\n")

        tetradron_mesh = Tetrahedron(self.nodes_coordinates_elements)

        print("The determinant of the jacobian evaluated at all quadra"+
        "ture points is:\n"+str(tetradron_mesh.det_J)+"\n")

        print("The derivatives of the shape functions at all quadratur"+
        "e point are:\n"+str(tetradron_mesh.shape_functions_derivatives))

# Runs all tests

if __name__=="__main__":

    unittest.main()