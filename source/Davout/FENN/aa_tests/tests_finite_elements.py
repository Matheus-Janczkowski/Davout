# Routine to store tests for the finite elements

# Routine to store some tests to evaluate convex input neural networks

import unittest

from ..finite_elements.volume_elements.tetrahedrons import Tetrahedron

from ..finite_elements.surface_elements.triangles import Triangle

from ..tool_box import mesh_tools

from ...MultiMech.tool_box.mesh_handling_tools import create_box_mesh

from ...PythonicUtilities.path_tools import get_parent_path_of_file

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

        self.dofs_per_elements = [[[0,1,2], [3,4,5], [6,7,8], [9,10,11], 
        [12,13,14], [15,16,17], [18,19,20], [21,22,23], [24,25,26], [27,
        28,29]], [[3,4,5], [0,1,2], [6,7,8], [30,31,32], [12,13,14], [24,
        25,26], [33,34,35], [36,37,38], [15,16,17], [39,40,41]]]

        # Defines a list of nodes coordinates for two triangle elements

        nodes_coordinates = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 
        0.5], [0.0, 0.5], [0.5, 0.0], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]

        # Gets a list of nodes coordinates per element

        self.nodes_coordinates_2Delements = []

        # Appends the nodes of the first element

        self.nodes_coordinates_2Delements.append([nodes_coordinates[0], 
        nodes_coordinates[1], nodes_coordinates[2], nodes_coordinates[3],
        nodes_coordinates[4], nodes_coordinates[5]])

        # Appends the nodes of the second element

        self.nodes_coordinates_2Delements.append([nodes_coordinates[1], 
        nodes_coordinates[0], nodes_coordinates[7], nodes_coordinates[3],
        nodes_coordinates[6], nodes_coordinates[8]])

        self.dofs_per_2Delements = [[[0,1,2], [3,4,5], [6,7,8], [9,10,11], 
        [12,13,14], [15,16,17]], [[3,4,5], [0,1,2], [21,22,23], [9,10,11], 
        [18,19,20], [24,25,26]]]

    # Defines a function to test the instantiation of tetrahedron class

    def test_quadratic_tetrahedron(self):

        print("\n#####################################################"+
        "###################\n#              Tests the second order te"+
        "trahedral element              #\n###########################"+
        "#############################################\n")

        tetradron_mesh = Tetrahedron(self.nodes_coordinates_elements,
        self.dofs_per_elements)

        print("The determinant of the jacobian evaluated at all quadra"+
        "ture points mutliplied by the quadrature weights is:\n"+str(
        tetradron_mesh.dx)+"\n")

        #print("The derivatives of the shape functions at all quadratur"+
        #"e point are:\n"+str(tetradron_mesh.shape_functions_derivatives))

    # Defines a function to test the instantiation of triangle class

    def test_quadratic_triangle(self):

        print("\n#####################################################"+
        "###################\n#                Tests the second order "+
        "triangle element               #\n###########################"+
        "#############################################\n")

        triangle_mesh = Triangle(self.nodes_coordinates_2Delements,
        self.dofs_per_2Delements)

        print("The determinant of the jacobian evaluated at all quadra"+
        "ture points mutliplied by the quadrature weights is:\n"+str(
        triangle_mesh.dx)+"\n")

        #print("The derivatives of the shape functions at all quadratur"+
        #"e point are:\n"+str(tetradron_mesh.shape_functions_derivatives))

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

        print("\nThe nodes coordinates are:\n"+str(
        mesh_data_class.nodes_coordinates)+"\n")

        print("The dictionary of domain physical groups is:\n"+str(
        mesh_data_class.domain_physicalGroupsNameToTag)+"\n")

        print("The dictionary of boundary physical groups is:\n"+str(
        mesh_data_class.boundary_physicalGroupsNameToTag)+"\n")

        print("The dictionary of domain elements' connectivities is:\n"+
        str(mesh_data_class.domain_connectivities)+"\n")

        print("The dictionary of boundary elements' connectivities is:"+
        "\n"+str(mesh_data_class.boundary_connectivities)+"\n")

        print("The dictionary of elements per domain physical group is"+
        ":\n"+str(mesh_data_class.domain_elements)+"\n")

        print("The tensor of DOFs per element is:")
        
        for field_name, element_dict in mesh_data_class.domain_elements.items():
            
            for physical_group, element_class in element_dict.items():

                print("\nField name: "+str(field_name)+"; physical gro"+
                "up: "+str(physical_group)+"; DOFs per element:\n"+str(
                element_class.dofs_per_element))

# Runs all tests

if __name__=="__main__":

    unittest.main()