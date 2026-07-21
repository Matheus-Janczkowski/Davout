# Routine to store some tests to evaluate SVD-based neural networks

import numpy as np

from ....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from ....Davout.PythonicUtilities.testing_tools import run_class_of_tests

from ....Davout.StochasticUtilities.tool_box import delaunay_tools

# Defines a function to test tools for Delaunay partition of the domain 
# and subsequent DOF catching

class TestDelaunayPartition:

    def __init__(self):

        # Defines some common information

        self.parent_path = get_parent_path_of_file()

        self.intervertebral_disc_mesh = (self.parent_path+"//intervert"+
        "ebral_disc_mesh.msh")

        self.L_x = 1.0

        self.L_y = 1.0

        self.L_z = 1.0

        self.cube_mesh = {"length x": self.L_x, "length y": self.L_y, 
        "length z": self.L_z, "number of divisions in x": 2, "number o"+
        "f divisions in y": 2, "number of divisions in z": 2, "verbose": 
        False, "mesh file name": "box_mesh", "mesh file directory": 
        self.parent_path}

        self.float_type = "float32"

        self.int_type = "int32"

    # Defines a function to test partitioning a brick mesh

    def test_delaunay_on_brick_mesh(self):

        print("\n#####################################################"+
        "###################\n#   Tests partitioning a brick-like mesh"+
        " using Delaunay's tetrahedra   #\n###########################"+
        "#############################################\n")

        # Sets the coordinates of the target points of the cube mesh

        target_points_coordinates = [[0.0, 0.0, 0.0], [self.L_x, 0.0, 
        0.0], [self.L_x, self.L_y, 0.0], [0.0, self.L_y, 0.0], [0.0, 
        0.0, self.L_z], [self.L_x, 0.0, self.L_z], [self.L_x, self.L_y, 
        self.L_z], [0.0, self.L_y, self.L_z]]

        # Gets the list of DOFs per tetrahedron

        dofs_per_tetrahedron = delaunay_tools.triangulate_domain_and_separate_dofs(
        self.cube_mesh, target_points_coordinates, "Displacement", {"D"+
        "isplacement": {"field type": "vector", "interpolation function":
        "CG", "polynomial degree": 2}}, verbose=True, int_dtype=
        self.int_type, float_type=self.float_type)

    # Defines a function to test the methodology for convex boundary 
    # section on the intervertebral disc mesh

    def test_delaunay_on_intervertebral_disc(self):

        flag_whole_disc_mesh = False

        list_of_markers = []

        # Defines the position of the target nodes

        # lower nodes

        if flag_whole_disc_mesh:

            list_of_markers.extend([
            {"position":[0.0, 0.002706666666657956, 0.005014005867379833], "color":"white"},
            {"position":[-0.009937066770880507, 0.002154044527756695, 0.005014005867379833], "color":"white"},
            {"position":[-0.02005598172531916, 0.00219532818557083, 0.005014005867379835], "color":"white"},
            {"position":[0.00986881339801569, 0.002157886334920429, 0.005014005867379833], "color":"white"},
            {"position":[0.0199910237665205, 0.002169405772122497, 0.005014005867379833], "color":"white"},
            {"position":[-4.534213541115486e-05, 0.009476875077660365, 0.005014005867379833], "color":"white"},
            {"position":[-4.318089313660243e-05, 0.01472370773312667, 0.005014005867379833], "color":"white"},
            {"position":[-4.549057055915243e-05, -0.005464843272612792, 0.005014005867379833], "color":"white"},
            {"position":[-4.331006004236785e-05, -0.01300714018473771, 0.005014005867379833], "color":"white"},
            {"position":[-0.008156171328816792, 0.01063828671865583, 0.005014005867379833], "color":"white"},
            {"position":[-0.01200284109465504, 0.01455883488091643, 0.005014005867379833], "color":"white"},
            {"position":[0.006604393387532548, -0.004133816343724023, 0.005014005867379833], "color":"white"},
            {"position":[0.01193196647380104, -0.009375813586500564, 0.005014005867379833], "color":"white"},
            {"position":[-0.00664049659193075, -0.004079153521379328, 0.005014005867379833], "color":"white"},
            {"position":[-0.01193276957183655, -0.009288741429231317, 0.005014005867379833], "color":"white"},
            {"position":[0.008056372734255662, 0.01066651450660219, 0.005014005867379833], "color":"white"},
            {"position":[0.01196729402242558, 0.01461209246369756, 0.005014005867379833], "color":"white"}])#16
        
        # Outer ring nodes lower

        list_of_markers.extend([{"position":[0.01435, 0.0170, 0.005014005867379833], "color":"yellow 5"},
            {"position":[-0.0143147, -0.0116766, 0.005014005867379833], "color":"yellow 5"},
            {"position":[0.0143127, -0.011763717, 0.005014005867379833], "color":"yellow 5"},
            {"position":[-0.014385546, 0.016946737, 0.005014005867379833], "color":"yellow 5"}])
        
        if flag_whole_disc_mesh:

            list_of_markers.extend([{"position":[-0.08e-3, -0.016395047, 0.005014005867379833], "color":"yellow 5"},
            {"position":[-0.08e-3, 0.017, 0.005014005867379833], "color":"yellow 5"},
            {"position":[0.02442, 0.0022, 0.005014005867379833], "color":"yellow 5"},
            {"position":[-0.02442, 0.0022, 0.005014005867379833], "color":"yellow 5"}])#24
        
        # Upper points

        if flag_whole_disc_mesh:

            list_of_markers.extend([
            {"position":[0.0, 0.002706666666657956, 0.008985994132581825], "color":"red 5"},
            {"position":[-0.009937066770880512, 0.002154044527756696, 0.008985994132581825], "color":"red 5"},
            {"position":[-0.02005598172531915, 0.002195328185570831, 0.008985994132581832], "color":"red 5"},
            {"position":[0.009868813398015692, 0.002157886334920429, 0.008985994132581825], "color":"red 5"},
            {"position":[0.0199910237665205, 0.002169405772122496, 0.008985994132581825], "color":"red 5"},
            {"position":[-4.534213541115486e-05, 0.009476875077660367, 0.008985994132581825], "color":"red 5"},
            {"position":[-4.318089313660243e-05, 0.01472370773312667, 0.008985994132581825], "color":"red 5"},
            {"position":[-4.549057055915243e-05, -0.005464843272612793, 0.008985994132581825], "color":"red 5"},
            {"position":[-4.331006004236786e-05, -0.01300714018473771, 0.008985994132581825], "color":"red 5"},
            {"position":[-0.008156171328816792, 0.01063828671865583, 0.008985994132581825], "color":"red 5"},
            {"position":[-0.01200284109465504, 0.01455883488091643, 0.008985994132581825], "color":"red 5"},
            {"position":[0.006604393387532548, -0.004133816343724024, 0.008985994132581825], "color":"red 5"},
            {"position":[0.01193196647380104, -0.009375813586500565, 0.008985994132581825], "color":"red 5"},
            {"position":[-0.00664049659193075, -0.004079153521379328, 0.008985994132581825], "color":"red 5"},
            {"position":[-0.01193276957183655, -0.009288741429231317, 0.008985994132581825], "color":"red 5"},
            {"position":[0.008056372734255662, 0.01066651450660219, 0.008985994132581825], "color":"red 5"},
            {"position":[0.01196729402242558, 0.01461209246369756, 0.008985994132581825], "color":"red 5"}])#41
        
        # Outer ring nodes upper

        list_of_markers.extend([
            {"position":[0.01435, 0.0170, 0.008985994132581825], "color":"yellow 5"},
            {"position":[-0.0143147, -0.0116766, 0.008985994132581825], "color":"yellow 5"},
            {"position":[0.0143127, -0.011763717, 0.008985994132581825], "color":"yellow 5"},
            {"position":[-0.014385546, 0.016946737, 0.008985994132581825], "color":"yellow 5"}])
        
        if flag_whole_disc_mesh:

            list_of_markers.extend([
            {"position":[-0.08e-3, -0.016395047, 0.008985994132581825], "color":"yellow 5"},
            {"position":[-0.08e-3, 0.017, 0.008985994132581825], "color":"yellow 5"},
            {"position":[0.02442, 0.0022, 0.008985994132581825], "color":"yellow 5"},
            {"position":[-0.02442, 0.0022, 0.008985994132581825], "color":"yellow 5"}])

        # Extracts only the values of the nodes position

        target_points_coordinates = np.array([dictionary["position"
        ] for dictionary in list_of_markers])

        # Gets the list of DOFs per tetrahedron

        dofs_per_tetrahedron = delaunay_tools.triangulate_domain_and_separate_dofs(
        self.intervertebral_disc_mesh, target_points_coordinates, "Dis"+
        "placement", {"Displacement": {"field type": "vector", "interp"+
        "olation function": "CG", "polynomial degree": 2}}, verbose=True, 
        int_dtype=self.int_type, float_type=self.float_type)

# Runs all tests

if __name__=="__main__":

    # Instantiates the class with the methods to be tested

    class_of_tests = TestDelaunayPartition()

    # Creates a list of methods (using their names) that are not to be
    # tested

    reserved_methods = []

    # Calls the function to run all the necessary tests

    run_class_of_tests(class_of_tests, reserved_methods=reserved_methods,
    sort_methods_alphabetically=False)