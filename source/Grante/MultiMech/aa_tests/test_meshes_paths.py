# Routine to get the paths to the test meshes

import os

from ..tool_box import file_handling_tools

# Defines a function to get the path to the desired mesh

def get_mesh_path(mesh_file):

    # Gets the path to the current file

    test_meshes_path = os.path.join(
    file_handling_tools.get_parent_path_of_file(), "test_meshes//"+str(
    mesh_file))

    return test_meshes_path