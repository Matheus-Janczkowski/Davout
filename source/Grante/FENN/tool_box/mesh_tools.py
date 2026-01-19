# Routine to store methods to read meshes generated in GMSH 

from ...PythonicUtilities.path_tools import verify_path, verify_file_existence, get_parent_path_of_file, take_outFileNameTermination

########################################################################
#                             Mesh reading                             #
########################################################################

# Defines a function to read a mesh .msh

def read_msh_mesh(file_name, parent_directory=None):

    # If the parent directory is None, get the parent path of the file 
    # where this function has been called

    if parent_directory is None:

        parent_directory = get_parent_path_of_file(
        function_calls_to_retrocede=2)

    # Takes out the file termination, and adds the correct .msh

    file_name = take_outFileNameTermination(file_name)+".msh"

    # Verifies the path

    file_name = verify_path(parent_directory, file_name)

    # Verifies if this file exists

    verify_file_existence(file_name)

    return file_name