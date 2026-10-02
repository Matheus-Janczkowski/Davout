# Routine to store methods to test ANN models that learn discrete apro-
# ximations given by the FEM

import tensorflow as tf

from ...PythonicUtilities.path_tools import join_path_and_verify_existence

# Defines a class to evaluate and compare the response of a FEM surroga-
# te ANN model against the original FEM data

class FEMSurrogateEvaluator:

    def __init__(self, input_file_name, output_file_name, parent_path=
    None):

        # Checks whether the file paths exist and returns the file paths
        # updated with the parent path

        input_file_name, output_file_name = join_path_and_verify_existence(
        [input_file_name, output_file_name], parent_path=parent_path,
        path_bits_to_be_excluded=3)