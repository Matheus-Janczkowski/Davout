# Routine to store some tests to evaluate LaTeXUtilities parser

from ....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from ....Davout.PythonicUtilities.testing_tools import run_class_of_tests

from ....Davout.LaTeXUtilities.latex_parser import substitute_utilities_by_raw_latex

# Defines a function to test the parser

class TestLaTeXParser:

    def __init__(self):

        # Defines the parent path for the data

        self.parent_path = get_parent_path_of_file()

        self.verbose = True

    # Defines a function to test parsing a zip file with LaTeXUtilities
    # already in it

    def test_parser_with_latexutilities(self):

        print("\n#####################################################"+
        "###################\n#           Tests parsing zip file with "+
        "LaTeXUtilities in it           #\n###########################"+
        "#############################################\n")

        substitute_utilities_by_raw_latex("RP02_neural_network_archite"+
        "cture_with_controlled_null_space.zip", parent_path=
        self.parent_path, verbose=self.verbose, files_to_be_compiled=
        "main.tex")

    # Defines a function to test parsing a zip file with LaTeXUtilities
    # given separately

    def test_parser_with_separate_latexutilities(self):

        print("\n#####################################################"+
        "###################\n#      Tests parsing zip file with LaTeX"+
        "Utilities given separately     #\n###########################"+
        "#############################################\n")

        substitute_utilities_by_raw_latex("RP02_neural_network_archite"+
        "cture_with_controlled_null_space.zip", parent_path=
        self.parent_path, verbose=self.verbose, 
        latex_utilities_full_path=self.parent_path+"//LaTeXUtilities.s"+
        "ty")

# Runs all tests

if __name__=="__main__":

    # Instantiates the class with the methods to be tested

    class_of_tests = TestLaTeXParser()

    # Creates a list of methods (using their names) that are not to be
    # tested

    reserved_methods = ["test_parser_with_separate_latexutilities"]

    # Calls the function to run all the necessary tests

    run_class_of_tests(class_of_tests, reserved_methods=reserved_methods,
    sort_methods_alphabetically=False)