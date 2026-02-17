# Routine to test the implementation of collage tools

from ...GraphUtilities.collage_tools import create_box_collage

from ...PythonicUtilities.programming_tools import get_attribute

from ...PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a class to store all tests

class TestCollage():

    def __init__(self):

        # Sets the file name and the input directory

        self.file_name = "displacement.xdmf"

        self.input_path = ("/home/matheus-janczkowski/Github/Davout/so"+
        "urce/Davout/MultiMech/aa_tests_and_examples/hyperelasticity")

        # Sets the output path

        self.output_path = get_parent_path_of_file()

    def test_box_collage(self):

        print("\n#####################################################"+
        "###################\n#                              Box colla"+
        "ge                             #\n###########################"+
        "#############################################\n")

        create_box_collage("collage.png", input_path=get_parent_path_of_file(),
        input_image_list=[{"file name": "warp.png", "position": [50.0, 200.0], 
        "size": 50.0, "trim transparent background": True, "origin point":
        "bottom-left"}, 
        {"file name": "warp_and_clip.png", "position": [107.0, 200.0], "size": 50.0, 
        "trim transparent background": True, "origin point": "bottom-left"}], 

        input_text_list=[{"text": "Lateral", "position": [100.0, 195.0], "font size": 
        4, "origin point": "bottom-right"}, 
        {"text": "Upper", "position": [157.0, 195.0], "font size": 4,
        "origin point": "bottom-right"},
        {"text": "$\MaterialDivOf{\\boldsymbol{P}}=\\boldsymbol{0}$", 
        "position": [50.0, 195.0], "font size": 4, "origin point": "bottom-left"}],

        boxes_list=[{"contour color": "black", "fill color": "red 2", "contour"+
        " thickness": 0.2, "position": [46.5, 193.0], "width": 114.0, "height": 58.0,
        "contour style": "solid", "origin point": "bottom-left"},
        {"contour color": "black", "fill color": "red 1", "contour"+
        " thickness": 0.2, "position": [47.5, 194.0], "width": 55.0, "height": 56.0,
        "contour style": "solid", "origin point": "bottom-left"},
        {"contour color": "black", "fill color": "red 1", "contour"+
        " thickness": 0.2, "position": [104.5, 194.0], "width": 55.0, "height": 56.0,
        "contour style": "solid", "origin point": "bottom-left"}], 

        arrows_and_lines_list=[{"start point": [99.0, 200.0], "end point": [109.0, 200.0],
        "polygonal points": [[104., 202.0]], "thickness": 0.2, "arrow style": "inkscape angular arrow",
        "line style": "dashed 7x3"},
        {"start point": [99.0, 240.0], "end point": [99.0, 240.0], 
        "spline points": [[102.3, 243.0], [105.6, 243.0], [109.0, 240.0], 
        [105.6, 237.0], [102.3, 237.0]], "closed path": True, "thickness": 0.2,
        "arrow style": "no arrow", "fill path with color": "red 3"}],

        vanishing_points_list=[
        {"coordinates": [105.0, -50.0], "rays central direction": [0.0, 1.0],
        "angle amplitude": 10.0, "number of rays": 15, "color": "black", 
        "thickness": 0.4}, 
        {"coordinates": [320.0, 250.0], "rays central direction": [-1.0, -0.1],
        "angle amplitude": 10.0, "number of rays": 15, "color": "black", 
        "thickness": 0.4}, 
        {"coordinates": [-110.0, 250.0], "rays central direction": [1.0, -0.1],
        "angle amplitude": 10.0, "number of rays": 15, "color": "black", 
        "thickness": 0.4}],

        verbose=True, no_padding=True, add_overlaying_grid=True, dpi=500,
        grid_annotation_length=10, save_lists_to_txt=True)

# Runs all tests

if __name__=="__main__":

    test_class = TestCollage()

    # Gets a dictionary of the methods inside this instance except for
    # the __init__ method

    methods_dictionary = get_attribute(test_class, None, None, 
    dictionary_of_methods=True, delete_init_key=True)

    # Initializes the success and failure counters

    success_counter = 0

    failure_counter = 0

    # Iterates through the methods 

    for name, method in methods_dictionary.items():

        print("\nRuns method '"+str(name)+"'\n")

        try:

            method()

            success_counter += 1

        except:

            failure_counter += 1

    print("\n#########################################################"+
    "###############\n#                             Execution log     "+
    "                       #\n#######################################"+
    "#################################\n")

    print(str(success_counter)+" methods were successfully executed\n")

    print(str(failure_counter)+" methods failed to be executed")