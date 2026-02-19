# Routine to test the implementation of paraview tools

import traceback

from ...GraphUtilities import paraview_tools

from ...PythonicUtilities.programming_tools import get_attribute

from ...PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a class to store all tests

class TestParaView():

    def __init__(self):

        # Sets the file name and the input directory

        self.file_name = "displacement.xdmf"

        self.input_path = ("/home/matheus-janczkowski/Github/Davout/so"+
        "urce/Davout/MultiMech/aa_tests_and_examples/hyperelasticity")

        # Sets the field name

        self.field_name = "Displacement"

        # Sets the output path

        self.output_path = get_parent_path_of_file()

    def test_warp_by_vector(self):

        print("\n#####################################################"+
        "###################\n#                            Warp by vec"+
        "tor                            #\n###########################"+
        "#############################################\n")
        
        # Takes a snapshot of a simulation saved at self.file name whose
        # field is called as 'Displacement' inside FEniCS. The edges of
        # the finite elements will be shown
        
        paraview_tools.frozen_snapshots(self.file_name, self.field_name, 
        input_path=self.input_path, time=1.0, representation_type=
        "Surface With Edges", axes_color=[0.0, 0.0, 0.0], 
        legend_bar_font="Times", zoom_factor=1.0, component_to_plot="2", 
        resolution_ratio=10, warp_by_vector=True, background_color=
        "WhiteBackground", display_reference_configuration=False, 
        transparent_background=True, legend_bar_font_color=[0.0, 0.0, 0.0],
        set_camera_interactively=False, output_imageFileName="warp.png",
        output_path=self.output_path,
        camera_position=[-2.2202601190413063, 2.7845260002185928, 2.412838032797255],
        camera_focal_point=[0.500121318497246, 0.8894379799910188, 0.03488656665544461],
        camera_up_direction=[0.19171813331267484, 0.8626133794219494, -0.46812638784984917],
        camera_parallel_scale=0.872714431199938,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.6048594377510041, 0.09555555555555562],
        legend_bar_length=0.7344444444444449,
        size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

    def test_clip(self):

        print("\n#####################################################"+
        "###################\n#                             Clip by pl"+
        "ane                            #\n###########################"+
        "#############################################\n")
        
        # Takes a snapshot of a simulation saved at self.file name whose
        # field is called as 'Displacement' inside FEniCS. The edges of
        # the finite elements will be shown
        
        paraview_tools.frozen_snapshots(self.file_name, self.field_name, 
        input_path=self.input_path, time=1.0, representation_type=
        "Surface With Edges", axes_color=[0.0, 0.0, 0.0], 
        legend_bar_font="Times", zoom_factor=1.0, component_to_plot="2", 
        resolution_ratio=10, clip=True, clip_plane_origin=[0.15, 0.5, 
        0.1], clip_plane_normal_vector=[1.0, 1.0, 1.0], background_color=
        "WhiteBackground", display_reference_configuration=False, 
        transparent_background=True, legend_bar_font_color=
        [0.0, 0.0, 0.0], set_camera_interactively=False, 
        output_imageFileName="clip.png", output_path=self.output_path,
        camera_position=[-2.2202601190413063, 2.7845260002185928, 2.412838032797255],
        camera_focal_point=[0.500121318497246, 0.8894379799910188, 0.03488656665544461],
        camera_up_direction=[0.19171813331267484, 0.8626133794219494, -0.46812638784984917],
        camera_parallel_scale=0.872714431199938,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.6048594377510041, 0.09555555555555562],
        legend_bar_length=0.7344444444444449,
        size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

    def test_warp_and_clip(self):

        print("\n#####################################################"+
        "###################\n#                   Warp by vector and c"+
        "lip by plane                   #\n###########################"+
        "#############################################\n")
        
        # Takes a snapshot of a simulation saved at self.file name whose
        # field is called as 'Displacement' inside FEniCS. The edges of
        # the finite elements will be shown
        
        paraview_tools.frozen_snapshots(self.file_name, self.field_name, 
        input_path=self.input_path, time=1.0, representation_type=
        "Surface With Edges", axes_color=[0.0, 0.0, 0.0], 
        legend_bar_font="Times", zoom_factor=1.0, component_to_plot="2", 
        warp_by_vector=True, resolution_ratio=10, clip=True, 
        clip_plane_origin=[0.15, 0.5, 0.1], clip_plane_normal_vector=
        [1.0, 1.0, 1.0], background_color="WhiteBackground", 
        display_reference_configuration=False, 
        transparent_background=True, legend_bar_font_color=
        [0.0, 0.0, 0.0], set_camera_interactively=False, 
        output_imageFileName="warp_and_clip.png", output_path=self.output_path,
        camera_position=[-2.2202601190413063, 2.7845260002185928, 2.412838032797255],
        camera_focal_point=[0.500121318497246, 0.8894379799910188, 0.03488656665544461],
        camera_up_direction=[0.19171813331267484, 0.8626133794219494, -0.46812638784984917],
        camera_parallel_scale=0.872714431199938,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.6048594377510041, 0.09555555555555562],
        legend_bar_length=0.7344444444444449,
        size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

    def test_glyph(self):

        print("\n#####################################################"+
        "###################\n#                                 Glyph "+
        "                               #\n###########################"+
        "#############################################\n")
        
        # Takes a snapshot of a simulation saved at self.file name whose
        # field is called as 'Displacement' inside FEniCS. The edges of
        # the finite elements will be shown
        
        paraview_tools.frozen_snapshots(self.file_name, self.field_name, 
        input_path=self.input_path, time=1.0, representation_type=
        "Surface With Edges", axes_color=[0.0, 0.0, 0.0], 
        legend_bar_font="Times", zoom_factor=1.0, component_to_plot="2", 
        resolution_ratio=10, glyph=True, glyph_scale=0.3, 
        background_color="WhiteBackground", 
        display_reference_configuration=False, 
        transparent_background=True, legend_bar_font_color=
        [0.0, 0.0, 0.0], set_camera_interactively=False, 
        output_imageFileName="glyph.png", output_path=self.output_path,
        camera_position=[-2.2202601190413063, 2.7845260002185928, 2.412838032797255],
        camera_focal_point=[0.500121318497246, 0.8894379799910188, 0.03488656665544461],
        camera_up_direction=[0.19171813331267484, 0.8626133794219494, -0.46812638784984917],
        camera_parallel_scale=0.872714431199938,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.6048594377510041, 0.09555555555555562],
        legend_bar_length=0.7344444444444449,
        size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

    def test_warp_and_glyph(self):

        print("\n#####################################################"+
        "###################\n#                            Warp and gl"+
        "yph                            #\n###########################"+
        "#############################################\n")
        
        # Takes a snapshot of a simulation saved at self.file name whose
        # field is called as 'Displacement' inside FEniCS. The edges of
        # the finite elements will be shown
        
        paraview_tools.frozen_snapshots(self.file_name, self.field_name, 
        input_path=self.input_path, time=1.0, representation_type=
        "Surface With Edges", axes_color=[0.0, 0.0, 0.0], 
        legend_bar_font="Times", zoom_factor=1.0, component_to_plot="2", 
        warp_by_vector=True, resolution_ratio=10, glyph=True, 
        glyph_scale=0.3, background_color="WhiteBackground", 
        display_reference_configuration=False, 
        transparent_background=True, legend_bar_font_color=
        [0.0, 0.0, 0.0], set_camera_interactively=False, 
        output_imageFileName="warp_and_glyph.png", output_path=self.output_path,
        camera_position=[-2.2202601190413063, 2.7845260002185928, 2.412838032797255],
        camera_focal_point=[0.500121318497246, 0.8894379799910188, 0.03488656665544461],
        camera_up_direction=[0.19171813331267484, 0.8626133794219494, -0.46812638784984917],
        camera_parallel_scale=0.872714431199938,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.6048594377510041, 0.09555555555555562],
        legend_bar_length=0.7344444444444449,
        size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

    def test_select_component(self):

        print("\n#####################################################"+
        "###################\n#                     Select component f"+
        "or plotting                    #\n###########################"+
        "#############################################\n")
        
        # Takes a snapshot of a simulation saved at self.file name whose
        # field is called as 'Displacement' inside FEniCS. The edges of
        # the finite elements will be shown
        
        paraview_tools.frozen_snapshots(self.file_name, self.field_name, 
        input_path=self.input_path, time=1.0, representation_type=
        "Surface With Edges", axes_color=[0.0, 0.0, 0.0], 
        legend_bar_font="Times", zoom_factor=1.0, component_to_plot="Magnitude", 
        warp_by_vector=True, resolution_ratio=10, background_color=
        "WhiteBackground", display_reference_configuration=False, 
        transparent_background=True, legend_bar_font_color=
        [0.0, 0.0, 0.0], set_camera_interactively=False, 
        output_imageFileName="magnitude_plot.png", output_path=self.output_path,
        camera_position=[-2.2202601190413063, 2.7845260002185928, 2.412838032797255],
        camera_focal_point=[0.500121318497246, 0.8894379799910188, 0.03488656665544461],
        camera_up_direction=[0.19171813331267484, 0.8626133794219494, -0.46812638784984917],
        camera_parallel_scale=0.872714431199938,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.6048594377510041, 0.09555555555555562],
        legend_bar_length=0.7344444444444449,
        size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

    def test_select_color_bar_bounds(self):

        print("\n#####################################################"+
        "###################\n#                    Select bounds for t"+
        "he color bar                   #\n###########################"+
        "#############################################\n")
        
        # Takes a snapshot of a simulation saved at self.file name whose
        # field is called as 'Displacement' inside FEniCS. The edges of
        # the finite elements will be shown
        
        paraview_tools.frozen_snapshots(self.file_name, self.field_name, 
        input_path=self.input_path, time=1.0, representation_type=
        "Surface With Edges", axes_color=[0.0, 0.0, 0.0], 
        legend_bar_font="Times", zoom_factor=1.0, component_to_plot="Magnitude", 
        warp_by_vector=True, resolution_ratio=10, background_color=
        "WhiteBackground", display_reference_configuration=False, 
        transparent_background=True, legend_bar_font_color=
        [0.0, 0.0, 0.0], set_camera_interactively=False, 
        color_bar_min_value=0.1, color_bar_max_value=0.6,
        output_imageFileName="magnitude_with_color_bounds.png", output_path=self.output_path,
        camera_position=[-2.2202601190413063, 2.7845260002185928, 2.412838032797255],
        camera_focal_point=[0.500121318497246, 0.8894379799910188, 0.03488656665544461],
        camera_up_direction=[0.19171813331267484, 0.8626133794219494, -0.46812638784984917],
        camera_parallel_scale=0.872714431199938,
        camera_rotation=[0.0, 0.0, 0.0],
        legend_bar_position=[0.6048594377510041, 0.09555555555555562],
        legend_bar_length=0.7344444444444449,
        size_in_pixels={'aspect ratio': 1.0, 'pixels in width': 400})

# Runs all tests

if __name__=="__main__":

    test_class = TestParaView()

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

        except Exception as e:

            print("Method '"+str(name)+"' failed:\n"+str(e)+"\n")

            traceback.print_exc()

            failure_counter += 1

    print("\n#########################################################"+
    "###############\n#                             Execution log     "+
    "                       #\n#######################################"+
    "#################################\n")

    print(str(success_counter)+" methods were successfully executed\n")

    print(str(failure_counter)+" methods failed to be executed")