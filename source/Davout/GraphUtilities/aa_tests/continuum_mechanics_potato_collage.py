# Figure for the kinematics of the continuum mechanics' potato

from ...GraphUtilities.collage_tools import create_box_collage

from ...PythonicUtilities.path_tools import get_parent_path_of_file

create_box_collage("continuum_mechanics_kinematics.png", input_path=
get_parent_path_of_file(),

boxes_list=[{"contour color": "black", "fill color": "grey 2", "contour"+
" thickness": 0.2, "position": [52.0, 99.0], "width": 105.0, "height": 52.0,
"contour style": "solid", "origin point": "bottom-left"},
{"contour color": "black", "fill color": "grey 1", "contour"+
" thickness": 0.2, "position": [54, 100.0], "width": 50.0, "height": 50.0,
"contour style": "solid", "origin point": "bottom-left"},
{"contour color": "black", "fill color": "grey 1", "contour"+
" thickness": 0.2, "position": [106.0, 100.0], "width": 50.0, "height": 50.0,
"contour style": "solid", "origin point": "bottom-left"}], 

arrows_and_lines_list=[],

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

verbose=True, no_padding=True, add_overlaying_grid=True, dpi=300,
grid_annotation_length=10, save_lists_to_txt=True, 
interactive_preview=True, size_template="A4", export_selection={
"origin point": "bottom-left", "position": [48.0, 94.0], "wid"+
"th": 115.0, "height": 60.0})