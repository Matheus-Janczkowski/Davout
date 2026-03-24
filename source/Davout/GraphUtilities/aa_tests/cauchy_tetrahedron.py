# Figure for the Cauchy tetrahedron

from ...GraphUtilities.collage_tools import create_box_collage

from ...PythonicUtilities.path_tools import get_parent_path_of_file

create_box_collage("cauchy_tetrahedron.pdf", input_path=
get_parent_path_of_file(),

boxes_list=[{"contour color": "black", "fill color": "grey 2", "contour"+
" thickness": 0.2, "position": [64.0, 101.0], "width": 61.0, "height": 27.0,
"contour style": "solid", "origin point": "bottom-left"},
{"contour color": "black", "fill color": "grey 1", "contour"+
" thickness": 0.2, "position": [65.0, 102.0], "width": 29.0, "height": 25.0,
"contour style": "solid", "origin point": "bottom-left"},
{"contour color": "black", "fill color": "grey 1", "contour"+
" thickness": 0.2, "position": [95.0, 102.0], "width": 29.0, "height": 25.0,
"contour style": "solid", "origin point": "bottom-left"}], 

arrows_and_lines_list=[],

#vanishing_points_list=[
#{"coordinates": [105.0, -50.0], "rays central direction": [0.0, 1.0],
#"angle amplitude": 10.0, "number of rays": 15, "color": "black", 
#"thickness": 0.4}, 
#{"coordinates": [320.0, 140.0], "rays central direction": [-1.0, -0.1],
#"angle amplitude": 10.0, "number of rays": 15, "color": "black", 
#"thickness": 0.4}, 
#{"coordinates": [-110.0, 140.0], "rays central direction": [1.0, -0.1],
#"angle amplitude": 10.0, "number of rays": 15, "color": "black", 
#"thickness": 0.4}],

verbose=True, no_padding=True, add_overlaying_grid=False, dpi=600,
grid_annotation_length=10, save_lists_to_txt=True, 
interactive_preview=True, size_template="A4",
export_selection={
"origin point": "bottom-left", "position": [64.0, 101.0], "width": 61.0, 
"height": 27.0})