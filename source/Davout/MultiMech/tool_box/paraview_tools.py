# Routine to externally control paraview and automate the extraction of
# simulation output

from ...PythonicUtilities import programming_tools

from ...MultiMech.tool_box import paraview_scripts

########################################################################
#                           Frozen snapshots                           #
########################################################################

# Defines a function to call the frozen_snapshots function,as it must be
# ran externally using the pvpython interpreter of ParaView

def frozen_snapshots(input_fileName, field_name, input_path=None,
output_path=None, camera_position=None, color_map="Cool to Warm", 
output_imageFileName="plot.png", execution_rootPath=None, time=None,
time_step_index=None, camera_focal_point=None, camera_up_direction=None,
representation_type=None, legend_bar_position=None, legend_bar_length=
None, axes_color=None, size_in_pixels=None, get_attributes_render=None,
camera_parallel_scale=None, camera_rotation=None, legend_bar_font=None,
legend_bar_font_file=None, zoom_factor=None, plot_x_axis=None, 
plot_y_axis=None, plot_z_axis=None, no_axes=None, component_to_plot=None,
resolution_ratio=None, transparent_background=None, 
LIBGL_ALWAYS_SOFTWARE=False, extra_execution_arguments=None,
python_interpreter="pvpython", warp_by_vector=None, warp_scale=None, 
glyph=None, glyph_scale=None, display_reference_configuration=True, clip=
None, clip_plane_origin=None, clip_plane_normal_vector=None,
set_camera_interactively=None):
    
    # Gets the path to the module where function LOCAL_frozenSnapshots is

    module_path = str(paraview_scripts.__file__)

    # Executes it as an external script
    
    programming_tools.script_executioner(module_path, python_interpreter=
    python_interpreter, function_name="LOCAL_frozenSnapshots", arguments_list=[
    input_fileName, field_name], keyword_argumentsDict={"input_path": 
    input_path, "output_path": output_path, "camera_position": 
    camera_position, "color_map": color_map, "output_imageFileName": 
    output_imageFileName, "time_step_index": time_step_index, "time": 
    time, "camera_focal_point": camera_focal_point, "camera_up_directi"+
    "on": camera_up_direction, "representation_type": 
    representation_type, "legend_bar_position": legend_bar_position, 
    "legend_bar_length": legend_bar_length, "axes_color": axes_color, 
    "size_in_pixels": size_in_pixels, "get_attributes_render": 
    get_attributes_render, "camera_parallel_scale":
    camera_parallel_scale, "camera_rotation": camera_rotation, "legend"+
    "_bar_font": legend_bar_font, "zoom_factor": zoom_factor, "legend_"+
    "bar_font_file": legend_bar_font_file, "plot_x_axis": plot_x_axis,
    "plot_y_axis": plot_y_axis, "plot_z_axis": plot_z_axis, "no_axes":
    no_axes, "component_to_plot": component_to_plot, "resolution_ratio":
    resolution_ratio, "transparent_background": transparent_background,
    "warp_by_vector": warp_by_vector, "warp_scale": warp_scale, "glyph":
    glyph, "glyph_scale": glyph_scale, "display_reference_configuration":
    display_reference_configuration, "clip": clip, "clip_plane_origin":
    clip_plane_origin, "clip_plane_normal_vector": 
    clip_plane_normal_vector, "set_camera_interactively": 
    set_camera_interactively},
    execution_rootPath=execution_rootPath, run_as_module=False,
    LIBGL_ALWAYS_SOFTWARE=LIBGL_ALWAYS_SOFTWARE, 
    extra_execution_arguments=extra_execution_arguments)