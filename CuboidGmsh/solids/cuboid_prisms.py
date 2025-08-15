# Routine to store cylindrical shapes

import numpy as np

from ..solids import cuboid_generator as cuboid

from ..tool_box import geometric_tools as geo

# Defines a function to create a rectangular prism with right corners

def right_rectangularPrism(length_x, length_y, length_z, axis_vector, 
base_point, transfinite_directions=[], bias_directions=dict(), 
shape_spin=0.0, geometric_data=[0, [[],[],[],[]], [[],[],[],[]], [[],[],
[]], dict(), [], dict(), [], [], [], 0.5, False]):
    
    # Creates the 8 corners as if the long axis is the Z axis proper, and 
    # the prism starts at the XY plane towards positive Z

    corner_points = [[length_x, length_x, 0.0, 0.0, length_x, length_x, 
    0.0, 0.0], [0.0, length_y, length_y, 0.0, 0.0, length_y, length_y, 
    0.0], [0.0, 0.0, 0.0, 0.0, length_z, length_z, length_z, length_z]]

    ####################################################################
    #                             Rotation                             #
    ####################################################################

    # Sets the native axis of this shape

    native_axis = [1.0, 0.0, 0.0]

    rotation_vector = geo.find_rotationToNewAxis(axis_vector, 
    native_axis, shape_spin)

    # Makes the shape

    geometric_data = cuboid.make_cuboid(corner_points, 
    transfinite_directions=transfinite_directions, bias_directions=
    bias_directions, rotation_vector=rotation_vector, translation_vector
    =base_point, geometric_data=geometric_data)

    return geometric_data