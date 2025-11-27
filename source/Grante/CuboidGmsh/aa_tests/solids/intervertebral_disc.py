# Routine to mesh the intervertebral disc given points

import numpy as np

from ...solids import cuboid_prisms as prisms

from ...tool_box import meshing_tools as tools

# Defines a function to construct the intervertebral disc using prisms

def mesh_disc():

    length_x = 1.0

    length_y = 1.5

    length_z = 2.5

    transfinite_directions = [7,30,11]

    ####################################################################
    #                     Boundary surfaces setting                    #
    ####################################################################

    # Sets the names of the surface regions

    surface_regionsNames = ['lower', 'upper']

    ####################################################################
    #                    Volumetric regions setting                    #
    ####################################################################

    # Sets the names of the volume regions

    volume_regionsNames = ['nucleus', 'annulus']

    ####################################################################
    #                              Cuboids                             #
    ####################################################################

    # Initializes the geometric data

    geometric_data = tools.gmsh_initialization(surface_regionsNames=
    surface_regionsNames, volume_regionsNames=volume_regionsNames)

    # Volume 1

    corner_points = [[length_x*1.2, 0.0, 0.0], [length_x, length_y, 0.0], [
    0.0, length_y, 0.0], [0.0, 0.0, 0.0], [length_x, 0.0, length_z], [
    length_x, length_y, length_z], [0.0, length_y, length_z], [0.0, 0.0, 
    length_z]]

    """edge_points = {"1": [[1.1*length_x, 0.2*length_y, 0.0], 
    [0.9*length_x, 0.4*length_y, 0.0], [1.2*length_x, 0.6*length_y, 0.0], 
    [0.8*length_x, 0.8*length_y, 0.0]], "5": [[1.1*length_x, 0.2*length_y, length_z], 
    [0.9*length_x, 0.4*length_y, length_z], [1.2*length_x, 0.6*length_y, length_z], 
    [0.8*length_x, 0.8*length_y, length_z]]}"""

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data,  explicit_volume_physical_group_name="nucleus",
    explicit_surface_physical_group_name={1: "lower", 6: "upper"})

    # Volume 2

    corner_points = [[0.0, 0.0, 0.0], [0.0, length_y, 0.0], [
    -length_x, length_y, 0.0], [-length_x, 0.0, 0.0], [0.0, 0.0, 
    length_z], [0.0, length_y, length_z], [-length_x, length_y, 
    length_z], [-length_x, 0.0, length_z]]

    edge_points = {"3": [[-1.1*length_x, 0.2*length_y, 0.0], 
    [-0.9*length_x, 0.4*length_y, 0.0], [-1.2*length_x, 0.6*length_y, 0.0], 
    [-0.8*length_x, 0.8*length_y, 0.0]]}

    # Explicitely tells the physical group of this volume

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions=transfinite_directions, geometric_data=
    geometric_data, edges_points=edge_points, 
    explicit_volume_physical_group_name='annulus',
    explicit_surface_physical_group_name={1: "lower", 6: "upper"})

    # Creates the geometry and meshes it

    tools.gmsh_finalize(geometric_data=geometric_data, file_name="inte"+
    "rvertebral_disc")

# Test block

if __name__=="__main__":

    mesh_disc()