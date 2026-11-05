# Routine to generate an interchangeble mesh both for thrird medium
# contact and for notch-based crack propagation.

import h5py

import numpy as np

from .....Davout.CuboidGmsh.solids import cuboid_prisms as prisms

from .....Davout.CuboidGmsh.tool_box import meshing_tools as tools

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

# Defines a function to generate the mesh from a set of geometrical pa-
# rametres. L is the length of the box, W is the width of the box, H is
# the height of the box, theta is the notch opening angle, and 
# half_W_notch is the notch width.

def generate_mesh(L, W, H, theta, half_W_notch, L_notch, transfinite_x,
transfinite_y, transfinite_z, transfinite_radial, mesh_file_name, 
bias_x, bias_y, bias_z, bias_radial, include_drittes_medium=True,
verbose=True):

    # Verifies whether the notch opening angle of the notch is between 0
    # and 90 degrees 

    if theta<0.0 or theta>90.0:

        raise ValueError("The notch opening angle should be between 0 "+
        "and 90 degrees. Currently, 'theta' is "+str(theta))

    # Converts the theta angle to radians

    theta = theta*(np.pi/180.0)

    # Verifies whether the notch width is confined within the box width  

    if (2*half_W_notch*np.sin(theta))>=W:

        raise ValueError("The notch width, 'half_W_notch', and the not"+
        "ch opening angle, 'theta', should be such that 2*half_W_notch"+
        "*sin(theta) < W. So that the notch fits into the width of the"+
        " box")

    # Verifies whether the notch length is confined within the box length

    # Defines the surfaces and volumes regions names for imposition of
    # boundary conditions and integration of constitutive models

    surfaces_regions_names = ["top", "front_solid", "front_third", "back", "bottom"]

    volumes_regions_names = ["third medium", "solid"]

    # Initializes the mesh 

    geometric_data = tools.gmsh_initialization(surface_regionsNames= 
    surfaces_regions_names, volume_regionsNames=volumes_regions_names)

    # Creates the upper half of the notch (filled with third medium)

    # Sets the corner points

    corner_points = np.array([[L-L_notch, 0.5*W, 0.0], [L, 0.5*W, 0.0],
    [L, (0.5*W)+(half_W_notch*np.sin(theta)), 0.0], [L-L_notch+(
    half_W_notch*np.cos(theta)), (0.5*W)+(half_W_notch*np.sin(theta)), 
    0.0],
    [L-L_notch, 0.5*W, H], [L, 0.5*W, H],
    [L, (0.5*W)+(half_W_notch*np.sin(theta)), H], [L-L_notch+(
    half_W_notch*np.cos(theta)), (0.5*W)+(half_W_notch*np.sin(theta)), 
    H]])

    # Generates this cuboid

    if include_drittes_medium:

        geometric_data = prisms.hexahedron_from_corners(corner_points, 
        transfinite_directions = [transfinite_y, transfinite_x, 
        transfinite_z], geometric_data=geometric_data,  
        explicit_volume_physical_group_name = "third medium", 
        explicit_surface_physical_group_name = {3: "front_third"},
        bias_directions = {"x": bias_y, "y": bias_x, "z": bias_z})

    # Creates the upper section of the solid

    # Sets the corner points
    
    corner_points = np.array([[L-L_notch+(half_W_notch*np.cos(theta)), 
    (0.5*W)+(half_W_notch*np.sin(theta)), 0.0],
    [L, (0.5*W)+(half_W_notch*np.sin(theta)), 0.0],
    [L, W, 0.0], [0.0, W, 0.0],
    [L-L_notch+(half_W_notch*np.cos(theta)), 
    (0.5*W)+(half_W_notch*np.sin(theta)), H],
    [L, (0.5*W)+(half_W_notch*np.sin(theta)), H],
    [L, W, H], [0.0, W, H]])

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions = [transfinite_radial, transfinite_x, 
    transfinite_z], geometric_data=geometric_data,  
    explicit_volume_physical_group_name = "solid", 
    explicit_surface_physical_group_name = {3: "front_solid", 4: "top"},
    bias_directions = {"x": bias_radial, "y": bias_x, "z": bias_z})

    # Creates the upper section of the back of the solid

    # Sets the corner points
        
    corner_points = np.array([[0.0, 0.5*W, 0.0], [L-L_notch, 0.5*W, 0.0],
    [L-L_notch+(half_W_notch*np.cos(theta)), (0.5*W)+(half_W_notch*
    np.sin(theta)), 0.0], [0.0, W, 0.0],
    [0.0, 0.5*W, H], [L-L_notch, 0.5*W, H],
    [L-L_notch+(half_W_notch*np.cos(theta)), (0.5*W)+(half_W_notch*
    np.sin(theta)), H], [0.0, W, H]])

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data=geometric_data,  
    explicit_volume_physical_group_name = "solid", 
    explicit_surface_physical_group_name = {5: "back"},
    bias_directions = {"x": bias_y, "y": bias_radial, "z": bias_z})

    # Creates the botton section of the back of the solid (6)

    # Sets the corner points
        
    corner_points = np.array([[0.0, 0.0, 0.0], [(L-L_notch)+(
    half_W_notch*np.cos(theta)), 0.5*W-(half_W_notch*np.sin(theta)), 0.0],
    [L-L_notch, 0.5*W, 0.0], [0.0, 0.5*W, 0.0],
    [0.0, 0.0, H], [(L-L_notch)+(half_W_notch*np.cos(theta)), 
    0.5*W-(half_W_notch*np.sin(theta)), H],
    [L-L_notch, 0.5*W, H], [0.0, 0.5*W, H]])

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions = [transfinite_y, transfinite_radial, 
    transfinite_z], geometric_data=geometric_data,  
    explicit_volume_physical_group_name = "solid", 
    explicit_surface_physical_group_name = {5: "back"},
    bias_directions = {"x": -bias_y, "y": bias_radial, "z": bias_z})

    # Creates the botton part of the third medium (2)

    # Sets the corner points
        
    corner_points = np.array([[(L-L_notch)+(half_W_notch*np.cos(theta)), 
    0.5*W-(half_W_notch*np.sin(theta)), 0.0], [L, 0.5*W-(half_W_notch*
    np.sin(theta)), 0.0], [L, 0.5*W, 0.0], [L-L_notch, 0.5*W, 0.0], 
    [(L-L_notch)+(half_W_notch*np.cos(theta)), 0.5*W-(half_W_notch*
    np.sin(theta)), H], [L, 0.5*W-(half_W_notch*np.sin(theta)), H],
    [L, 0.5*W, H], [L-L_notch, 0.5*W, H]])

    # Generates this cuboid

    if include_drittes_medium:

        geometric_data = prisms.hexahedron_from_corners(corner_points, 
        transfinite_directions = [transfinite_y, transfinite_x, 
        transfinite_z], geometric_data=geometric_data,  
        explicit_volume_physical_group_name = "third medium", 
        explicit_surface_physical_group_name = {3: "front_third"},
        bias_directions = {"x": -bias_y, "y": bias_x, "z": bias_z})

    # Creates the botton part of the solid (4)

    # Sets the corner points
    
    corner_points = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], 
    [L, (0.5*W)-(half_W_notch*np.sin(theta)), 0.0], [L-L_notch+(
    half_W_notch*np.cos(theta)), (0.5*W)-(half_W_notch*np.sin(theta)), 0.0], 
    [0.0, 0.0, H], [L, 0.0, H], [L, (0.5*W)-(half_W_notch*np.sin(theta)), 
    H], [L-L_notch+(half_W_notch*np.cos(theta)), (0.5*W)-(half_W_notch*
    np.sin(theta)), H]])

    # Generates this cuboid

    geometric_data = prisms.hexahedron_from_corners(corner_points, 
    transfinite_directions = [transfinite_y, transfinite_x, 
    transfinite_z], geometric_data=geometric_data,  
    explicit_volume_physical_group_name = "solid", 
    explicit_surface_physical_group_name = {3: "front_solid", 2: "bottom"},
    bias_directions = {"x": -bias_radial, "y": bias_x, "z": bias_z})

    ####################################################################
    #                          Mesh generation                         #
    ####################################################################
    
    tools.gmsh_finalize(geometric_data=geometric_data, file_name=
    mesh_file_name, file_directory="", verbose=verbose)

# Testing block

if __name__ == "__main__":

    # Example parameters for the mesh generation

    L = 10.0

    W = 5.0
    
    H = 3.0
    
    theta = 30.0

    half_W_notch = 1.0
    
    L_notch = 8.0

    # Sets the number of elements of each direction

    transfinite_x = 10

    transfinite_y = 4
    
    transfinite_z = 5
    
    transfinite_radial = 5

    # Sets the bias factors. When the bias is 1.0, nothing happens and
    # the finite elements are uniformly distributed

    bias_x = 1.1

    bias_y = 1.2

    bias_z = 1.0

    bias_radial = -1.5

    # Defines the name of the mesh and appends the whole path to it.
    # Uses a function from Davout to automatically retrieve the path to
    # this file (notch_mesh.py)

    mesh_file_name = get_parent_path_of_file()+"//notch_mesh"

    # Sets a flag to add or not the mesh of the third medium

    include_drittes_medium = True

    # Calls the mesh generation function 

    generate_mesh(L, W, H, theta, half_W_notch, L_notch, transfinite_x,
    transfinite_y, transfinite_z, transfinite_radial, mesh_file_name,
    bias_x, bias_y, bias_z, bias_radial, include_drittes_medium=
    include_drittes_medium)