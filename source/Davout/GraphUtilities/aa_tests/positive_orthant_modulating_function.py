# Routine to test the modulating function that sends vectors to the po-
# sitive orthant. Note that this implementation is carried out using 
# numpy, and, for this reason, is a mere test

import numpy as np

from time import time

from ....Davout.GraphUtilities.plotting_tools import plane_plot

from ....Davout.GraphUtilities.collage_tools import create_box_collage

from ....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from ....Davout.PythonicUtilities.tensor_and_math_tools import tridimensional_rotation_tensor

# Defines a function to evaluate a vector normal to the identity line in
# n-dimensional real space that belongs to the subspace spanned by the 
# identity line direction vector and a given vector

def get_normal_vector_to_identity_line(u_vector, space_dimension):

    # Gets the inner product of the vector u by the direction of the i-
    # dentity line, d

    u_dot_d = np.sum(u_vector)/np.sqrt(space_dimension)

    # Computes the denominator common to all coefficients of the fol-
    # lowing linear combination

    denominator = np.sqrt(np.dot(u_vector, u_vector)-(u_dot_d*u_dot_d))

    # Checks if the denominator approaches zero

    zero_denominator = np.equal(denominator, 0.0)

    # Computes the coefficients of the combination of the given vector u
    # and of the identity vector d. If u is colinear to the identity, 
    # makes the first coefficient 0 and the second 1

    coefficient_u = np.where(zero_denominator, np.zeros_like(denominator
    ), 1.0/denominator)

    coefficient_d = np.where(zero_denominator, 1.0, -((coefficient_u*
    u_dot_d)/np.sqrt(space_dimension)))

    # Returns the combination

    return (coefficient_u*u_vector)+(coefficient_d*np.ones(
    space_dimension))

# Defines a function to get any vector u in n-dimensional real space,
# make it unitary, and then rotate it to the positive orthant inside a
# 2-dimensional subspace spanned by the original vector u and the posi-
# tive and unit identity vector

def normalize_and_rotate_vector_to_positive_orthant_cone(u_vector,
space_dimension, tolerance=1E-10):

    # Gets the dot product of the given u vector by the positive identi-
    # ty vector

    dimensionality_square_root = np.sqrt(space_dimension)

    u_dot_d = np.sum(u_vector)/dimensionality_square_root

    # Computes the square norm of u

    u_dot_u = np.dot(u_vector, u_vector)

    # Computes beta, that is the argument of the arccosine function

    beta = u_dot_d/np.sqrt(u_dot_u)

    # Computes the quarter cosine and the quarter sine

    inner_square_root = np.sqrt(0.5*(1.0+beta))

    quarter_cosine = np.sqrt(0.5*(1+inner_square_root))

    quarter_sine = np.sqrt(0.5*(1-inner_square_root))

    # Evaluates the common denominator of the coefficients to generate
    # a vector perpendicular to the identity line in the subspace span-
    # ned by u and d
    
    denominator = np.sqrt(u_dot_u-(u_dot_d*u_dot_d)) 

    # Checks if the denominator approaches zero
    
    zero_denominator = np.abs(beta+1.0)<=tolerance

    # Computes the coefficients of the combination of the given vector u
    # and of the identity vector d. If u is colinear to the identity, 
    # makes the first coefficient 0 and the second 1

    coefficient_u = np.where(zero_denominator, np.zeros_like(denominator
    ), 1.0/denominator)

    coefficient_d = -((coefficient_u*u_dot_d)/dimensionality_square_root)

    # Assembles the resulting rotated vector

    return (((quarter_cosine+(coefficient_d*quarter_sine))/
    dimensionality_square_root)*np.ones(space_dimension)+(coefficient_u*
    quarter_sine*u_vector))

# Defines a function to get a vector in the boundary of the positive or-
# thant that intercepts the subspace spanned by the given vector u and 
# the identity line. Then, this vector is used to interpolate another 
# vector inside the positive orthant. The incoming vector must be an ar-
# ray (n_samples, space_dimension)

def normalize_and_rotate_vector_to_positive_orthant(u_vector,
dimension_axis, tolerance=1E-12):

    # Gets the space dimension

    space_dimension = u_vector.shape[dimension_axis]

    # Gets the dot product of the given u vector by the positive identi-
    # ty vector

    dimensionality_square_root = np.sqrt(space_dimension)

    u_dot_d = (np.sum(u_vector, axis=dimension_axis)/
    dimensionality_square_root)

    # Computes the square norm of u

    u_dot_u = np.sum(u_vector**2, axis=dimension_axis)

    # Computes beta, that is the argument of the arccosine function

    beta = u_dot_d/np.sqrt(u_dot_u)

    # Evaluates the common denominator of the coefficients to generate
    # a vector perpendicular to the identity line in the subspace span-
    # ned by u and d
    
    denominator = np.sqrt(u_dot_u-(u_dot_d*u_dot_d)+tolerance)

    # Checks if the denominator approaches zero
    
    zero_denominator_condition = denominator<=tolerance

    # Computes the coefficients of the combination of the given vector u
    # and of the identity vector d. If u is colinear to the identity, 
    # makes the first coefficient 0 and the second 1

    coefficient_u = np.where(zero_denominator_condition, np.zeros_like(
    denominator), 1.0/denominator)

    coefficient_d = -(coefficient_u*u_dot_d)

    # Builds the vector colinear to the identity line

    d_vector = (1.0/dimensionality_square_root)*np.ones(space_dimension)

    # Gets the orthonormal vector to the identity line. The shape of the
    # coefficients must be expanded to include the dimension of the spa-
    # ce, that was lost during summation operations

    c_vector = (np.expand_dims(coefficient_u, axis=dimension_axis)*
    u_vector)+(np.expand_dims(coefficient_d, axis=dimension_axis)*
    d_vector)

    # Evaluates the inequation to determine the mu factor

    inequation_numerator = np.abs(c_vector)-c_vector

    inequation_denominator = (2.0*(d_vector-c_vector))

    mu_inequation = np.divide(inequation_numerator, 
    inequation_denominator, out=np.zeros_like(inequation_numerator), 
    where=inequation_denominator!=0)

    # Gets the maximum mu factor and the corresponding boundary vector

    mu = np.max(mu_inequation, axis=dimension_axis)

    # Gets the relative position of the vector u with respect to the i-
    # dentity line. The relative position must be 1 when it lies on the
    # identity line and 0 when it lies on the negative identity line. For
    # this purpose, we use beta, since it is contained within the inter-
    # val [-1,1]

    ratio = 0.5*(beta+1.0)

    # Gets the mu corresponding to the final vector inside the positive
    # orthant

    final_mu = np.where(ratio<tolerance, np.ones_like(ratio), (mu*(1-
    ratio))+ratio)

    denominator = np.sqrt((final_mu**2)+((1.0-final_mu)**2))

    # Expands the shape of the final mu coefficient and of the common 
    # denominator to account for the space dimension that was lost du-
    # ring summation operations

    final_mu = np.expand_dims(final_mu, axis=dimension_axis)

    denominator = np.expand_dims(denominator, axis=dimension_axis)

    # Constructs the final vector as a linear interpolation of the iden-
    # tity line and the orthonormal vector c. Returns an array 
    # (n_samples, space_dimension)

    return ((final_mu/denominator)*d_vector)+(((1.0-final_mu)/
    denominator)*c_vector)

# Defines a function to plot a circle to test the modulating effect in
# 2D space

def plot_2D_modulating_function():

    # Sets the number of position vectors to be plotted

    n_samples = 500

    # Creates a range of theta

    theta = np.linspace(-(3/4)*np.pi, (5/4)*np.pi, n_samples)

    # Gets the x and y coordinates

    x_coordinates = np.cos(theta)

    y_coordinates = np.sin(theta)

    # Concatenates the two coordinates into a matrix (n_samples, 2)

    position_vectors = np.column_stack([x_coordinates, y_coordinates])

    # Assembles the x and y data for plotting the original distribution 
    # of position vectors. The first column corresponds to the coordina-
    # tes of the origin, whereas the second column to the coordinates in
    # the circumference of a unit circle centered in the origin

    x_data = np.column_stack([np.zeros_like(x_coordinates), 
    x_coordinates])

    y_data = np.column_stack([np.zeros_like(y_coordinates), 
    y_coordinates])

    # Plots the circle

    color = np.linspace(0, 1, n_samples)

    plane_plot("original_unit_circle.png", x_data=x_data, y_data=y_data,
    color_map="coolwarm", color=color, parent_path=
    get_parent_path_of_file(), aspect_ratio='equal', dpi=1000)

    # Uses the modulating function to transform all vectors to the posi-
    # tive quadrant

    modulated_vectors = normalize_and_rotate_vector_to_positive_orthant(
    position_vectors, dimension_axis=1)

    # Constrcuts the corresponding coordinates

    modulated_x_data = np.column_stack([np.zeros_like(x_coordinates), 
    modulated_vectors[:,0]])
    
    modulated_y_data = np.column_stack([np.zeros_like(y_coordinates), 
    modulated_vectors[:,1]])

    # Plots the modulated circle

    common_ticks = np.linspace(-1.0, 1.0, 5)

    plane_plot("modulated_unit_circle.png", x_data=modulated_x_data, 
    y_data=modulated_y_data, color_map="coolwarm", color=color, 
    parent_path=get_parent_path_of_file(), aspect_ratio='equal', dpi=
    1000, x_ticksLabels=common_ticks, y_ticksLabels=common_ticks)

# Defines a function to plot a sphere to test the modulating effect in
# 3D space

def plot_3D_modulating_function():

    # Sets the number of position vectors to be plotted

    n_samples_radial = 96

    n_samples_elevation = 100

    # Sets the elevation and azimuth angle for visualization

    elevation_angle = 30#35.26

    azimuth_angle = -15

    # Creates a range of the polar angle. The final point is not inclu-
    # ded to avoid repeating the first angle

    theta = np.linspace(-(3/4)*np.pi, (5/4)*np.pi, n_samples_radial, 
    endpoint=False)

    # And the range for the azimuthal angle

    phi = np.linspace(-0.5*np.pi, 0.5*np.pi, n_samples_elevation)

    # Generates a grid of the angles

    grid_theta, grid_phi = np.meshgrid(theta, phi)

    # Flattens the vectors of combinations

    grid_theta = grid_theta.ravel()

    grid_phi = grid_phi.ravel()

    # Gets the x, y, and z coordinates

    x_coordinates = np.cos(grid_theta)*np.cos(grid_phi)

    y_coordinates = np.sin(grid_theta)*np.cos(grid_phi)

    z_coordinates = np.sin(grid_phi)

    # Concatenates the two coordinates into a matrix (n_samples, 3)

    position_vectors = np.column_stack([x_coordinates, y_coordinates,
    z_coordinates])

    # Gets the rotation matrix to rotate this sphere to the identity li-
    # ne

    R = tridimensional_rotation_tensor(np.array([-0.5*np.sqrt(2.0), 0.5*
    np.sqrt(2.0), 0.0])*1.0*np.arccos((np.sqrt(3)/3.0)))

    # Rotates the position vectors

    position_vectors = np.einsum('ij,nj->ni', R, position_vectors)

    # Adds points that will map to the coordinate axes

    epsilon = 1E-3

    coordinates_axes_vectors = np.asarray([[-1.0+epsilon, -1.0, -1.0], 
    [-1.0, -1.0+epsilon, -1.0], [-1.0, -1.0, -1.0+epsilon]])

    row_norms = np.linalg.norm(coordinates_axes_vectors, axis=1, 
    keepdims=True)
        
    coordinates_axes_vectors = np.divide(coordinates_axes_vectors, 
    row_norms, out=np.zeros_like(coordinates_axes_vectors), where=
    row_norms!=0)

    position_vectors = np.row_stack([coordinates_axes_vectors, 
    position_vectors])

    # Assembles the x, y, and z data for plotting the original distribu-
    # tion of position vectors. The first column corresponds to the co-
    # ordinates of the origin, whereas the second column to the coordi-
    # nates in the circumference of a unit circle centered in the origin

    x_data = np.column_stack([np.zeros_like(position_vectors[:,0]), 
    position_vectors[:,0]])

    y_data = np.column_stack([np.zeros_like(position_vectors[:,1]), 
    position_vectors[:,1]])

    z_data = np.column_stack([np.zeros_like(position_vectors[:,2]), 
    position_vectors[:,2]])

    x_data = position_vectors[:,0]

    y_data = position_vectors[:,1]

    z_data = position_vectors[:,2]

    # Plots the identity line

    color = np.linspace(0, 1, z_data.shape[0])

    common_ticks = np.linspace(-1.0, 1.0, 5)

    n_points_boundary_edge = 100

    n_points_first_section = 10
    
    n_points_second_section = 20

    common_identity_line_points = np.concatenate([np.linspace(-1.0, 
    ((n_points_first_section/n_points_boundary_edge)*2.0)-1.0,
    n_points_first_section), np.linspace(1.0-((n_points_second_section/
    n_points_boundary_edge)*2.0), 1.0, n_points_second_section)])

    identity_line_plot = plane_plot("original_unit_sphere.png", 
    x_data=common_identity_line_points, y_data=
    common_identity_line_points, z_data=common_identity_line_points, 
    color="black", parent_path=get_parent_path_of_file(), dpi=1000, 
    aspect_ratio='equal', x_ticksLabels=common_ticks, y_ticksLabels=
    common_ticks, z_ticksLabels=common_ticks, x_label="$x$", y_label="$y$", 
    z_label="$z$", elevation_angle=elevation_angle, azimuth_angle=
    azimuth_angle, plot_type="scatter", verbose=True, element_size=0.7,
    transparent_background=True)

    # Plots the sphere

    plane_plot("original_unit_sphere.png", x_data=x_data, y_data=y_data,
    z_data=z_data, color_map="coolwarm", color=color, parent_path=
    get_parent_path_of_file(), dpi=1000, aspect_ratio='equal',
    x_ticksLabels=common_ticks, y_ticksLabels=common_ticks, 
    z_ticksLabels=common_ticks, x_label="$x$", y_label="$y$", z_label="$z$",
    elevation_angle=elevation_angle, azimuth_angle=azimuth_angle,
    plot_type="scatter", verbose=True, plot_object=identity_line_plot,
    right_padding_in_mm=7.5, top_padding_in_mm=-5.0, 
    left_padding_in_mm=-9.0, bottom_padding_in_mm=3.0, 
    transparent_background=True)

    # Uses the modulating function to transform all vectors to the posi-
    # tive quadrant

    initial_time = time()

    modulated_vectors = normalize_and_rotate_vector_to_positive_orthant(
    position_vectors, dimension_axis=1)

    final_time = time()

    print("\nTo modulate a "+str(position_vectors.shape)+" matrix, it "+
    "took "+str(final_time-initial_time)+" seconds")

    # Constructs the corresponding coordinates

    modulated_x_data = np.column_stack([np.zeros_like(modulated_vectors[
    :,0]), modulated_vectors[:,0]])
    
    modulated_y_data = np.column_stack([np.zeros_like(modulated_vectors[
    :,1]), modulated_vectors[:,1]])
    
    modulated_z_data = np.column_stack([np.zeros_like(modulated_vectors[
    :,2]), modulated_vectors[:,2]])

    modulated_x_data = modulated_vectors[:,0]

    modulated_y_data = modulated_vectors[:,1]

    modulated_z_data = modulated_vectors[:,2]

    # Plots the identity line again, but adds the points of the edges of
    # this octant

    n_points_first_section = 59

    n_points_second_section = 20

    n_points_boundary_curved_edge = 30

    common_identity_line_points = np.concatenate([np.linspace(-1.0, 
    ((n_points_first_section/n_points_boundary_edge)*2.0)-1.0,
    n_points_first_section), np.linspace(1.0-((n_points_second_section/
    n_points_boundary_edge)*2.0), 1.0, n_points_second_section)])

    # Sets the initial and final angles of the bottom and back edges

    initial_angle_back_edge = (25/180)*np.pi

    final_angle_bottom_edge = (70/180)*np.pi

    x_data = np.concatenate([common_identity_line_points, np.cos(
    np.linspace(0.5*np.pi, 0.0, n_points_boundary_curved_edge)), np.cos(
    np.linspace(0.0, final_angle_bottom_edge, 
    n_points_boundary_curved_edge)), np.linspace(0.0, 0.0, 
    n_points_boundary_curved_edge)])

    y_data = np.concatenate([common_identity_line_points, np.linspace(
    0.0, 0.0, n_points_boundary_curved_edge), np.sin(np.linspace(0.0, 
    final_angle_bottom_edge, n_points_boundary_curved_edge)), np.cos(
    np.linspace(initial_angle_back_edge, 0.5*np.pi, 
    n_points_boundary_curved_edge))])

    z_data = np.concatenate([common_identity_line_points, np.sin(
    np.linspace(0.5*np.pi, 0.0, n_points_boundary_curved_edge)), 
    np.linspace(0.0, 0.0, n_points_boundary_curved_edge), np.sin(
    np.linspace(initial_angle_back_edge, 0.5*np.pi, 
    n_points_boundary_curved_edge))])

    identity_line_plot = plane_plot("modulated_unit_sphere.png", 
    x_data=x_data, y_data=y_data, z_data=z_data, 
    color="black", parent_path=get_parent_path_of_file(), dpi=1000, 
    aspect_ratio='equal', x_ticksLabels=common_ticks, y_ticksLabels=
    common_ticks, z_ticksLabels=common_ticks, x_label="$x$", y_label="$y$", 
    z_label="$z$", elevation_angle=elevation_angle, azimuth_angle=
    azimuth_angle, plot_type="scatter", verbose=True, element_size=0.7, 
    transparent_background=True)

    plane_plot("modulated_unit_sphere.png", x_data=modulated_x_data, 
    y_data=modulated_y_data, z_data=modulated_z_data, color_map=
    "coolwarm", color=color, parent_path=get_parent_path_of_file(), 
    aspect_ratio='equal', dpi=1000, x_ticksLabels=common_ticks, 
    y_ticksLabels=common_ticks, z_ticksLabels=common_ticks, x_label=
    "$x$", y_label="$y$", z_label="$z$", elevation_angle=elevation_angle,
    azimuth_angle=azimuth_angle, plot_type="scatter", verbose=True,
    plot_object=identity_line_plot, right_padding_in_mm=7.5, 
    top_padding_in_mm=-5.0, left_padding_in_mm=-9.0, 
    bottom_padding_in_mm=3.0, transparent_background=True)

# Defines a function to create a collage

def create_collage():

    create_box_collage("collage_modulating_function.pdf", input_path=
    get_parent_path_of_file(),

    input_image_list=[{"file name": "original_unit_sphere.png", 
    "position": [50.0, 249.0], 
    "size": 50.0, "trim transparent background": True, "origin point":
    "top-left"}, 
    {"file name": "modulated_unit_sphere.png", 
    "position": [103.0, 249.0], "size": 50.0, 
    "trim transparent background": True, "origin point": "top-left"}], 

    input_text_list=[{"text": "$\\Omega_{\\mathcal{B}}\\in\\RealSpace{3}$", "position":
    [100.0, 198.8], "font size": 4, "origin point": "top-right", 
    "rendering method": "matplotlib text", "object name": "lateral sign"},
    {"text": "$\\Omega_{\\mathrm{\\mathcal{M}}}\\in\\PositiveRealSpace{3}$", "position":
    [153.0, 199.0], "font size": 4, "origin point": "top-right", 
    "rendering method": "matplotlib text", "object name": "modulated title"},
    {"text": "$\\mathcal{M}:\\RealSpace{n}\\rightarrow\\PositiveRealSpace{n}$", "position":
    [103.0, 248.0], "font size": 4, "origin point": "top-left", 
    "rendering method": "matplotlib text", "object name": "modulated title"}],

    boxes_list=[{"contour color": "black", "fill color": "grey 3", "contour"+
    " thickness": 0.2, "position": [48.0, 251.0], "width": 107.0, "height": 58.0,
    "contour style": "solid", "origin point": "top-left"},
    {"contour color": "black", "fill color": "grey 1", "contour"+
    " thickness": 0.2, "position": [49.0, 250.0], "width": 52.0, "height": 56.0,
    "contour style": "solid", "origin point": "top-left"},
    {"contour color": "black", "fill color": "grey 1", "contour"+
    " thickness": 0.2, "position": [102.0, 250.0], "width": 52.0, "height": 56.0,
    "contour style": "solid", "origin point": "top-left"}], 

    arrows_and_lines_list=[{"start point": [92.0, 240.0], "end point": [109.0, 240.0], 
    "spline points": [[98.0, 244], [104.0, 244]], "thickness": 0.2,
    "arrow style": "inkscape angular arrow"}],

    verbose=True, no_padding=True, add_overlaying_grid=False, dpi=1000,
    grid_annotation_length=10, save_lists_to_txt=False, 
    interactive_preview=False, size_template="A4",
    export_selection={"origin point": "bottom-right", "position": [
    155.0, 193.0], "width": 107.0, "height": 58.0}, 
    )

# Testing block

if __name__=="__main__":

    # Gets the vector u

    u = np.asarray([-0.5, -0.5, 1.0])

    # Gets the normal vector to the identity line that is spanned by u
    # and the vector u

    c = get_normal_vector_to_identity_line(u, u.shape[0])

    print("\nThe normal vector given by "+str(u)+" is: "+str(c))

    # Gets the vector u in a batched format, i.e., an array (n_samples,
    # space_dimension)

    epsilon = 1E-3
    
    u = np.asarray([[-1.0+epsilon, -1.0, -1.0], [-1.0, -1.0+epsilon, 
    -1.0], [-1.0, -1.0, -1.0+epsilon]])

    # Evaluates the corresponding vector that shall be rotated to the 
    # positive orthant

    positive_u = normalize_and_rotate_vector_to_positive_orthant(u,
    dimension_axis=1)

    print("\nThe rotated vector to the positive orthant given by\n"+str(
    u)+"\nis\n"+str(positive_u)+"\n\nwhose norm is "+str(np.linalg.norm(
    positive_u, axis=1)))

    # Sets a flag for plotting

    flag_plot = False 

    flag_collage = True

    if flag_plot:

        # Plots a visualization of the modulating function in 2D space

        plot_2D_modulating_function()

        # And in 3D space

        plot_3D_modulating_function()

    if flag_collage:

        create_collage()