# Routine to test the modulating function that sends vectors to the po-
# sitive orthant. Note that this implementation is carried out using 
# numpy, and, for this reason, is a mere test

import numpy as np

from ....Davout.GraphUtilities.plotting_tools import plane_plot

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
dimension_axis, tolerance=1E-8):

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
    
    denominator = np.sqrt(u_dot_u-(u_dot_d*u_dot_d)) 

    # Checks if the denominator approaches zero
    
    zero_denominator = np.abs(beta+1.0)<=tolerance

    # Computes the coefficients of the combination of the given vector u
    # and of the identity vector d. If u is colinear to the identity, 
    # makes the first coefficient 0 and the second 1

    coefficient_u = np.where(zero_denominator, np.zeros_like(denominator
    ), 1.0/denominator)

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

    final_mu = (mu*(1-ratio))+ratio

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

    n_samples = 50

    # Creates a range of the polar angle

    theta = np.linspace(-np.pi, np.pi, n_samples)

    # And the range for the azimuthal angle

    phi = np.linspace(-0.5*np.pi, 0.5*np.pi, n_samples)

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

    R = tridimensional_rotation_tensor(np.array([0.5*np.sqrt(2.0), -0.5*
    np.sqrt(2.0), 0.0])*np.arccos(-(np.sqrt(3)/3.0)))

    # Rotates the position vectors

    position_vectors = np.einsum('ij,nj->ni', R, position_vectors)

    # Assembles the x, y, and z data for plotting the original distribu-
    # tion of position vectors. The first column corresponds to the co-
    # ordinates of the origin, whereas the second column to the coordi-
    # nates in the circumference of a unit circle centered in the origin

    x_data = np.column_stack([np.zeros_like(x_coordinates), 
    position_vectors[:,0]])

    y_data = np.column_stack([np.zeros_like(y_coordinates), 
    position_vectors[:,1]])

    z_data = np.column_stack([np.zeros_like(z_coordinates), 
    position_vectors[:,2]])

    # Plots the circle

    color = np.linspace(0, 1, z_data.shape[0])

    common_ticks = np.linspace(-1.0, 1.0, 5)

    plane_plot("original_unit_sphere.png", x_data=x_data, y_data=y_data,
    z_data=z_data, color_map="coolwarm", color=color, parent_path=
    get_parent_path_of_file(), dpi=1000, aspect_ratio='equal',
    x_ticksLabels=common_ticks, y_ticksLabels=common_ticks, 
    z_ticksLabels=common_ticks, x_label="x", y_label="y", z_label="z",
    azimuth_angle=30)

    # Uses the modulating function to transform all vectors to the posi-
    # tive quadrant

    modulated_vectors = normalize_and_rotate_vector_to_positive_orthant(
    position_vectors, dimension_axis=1)

    # Constructs the corresponding coordinates

    modulated_x_data = np.column_stack([np.zeros_like(x_coordinates), 
    modulated_vectors[:,0]])
    
    modulated_y_data = np.column_stack([np.zeros_like(y_coordinates), 
    modulated_vectors[:,1]])
    
    modulated_z_data = np.column_stack([np.zeros_like(z_coordinates), 
    modulated_vectors[:,2]])

    # Plots the modulated circle

    plane_plot("modulated_unit_sphere.png", x_data=modulated_x_data, 
    y_data=modulated_y_data, z_data=modulated_z_data, color_map=
    "coolwarm", color=color, parent_path=get_parent_path_of_file(), 
    aspect_ratio='equal', dpi=1000, x_ticksLabels=common_ticks, 
    y_ticksLabels=common_ticks, z_ticksLabels=common_ticks, x_label=
    "x", y_label="y", z_label="z",
    azimuth_angle=-10)

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
    
    u = np.asarray([[-0.5, -0.5, 1.0], [-0.25, -0.25, 1.0], [-0.1, 0.25, 
    1.0]])

    # Evaluates the corresponding vector that shall be rotated to the 
    # positive orthant

    positive_u = normalize_and_rotate_vector_to_positive_orthant(u,
    dimension_axis=1)

    print("\nThe rotated vector to the positive orthant given by "+str(u
    )+" is: "+str(positive_u)+"\nwhose norm is "+str(np.linalg.norm(
    positive_u, axis=1)))

    # Plots a visualization of the modulating function in 2D space

    plot_2D_modulating_function()

    # And in 3D space

    plot_3D_modulating_function()