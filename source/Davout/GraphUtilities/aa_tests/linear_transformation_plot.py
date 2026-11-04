# Routine to plot the result of a linear transformation applied to vec-
# tors in the origin-centered circumference of the XY plane

import numpy as np

from ....Davout.GraphUtilities.plotting_tools import plane_plot

# Defines a function to plot the 2D transformation. 
# 'column_eigenvectors_array' is a [2,2] numpy array, where the columns 
# are the column eigenvectors. 
# 'eigenvalues_array' is a [2] numpy array, where each row is the eigen-
# value to the corresponding eigenvector

def generate_linear_2D_transformation(column_eigenvectors_array, 
eigenvalues_array, p=2, n_points=300):

    print("The original eigenvectors are:\n"+str(
    column_eigenvectors_array)+"\n")

    # Normalizes the eigenvectors

    eigenvectors_norms = np.linalg.norm(column_eigenvectors_array, 
    axis=0)

    column_eigenvectors_array = (column_eigenvectors_array/
    eigenvectors_norms)

    print("The normalized eigenvectors are:\n"+str(
    column_eigenvectors_array)+"\n")

    # Gets the polar angles of the eigenvectors

    polar_angles = np.arctan2(column_eigenvectors_array[1,:],
    column_eigenvectors_array[0,:])

    # Makes sure they are in the first polar cycle

    polar_angles = np.mod(polar_angles, 2.0*np.pi)

    # Makes sure the first polar angle is lower than the second one

    polar_angle_1, polar_angle_2 = np.sort(polar_angles)

    # Gets the indices of precedence of the eigenvectors

    eigenvector_polar_indices = np.argsort(polar_angles)

    print("Polar angle of the first eigenvector: "+str(polar_angle_1*(
    180.0/np.pi))+" degrees or "+str(polar_angle_1)+" radians\nPolar a"+
    "ngle of the second eigenvector: "+str(polar_angle_2*(180.0/np.pi))+
    " degrees or "+str(polar_angle_2)+" radians\n")

    # Computes the inverse of the eigenvectors matrix

    eigenvectors_inverse = np.linalg.inv(column_eigenvectors_array)

    print("The inverse of the matrix of eigenvectors is:\n"+str(
    eigenvectors_inverse)+"\n")

    # Reconstructs the transformation matrix using the spectral decompo-
    # sition

    print("The diagonal format of the transformation matrix is: "+str(
    np.diag(eigenvalues_array))+"\n")

    transformation_matrix = np.einsum('ij,jl,lk->ik', 
    column_eigenvectors_array, np.diag(eigenvalues_array), 
    eigenvectors_inverse)

    print("The transformation matrix is:\n"+str(transformation_matrix)+
    "\n")

    # Checks the eigenvalues and eigenvectors

    _, eigenvectors = np.linalg.eig(transformation_matrix)

    print("The eigenvalues and eigenvectors of the reconstructed trans"+
    "formation matrix are respectively:")

    for eigenvector_index in range(eigenvectors.shape[1]):

        eigenvector = eigenvectors[:,eigenvector_index]

        eigenvalue = 0.0

        for index, component in enumerate(eigenvector):

            if np.abs(component)>1E-10:

                eigenvalue = np.einsum('ij,j->i', transformation_matrix, 
                eigenvector)[index]/component

        print(str(eigenvalue)+" <-> "+str(eigenvector))

    # Initializes the lists of coordinates of the points on the boundary
    # of the transformation

    x_data = []

    y_data = []

    colors = []

    # Iterates through the points

    for j in range(n_points):

        # Gets the angle of this point

        angle = (j*(2*np.pi)/(n_points-1))+polar_angle_1

        if angle<=polar_angle_2:

            colors.append((angle-polar_angle_1)/(polar_angle_2-
            polar_angle_1))

        else:

            colors.append(1.0-(angle-polar_angle_2)/((2*np.pi)-
            polar_angle_2+polar_angle_1))

        # Computes the polar coordinates and generates the input vector

        input_vector = np.array([np.cos(angle), np.sin(angle)])

        # Maps the input vector to the transformed coordinates

        transformed_vector = np.einsum('ij,j->i', transformation_matrix,
        input_vector)

        # Appends the coordinates

        x_data.append(transformed_vector[0])

        y_data.append(transformed_vector[1])

    plot_object = plane_plot("2D_linear_transformation", x_data=x_data, 
    y_data=y_data, plot_type="scatter", element_size=2.5, color=colors,
    color_map="coolwarm", aspect_ratio=1.0)

    # Plots the eigenvectors

    colors = np.array([0.0, 1.0])

    colors = colors[eigenvector_polar_indices].tolist()

    plot_object = plane_plot("2D_linear_transformation", x_data=[[0.0, 
    column_eigenvectors_array[0,0]], [0.0, column_eigenvectors_array[0,1]]], 
    y_data=[[0.0, column_eigenvectors_array[1,0]], [0.0, 
    column_eigenvectors_array[1,1]]], plot_type="line", element_size=2.5, 
    color=colors, color_map="coolwarm", aspect_ratio=1.0, plot_object=
    plot_object)

if __name__=="__main__":

    column_eigenvectors_array_non_orthogonal = np.transpose(np.array([
    [1.0, 1.0], [1.0, 0.0]]))

    column_eigenvectors_array_orthogonal = np.transpose(np.array([
    [1.0, 1.0], [-1.0, 1.0]]))

    eigenvalues_array = np.array([1.0, 2.0])

    generate_linear_2D_transformation(
    column_eigenvectors_array_non_orthogonal, eigenvalues_array)