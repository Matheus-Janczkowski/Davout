# Routine to test ordering nodes by distance from the centroid of the
# region

import numpy as np

from scipy.stats import pearsonr

from ...GraphUtilities.plotting_tools import plot_matrix

from ...PythonicUtilities.path_tools import get_parent_path_of_file

def experiment():

    ####################################################################
    #              Node ordering with respect to centroid              #
    ####################################################################

    # Builds the nodes coordinates in a mesh of linear hexahedrons

    nodes_coordinates = []

    length_elements_in_x = np.array([1.0, 1.5])

    length_elements_in_y = np.array([2.0, 2.2])

    length_elements_in_z = np.array([0.5, 0.7])

    for k in range(3):

        for j in range(3):

            for i in range(3):

                # Calculates the coordinates of the node at division 
                # (i+1, j+1, k+1)

                nodes_coordinates.append([np.sum(length_elements_in_x[0:
                i]), np.sum(length_elements_in_y[0:j]), np.sum(
                length_elements_in_z[0:k])])

    nodes_coordinates = np.array(nodes_coordinates)

    print("Nodes coordinates:\n"+str(nodes_coordinates))

    # Calculates the centroid

    region_centroid = np.mean(nodes_coordinates, axis=0)

    print("\nThe region centroid coordinates are:\n"+str(region_centroid))

    # Evaluates the vectors from the centroid of the region to each one
    # of the nodes

    vectors_centroid_to_nodes = nodes_coordinates-region_centroid

    print("\nThe vectors from the centroid to each node are:\n"+str(
    vectors_centroid_to_nodes))

    # Evaluates the distances from the centroid to each node

    distances_centroid_to_nodes = np.linalg.norm(
    vectors_centroid_to_nodes, axis=1)

    print("\nThe distances from the centroid to each node are:\n"+str(
    distances_centroid_to_nodes))

    # Gets the indices of the nodes such that the distances to the cen-
    # troid of the region are ordered in ascending manner

    nodes_ordering_from_centroid = np.argsort(distances_centroid_to_nodes)

    # Reorders the nodes so that the indices are centered around the clo-
    # sest node to the centroid

    indices_of_nodes_ordering_from_centroid = [
    nodes_ordering_from_centroid[0]]

    for counter, node_index in enumerate(nodes_ordering_from_centroid[
    1:len(nodes_ordering_from_centroid)]):
        
        # If the counter is even, the index is appended to the right

        if counter%2==0:

            indices_of_nodes_ordering_from_centroid.append(node_index)

        # Otherwise, if it is odd, appends to the left

        else:

            indices_of_nodes_ordering_from_centroid = [node_index,
            *indices_of_nodes_ordering_from_centroid]

    # Reorders the node coordinates using the indices lastly gathered for
    # ascending order of distance to the centroid

    nodes_coordinates = nodes_coordinates[
    indices_of_nodes_ordering_from_centroid,:]

    distances_centroid_to_nodes = distances_centroid_to_nodes[
    indices_of_nodes_ordering_from_centroid]

    print("\nThe new nodes coordinates, after reordering with respect "+
    "to the distance to the centroid, are:\n")

    counter = 1

    for node_coordinate, distance in zip(nodes_coordinates, 
    distances_centroid_to_nodes):
        
        print(str(counter)+": Distance to centroid: "+str(distance)+";"+
        "            node coordinates: "+str(node_coordinate))

        counter += 1

    print("\nThe region centroid coordinates are:\n"+str(region_centroid))

    ####################################################################
    #                          Data processing                         #
    ####################################################################

    # Suppose we have a matrix of data, where each row corresponds to a 
    # sample, and each column corresponds to a node

    n_nodes = len(nodes_coordinates)

    data = np.random.randn(10, n_nodes)

    # Initializes the Pearson correlation matrix

    pearson_matrix = np.zeros((n_nodes, n_nodes))

    # Iterates through the rows of the Pearson correlation matrix

    for i in range(n_nodes):

        # Recovers the original index of the node (prior to distance or-
        # dering)

        original_row_node_index = indices_of_nodes_ordering_from_centroid[i]

        # Iterates through the columns of the Pearson correlation matrix,
        # but starts at i, because it is symmetric

        for j in range(i, n_nodes):

            # Recovers the original index of the node (prior to distance 
            # ordering), but this time for the column node

            original_column_node_index = (
            indices_of_nodes_ordering_from_centroid[j])

            # Recovers the data, and evaluates the Pearson correlation

            pearson_value, _ = pearsonr(data[:, original_row_node_index],
            data[:, original_column_node_index])

            # Updates the Pearson matrix

            pearson_matrix[i,j] = pearson_value*1.0

    plot_matrix([[0.0, pearson_matrix]], get_parent_path_of_file(), "p"+
    "earson")

experiment()