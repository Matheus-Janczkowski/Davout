# Routine to partition a mesh in tetrahedra given a set of points using 
# Delaunay's triangulation. The DOFs of the mesh are divided in the cor-
# responding envolving tetrahedra.
#
# Code written solely by Rafael Prado in July 2026

import tensorflow as tf

import numpy as np

from scipy.spatial import Delaunay

from ...MultiMech.tool_box.mesh_handling_tools import read_mshMesh, get_dofs_in_field_and_coordinates

from ...MultiMech.tool_box.functional_tools import construct_monolithicFunctionSpace, FunctionalData

# Defines a function to get a mesh file, read the mesh, and then perform
# Delaunay's triangulation. The output is a list of lists, such that each
# sublist corresponds to a tetrahedron and the contents are the DOFs of 
# the mesh that lie in this tetrahedron. The tetrahedra vertices are the
# given target points

def triangulate_domain_and_separate_dofs(mesh_file_name, 
target_points_coordinates, field_name, functional_data_class, verbose=
False, return_as_tensor=False, int_dtype="int32", float_type="float64", 
area_tolerance_of_degenerate_triangles=1E-14):

    # Reads the mesh and returns a class of mesh data

    mesh_data_class = read_mshMesh(mesh_file_name)

    # Verifies if functional data is a dictionary with finite element
    # information

    if isinstance(functional_data_class, dict):

        # Verifies if the given field name is a key in the dictionary of
        # finite elements

        if not (field_name in functional_data_class):

            available_field_names = ""

            for field_key in functional_data_class.keys():

                available_field_names += "\n'"+str(field_key)+"'"

            raise ValueError("'functional_data_class' argument was giv"+
            "en as a dictionary, but the given 'field_name', '"+str(
            field_name)+"', is not a key in the aforementioned diction"+
            "ary")

        # From the given information, constructs and instance of the 
        # functional data class

        functional_data_class = construct_monolithicFunctionSpace(
        functional_data_class, mesh_data_class)

    # Verifies if it is an instance of the FunctionalData class

    elif not isinstance(functional_data_class, FunctionalData):

        raise TypeError("'functional_data_class' in 'triangulate_domai"+
        "n_and_separate_dofs' must be a dictionary with finite element"+
        " information for each field or it must be an instance of the "+
        " class 'FunctionalData'. Currently, it is neither, but:\n"+str(
        functional_data_class))
    
    # Gets the numpy float type

    numpy_float_type = np.dtype(float_type)

    # Converts the array of coordinates to a numpy float array of shape 
    # (n_target_points, 3)

    target_points = np.asarray(target_points_coordinates, dtype=
    numpy_float_type)

    # Gets an array of indices of all DOFs of the mesh with shape (
    # number_of_dofs,) and an array with the coordinates of the corres-
    # ponding DOFs with shape (number_of_dofs,3). This function also 
    # captures DOFs linked to higher-order elements

    dofs_indices, dofs_coords = get_dofs_in_field_and_coordinates(
    functional_data_class, field_name, return_as_list=False)

    # Calls the Delaunay class to separate the domain in tetrahedra

    delaunay_class = Delaunay(target_points)

    # Recovers a list of lists with the connectivity of each tetrahedron.
    # Each sublist corresponds to a tetrahedron and tells the indices of
    # the target points at the vertices
    
    tetrahedra_list = delaunay_class.simplices.tolist()

    if verbose:

        print("The list of the indices of the target points that form "+
        "each tetrahedron is:\n"+str(tetrahedra_list)+"\nThere are "+str(
        len(tetrahedra_list))+" sublists in the list of the indices of"+
        " the target points\n")

    # Initializes a list of lists. Each sublist will contain the indices
    # of the tetrahedra connected to each target point

    tetrahedra_per_target_points = [[] for _ in range (len(target_points
    ))]

    # Iterates over the lists of conectivities of the tetrahedra

    for tetrahedron_index, tetrahedron_vertices in enumerate(
    tetrahedra_list):

        # Iterates over the vertices of this tetrahedron

        for node_index in tetrahedron_vertices:

            tetrahedra_per_target_points[node_index].append(
            tetrahedron_index)

    if verbose:

        print("The list of tetrahedra indices per target points is:\n"+
        str(tetrahedra_per_target_points)+"\nThere are "+str(len(
        tetrahedra_per_target_points))+" sublists in this list\n")

    # Creates an array with the tetrahedron index of each DOF of the mesh

    tetrahedron_index_per_dof = delaunay_class.find_simplex(dofs_coords)

    # Initializes an empty list to store the DOFs of the mesh that lie 
    # in each tetrahedron

    dofs_per_tetrahedron = [[] for _ in range(len(tetrahedra_list))]
    
    # Initializes an empty list to store the DOFs that were not found in
    # the tetrahedra created by the Delaunay's triangulation of the tar-
    # get points. Initializes another list to store the corresponding 
    # coordinates

    outsite_domain_dofs = []

    outside_domain_coords = []

    # Simultaneously, iterates over the indices of tetrahedron per DOF,
    # and the DOFs' indices and coordinates

    for dof_index, dof_coordinates, tetrahedron_index in zip(
    dofs_indices, dofs_coords, tetrahedron_index_per_dof):
        
        # If the tetrahedron index is -1, it means this DOF was not 
        # found in the triangulated domain

        if tetrahedron_index != -1:

            dofs_per_tetrahedron[tetrahedron_index].append(dof_index)
        
        else:

            # Appends the missing DOF into the lists of DOFs cast out of
            # the triangulated domain

            outsite_domain_dofs.append(dof_index)

            outside_domain_coords.append(dof_coordinates)

    # Converts the out-cast coordinates to numpy array (n_missing_dofs,
    # 3)
    
    if outsite_domain_dofs:

        outside_domain_coords = np.array(outside_domain_coords, dtype=
        numpy_float_type)

    if verbose:

        print(f"There are {len(outsite_domain_dofs)} DOFs not found in"+
        " the domain\n")
    
    # Verifies if there are dofs that fell outside the Delaunay triangu-
    # lation
     
    if outsite_domain_dofs:

        dofs_per_tetrahedron = get_dofs_outside_delaunay_triangulation(
        dofs_per_tetrahedron, target_points, delaunay_class, 
        tetrahedra_list, numpy_float_type, 
        area_tolerance_of_degenerate_triangles, outsite_domain_dofs, 
        outside_domain_coords)
    
    # Converts the list of DOFs per tetrahedron to a ragged tensor if
    # needed. A ragged tensor is used since different tetrahedra might
    # own different numbers of DOFs

    if return_as_tensor:

        dofs_per_tetrahedron = tf.ragged.constant(dofs_per_tetrahedron, 
        dtype=tf.as_dtype(int_dtype))
    
    return dofs_per_tetrahedron

# Defines a function to recover DOFs that are outside of the Delaunay 
# triangulation due to convex boundary sections. This function uses a
# projection-based methodology, such that the coordinates of the missing
# DOFs are projected onto the boundary of the Delaunay's triangulation.
# The tetrahedron whose boundary owns the projection owns the correspon-
# ding DOF

def get_dofs_outside_delaunay_triangulation(dofs_per_tetrahedron, 
target_points, delaunay_class, tetrahedra_list, numpy_float_type,
area_tolerance_of_degenerate_triangles, outsite_domain_dofs, 
outside_domain_coords):

    # Each facet must have a particular order of origin points to ensure
    # that the normal vector of the facet point outwards. For each or-
    # dered facet, the first element is the local number of the origin
    # point; the second element is the local number of the vertex to
    # where the first vector goes; the third element gives the same in-
    # formation for the second edge vector

    vertex_orders = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]

    # Initializes a list of tetrahedra that lies in the boundary. Note,
    # however, that the occurences of a single tetrahedron is dependent 
    # on the number of facets that lie in the boundary. For example, if 
    # a tetrahedron has 3 facets in the boundary, its index will show up 
    # 3 times in the following list. This is done so that each facet in 
    # the forthcoming arrays of coefficients has a corresponding address 
    # in the list of boundary tetrahedra

    boundary_tetrahedra_indices = []

    # Initializes a list for each one of the vectors of each boundary
    # facets

    vector_x_1_boundary_facets = []

    vector_x_2_boundary_facets = []
    
    # Initializes a list that stores the indices of the boundary facets
    # that connects to each target point

    boundary_facets_per_target_point = [[] for _ in range(
    target_points.shape[0])]

    # Initializes a list that stores the origin point of each facet

    origin_point_boundary_facets = []

    # Initializes a counter of boundary facets

    n_boundary_facets = 0

    # Iterates over the matrix of neighbors of the Delaunay class. This
    # matrix tells the neighboring tetrahedra for each tetrahedron. As
    # each tetrahedron has 4 faces, it can have at most 4 neighbors

    for tetrahedron_index, neighbors in enumerate(
    delaunay_class.neighbors):

        # Gets the four vertices of this tetrahedron.

        local_vertices = tetrahedra_list[tetrahedron_index]

        # Iterates over the list of neighboring tetrahedra of this te-
        # trahedron and the local indices of its facets

        for facet_index, neighbor_tetrahedron in enumerate(neighbors):

            # If the neighbor tetrahedron is -1, it signals that this
            # facet is at the boundary of the Delaunay's triangulation

            if neighbor_tetrahedron==-1:

                # Gets the vertices of this facet, but disconsiders the 
                # node opposite to this facet. The opposite node is dis-
                # considered since it also enumerates this facet

                facet_vertices = (local_vertices[:facet_index]+
                local_vertices[facet_index+1:])

                # Updates the connectivities of these vertices

                for vertex_index in facet_vertices:

                    # Appends the index of this boundary facet to the 
                    # list of connectivities of this target point

                    boundary_facets_per_target_point[vertex_index
                    ].append(n_boundary_facets)

                # Stores the index of the tetrahedron that owns this fa-
                # cet

                boundary_tetrahedra_indices.append(tetrahedron_index)

                # Gets the local order of the origin vertex, of the 
                # ending point of the first edge, and of the ending 
                # point of the second edge. This ensures that the cross 
                # product of the first edge with the second edge always 
                # points outwards

                (origin_local_index, edge_1_local_index, 
                edge_2_local_index) = vertex_orders[facet_index]

                # Gets the origin point of the local coordinate system 
                # centered in the first vertix of this facet and copla-
                # nar to it

                origin_point_boundary_facets.append(target_points[
                local_vertices[origin_local_index]])

                # Gets the two vectors that lie in two sides of this 
                # triangular facet

                vector_x_1_boundary_facets.append(target_points[
                local_vertices[edge_1_local_index]]-target_points[
                local_vertices[origin_local_index]])

                vector_x_2_boundary_facets.append(target_points[
                local_vertices[edge_2_local_index]]-target_points[
                local_vertices[origin_local_index]])

                # Updates the counter of boundary facets

                n_boundary_facets += 1

    # Converts the lists of vectors of the boundary facets to pure numpy
    # arrays

    vector_x_1_boundary_facets = np.asarray(vector_x_1_boundary_facets,
    dtype=numpy_float_type)

    vector_x_2_boundary_facets = np.asarray(vector_x_2_boundary_facets,
    dtype=numpy_float_type)

    vector_x_3_boundary_facets = np.cross(vector_x_1_boundary_facets, 
    vector_x_2_boundary_facets)

    origin_point_boundary_facets = np.asarray(
    origin_point_boundary_facets, dtype=numpy_float_type)

    # Normalizes the outward normal vector

    vector_x_3_boundary_facets = (vector_x_3_boundary_facets/
    np.linalg.norm(vector_x_3_boundary_facets, axis=1, keepdims=True))

    # Computes the average normal vector for each target point in the 
    # boundary

    for i, boundary_facets_list in enumerate(
    boundary_facets_per_target_point):

        # If this target point is in the boundary, the list of boundary 
        # facets indices attached to it is not empty

        if len(boundary_facets_list)>0:

            # Initializes a normal vector

            average_normal = np.asarray([0.0, 0.0, 0.0], dtype=
            numpy_float_type)

            # Adds the normal vector of each facet attached to this tar-
            # get point

            for facet_index in boundary_facets_list:

                average_normal += vector_x_3_boundary_facets[
                facet_index,:]

            # Divides by the length to get the average

            average_normal = (1/len(boundary_facets_list))*average_normal

            # Gets the average normal vector from this target point and 
            # modifies it in place, but makes it unitary first

            boundary_facets_per_target_point[i] = (average_normal/
            np.linalg.norm(average_normal))

    # Evaluates the batched dot product between the vectors of the boundary
    # facets

    w_11 = np.einsum('ij,ij->i', vector_x_1_boundary_facets, 
    vector_x_1_boundary_facets)

    w_22 = np.einsum('ij,ij->i', vector_x_2_boundary_facets, 
    vector_x_2_boundary_facets)

    w_12 = np.einsum('ij,ij->i', vector_x_1_boundary_facets, 
    vector_x_2_boundary_facets)

    # Computes the denominator common to both coefficient matrices v_1
    # and v_2. The resulting tensor is [n_boundary_facets]

    denominator = (w_11*w_22)-(w_12*w_12)

    # Verifies if there are degenerate triangles

    degenerate_facets = np.where(np.abs(denominator)<(
    area_tolerance_of_degenerate_triangles))[0]

    if len(degenerate_facets)>0:

        # Gets the vertices of the tetrahedron that owns the first 
        # generate facet

        ill_placed_target_points = tetrahedra_list[
        boundary_tetrahedra_indices[degenerate_facets[0]]]

        raise ValueError("There are "+str(len(degenerate_facets))+
        " degenerate triangles in the decomposition of the domain "+
        "into stochastic supports. This means that there are ill-p"+
        "laced target points or 'area_tolerance_of_degenerate_tria"+
        "ngles'="+str(area_tolerance_of_degenerate_triangles)+" is"+
        " too large fo the order of magnitude of the given mesh co"+
        "ordinates.\nThe vertices of the tetrahedron that own the "+
        "first degenerate facets are:\n"+str(
        ill_placed_target_points))

    # Computes the coefficient matrices v_1 and v_2

    v_1 = np.column_stack([w_22/denominator, -w_12/denominator])

    v_2 = np.column_stack([-w_12/denominator, w_11/denominator])

    # Batching and allocation
    
    for leftover_dof, leftover_coord in zip(outsite_domain_dofs, outside_domain_coords):

        # Distance vectors 'y' from the origin of each boundary facet to the 
        # orphan DOF

        y_vectors = leftover_coord - origin_point_boundary_facets 

        # Dot products (y * x1) and (y * x2) evaluated for all boundary facets
        # at once

        dot_y_x1 = np.einsum('ij,ij->i', y_vectors, vector_x_1_boundary_facets)

        dot_y_x2 = np.einsum('ij,ij->i', y_vectors, vector_x_2_boundary_facets)

        # Solution of the normal equations to find the barycentrc coordinates (a1, a2)

        a_1 = (v_1[:, 0]*dot_y_x1)+(v_1[:, 1]*dot_y_x2)

        a_2 = (v_2[:, 0]*dot_y_x1)+(v_2[:, 1]*dot_y_x2)

        # Validation: checks if the projection falls inside the triangle
        # bounds

        tol = -1e-6

        valid_mask = (a_1 >= tol) & (a_2 >= tol) & ((a_1 + a_2) <= (1.0 - tol))

        # Orthogonal distance from the point to the facet's plane

        projected_points = (origin_point_boundary_facets + 
        a_1[:, np.newaxis]*vector_x_1_boundary_facets +
        a_2[:, np.newaxis]*vector_x_2_boundary_facets)

        # Solve the linear problem

        distances = np.linalg.norm(leftover_coord - projected_points, axis=1)

        # Penalize facets where the projection fell outside the boundaries.
        # ~ means a boolean tool (invert True or False). np.inf defines infinity
        # the distances that are invalid as a penalization, because we want only
        # the less distance.

        distances[~valid_mask] = np.inf

        # Projection heuristics. Finds the nearest valid facet

        best_facet_idx = np.argmin(distances)

        if distances[best_facet_idx] != np.inf:

            # the point meets the criteria, map to the corresponding
            # tetrahedron

            index_tetrahedron = boundary_tetrahedra_indices[best_facet_idx]
        
        else:

            # Fallback for extreme convex corners: allocate to the
            # facet with the closest origin
            
            fallback_idx = np.argmin(np.linalg.norm(leftover_coord-origin_point_boundary_facets, axis=1))

            index_tetrahedron = boundary_tetrahedra_indices[fallback_idx]

        # Allocates the orphan DOF into the correct stochastic support

        dofs_per_tetrahedron[index_tetrahedron].append(leftover_dof)

    # Returns the updated list of DOFs per tetrahedron

    return dofs_per_tetrahedron