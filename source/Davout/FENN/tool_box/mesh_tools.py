# Routine to store methods to read meshes generated in GMSH 

import tensorflow as tf

import numpy as np

from time import time

from ...PythonicUtilities.path_tools import verify_path, verify_file_existence, get_parent_path_of_file, take_outFileNameTermination

from ...PythonicUtilities.string_tools import string_toList

from ..finite_elements.finite_element_dispatcher import dispatch_region_elements

# Defines a class to inform mesh data

class MshMeshData:

    def __init__(self, nodes_coordinates, domain_physicalGroupsNameToTag,
    boundary_physicalGroupsNameToTag, domain_connectivities, 
    boundary_connectivities, quadrature_degree, dtype, integer_dtype,
    global_number_dofs=None, domain_elements=None, boundary_elements=
    None, dofs_node_dict=None):
    
        self.nodes_coordinates = nodes_coordinates

        self.domain_physicalGroupsNameToTag = domain_physicalGroupsNameToTag
        
        self.boundary_physicalGroupsNameToTag = boundary_physicalGroupsNameToTag

        self.domain_connectivities = domain_connectivities
        
        self.boundary_connectivities = boundary_connectivities

        self.quadrature_degree = quadrature_degree

        self.dtype = dtype

        self.integer_dtype = integer_dtype

        self.global_number_dofs = global_number_dofs

        self.domain_elements = domain_elements 

        self.boundary_elements = boundary_elements

        self.dofs_node_dict = dofs_node_dict

########################################################################
#                             Mesh reading                             #
########################################################################

# Defines a class with information about gmsh versions and the start-end
# keys within the file

class GmshVersions:

    def __init__(self):
        
        self.versions = dict()

        # Adds information about version '2.2 0 8'

        self.versions["2.2 0 8"] = {"physical groups": ["$PhysicalNames", 
        "$EndPhysicalNames"], "nodes": ["$Nodes", "$EndNodes"], "eleme"+
        "nts": ["$Elements", "$EndElements"]}

# Defines a function to read a mesh .msh or a list of them

def read_msh_mesh(file_name, quadrature_degree, elements_per_field, 
parent_directory=None, verbose=False, dtype=tf.float32, integer_dtype=
tf.int32):
    
    # Verifies if file name is a list

    if isinstance(file_name, list):

        # Initializes a list of instances of the mesh data class

        meshes_list = []

        # Iterates through the list of file names

        for file in file_name:
    
            # Calls the reader function and appends the mesh data class 
            # instance

            meshes_list.append(reader_msh_mesh(file, quadrature_degree, 
            elements_per_field, parent_directory=parent_directory, 
            verbose=verbose, dtype=dtype, integer_dtype=integer_dtype))

        # Returns the list

        return meshes_list
    
    # Otherwise, reads the single mesh

    else:

        return reader_msh_mesh(file_name, quadrature_degree, 
        elements_per_field, parent_directory=parent_directory, verbose=
        verbose, dtype=dtype, integer_dtype=integer_dtype)

# Defines a function to read a single mesh .msh

def reader_msh_mesh(file_name, quadrature_degree, elements_per_field, 
parent_directory=None, verbose=False, dtype=tf.float32, integer_dtype=
tf.int32):

    # If the parent directory is None, get the parent path of the file 
    # where this function has been called

    if parent_directory is None:

        parent_directory = get_parent_path_of_file(
        function_calls_to_retrocede=3)

    # Takes out the file termination, and adds the correct .msh

    file_name = take_outFileNameTermination(file_name)+".msh"

    # Verifies the path

    file_name = verify_path(parent_directory, file_name)

    # Reads the msh file into a list of strings. Each element is a line

    start_time = time()

    lines_list = []

    try:

        with open(file_name, "r") as infile:

            for line in infile:

                lines_list.append(line.strip())

    except:

        raise FileNotFoundError("The file "+file_name+" was not found "+
        "while evaluating trying to read a msh mesh.")

    lines_reading_time = time()-start_time
    
    # Gets the gmsh version with which the file was written

    gmsh_version, start_reading_at_index = read_gmsh_version(lines_list, 
    0)

    # Instantiates the class of gmsh versions and verifies if the captu-
    # red version is one of the allowed versions

    version_info = GmshVersions().versions

    if gmsh_version in version_info:

        # Captures only this version information

        version_info = version_info[gmsh_version]

    else:

        raise NameError("The captured gmsh version is '"+str(gmsh_version
        )+"', but it is not one of the allowed versions. See a list of"+
        " the available versions:\n"+str(list(version_info.keys())))

    # Reads the physical groups. Gets the dictionaries of physical groups
    # of names to tags

    start_time = time()

    (domain_physicalGroupsNameToTag, 
    boundary_physicalGroupsNameToTag, start_reading_at_index
    ) = read_physical_groups(lines_list, start_reading_at_index, 
    start_key=version_info["physical groups"][0], end_key=version_info[
    "physical groups"][1])

    physical_groups_reading_time = time()-start_time

    # Reads the node coordinates

    start_time = time()

    nodes_coordinates, start_reading_at_index = read_nodes(lines_list,
    start_reading_at_index, start_key=version_info["nodes"][0], end_key=
    version_info["nodes"][1])

    nodes_reading_time = time()-start_time
    
    # Reads the finite elements connectivities

    start_time = time()

    (domain_connectivities, boundary_connectivities, 
    start_reading_at_index) = read_elements(lines_list, 
    start_reading_at_index, list(domain_physicalGroupsNameToTag.values()
    ), list(boundary_physicalGroupsNameToTag.values()), start_key=
    version_info["elements"][0], end_key=version_info["elements"][1])

    connectivities_reading_time = time()-start_time

    # Instantiates the class of mesh data and returns it

    mesh_data_class = MshMeshData(nodes_coordinates, 
    domain_physicalGroupsNameToTag, boundary_physicalGroupsNameToTag, 
    domain_connectivities, boundary_connectivities, quadrature_degree,
    dtype, integer_dtype)

    # Dispatches the elements of the domain. Generates a dictionary of
    # physical group tag whose values are dictionaries of element types

    start_time = time()

    mesh_data_class = dispatch_region_elements(mesh_data_class,
    elements_per_field, dtype, "domain", integer_dtype)

    domain_dispatching_reading_time = time()-start_time

    # Dispatches the elements of the boundary in the same way

    start_time = time()

    mesh_data_class = dispatch_region_elements(mesh_data_class,
    elements_per_field, dtype, "boundary", integer_dtype)

    boundary_dispatching_reading_time = time()-start_time

    if verbose:

        print("#######################################################"+
        "#################\n#                             Elapsed time"+
        "                             #\n#############################"+
        "###########################################\n")

        print("Reading the msh file took:                 "+str(
        lines_reading_time))

        print("Reading physical groups took:              "+str(
        physical_groups_reading_time)+" s")

        print("Reading nodes took:                        "+str(
        nodes_reading_time)+" s")

        print("Reading connectivitites took:              "+str(
        connectivities_reading_time)+" s")

        print("Dispatching domain finite elements took:   "+str(
        domain_dispatching_reading_time)+" s")

        print("Dispatching boundary finite elements took: "+str(
        boundary_dispatching_reading_time)+" s\n")

        print("#######################################################"+
        "#################\n#                        Domain physical g"+
        "roups                        #\n#############################"+
        "###########################################\n")

        for name, tag in domain_physicalGroupsNameToTag.items():

            print("Domain physical group name: "+str(name)+"; tag: "+str(
            tag))

            for field, dictionary in mesh_data_class.domain_elements.items():

                print("    field '"+str(field)+"' has "+str(dictionary[
                tag].number_elements)+" finite elements")

            print("")

        print("\n#####################################################"+
        "###################\n#                       Boundary physica"+
        "l groups                       #\n###########################"+
        "#############################################\n")

        for name, tag in boundary_physicalGroupsNameToTag.items():

            print("Boundary physical group name: "+str(name)+"; tag: "+
            str(tag))

            for field, dictionary in mesh_data_class.boundary_elements.items():

                print("    field '"+str(field)+"' has "+str(dictionary[
                tag].number_elements)+" finite elements")

            print("")

    return mesh_data_class

# Defines a function to read the bit about the Gmsh output file version

def read_gmsh_version(lines_list, start_reading_at_index, start_key=
"$MeshFormat", end_key="$EndMeshFormat"):

    # Initializes a variable with the gmsh version 

    gmsh_version = None

    # Initializes a flag to tell if reading is allowed already

    flag_reading = False

    # Iterates through the lines

    for i in range(start_reading_at_index, len(lines_list)):

        # Gets the line

        line = lines_list[i]

        # Verifies if reading is not allowed yet

        if not flag_reading:

            # Verifies if this line is equal to the start key

            if line==start_key:

                # Updates the flag reading to allow for reading

                flag_reading = True

            continue

        # Verifies if the end key has been reached

        elif line==end_key:

            # Modifies the start index for the next index

            start_reading_at_index = i+1

            break 

        gmsh_version = line

    return gmsh_version, start_reading_at_index
    
# Defines a function to read the bit about physical groups. The output is
# a dictionary of domain physical groups names to tags and another to 
# boundary information

def read_physical_groups(lines_list, start_reading_at_index, start_key=
"$PhysicalNames", end_key="$EndPhysicalNames"):

    # Initializes a dictionary where the keys corresponds to the topolo-
    # gycal dimension of the physical group and the values are dictiona-
    # ry themselves. The dictionaries as values store key-value pairs of
    # physical groups names to the respective physical groups tags

    physical_groups_dicts = {}

    # Initializes a flag to tell if reading is allowed already

    flag_reading = False

    # Iterates through the lines

    for i in range(start_reading_at_index, len(lines_list)):

        # Gets the line

        line = lines_list[i]

        # Verifies if reading is not allowed yet

        if not flag_reading:

            # Verifies if this line is equal to the start key

            if line==start_key:

                # Updates the flag reading to allow for reading

                flag_reading = True

            continue

        # Verifies if the end key has been reached

        elif line==end_key:

            # Modifies the start index for the next index

            start_reading_at_index = i+1

            break 

        # Iterates through the line looking for '"", that tell a physi-
        # cal group name

        line_numerical_info = ""

        physical_group_name = ""

        for j in range(len(line)):

            if line[j]=='"':

                physical_group_name = line[(j+1):(len(line)-1)]

                line_numerical_info = line[0:(j-1)]

                break 

        # Transforms this line to a list to get the different informa-
        # tion readily available, and independently

        line_info = string_toList("["+line_numerical_info+"]", 
        element_separator=" ")

        # If there are two elements in the list, the first is the physi-
        # cal group topological dimension, whereas the second one is the
        # physical group tag

        if len(line_info)==2:

            # Verifies if this topological dimension has already been
            # registered

            if line_info[0] in physical_groups_dicts:

                # Updates the dictionary of physical groups

                physical_groups_dicts[line_info[0]][physical_group_name
                ] = line_info[1]

            # Otherwise, creates the inner dictionary

            else:

                physical_groups_dicts[line_info[0]] = {
                physical_group_name: line_info[1]}

    # Verifies the keys of the greater dictionary

    keys = list(physical_groups_dicts.keys())

    keys.sort()

    # If there are not two keys throws an error

    if len(keys)!=2:

        raise ValueError("The number of categories of physical groups "+
        "using the topological dimension as criterion is "+str(len(keys)
        )+". There should be 2: one for the domain and another for the"+
        " boundary")
    
    # Separates the domain from the surface physical groups using the 
    # topological dimension

    boundary_physicalGroupsNameToTag = physical_groups_dicts[keys[0]]

    domain_physicalGroupsNameToTag = physical_groups_dicts[keys[1]]

    # Returns the dictionaries and the new index to start reading

    return (domain_physicalGroupsNameToTag, 
    boundary_physicalGroupsNameToTag, start_reading_at_index)
    
# Defines a function to read the bit about nodes. The output is a list of
# lists, where each list corresponds to a node with that index and the 
# components are the coordinates

def read_nodes(lines_list, start_reading_at_index, start_key="$Nodes", 
end_key="$EndNodes"):

    # Initializes a list of nodes

    nodes_coordinates = None

    node_counter = 0

    # Initializes a flag to tell if reading is allowed already

    flag_reading = False

    # Iterates through the lines

    for i in range(start_reading_at_index, len(lines_list)):

        # Gets the line

        line = lines_list[i]

        # Verifies if reading is not allowed yet

        if not flag_reading:

            # Verifies if this line is equal to the start key

            if line==start_key:

                # Updates the flag reading to allow for reading

                flag_reading = True

            continue

        # Verifies if the end key has been reached

        elif line==end_key:

            # Modifies the start index for the next index

            start_reading_at_index = i+1

            break 

        # Transforms this line to a list to get the different informa-
        # tion readily available, and independently

        line_info = string_toList("["+line+"]", element_separator=" ")

        # If there are four elements in the list, the first is the node
        # index; the second, the third, and the fourth elements are the
        # coordinates

        if len(line_info)>1:

            # Adds the node coordinates, skipping the first element (
            # which is node index)

            nodes_coordinates[node_counter, :] = np.asarray(line_info[1:
            len(line_info)], dtype=float)

            node_counter += 1

        # Otherwise it is the number of nodes

        elif len(line_info)==1:

            nodes_coordinates = np.zeros((line_info[0], 3))

    return nodes_coordinates, start_reading_at_index
    
# Defines a function to read the bit about finite element connectivity. 
# The output is a list of lists, where each list corresponds to a finite
# element with that index, and the components are the node numbers (sub-
# tracts one off the number given by Gmsh to compatibilize with python
# indexing)

def read_elements(lines_list, start_reading_at_index, 
domain_physical_groups_tags, boundary_physical_groups_tags, start_key=
"$Elements", end_key="$EndElements"):

    # Initializes a dictionary whose keys are the physical groups. The 
    # values are dictionaries whose keys are the element type, and the 
    # values are the element connectivities

    connectivity_dict = dict()

    # Initializes a flag to tell if reading is allowed already

    flag_reading = False

    # Iterates through the lines

    for i in range(start_reading_at_index, len(lines_list)):

        # Gets the line

        line = lines_list[i]

        # Verifies if reading is not allowed yet

        if not flag_reading:

            # Verifies if this line is equal to the start key

            if line==start_key:

                # Updates the flag reading to allow for reading

                flag_reading = True

            continue

        # Verifies if the end key has been reached

        elif line==end_key:

            # Modifies the start index for the next index

            start_reading_at_index = i+1

            break 

        # Transforms this line to a list to get the different informa-
        # tion readily available, and independently

        line_info = string_toList("["+line+"]", element_separator=" ")

        # If there are more than one element in the list, it is a valid
        # finite element. The first component is the element index; the
        # second one represents the element type; the third represents
        # the number of tags for the physical groups; the fourth compo-
        # nent gives the physical group tag given by the user; the fifth
        # one is the tag given by gmsh; the sixth onwards are nodes in-
        # dices

        if len(line_info)>1:

            # Gets the element type, and the physical group tag

            element_type = line_info[1]

            physical_group_tag = line_info[3]

            nodes = [(node_index-1) for node_index in line_info[5:len(
            line_info)]]

            # Verifies if this physical group tag has already been regis-
            # tered

            if physical_group_tag in connectivity_dict:

                # Verifies if this element type has already been regis-
                # tered

                if element_type in connectivity_dict[physical_group_tag]:

                    # Appends the connectivity

                    connectivity_dict[physical_group_tag][element_type
                    ].append(nodes)

                else:

                    connectivity_dict[physical_group_tag][element_type
                    ] = [nodes]

            else:

                # Registers the tag

                connectivity_dict[physical_group_tag] = {element_type:
                [nodes]}

    # Separates the connectivity dictionaries into domain and boundary

    domain_connectivity = dict()

    boundary_connectivity = dict()

    for physical_group_tag, connectivities in connectivity_dict.items():

        # Verifies if the physical group tag is one of the domain's

        if physical_group_tag in domain_physical_groups_tags:

            domain_connectivity[physical_group_tag] = connectivities

        # Verifies if the physical group tag is one of the boundary's

        elif physical_group_tag in boundary_physical_groups_tags:

            boundary_connectivity[physical_group_tag] = connectivities

        else:

            raise NameError("Physical group tag '"+str(physical_group_tag
            )+"' is not a valid physical group tag. Check out the avai"+
            "lable tags for domain:\n"+str(domain_physical_groups_tags)+
            "\nand for boundary:\n"+str(boundary_physical_groups_tags))

    return (domain_connectivity, boundary_connectivity, 
    start_reading_at_index)

########################################################################
#                          Mesh normal vectors                         #
########################################################################