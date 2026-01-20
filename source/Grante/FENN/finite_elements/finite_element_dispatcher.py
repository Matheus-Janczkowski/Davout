# Routine to dispatch and instantiate finite element classes given the 
# tag read by the mesh reader

from ..finite_elements.tetrahedrons import Tetrahedron

########################################################################
#                            Domain elements                           #
########################################################################

# Defines a class with information about domain elements

class DomainElements:

    def __init__(self, nodes_coordinates, quadrature_degree, 
    element_per_field):
        
        # Defines a dictionary with domain elements. The keys are the
        # integer tags of the elements in GMSH convention. The values 
        # are the classes and other information

        self.finite_elements_classes = dict()

        self.finite_elements_classes[11] = {"class": Tetrahedron, "pol"+
        "ynomial degree": 2, "number of nodes": 10, "name": "tetrahedr"+
        "on of 10 nodes", "indices of the gmsh connectivity": [1, 2, 3,
        0, 5, 8, 7, 4, 9, 6]}

        # TODO
        # 4  - Tetrahedron of 4 nodes;
        # 29 - Tetrahedron of 20 nodes;
        # 30 - Tetrahedron of 35 nodes;
        # 5  - Hexahedron of 8 nodes;
        # 12 - Hexahedron of 27 nodes;
        # 92 - Hexahedron of 64 nodes;
        # 93 - Hexahedron of 125 nodes.

        # Saves necessary information

        self.nodes_coordinates = nodes_coordinates

        self.quadrature_degree = quadrature_degree

        self.element_per_field = element_per_field

        # Verifies if element per field has the necessary keys

        necessary_keys = ["number of DOFs per node", "required element"+
        " type"]

        for field_name, value_dict in self.element_per_field.items():

            for key in necessary_keys:

                if not (key in value_dict):

                    raise ValueError("The key '"+str(key)+"' is not in"+
                    " the dictionary of finite element information for"+
                    " the '"+str(field_name)+"' field. The obligatory "+
                    "keys are:\n"+str(necessary_keys))

        # Initializes a dictionary of dispatched finite elements. The
        # keys are the names of the fields, whose values are dictionaries
        # themselves. The keys of the inner dictionaries are physical 
        # groups tags, and the respective values are instances of element
        # classes

        self.elements_dictionaries = {field_name: {} for field_name in (
        self.element_per_field.keys())}

        # Initializes a DOFs counter

        self.dofs_counter = 0

    # Defines a function to instantiate the finite element class

    def dispatch_element(self, tag, connectivities, physical_group_tag):

        # Iterates through the fields to add an empty list with a sublist
        # for each dimension (each DOF in the node)

        for field_name, info_dict in self.element_per_field.items():

            # Gets the base list of degrees of freedom per element

            n_dofs_per_node = info_dict["number of DOFs per node"]

            # Initializes a set of node indexes that are really used by
            # this field

            used_nodes_set = set()

            # Gets the type of the element required by the field

            required_element_type = info_dict["required element type"]

            # If the type is not an integer tries to recover the corres-
            # ponding integer type

            if not isinstance(required_element_type, int):

                for type, info in self.finite_elements_classes.items():

                    if info["name"]==required_element_type:

                        required_element_type = type 

                        break 

            # Gets the element dictionary of information

            element_info = self.get_element(required_element_type)

            # And adds a list of nodes for each element

            nodes_in_elements = []

            # Initializes a list of lists. Each sublist corresponds to a 
            # finite element. Each sublist contains n other sublists, 
            # where n is the number of nodes in each element. The inner-
            # most sublists contain the nodes coordinates for that ele-
            # ment

            element_nodes_coordinates = []

            # Iterates through the connectivities

            for connectivity in connectivities:

                # Appends another sublist corresponding to the element

                element_nodes_coordinates.append([])

                # Initializes a list of nodes per element

                nodes_in_elements.append([])

                # Verifies if the element has enough nodes

                if len(connectivity)<len(element_info["indices of the "+
                "gmsh connectivity"]):
                    
                    received_element_name = str(tag)

                    if tag in self.finite_elements_classes:

                        received_element_name = str(
                        self.finite_elements_classes[tag]["name"])
                    
                    raise IndexError("The element recovered from the m"+
                    "esh has a connectivity of "+str(connectivity)+". "+
                    "This connectivity list has "+str(len(connectivity)
                    )+" nodes; but "+str(len(element_info["indices of "+
                    "the gmsh connectivity"]))+" nodes are required by"+
                    " element type '"+str(element_info["name"])+"'. Th"+
                    "e generated mesh has a '"+received_element_name+
                    "' element")

                # Iterates through the nodes indices in the list of con-
                # nectivity

                for connectivity_real_index in element_info["indices o"+
                "f the gmsh connectivity"]:
                    
                    # Gets the node index

                    node_index = connectivity[connectivity_real_index]

                    # Gets the node coordinates and adds them to the list 
                    # of element nodes coordinates

                    element_nodes_coordinates[-1].append(
                    self.nodes_coordinates[node_index])

                    # Adds the node index to the set of used nodes

                    used_nodes_set.add(node_index)

                    # Adds the node index to the list of nodes in the 
                    # element

                    nodes_in_elements[-1].append(node_index)

            # Sorts the set of used nodes

            used_nodes_set = sorted(used_nodes_set)

            # Gets a dictionary of DOFs per node

            dofs_node_dict = dict()

            max_dof_number = 0

            for set_index in range(len(used_nodes_set)):

                # Adds a list of the DOFs for this node

                dofs_node_dict[used_nodes_set[set_index]] = []

                # And iterates through the local numbers of DOFs
                
                for local_dof in range(n_dofs_per_node):

                    DOF_number = ((set_index*n_dofs_per_node)+local_dof+
                    self.dofs_counter)
                    
                    dofs_node_dict[used_nodes_set[set_index]].append(
                    DOF_number)

                    # Updates the maximum DOF number

                    max_dof_number = max(DOF_number, max_dof_number)

            # Uses the dictionary of nodes to DOFs to create a list of
            # elements with nested lists for nodes, which, in turn, have
            # nested lists for dimensions (local DOFs)

            for element_index in range(len(nodes_in_elements)):

                for node_index in range(len(nodes_in_elements[
                element_index])):
                    
                    nodes_in_elements[element_index][node_index] = (
                    dofs_node_dict[nodes_in_elements[element_index][
                    node_index]])

            # Updates the DOFs couter

            self.dofs_counter = max_dof_number+1

            # Instantiates the finite element class

            self.elements_dictionaries[field_name][physical_group_tag
            ] = element_info["class"](element_nodes_coordinates, 
            nodes_in_elements, polynomial_degree=element_info["polynom"+
            "ial degree"], quadrature_degree=self.quadrature_degree)

    # Defines a function to verify is the element type tag is available

    def get_element(self, tag):

        # Verifies if this tag is one of the implemented elements

        if not (tag in self.finite_elements_classes):

            elements_list = ""

            for tag, element_info in self.finite_elements_classes.items():

                elements_list += "\ntag: "+str(tag)+"; name: "+str(
                element_info["name"])

            raise NotImplementedError("The element tag '"+str(tag)+"' "+
            "has not been implemented yet for the domain. Check out th"+
            "e available elements tags and their names:"+elements_list)
        
        # Gets the element info

        return self.finite_elements_classes[tag]

# Defines a function to receive the dictionary of domain connectivities
# and to dispatch the respective finite element classes

def dispatch_domain_elements(mesh_data_class, element_per_field):

    # Instantiates the class with the dictionary of finite elements 
    # classes

    domain_finite_elements = DomainElements(
    mesh_data_class.nodes_coordinates, mesh_data_class.quadrature_degree,
    element_per_field)

    # Iterates through the physical groups

    for physical_group_tag, finite_elements_dict in (
    mesh_data_class.domain_connectivities.items()):

        # TODO: just a single type of element is allowed now

        if len(list(finite_elements_dict.keys()))>1:

            raise NotImplementedError("More than one finite element ty"+
            "pe has been asked for the domain: "+str(list(
            finite_elements_dict.keys()))+". Currently, a single type "+
            "of element is allowed in the domain.")
        
        # Iterates through the finite element types

        for element_type, connectivities in finite_elements_dict.items():

            # Verifies if this element is in the class of finite ele-
            # ments and dispatches the class of this element

            domain_finite_elements.dispatch_element(element_type, 
            connectivities, physical_group_tag)

    # Returns the class of finite elements for the domain

    return domain_finite_elements

########################################################################
#                           Boundary elements                          #
########################################################################

# Defines a class with information about domain elements

class BoundaryElements:

    def __init__(self, nodes_coordinates, quadrature_degree):
        
        # Defines a dictionary with boundary elements. The keys are the
        # integer tags of the elements in GMSH convention. The values 
        # are the classes and other information

        self.finite_elements_classes = dict()

        self.finite_elements_classes[9] = {"class": "Triangle", "pol"+
        "ynomial degree": 2, "number of nodes": 6, "name": "triangle o"+
        "f 6 nodes"}

        # TODO
        # 2  - triangle of 3 nodes;
        # 9  - triangle of 6 nodes;
        # 21 - triangle of 10 nodes;
        # 23 - triangle of 15 nodes;
        # 3  - quadrilateral of 4 nodes;
        # 10 - quadrilateral of 9 nodes;
        # 36 - quadrilateral of 16 nodes;
        # 37 - quadrilateral of 25 nodes.

        # Saves necessary information

        self.nodes_coordinates = nodes_coordinates

        self.quadrature_degree = quadrature_degree

        # Initializes a dictionary of dispatched finite elements. The
        # keys are the physical groups tags

        self.physical_groups_elements = dict()

    # Defines a function to instantiate the finite element class

    def dispatch_element(self, tag, connectivities, physical_group_tag):

        # Gets the element dictionary of information

        element_info = self.get_element(tag)

        # Initializes a list of lists. Each sublist corresponds to a fi-
        # nite element. Each sublist contains n other sublists, where n
        # is the number of nodes in each element. The inner-most sublists
        # contain the nodes coordinates for that element

        element_nodes_coordinates = []

        # Iterates through the connectivities

        for connectivity in connectivities:

            # Appends another sublist corresponding to the element

            element_nodes_coordinates.append([])

            # Iterates through the nodes indices in the list of connec-
            # tivity

            for node_index in connectivity:

                # Gets the node coordinates and adds them to the list of
                # element nodes coordinates

                element_nodes_coordinates[-1].append(
                self.nodes_coordinates[node_index])

        # Dispatch the class

        self.physical_groups_elements[physical_group_tag] = element_info[
        "class"](element_nodes_coordinates, polynomial_degree=
        element_info["polynomial degree"], quadrature_degree=
        self.quadrature_degree)

    # Defines a function to verify is the element type tag is available

    def get_element(self, tag):

        # Verifies if this tag is one of the implemented elements

        if not (tag in self.finite_elements_classes):

            elements_list = ""

            for tag, element_info in self.finite_elements_classes.items():

                elements_list += "\ntag: "+str(tag)+"; name: "+str(
                element_info["name"])

            raise NotImplementedError("The element tag '"+str(tag)+"' "+
            "has not been implemented yet for the boundary. Check out "+
            "the available elements tags and their names:"+elements_list)
        
        # Gets the element info

        return self.finite_elements_classes[tag]