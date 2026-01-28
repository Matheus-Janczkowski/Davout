# Routine to dispatch and instantiate finite element classes given the 
# tag read by the mesh reader

from ..finite_elements import volume_elements

from ..finite_elements import surface_elements

from ...PythonicUtilities.package_tools import load_classes_from_package

########################################################################
#                            Domain elements                           #
########################################################################

# Defines a class with information about domain elements

class DomainElements:

    def __init__(self, nodes_coordinates, quadrature_degree, 
    element_per_field):
        
        # Initializes the dictionary of DOFs per node. This dictionary
        # will have field names as keys, whose values will have other
        # dictionaries with node-DOFs pairing

        self.dofs_node_dict = {}
        
        # Defines a dictionary with domain elements. The keys are the
        # integer tags of the elements in GMSH convention. The values 
        # are the classes and other information. Automatically retrieves
        # the finite element classes from the volume_elements and
        # surface_elements packages

        self.finite_elements_classes = automatic_import_finite_element_classes()

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
                    " the dictionary of domain finite element informat"+
                    "ion for the '"+str(field_name)+"' field. The obli"+
                    "gatory keys are:\n"+str(necessary_keys))

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

    def dispatch_element(self, tag, connectivities, physical_group_tag,
    dtype):

        # Calls the template function to perform this

        (self.elements_dictionaries, self.dofs_counter, 
        self.dofs_node_dict) = dispatch_element_template(tag, connectivities, 
        physical_group_tag, dtype, self.element_per_field, 
        self.finite_elements_classes, self.nodes_coordinates, 
        self.quadrature_degree, self.elements_dictionaries, "domain", 
        self.dofs_counter, dofs_node_dict=self.dofs_node_dict,
        flag_building_dofs_dict=True)

########################################################################
#                           Boundary elements                          #
########################################################################

# Defines a class with information about domain elements

class BoundaryElements:

    def __init__(self, nodes_coordinates, quadrature_degree, 
    element_per_field, dofs_node_dict):
        
        # Saves the dictionary of DOFs per node since the DOFs were al-
        # ready mapped with the domain mesh

        self.dofs_node_dict = dofs_node_dict

        # Initializes a dictionary of dispatched finite elements. The
        # keys are the physical groups tags

        self.physical_groups_elements = dict()
        
        # Defines a dictionary with boundary elements. The keys are the
        # integer tags of the elements in GMSH convention. The values 
        # are the classes and other information. Automatically retrieves
        # the finite element classes from the volume_elements and
        # surface_elements packages. Also gets a dictionary that pairs
        # the element tags to their suitable boundary elements

        (self.finite_elements_classes, self.suitable_boundary_elements
        ) = automatic_import_finite_element_classes(
        get_suitable_boundary_elements=True)

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
                    " the dictionary of boundary finite element inform"+
                    "ation for the '"+str(field_name)+"' field. The ob"+
                    "ligatory keys are:\n"+str(necessary_keys))

        # Initializes a dictionary of dispatched finite elements. The
        # keys are the names of the fields, whose values are dictionaries
        # themselves. The keys of the inner dictionaries are physical 
        # groups tags, and the respective values are instances of element
        # classes

        self.elements_dictionaries = {field_name: {} for field_name in (
        self.element_per_field.keys())}

    # Defines a function to dispatch each element class 

    def dispatch_element(self, tag, connectivities, physical_group_tag,
    dtype):

        # Calls the template function to perform this

        (self.elements_dictionaries, self.dofs_counter, _) = dispatch_element_template(tag, connectivities, 
        physical_group_tag, dtype, self.element_per_field, 
        self.finite_elements_classes, self.nodes_coordinates, 
        self.quadrature_degree, self.elements_dictionaries, "boundary", 
        0, self.dofs_node_dict, flag_building_dofs_dict=False, 
        suitable_boundary_elements=self.suitable_boundary_elements)

########################################################################
#              Element dispatching and class instantiation             #
########################################################################

# Defines a template of function to instantiate the finite element class

def dispatch_element_template(tag, connectivities, physical_group_tag,
dtype, element_per_field, finite_elements_classes, nodes_coordinates,
quadrature_degree, elements_dictionaries, region_name, dofs_counter, 
dofs_node_dict, flag_building_dofs_dict=True, suitable_boundary_elements=
None):

    # Iterates through the fields to add an empty list with a sublist
    # for each dimension (each DOF in the node)

    for field_name, info_dict in element_per_field.items():

        # Gets the base list of degrees of freedom per element

        n_dofs_per_node = info_dict["number of DOFs per node"]

        # Initializes a set of node indexes that are really used by this
        # field

        used_nodes_set = set()

        # Gets the type of the element required by the field

        required_element_type = info_dict["required element type"]

        # If the type is not an integer tries to recover the correspon-
        # ding integer type

        if not isinstance(required_element_type, int):

            for given_type, info in finite_elements_classes.items():

                if info.stored_elements[given_type]["name"]==required_element_type:

                    required_element_type = given_type 

                    break 

        # If the region is the boundary, the suitable element in the 
        # boundary must be taken

        if region_name=="boundary":

            required_element_type = suitable_boundary_elements[
            required_element_type]

        # Gets the element dictionary of information

        element_class, element_info = get_element(required_element_type, 
        finite_elements_classes, region_name)

        # And adds a list of nodes for each element

        nodes_in_elements = []

        # Initializes a list of lists. Each sublist corresponds to a fi-
        # nite element. Each sublist contains n other sublists, where n 
        # is the number of nodes in each element. The inner-most sublists 
        # contain the nodes coordinates for that element

        element_nodes_coordinates = []

        # Iterates through the connectivities

        for connectivity in connectivities:

            # Appends another sublist corresponding to the element

            element_nodes_coordinates.append([])

            # Initializes a list of nodes per element

            nodes_in_elements.append([])

            # Verifies if the element has enough nodes

            if len(connectivity)<len(element_info["indices of the gmsh"+
            " connectivity"]):
                
                received_element_name = str(tag)

                if tag in finite_elements_classes:

                    received_element_name = str(finite_elements_classes[
                    tag].stored_elements[tag]["name"])
                
                raise IndexError("The element recovered from the mesh "+
                "has a connectivity of "+str(connectivity)+". This con"+
                "nectivity list has "+str(len(connectivity))+" nodes; "+
                "but "+str(len(element_info["indices of the gmsh conne"+
                "ctivity"]))+" nodes are required by element type '"+str(
                element_info["name"])+"'. The generated mesh has a '"+
                received_element_name+"' element. This happened at the"+
                " '"+str(region_name)+"' region")

            # Iterates through the nodes indices in the list of connec-
            # tivity

            for connectivity_real_index in element_info["indices of th"+
            "e gmsh connectivity"]:
                
                # Gets the node index

                node_index = connectivity[connectivity_real_index]

                # Gets the node coordinates and adds them to the list of
                # element nodes coordinates

                element_nodes_coordinates[-1].append(
                nodes_coordinates[node_index])

                # Adds the node index to the set of used nodes

                used_nodes_set.add(node_index)

                # Adds the node index to the list of nodes in the element

                nodes_in_elements[-1].append(node_index)

        # If the dictionary of DOFs per node is to be updated

        if flag_building_dofs_dict:

            # Verifies if there is a key for this field

            if not (field_name in dofs_node_dict):

                dofs_node_dict[field_name] = {}

            # Sorts the set of used nodes

            used_nodes_set = sorted(used_nodes_set)

            max_dof_number = 0

            for set_index in range(len(used_nodes_set)):

                # Adds a list of the DOFs for this node

                dofs_node_dict[field_name][used_nodes_set[set_index]] = []

                # And iterates through the local numbers of DOFs
                
                for local_dof in range(n_dofs_per_node):

                    DOF_number = ((set_index*n_dofs_per_node)+local_dof+
                    dofs_counter)
                    
                    dofs_node_dict[field_name][used_nodes_set[set_index]
                    ].append(DOF_number)

                    # Updates the maximum DOF number

                    max_dof_number = max(DOF_number, max_dof_number)

            # Updates the DOFs couter

            dofs_counter = max_dof_number+1

        # Uses the dictionary of nodes to DOFs to create a list of ele-
        # ments with nested lists for nodes, which, in turn, have nested
        # lists for dimensions (local DOFs)

        for element_index in range(len(nodes_in_elements)):

            for node_index in range(len(nodes_in_elements[
            element_index])):
                
                nodes_in_elements[element_index][node_index] = (
                dofs_node_dict[field_name][nodes_in_elements[
                element_index][node_index]])

        # Instantiates the finite element class

        elements_dictionaries[field_name][physical_group_tag
        ] = element_class(element_nodes_coordinates,  nodes_in_elements, 
        polynomial_degree=element_info["polynomial degree"], 
        quadrature_degree=quadrature_degree, dtype=dtype)

    return elements_dictionaries, dofs_counter, dofs_node_dict

########################################################################
#                             Verification                             #
########################################################################

# Defines a function to verify is the element type tag is available

def get_element(tag, finite_elements_classes, region):

    # Verifies if this tag is one of the implemented elements

    if not (tag in finite_elements_classes):

        elements_list = ""

        for tag, element_info in finite_elements_classes.items():

            elements_list += "\ntag: "+str(tag)+"; name: "+str(
            element_info.stored_elements[tag]["name"])

        raise NotImplementedError("The element tag '"+str(tag)+"' has "+
        "not been implemented yet for the "+str(region)+". Check out t"+
        "he available elements tags and their names:"+elements_list)
    
    # Gets the element info

    return finite_elements_classes[tag], finite_elements_classes[tag
    ].stored_elements[tag]

########################################################################
#                      Automatic element importing                     #
########################################################################

# Defines a function to import elements automatically from the volume 
# and surface elements packages. Then, it creates a dictionary of finite
# elements classes, where the keys are the elements GMSH integer tag and
# the respective values are the classes objects

def automatic_import_finite_element_classes(
get_suitable_boundary_elements=False):

    # Imports the classes of the volumetric finite elements

    classes_list = load_classes_from_package(volume_elements, 
    necessary_attributes=["stored_elements"])

    # Imports the classes of the surface finite elements

    classes_list = load_classes_from_package(surface_elements, 
    necessary_attributes=["stored_elements"], classes_list=
    classes_list)

    # If a dictionary of suitable boundary elements is to be constructed,
    # initializes a dictionary whose keys are gmsh integer tags, and the
    # values are gmsh integer tags of the elements that form a suitable
    # boundary to the latter ones

    suitable_boundary_elements = None 

    if get_suitable_boundary_elements:

        suitable_boundary_elements = dict()

    # Constructs the dictionary 

    finite_elements_dictionary = dict()

    for class_object in classes_list:

        # Iterates through the dictionary of stored elements

        for element_gmsh_tag, element_info in (
        class_object.stored_elements.items()):
            
            # Adds to the dictionary of finite elements

            finite_elements_dictionary[element_gmsh_tag] = class_object

            # If the suitable boundary element is to be registered

            if get_suitable_boundary_elements and ("suitable boundary "+
            "element type tag" in element_info):
                
                suitable_boundary_elements[element_gmsh_tag] = (
                element_info["suitable boundary element type tag"])

    # Returns the dictionary of finite elements

    if get_suitable_boundary_elements:

        return finite_elements_dictionary, suitable_boundary_elements

    return finite_elements_dictionary