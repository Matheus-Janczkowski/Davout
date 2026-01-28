# Routine to dispatch and instantiate finite element classes given the 
# tag read by the mesh reader

from ..finite_elements.elements_manager import DomainElements, BoundaryElements

# Defines a function to receive the mesh data class, the dictionary of
# element information per field, and the tensorflow type. This function
# outputs the updated mesh data class

def dispatch_region_elements(mesh_data_class, element_per_field, dtype,
region):

    # Instantiates the class with the dictionary of finite elements 
    # classes

    region_finite_elements = None

    connectivities_name = None

    elements_attribute_name = None

    if region=="domain":

        region_finite_elements = DomainElements(
        mesh_data_class.nodes_coordinates, 
        mesh_data_class.quadrature_degree, element_per_field)

        # Updates the name of the connectivities inside the domain class

        connectivities_name = "domain_connectivities"

        # Updates the name of the attribute for the dispatched element 
        # classes in the mesh data class

        elements_attribute_name = "domain_elements"

    elif region=="boundary":

        region_finite_elements = BoundaryElements(
        mesh_data_class.nodes_coordinates, 
        mesh_data_class.quadrature_degree, element_per_field,
        mesh_data_class.dofs_node_dict)

        # Updates the name of the connectivities inside the domain class

        connectivities_name = "boundary_connectivities"

        # Updates the name of the attribute for the dispatched element 
        # classes in the mesh data class

        elements_attribute_name = "boundary_elements"

    else: 

        raise KeyError("'dispatch_region_elements' can only dispatch a"+
        "region named 'domain' or 'boundary'. This, however, has nothi"+
        "ng to do with the names of the physical groups.")

    # Iterates through the physical groups

    for physical_group_tag, finite_elements_dict in (getattr(
    mesh_data_class, connectivities_name).items()):

        # TODO: just a single type of element is allowed now

        if len(list(finite_elements_dict.keys()))>1:

            raise NotImplementedError("More than one finite element ty"+
            "pe has been asked for the "+str(region)+": "+str(list(
            finite_elements_dict.keys()))+". Currently, a single type "+
            "of element is allowed in the "+str(region))
        
        # Iterates through the finite element types

        for element_type, connectivities in finite_elements_dict.items():

            # Verifies if this element is in the class of finite ele-
            # ments and dispatches the class of this element

            region_finite_elements.dispatch_element(element_type, 
            connectivities, physical_group_tag, dtype)

    # Stores the global number of DOFs if the current region is the do-
    # main

    if region=="domain":

        mesh_data_class.global_number_dofs = region_finite_elements.dofs_counter

        mesh_data_class.dofs_node_dict = region_finite_elements.dofs_node_dict

    # Stores the class of finite elements for the region into the mesh
    # data class and returns it

    setattr(mesh_data_class, elements_attribute_name,
    region_finite_elements.elements_dictionaries)

    return mesh_data_class