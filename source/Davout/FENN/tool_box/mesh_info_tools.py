# Routine to gather information across multiple meshes

########################################################################
#                          Volumetric elements                         #
########################################################################

# Defines a function to get information from the mesh data class concer-
# ning volume elements

def get_volume_info_from_mesh_data_class(mesh_data_class, 
physical_group_name, dictionary_name, class_name, field_name):

    # Initializes the numerical types

    integer_dtype = None

    float_dtype = None

    # Initializes the tag of the physical group

    physical_group_tag = None

    mesh_dict = None

    # Verifies if multiple meshes were given

    if isinstance(mesh_data_class, list):

        # Gets the integer and float types from the first mesh

        integer_dtype = mesh_data_class[0].integer_dtype

        float_dtype = mesh_data_class[0].dtype

        # Verifies if the dictionary of domain elements has this field
        # name as key

        if not (field_name in mesh_data_class[0].domain_elements):

            raise KeyError("The first mesh does not have in its dictio"+
            "nary of domain elements the key for the '"+str(field_name)+
            "' field")
        
        # Recovers the dictionary of elements in the domain concerned
        # with displacement field from the first mesh
        
        mesh_dict = mesh_data_class[0].domain_elements[field_name]
        
        # Recovers the dictionary of physical groups names to their nu-
        # merical tags

        domain_physical_groups_dict = mesh_data_class[0
        ].domain_physicalGroupsNameToTag

        # Verifies if the physical group name is in this dictionary

        if not (physical_group_name in domain_physical_groups_dict):

            raise NameError("The physical group name '"+str(
            physical_group_name)+"' in the dictionary of "+str(
            dictionary_name)+" is not a valid physical group name. Che"+
            "ck the available names:\n"+str(list(
            domain_physical_groups_dict.keys())))
        
        else:

            physical_group_tag = domain_physical_groups_dict[
            physical_group_name]

        # Verifies if all meshes have this physical group name in their
        # dictionaries of domain physical groups

        for index, mesh in enumerate(mesh_data_class):

            # Recovers the dictionary of physical groups names to their 
            # numerical tags

            domain_physical_groups_dict = (
            mesh.domain_physicalGroupsNameToTag)

            # Verifies if the physical group name is in this dictionary

            if not (physical_group_name in domain_physical_groups_dict):

                raise NameError("The physical group name '"+str(
                physical_group_name)+"' in the dictionary of "+str(
                dictionary_name)+" is not a valid physical group name "+
                "for the "+str(index+1)+"-th mesh. Even though it was "+
                "found in the first mesh. Check the available names fo"+
                "r this mesh:\n"+str(list(
                domain_physical_groups_dict.keys())))
            
            # Verifies if the physical group tag is the same across me-
            # shes
            
            elif domain_physical_groups_dict[physical_group_name]!=(
            physical_group_tag):

                raise ValueError("The physical group name '"+str(
                physical_group_name)+"' in the dictionary of "+str(
                dictionary_name)+" has the numerical tag "+str(
                domain_physical_groups_dict[physical_group_name])+" "+
                "for the "+str(index+1)+"-th mesh. Whereas the tag for"+
                " the first mesh is "+str(physical_group_tag)+". The t"+
                "ags must be the same across meshes")

    # Otherwise, if it is a single mesh

    else:

        # Gets the integer and float types directly from the mesh

        integer_dtype = mesh_data_class.integer_dtype

        float_dtype = mesh_data_class.dtype

        # Verifies if the dictionary of domain elements has this field
        # name as key

        if not (field_name in mesh_data_class.domain_elements):

            raise KeyError("The mesh does not have in its dictionary o"+
            "f domain elements the key for the '"+str(field_name)+"' f"+
            "ield")
        
        # Recovers the dictionary of elements in the domain concerned
        # with displacement field
        
        mesh_dict = mesh_data_class.domain_elements[field_name]
        
        # Recovers the dictionary of physical groups names to their nu-
        # merical tags

        domain_physical_groups_dict = (
        mesh_data_class.domain_physicalGroupsNameToTag)

        # Verifies if the physical group name is in this dictionary

        if not (physical_group_name in domain_physical_groups_dict):

            raise NameError("The physical group name '"+str(
            physical_group_name)+"' in the dictionary of "+str(
            dictionary_name)+" is not a valid physical group name. Che"+
            "ck the available names:\n"+str(list(
            domain_physical_groups_dict.keys())))
        
        else:

            physical_group_tag = domain_physical_groups_dict[
            physical_group_name]

    # Gets the instance of the mesh data

    if isinstance(mesh_dict[physical_group_tag], dict):

        raise NotImplementedError("Multiple element types per physical"+
        " group has not yet been updated to compute "+str(class_name))
            
    # Gets the first element type

    mesh_data = mesh_dict[physical_group_tag]

    return (integer_dtype, float_dtype, mesh_data, physical_group_tag)

########################################################################
#                           Boundary elements                          #
########################################################################

# Defines a function to get information from the mesh data class concer-
# ning boundary elements

def get_boundary_info_from_mesh_data_class(mesh_data_class, 
physical_group_name, dictionary_name, class_name, field_name):

    # Initializes the numerical types

    integer_dtype = None

    float_dtype = None

    # Initializes the tag of the physical group

    physical_group_tag = None

    mesh_dict = None

    # Verifies if multiple meshes were given

    if isinstance(mesh_data_class, list):

        # Gets the integer and float types from the first mesh

        integer_dtype = mesh_data_class[0].integer_dtype

        float_dtype = mesh_data_class[0].dtype

        # Verifies if the dictionary of boundary elements has this field
        # name as key

        if not (field_name in mesh_data_class[0].boundary_elements):

            raise KeyError("The first mesh does not have in its dictio"+
            "nary of boundary elements the key for the '"+str(field_name
            )+"' field")
        
        # Recovers the dictionary of elements in the boundary concerned
        # with displacement field from the first mesh
        
        mesh_dict = mesh_data_class[0].boundary_elements[field_name]
        
        # Recovers the dictionary of physical groups names to their nu-
        # merical tags

        boundary_physical_groups_dict = mesh_data_class[0
        ].boundary_physicalGroupsNameToTag

        # Verifies if the physical group name is in this dictionary

        if not (physical_group_name in boundary_physical_groups_dict):

            raise NameError("The physical group name '"+str(
            physical_group_name)+"' in the dictionary of "+str(
            dictionary_name)+" is not a valid physical group name. Che"+
            "ck the available names:\n"+str(list(
            boundary_physical_groups_dict.keys())))
        
        else:

            physical_group_tag = boundary_physical_groups_dict[
            physical_group_name]

        # Verifies if all meshes have this physical group name in their
        # dictionaries of boundary physical groups

        for index, mesh in enumerate(mesh_data_class):

            # Recovers the dictionary of physical groups names to their 
            # numerical tags

            boundary_physical_groups_dict = (
            mesh.boundary_physicalGroupsNameToTag)

            # Verifies if the physical group name is in this dictionary

            if not (physical_group_name in boundary_physical_groups_dict):

                raise NameError("The physical group name '"+str(
                physical_group_name)+"' in the dictionary of "+str(
                dictionary_name)+" is not a valid physical group name "+
                "for the "+str(index+1)+"-th mesh. Even though it was "+
                "found in the first mesh. Check the available names fo"+
                "r this mesh:\n"+str(list(
                boundary_physical_groups_dict.keys())))
            
            # Verifies if the physical group tag is the same across me-
            # shes
            
            elif boundary_physical_groups_dict[physical_group_name]!=(
            physical_group_tag):

                raise ValueError("The physical group name '"+str(
                physical_group_name)+"' in the dictionary of "+str(
                dictionary_name)+" has the numerical tag "+str(
                boundary_physical_groups_dict[physical_group_name])+" "+
                "for the "+str(index+1)+"-th mesh. Whereas the tag for"+
                " the first mesh is "+str(physical_group_tag)+". The t"+
                "ags must be the same across meshes")

    # Otherwise, if it is a single mesh

    else:

        # Gets the integer and float types directly from the mesh

        integer_dtype = mesh_data_class.integer_dtype

        float_dtype = mesh_data_class.dtype

        # Verifies if the dictionary of boundary elements has this field
        # name as key

        if not (field_name in mesh_data_class.boundary_elements):

            raise KeyError("The mesh does not have in its dictionary o"+
            "f boundary elements the key for the '"+str(field_name)+"'"+
            " field")
        
        # Recovers the dictionary of elements in the boundary concerned
        # with displacement field
        
        mesh_dict = mesh_data_class.boundary_elements[field_name]
        
        # Recovers the dictionary of physical groups names to their nu-
        # merical tags

        boundary_physical_groups_dict = (
        mesh_data_class.boundary_physicalGroupsNameToTag)

        # Verifies if the physical group name is in this dictionary

        if not (physical_group_name in boundary_physical_groups_dict):

            raise NameError("The physical group name '"+str(
            physical_group_name)+"' in the dictionary of "+str(
            dictionary_name)+" is not a valid physical group name. Che"+
            "ck the available names:\n"+str(list(
            boundary_physical_groups_dict.keys())))
        
        else:

            physical_group_tag = boundary_physical_groups_dict[
            physical_group_name]

    # Gets the instance of the mesh data

    if isinstance(mesh_dict[physical_group_tag], dict):

        raise NotImplementedError("Multiple element types per physical"+
        " group has not yet been updated to compute "+str(class_name))
            
    # Gets the first element type

    mesh_data = mesh_dict[physical_group_tag]

    # Gets the tensor of DOFs per element directly from the mesh

    dofs_per_element = mesh_data.dofs_per_element

    return (integer_dtype, float_dtype, mesh_data, physical_group_tag)