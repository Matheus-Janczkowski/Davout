# Routine to gather information across multiple meshes

from ..tool_box.mesh_tools import MshMeshData

# Defines a class to store information gathered from different instances
# of the mesh data class. This information, however, must be equal all
# over the different meshes, in different realizations

class MeshRealizationsCommonInfo:

    def __init__(self, float_dtype, integer_dtype, dofs_per_element, 
    number_elements, number_quadrature_points, shape_functions_tensor):
        
        self.float_dtype = float_dtype
        
        self.integer_dtype = integer_dtype
        
        self.dofs_per_element = dofs_per_element
        
        self.number_elements = number_elements
        
        self.number_quadrature_points = number_quadrature_points

        self.shape_functions_tensor = shape_functions_tensor

########################################################################
#                             Verification                             #
########################################################################

# Defines a function to verify if a mesh data class is a list with many
# realizations of the same mesh or a single mesh instance

def verify_mesh_realizations(mesh_data_class, n_realizations, field_name,
class_name):

    # Initializes the global number of DOFs and the numerical types

    global_number_dofs = None 

    dtype = None

    integer_dtype = None

    # Verifies if mesh data class is an instance of the class of mesh
    # data

    if isinstance(mesh_data_class, MshMeshData):
    
        global_number_dofs = mesh_data_class.global_number_dofs

        dtype = mesh_data_class.dtype

        integer_dtype = mesh_data_class.integer_dtype

        # Verifies if the domain elements have the frequired field

        if not (field_name in mesh_data_class.domain_elements):

            raise NameError("There is no field named '"+str(field_name)+
            "' in the mesh. Thus, it is not possible to assemble the c"+
            "lass "+str(class_name))

    # Otherwise, if it is a list, there will be a mesh for multiple
    # realizations
    
    elif isinstance(mesh_data_class, list):

        # Verifies if the number of meshes is within bounds of the
        # number of given realizations

        if len(mesh_data_class)>n_realizations:

            raise IndexError(str(len(mesh_data_class))+" meshes were p"+
            "rovided, but only "+str(n_realizations)+" realizations we"+
            "re given. The number of meshes must be at most equal to t"+
            "he number of realizations")

        # Gets the number of DOFs and the type from the first mesh 
    
        global_number_dofs = mesh_data_class[0].global_number_dofs

        dtype = mesh_data_class[0].dtype

        integer_dtype = mesh_data_class[0].integer_dtype

        # Iterates through the other meshes to verify if the number 
        # of DOFs and the float type are the same

        for i in range(len(mesh_data_class)):

            # Verifies the number of global DOFs

            if mesh_data_class[i].global_number_dofs!=(
            global_number_dofs):
                
                raise ValueError("The "+str(i+1)+"-th mesh has "+
                str(mesh_data_class[i].global_number_dofs)+" DOFs, whe"+
                "reas the first mesh has "+str(global_number_dofs)+" D"+
                "OFs. All meshes, across all realizations must have th"+
                "e same number of DOFs and the same conectivities")
            
            # Verifies the float type

            if mesh_data_class[i].dtype!=dtype:
                
                raise ValueError("The "+str(i+1)+"-th mesh has dtype="+
                str(mesh_data_class[i].dtype)+", whereas the first mes"+
                "h has dtype="+str(dtype)+". All meshes, across all re"+
                "alizations must have the same numerical type dtype")
            
            # Verifies the integer type

            if mesh_data_class[i].integer_dtype!=integer_dtype:
                
                raise ValueError("The "+str(i+1)+"-th mesh has integer"+
                "_dtype="+str(mesh_data_class[i].integer_dtype)+", whe"+
                "reas the first mesh has integer_dtype="+str(
                integer_dtype)+". All meshes, across all realizations "+
                "must have the same integer type")
            
            # Verifies if the domain elements have the required field

            if not (field_name in mesh_data_class[i].domain_elements):

                raise NameError("There is no field named '"+str(
                field_name)+"' in the "+str(i+1)+"-th mesh. Thus, it i"+
                "s not possible to assemble class "+str(class_name))
            
    # If it is not a list, throws an error

    else:

        raise TypeError("'mesh_data_class' in '"+str(class_name)+"' is"+
        " neither an instance of the class 'MshMeshData' neither is it"+
        " a list")
    
    # Returns the gathered information

    return global_number_dofs, dtype, integer_dtype

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

    # Initializes the tensor of DOF indices per element

    dofs_per_element = None

    # Initializes the numbers of elements and of quadrature points common
    # to all meshes

    number_elements = None 

    number_quadrature_points = None

    # Initializes the object with information of the finite element class
    # for this physical group
    
    mesh_data = None

    # Initializes the tensor of shape functions (Lagrangian shape func-
    # tions, for they do not depend upon geometric parameters of the fi-
    # nite element)

    shape_functions_tensor = None

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
        
        # Initializes a list to store the element types concerned with
        # the required field at the current physical group
        
        mesh_data = []
        
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
            
            # Appends the domain elements dictionary of this mesh

            domain_elements_dict = mesh.domain_elements[field_name]
            
            # Verifies if this dictionary of domain elements has more 
            # than one element type

            if isinstance(domain_elements_dict[physical_group_tag], 
            dict):

                raise NotImplementedError("Multiple element types per "+
                "physical group has not yet been updated to compute "+
                str(class_name))
                
            # Gets the elements defined for this physical group

            mesh_data.append(domain_elements_dict[physical_group_tag])

        # Gets the tensor of DOFs per element from the first mesh, since
        # all meshes share the same connectivities and DOF enumeration
        # anyway

        dofs_per_element = mesh_data[0].dofs_per_element

        # Gathers other common information to all meshes

        number_elements = mesh_data[0].number_elements

        number_quadrature_points = mesh_data[0].number_quadrature_points

        shape_functions_tensor = mesh_data[0].shape_functions_tensor

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

            raise NotImplementedError("Multiple element types per phys"+
            "ical group has not yet been updated to compute "+str(
            class_name))
                
        # Gets the elements defined for this physical group

        mesh_data = mesh_dict[physical_group_tag]

        # Gets the tensor of DOFs per element from this mesh

        dofs_per_element = mesh_data.dofs_per_element

        # Gathers other mesh information

        number_elements = mesh_data.number_elements

        number_quadrature_points = mesh_data.number_quadrature_points

        shape_functions_tensor = mesh_data.shape_functions_tensor

    # Creates an instance of the class for common information across 
    # mesh realizations

    mesh_common_info = MeshRealizationsCommonInfo(float_dtype, 
    integer_dtype, dofs_per_element, number_elements, 
    number_quadrature_points, shape_functions_tensor)

    return (mesh_data, physical_group_tag, mesh_common_info)

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

    # Initializes the tensor of DOF indices per element

    dofs_per_element = None

    # Initializes the numbers of elements and of quadrature points common
    # to all meshes

    number_elements = None 

    number_quadrature_points = None

    # Initializes the object with information of the finite element class
    # for this physical group
    
    mesh_data = None

    # Initializes the tensor of shape functions (Lagrangian shape func-
    # tions, for they do not depend upon geometric parameters of the fi-
    # nite element)

    shape_functions_tensor = None

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
        
        # Initializes a list to store the element types concerned with
        # the required field at the current physical group
        
        mesh_data = []
        
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
            
            # Appends the boundary elements dictionary of this mesh

            boundary_elements_dict = mesh.boundary_elements[field_name]
            
            # Verifies if this dictionary of boundary elements has more 
            # than one element type

            if isinstance(boundary_elements_dict[physical_group_tag], 
            dict):

                raise NotImplementedError("Multiple element types per "+
                "physical group has not yet been updated to compute "+
                str(class_name))
                
            # Gets the elements defined for this physical group

            mesh_data.append(boundary_elements_dict[physical_group_tag])

        # Gets the tensor of DOFs per element from the first mesh, since
        # all meshes share the same connectivities and DOF enumeration
        # anyway

        dofs_per_element = mesh_data[0].dofs_per_element

        # Gathers other common information to all meshes

        number_elements = mesh_data[0].number_elements

        number_quadrature_points = mesh_data[0].number_quadrature_points

        shape_functions_tensor = mesh_data[0].shape_functions_tensor

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

            raise NotImplementedError("Multiple element types per phys"+
            "ical group has not yet been updated to compute "+str(
            class_name))
                
        # Gets the elements defined for this physical group

        mesh_data = mesh_dict[physical_group_tag]

        # Gets the tensor of DOFs per element from this mesh

        dofs_per_element = mesh_data.dofs_per_element

        # Gathers other mesh information

        number_elements = mesh_data.number_elements

        number_quadrature_points = mesh_data.number_quadrature_points

        shape_functions_tensor = mesh_data.shape_functions_tensor

    # Creates an instance of the class for common information across 
    # mesh realizations

    mesh_common_info = MeshRealizationsCommonInfo(float_dtype, 
    integer_dtype, dofs_per_element, number_elements, 
    number_quadrature_points, shape_functions_tensor)

    return (mesh_data, physical_group_tag, mesh_common_info)