# Routine to store some methods to work with meshes

from dolfin import *

import meshio

import numpy as np

from copy import copy

from scipy.spatial import KDTree

from ..tool_box import programming_tools

# Defines a class for the mesh data

class MeshData:

    def __init__(self, mesh, dx, ds, n, x, domain_meshCollection, 
    domain_meshFunction, boundary_meshCollection, boundary_meshFunction, 
    domain_physicalGroupsNameToTag, boundary_physicalGroupsNameToTag,
    verbose):
        
        # Saves the mesh parameters

        self.mesh = mesh
        
        self.dx = dx
        
        self.ds = ds
        
        self.n = n

        self.x = x
        
        self.domain_meshCollection = domain_meshCollection 

        self.domain_meshFunction = domain_meshFunction
        
        self.boundary_meshCollection = boundary_meshCollection
        
        self.boundary_meshFunction = boundary_meshFunction

        self.domain_physicalGroupsNameToTag = domain_physicalGroupsNameToTag
        
        self.boundary_physicalGroupsNameToTag = boundary_physicalGroupsNameToTag
        
        self.verbose = verbose

########################################################################
#                              Mesh files                              #
########################################################################

# Defines a function to read a mesh from a msh file

@programming_tools.optional_argumentsInitializer({'desired_elements':
lambda: ['tetra', 'triangle'], 'data_sets': lambda: ["domain", ("bound"+
"ary")]})

def read_mshMesh(file_name, desired_elements=None, data_sets=None, 
quadrature_degree=2, verbose=False):

    # Reads the saved gmsh mesh using meshio

    mesh_reading = meshio.read(file_name+".msh")

    # Initializes the dictionary of cells and the list of cell data

    cells_dictionary = dict()

    cell_dataList = []

    # Recovers the physical groups

    domain_physicalGroupsNameToTag = dict()

    boundary_physicalGroupsNameToTag = dict()

    # Gets the physical group through the field data

    field_data = mesh_reading.field_data

    # Iterates through its keys

    for physical_group, tag_dimensionality in field_data.items():

        # If the dimensionality is 2, gets the boundary

        if tag_dimensionality[1]==2:

            boundary_physicalGroupsNameToTag[physical_group] = int(
            tag_dimensionality[0])

        # If the dimensionality is 3, gets the domain

        elif tag_dimensionality[1]==3:

            domain_physicalGroupsNameToTag[physical_group] = int(
            tag_dimensionality[0])

    # Initializes a dictionary whose keys are the element types and the
    # values are the list of physical groups tags to which each element
    # belongs

    physical_groupsElements = dict()

    # Iterates through the elements

    for element in desired_elements:

        # Gets a list of tags

        list_ofTags = []

        try:

            list_ofTags = list(mesh_reading.cell_data_dict["gmsh:physi"+
            "cal"][element])

        except:

            raise KeyError("The required element '"+str(element)+"' ha"+
            "s not been found. Check out the mesh you are providing. P"+
            "robably, either the volume or the boundary hadn't been sa"+
            "ved")

        # Iterates through the domain physical groups
        
        for physical_groupName, physical_groupTag in domain_physicalGroupsNameToTag.items():
        
            # Counts the number of elements that belong to this physical
            # group

            n_elements = list_ofTags.count(physical_groupTag)

            # If there is at least one element, saves the number into 
            # the dictionary of physical groups

            if n_elements>0:

                try:

                    physical_groupsElements[physical_groupName] += (";"+
                    " "+str(n_elements)+" "+str(element)+" elements")

                except:

                    physical_groupsElements[physical_groupName] = str(
                    n_elements)+" "+str(element)+" elements" 

        # Iterates through the domain physical groups
        
        for physical_groupName, physical_groupTag in boundary_physicalGroupsNameToTag.items():
        
            # Counts the number of elements that belong to this physical
            # group

            n_elements = list_ofTags.count(physical_groupTag)

            # If there is at least one element, saves the number into 
            # the dictionary of physical groups

            if n_elements>0:

                try:

                    physical_groupsElements[physical_groupName] += (";"+
                    " "+str(n_elements)+" "+str(element)+" elements")

                except:

                    physical_groupsElements[physical_groupName] = str(
                    n_elements)+" "+str(element)+" elements" 

    print("###########################################################"+
    "#############\n#                        Mesh - physical groups   "+
    "                     #\n#########################################"+
    "###############################\n")

    print("Finds the following domain physical groups with their respe"+
    "ctive tags:")

    for physical_group, tag in domain_physicalGroupsNameToTag.items():

        print(physical_group, "-", tag)

        try: 

            print("      - "+physical_groupsElements[physical_group])

        except:

            raise ValueError("The physical group "+str(physical_group)+
            " does not have any elements in it.")
        
        print("")

    print("\n\nFinds the following boundary physical groups with their r"+
    "espective tags:")

    for physical_group, tag in boundary_physicalGroupsNameToTag.items():

        print(physical_group, "-", tag)

        try: 

            print("      - "+physical_groupsElements[physical_group])

        except:

            raise ValueError("The physical group "+str(physical_group)+
            " does not have any elements in it.")
        
        print("")

    print("\n")

    # Gets the cells which consist of the desired element

    for i in range(len(desired_elements)):

        print("Saves the mesh of", data_sets[i], "dataset\n")

        cells_dictionary[desired_elements[i]] = (
        mesh_reading.get_cells_type(desired_elements[i]))

        print("There are "+str(len(cells_dictionary[desired_elements[i]]
        ))+" "+str(desired_elements[i])+" elements in the mesh.\n")

        # Gets the physical cell data for this element

        cell_dataPhysical = mesh_reading.get_cell_data("gmsh:physical", 
        desired_elements[i])

        # Gets the geometric cell data for this element

        cell_dataGeometric = mesh_reading.get_cell_data("gmsh:geometri"+
        "cal", desired_elements[i])

        # Adds the geometric cell data to the list of cell data

        cell_dataList.append(cell_dataGeometric)

        # Creates a mesh for this data set and saves it

        mesh_set = meshio.Mesh(points=mesh_reading.points, cells={
        desired_elements[i]: cells_dictionary[desired_elements[i]]},
        cell_data={data_sets[i]: [cell_dataPhysical]})

        meshio.write(file_name+"_"+data_sets[i]+".xdmf", mesh_set)

    # Creates the mesh with all information, including geometric infor-
    # mation and saves it
    
    whole_mesh = meshio.Mesh(points=mesh_reading.points, cells=
    cells_dictionary, cell_data={"whole_mesh": cell_dataList})

    meshio.write(file_name+".xdmf", whole_mesh)

    print("###########################################################"+
    "#############\n#                    Mesh reading has been finaliz"+
    "ed                   #\n#########################################"+
    "###############################\n")

    # Then, calls the xdmf reader and returns its output

    return read_xdmfMesh(file_name, domain_physicalGroupsNameToTag=
    domain_physicalGroupsNameToTag, boundary_physicalGroupsNameToTag=
    boundary_physicalGroupsNameToTag, quadrature_degree=
    quadrature_degree, verbose=verbose)

# Defines a function to read a mesh from a xdmf file

@programming_tools.optional_argumentsInitializer({('domain_physicalGro'+
'upsNameToTag'): lambda: dict(), 'boundary_physicalGroupsNameToTag': 
lambda: dict()})

def read_xdmfMesh(file_name, domain_physicalGroupsNameToTag=None, 
boundary_physicalGroupsNameToTag=None, quadrature_degree=2, verbose=
False):
    
    # Sets the compiler parameters

    parameters["form_compiler"]["representation"] = "uflacs"

    parameters["allow_extrapolation"] = True

    parameters["form_compiler"]["cpp_optimize"] = True

    parameters["form_compiler"]["quadrature_degree"] = quadrature_degree

    # Verifies the dictionaries of names to tags
    
    if len(domain_physicalGroupsNameToTag.keys())==0 or (len(
    boundary_physicalGroupsNameToTag.keys())==0):
        
        print("WARNING: the dictionaries of physical groups' names to "+
        "tags are empty, thus, it won't be possible to use the names i"+
        "n the variational forms. Use a .msh mesh file directly instea"+
        "d.")

    # Verifies whether there is an extension in the file name

    if ("." in file_name) or ('.' in file_name):

        raise NameError("The name of the mesh file, "+file_name+", has"+
        " an extension. This name musn't have a extension, for xdmf fi"+
        "les only can be read with this method.\n")

    # Initializes the mesh object and reads the xdmf file

    mesh = Mesh()

    # Initializes a mesh value collection to store mesh data of the do-
    # main

    domain_meshCollection = MeshValueCollection("size_t", mesh, 
    mesh.topology().dim())

    # Reads the mesh with domain physical groups

    with XDMFFile(file_name+"_domain.xdmf") as infile:

        infile.read(mesh)

        infile.read(domain_meshCollection, "domain")

    # Converts the mesh value collection to mesh function, for mesh va-
    # lue collections are low level and cannot be used for FEM integra-
    # tion and other higher level operations inside FEniCS

    domain_meshFunction = MeshFunction("size_t", mesh, 
    domain_meshCollection)

    # Reinitializes the mesh value colection to the boundary data

    boundary_meshCollection = MeshValueCollection("size_t", mesh,
    mesh.topology().dim()-1)

    # Reads the mesh with surface physical groups

    with XDMFFile(file_name+"_boundary.xdmf") as infile:
    
        infile.read(boundary_meshCollection, "boundary")

    # Converts the mesh value collection to mesh function

    boundary_meshFunction = MeshFunction("size_t", mesh, 
    boundary_meshCollection)

    # Sets the integration differentials

    dx = Measure("dx", domain=mesh, subdomain_data=domain_meshFunction,
    metadata={"quadrature_degree": quadrature_degree})

    ds = Measure("ds", domain=mesh, subdomain_data=boundary_meshFunction)#,
    #metadata={"quadrature_degree": quadrature_degree})

    # Sets the normal vector to the mesh's boundary

    n  = FacetNormal(mesh)

    # Sets the position vector

    x_position = SpatialCoordinate(mesh)

    print("Finishes creating the mesh functions, measures, and tags di"+
    "ctionaries.\n")

    # Stores these objects inside a class and returns it

    return MeshData(mesh, dx, ds, n, x_position, domain_meshCollection, 
    domain_meshFunction, boundary_meshCollection, boundary_meshFunction, 
    domain_physicalGroupsNameToTag, boundary_physicalGroupsNameToTag,
    verbose)

########################################################################
#                              Submeshing                              #
########################################################################

# Defines a function to generate a submesh using MeshView and creates

def create_submesh(domain_meshCollection, domain_meshFunction,
volume_physGroupsTags, parent_functionSpace, 
domain_physicalGroupsNameToTag=None):
    
    # Verifies if the volume physical groups are a list of strings

    for i in range(len(volume_physGroupsTags)):

        if isinstance(volume_physGroupsTags[i], str):

            # Converts the string to integer tag

            try:

                volume_physGroupsTags[i] = domain_physicalGroupsNameToTag[
                volume_physGroupsTags[i]]

            except TypeError:

                raise TypeError("The dictionary of volumetric physical"+
                " groups to integer tags has not been provided")
            
            except KeyError:

                raise KeyError("The '"+str(volume_physGroupsTags[i])+
                "' is not a valid tag for a volumetric physical group."+
                " The valid tags are: "+str(domain_physicalGroupsNameToTag.keys()))

    # Gets the mesh, the polynomial degree, and the shape function

    mesh = parent_functionSpace.mesh()

    polynomial_degree = parent_functionSpace.ufl_element().degree()

    shape_function = parent_functionSpace.ufl_element().family()

    # Gets the dimensionality of the parent function space

    n_dimsParentFunctionSpace = len(parent_functionSpace.ufl_element(
    ).value_shape())

    # Creates a new cell markers object to not affect the mesh function
    # at other parts of the code

    submesh_cellMarkers = cpp.mesh.MeshFunctionSizet(mesh, 
    domain_meshCollection)

    # If the submesh is meant to be constructed from different volume 
    # physical groups of the mesh, the object of cell_markers has to be
    # changed, because the MeshView create function works only with a 
    # single marker to define a subdomain and, from there, a submesh. 
    # Therefore, the physical groups that contain the RVE are collapsed 
    # into a single marker; conventionally the first selected marker

    if isinstance(volume_physGroupsTags, list):

        # Iterates through the elements of the parent mesh

        if len(volume_physGroupsTags)>1:

            for element in cells(mesh):

                # Tests if the cell marker is in the list of required 
                # cell markers

                if submesh_cellMarkers[element] in volume_physGroupsTags:

                    # Changes the cell marker to the first required one

                    submesh_cellMarkers[element] = volume_physGroupsTags[
                    0]

        # Collapses the volume physical groups tags list to the first 
        # component

        volume_physGroupsTags = volume_physGroupsTags[0]

    # Creates a submesh for the RVE

    sub_mesh = MeshView.create(submesh_cellMarkers, volume_physGroupsTags)

    # Creates the mapping of elements from the submesh to the parent 
    # mesh, i.e. given the element index in the submesh, it throws the 
    # index in the parent mesh

    sub_toParentCellMap = sub_mesh.topology().mapping()[mesh.id()
    ].cell_map()

    # Creates a new mesh function with physical groups for the submesh

    submesh_meshFunction = MeshFunction("size_t", sub_mesh, 
    sub_mesh.topology().dim(), 0)

    # Iterates over the elements in the cell mapping of the submesh to 
    # the parent mesh to retrieve elements physical groups

    for submesh_cellIndex, parent_cellIndex in enumerate(
    sub_toParentCellMap):

        submesh_meshFunction[submesh_cellIndex] = domain_meshFunction[
        parent_cellIndex]

    # Creates the function spaces

    submesh_functionSpace = 0

    # If the element is mixed, the .family() will return 'Mixed'

    if shape_function=='Mixed':

        submesh_functionSpace = FunctionSpace(sub_mesh,
        parent_functionSpace.ufl_element())

    else:

        # If the field is scalar, the dimensionality is 0

        if n_dimsParentFunctionSpace==0:

            submesh_functionSpace = FunctionSpace(sub_mesh, 
            shape_function, polynomial_degree)

        # If the field is vector function, the dimensionality is 1

        elif n_dimsParentFunctionSpace==1:

            submesh_functionSpace = VectorFunctionSpace(sub_mesh,
            shape_function, polynomial_degree)

        # If the field is a second order tensor function

        elif n_dimsParentFunctionSpace==2:

            submesh_functionSpace = TensorFunctionSpace(sub_mesh, 
            shape_function, polynomial_degree)

        # Handle not implemented function spaces

        else:

            raise NameError("create_submesh in mesh_handling_tools.py "+
            "does not support a function space with dimensionality lar"+
            "ger than "+str(n_dimsParentFunctionSpace)+", for it's not"+
            " been implemented yet or it's not possible to implement.")
        
    # Initializes the function to the solution at the submesh

    submesh_function = Function(submesh_functionSpace)

    # Initializes the DOF mappings for the RVE and for the original mesh

    sub_meshMapping = []

    parent_meshMapping = []

    # Verifies whether there is only on field

    if shape_function!='Mixed':

        sub_meshMapping.append(submesh_functionSpace.dofmap())

        parent_meshMapping.append(parent_functionSpace.dofmap())

    # If there are multiple fields

    else:

        # Iterates through the number of fields

        for i in range(parent_functionSpace.ufl_element().num_sub_elements()):

            # Adds the submesh mapping and the parent mesh mapping

            sub_meshMapping.append(submesh_functionSpace.sub(i).dofmap())

            parent_meshMapping.append(parent_functionSpace.sub(i).dofmap(
            ))
            
    # Sets the integration differential in the submesh

    dx_submesh = Measure("dx", domain=sub_mesh, subdomain_data=
    submesh_meshFunction)

    submesh_physicalGroups = set(dx_submesh.subdomain_data().array())

    physical_groupsList = ""

    for physical_group in submesh_physicalGroups:

        physical_groupsList += str(physical_group)+"\n"

    print("The submesh has the following physical groups:\n"+
    physical_groupsList)

    # Creates a position vector field for the submesh

    V_positionVector = VectorFunctionSpace(sub_mesh, "CG", 1)

    x_submesh = Function(V_positionVector)

    # Assign the mesh coordinates to the function

    x_submesh.interpolate(Expression(("x[0]", "x[1]", "x[2]"), element=
    V_positionVector.ufl_element()))

    # Returns the submesh, the updated cell markers, and the DOF mappings

    return (sub_mesh, submesh_meshFunction, submesh_functionSpace, 
    sub_meshMapping, parent_meshMapping, submesh_function, 
    sub_toParentCellMap, dx_submesh, x_submesh)

# Defines a function to update the field parameters vector of a submesh 
# given the corresponding vector at the parent mesh

def field_parentToSubmesh(submesh, field_parentMesh, sub_toParentCellMap, 
sub_meshMapping=None, parent_meshMapping=None, field_submesh=None):
    
    # If the field of the submesh is not explicitely given, creates it
    # as a copy of the parent field jsut with a different mesh

    if field_submesh is None:

        field_submesh = Function(FunctionSpace(submesh, 
        field_parentMesh.ufl_element()))

    # If the mesh mappings have not been provided

    if (sub_meshMapping is None) or (parent_meshMapping is None):

        # Gets the parent field function space and its shape function

        submesh_functionSpace = field_submesh.function_space()

        parent_functionSpace = field_parentMesh.function_space()

        shape_function = parent_functionSpace.ufl_element().family()

        # Initializes the DOF mappings for the RVE and for the original 
        # mesh

        sub_meshMapping = []

        parent_meshMapping = []

        # Verifies whether there is only on field

        if shape_function!='Mixed':

            sub_meshMapping.append(submesh_functionSpace.dofmap())

            parent_meshMapping.append(parent_functionSpace.dofmap())

        # If there are multiple fields

        else:

            # Iterates through the number of fields

            for i in range(parent_functionSpace.ufl_element(
            ).num_sub_elements()):

                # Adds the submesh mapping and the parent mesh mapping

                sub_meshMapping.append(submesh_functionSpace.sub(i
                ).dofmap())

                parent_meshMapping.append(parent_functionSpace.sub(i
                ).dofmap())

    # Iterates through the elements of the submesh

    for element in cells(submesh):

        # Gets the index of the element in the submesh

        submesh_index = element.index()

        # Gets the index of the element in the parent mesh

        parent_index = sub_toParentCellMap[submesh_index]

        # Iterates through the fields of the meshes. If the mesh has on-
        # ly one field, displacement for instance, the list of DOF map-
        # ping has only one component

        for i in range(len(sub_meshMapping)):

            # Translates the values of the solution using the DOFs map-
            # ping

            field_submesh.vector()[sub_meshMapping[i].cell_dofs(
            submesh_index)] = field_parentMesh.vector()[
            parent_meshMapping[i].cell_dofs(parent_index)] 

    # Returns the submesh field

    return field_submesh
        
########################################################################
#                             Node finding                             #
########################################################################

# Defines a function to find a node of the mesh nearest to a given point.
# You can provide a class with a .mesh attribute or you can provide the
# mesh proper

def find_nodeClosestToPoint(mesh_dataClass, point_coordinates, 
node_number, node_coordinates, set_ofNodes=None):

    # Tests if the node has already been found

    if (node_number is None) or (node_coordinates is None):

        # Verifies if the node coordinates is a list

        if (not isinstance(point_coordinates, list)) and (not 
        isinstance(point_coordinates, np.ndarray)):

            raise TypeError("point_coordinates must be a list to find "+
            "a node near this coordinates, whereas it is: "+str(
            point_coordinates)+", whose type is: "+str(type(
            point_coordinates)))

        # Gets the coordinates of the mesh

        mesh_coordinates = 0

        # If a special set of nodes has been required

        if set_ofNodes is None:

            # Verifies if the object is already the mesh

            if hasattr(mesh_dataClass, "coordinates"):

                mesh_coordinates = mesh_dataClass.coordinates()

            else:

                mesh_coordinates = mesh_dataClass.mesh.coordinates()

        elif isinstance(set_ofNodes, list):

            # Verifies if the object is already the mesh

            if hasattr(mesh_dataClass, "coordinates"):

                mesh_coordinates = mesh_dataClass.coordinates()[
                set_ofNodes]

            else:

                mesh_coordinates = mesh_dataClass.mesh.coordinates()[
                set_ofNodes]

        else:

            raise TypeError("The set of nodes to find a node closest t"+
            "o a point from must be a list. The provided set of nodes,"+
            " however, is not a list: "+str(set_ofNodes))

        # Gets a tree of these coordinates

        coordinates_tree = KDTree(mesh_coordinates)

        # Gets the number of the node that is closest to the given coor-
        # dinates

        _, node_number = coordinates_tree.query(point_coordinates)

        # Returns the node number

        if set_ofNodes is None:

            node_number = int(node_number)

            return node_number, mesh_coordinates[node_number]

        else:

            # The node number given by the query is not the actual node
            # number, rather the index inside the point coordinates list.
            # Hence, this index must be mapped back to the global index
            # system

            global_nodeNumber = set_ofNodes[int(node_number)]

            return global_nodeNumber, mesh_coordinates[int(node_number)]
    
    else:

        # Returns the node number as it's been given

        return node_number, node_coordinates
    
# Defines a function to create a list of nodes indexes that lie on the
# boundary of surface

def find_nodesOnSurfaceBoundary(mesh_dataClass, physical_group):

    # Verifies if the physical group is a string

    if isinstance(physical_group, str):

        # Tests if it is in the dictionary of physical groups of the 
        # boundary

        if physical_group in mesh_dataClass.boundary_physicalGroupsNameToTag:

            # Converts it

            physical_group = mesh_dataClass.boundary_physicalGroupsNameToTag[
            physical_group]

        else:

            raise KeyError("The physical group '"+str(physical_group)+
            "' is not in the dictionary of boundary physical groups. T"+
            "hus, cannot be used to find the nodes in the boundary of "+
            "surface. Check out the available options of physical grou"+
            "ps in the boundary: "+str(
            mesh_dataClass.boundary_physicalGroupsNameToTag.keys()))
        
    # Do not accept other formats than integer

    elif not isinstance(physical_group, int):

        raise TypeError("The physical group "+str(physical_group)+" is"+
        " not an integer, thus cannot be used to find the nodes in the"+
        " boundary of a surface")
    
    # Gets the 2D elements in the mesh that lie on this physical group

    bidimensional_elements = [element for element in facets(
    mesh_dataClass.mesh) if mesh_dataClass.boundary_meshFunction[
    element.index()]==physical_group]

    # Initializes a dictionary of element contours to count the number 
    # of elements where they appear in

    element_contoursCounter = dict()

    # Iterates through the 2D elements

    for element in bidimensional_elements:

        # Iterates through the contour lines

        for contour_line in edges(element):

            # Gets the index of the contour

            contour_index = contour_line.index()

            # Adds this to the counter. Uses the get function to avoid
            # key error, thus, get 0 if the key is not found

            element_contoursCounter[contour_index] = element_contoursCounter.get(
            contour_index, 0)+1

    # Initializes a set to guard the set of nodes that lie on the boun-
    # dary

    boundary_nodes = set()

    # Harvests just the contour lines that appear only once, because the
    # lines that appear more than one are shared between elements and,
    # hence, are inside the region

    for contour_index, elements_count in element_contoursCounter.items():

        if elements_count==1:

            # Iterates through the nodes at the edges of this line

            for node in vertices(Edge(mesh_dataClass.mesh, contour_index
            )):
                
                # Adds this element to the set

                boundary_nodes.add(node.index())

    # Returns the set of nodes' indices at the boundary and converts to
    # a list

    return list(boundary_nodes)

# Defines a function to create a list of nodes indexes that lie on the 
# vertices of elements around a node

def find_nodesOnSurfaceAroundNode(mesh_dataClass, physical_group, 
node_number=None, node_coordinates=None):

    # Verifies if the physical group is a string

    if isinstance(physical_group, str):

        # Tests if it is in the dictionary of physical groups of the 
        # boundary

        if physical_group in mesh_dataClass.boundary_physicalGroupsNameToTag:

            # Converts it

            physical_group = mesh_dataClass.boundary_physicalGroupsNameToTag[
            physical_group]

        else:

            raise KeyError("The physical group '"+str(physical_group)+
            "' is not in the dictionary of boundary physical groups. T"+
            "hus, cannot be used to find the nodes around a node on a "+
            "surface. Check out the available options of physical grou"+
            "ps in the boundary: "+str(
            mesh_dataClass.boundary_physicalGroupsNameToTag.keys()))
        
    # Do not accept other formats than integer

    elif not isinstance(physical_group, int):

        raise TypeError("The physical group "+str(physical_group)+" is"+
        " not an integer, thus cannot be used to find the nodes around"+
        " a node on a surface")

    # Verifies if both the node number and the node coordinates are None

    if (node_number is None) and (node_coordinates is None):

        raise ValueError("The node_number and the node_coordinates can"+
        "not be both None, at least one must be provided to find the a"+
        "djacent nodes to a node on a surface")

    elif not (node_coordinates is None):

        # Finds the node closest to the given coordinates

        node_number, node_coordinates = find_nodeClosestToPoint(
        mesh_dataClass, node_coordinates, None, None)

    else:

        # Checks if the node number is an integer

        if not isinstance(node_number, int):

            raise TypeError("The number of the node given to find the"+
            " adjacent nodes to itself on a boundary surface is not a"+
            "n integer")

        # Checks if the number of the node is inbound with the number of
        # nodes in the mesh

        elif node_number>(len(mesh_dataClass.mesh.coordinates())-1):

            raise ValueError("The number of the node given to find the"+
            " adjacent nodes to itself on a boundary surface is larger"+
            " than the number of nodes in the mesh")
    
    # Gets the 2D elements in the mesh that lie on this physical group

    bidimensional_elements = [element for element in facets(
    mesh_dataClass.mesh) if mesh_dataClass.boundary_meshFunction[
    element.index()]==physical_group]

    # Initializes a list of nodes' indices whose nodes lie on elements
    # that are adjacent to the desired node

    adjacent_nodes = set()

    # Iterates through the 2D elements

    for element in bidimensional_elements:

        # Gets the nodes of this element

        elements_nodes = element.entities(0)

        # Verifies if the sought after node is inside this set

        if node_number in elements_nodes:

            # Adds them to the set

            adjacent_nodes.update(elements_nodes)

    # Transforms the adjacent nodes' numbers to a list and gets their 
    # coordinates

    adjacent_nodes = list(adjacent_nodes)

    adjacent_nodesCoordinates = mesh_dataClass.mesh.coordinates()[
    adjacent_nodes]

    return adjacent_nodes, adjacent_nodesCoordinates

########################################################################
#                            Physical groups                           #
########################################################################

# Defines a function to convert a (possibly) string physical group to 
# the corresponding integer physical group

def convert_physicalGroup(physical_group, mesh_dataClass, region):

    # Makes a copy of the physical group

    original_physicalGroup = copy(physical_group)

    # Verifies if region is either domain or boundary

    if region=="boundary":

        if isinstance(physical_group, str):

            # Tests if it is in the dictionary of physical groups of the 
            # boundary

            if physical_group in mesh_dataClass.boundary_physicalGroupsNameToTag:

                # Converts it

                physical_group = mesh_dataClass.boundary_physicalGroupsNameToTag[
                physical_group]

            else:

                raise KeyError("The physical group '"+str(physical_group
                )+"' is not in the dictionary of boundary physical gro"+
                "ups. Check out the available options of physical grou"+
                "ps in the boundary: "+str(
                mesh_dataClass.boundary_physicalGroupsNameToTag.keys()))
            
        # Does not accept any other formats than integer

        elif not isinstance(physical_group, int):

            raise TypeError("The physical group "+str(physical_group)+
            " is not an integer nor a string. Thus, cannot be converte"+
            "d into a numeric physical group")

    elif region=="domain":

        if isinstance(physical_group, str):

            # Tests if it is in the dictionary of physical groups of the 
            # domain

            if physical_group in mesh_dataClass.domain_physicalGroupsNameToTag:

                # Converts it

                physical_group = mesh_dataClass.domain_physicalGroupsNameToTag[
                physical_group]

            else:

                raise KeyError("The physical group '"+str(physical_group
                )+"' is not in the dictionary of domain physical group"+
                "s. Check out the available options of physical groups"+
                " in the domain: "+str(
                mesh_dataClass.domain_physicalGroupsNameToTag.keys()))
            
        # Does not accept any other formats than integer

        elif not isinstance(physical_group, int):

            raise TypeError("The physical group "+str(physical_group)+
            " is not an integer nor a string. Thus, cannot be converte"+
            "d into a numeric physical group")

    else:

        raise NameError("The region flag must be either 'domain' or 'b"+
        "oundary' to be used to convert a physical group to its true n"+
        "umerical counterpart")

    return physical_group, original_physicalGroup

# Defines a function to evaluate the centroid of a surface region given
# by an integer physical group

def evaluate_centroidSurface(physical_group, mesh_dataClass):

    if not isinstance(physical_group, int):

        # Tries to convert is

        physical_group = convert_physicalGroup(physical_group, 
        mesh_dataClass, "boundary")

    # Gets the position vector from the mesh

    position_vector = mesh_dataClass.x

    # Evaluates the area of this physical group

    area_inverse = (1.0/float(assemble(1.0*mesh_dataClass.ds(
    physical_group))))

    # Evaluates the centroid coordinates

    centroid_x = (area_inverse*float(assemble(position_vector[0]*
    mesh_dataClass.ds(physical_group))))

    centroid_y = (area_inverse*float(assemble(position_vector[1]*
    mesh_dataClass.ds(physical_group))))

    centroid_z = (area_inverse*float(assemble(position_vector[2]*
    mesh_dataClass.ds(physical_group))))

    return [centroid_x, centroid_y, centroid_z], area_inverse