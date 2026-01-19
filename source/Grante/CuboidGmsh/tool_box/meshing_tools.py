# Routine to store different and useful methods for gmsh using Python

from dolfin import *

import gmsh

import meshio

from ..tool_box import region_finder

from ..tool_box import mesh_data_retriever

from ...PythonicUtilities import path_tools

########################################################################
#                         Meshio - mesh writing                        #
########################################################################

# Defines a function to convert a mesh read by meshio to xdmf. Separates
# all the physical groups, both 3D and 2D. The desired_elements is a
# list of element types that are desired in the FEM mesh. The data_sets 
# is a list of names for data sets of physical groups

def create_meshioMesh(original_mesh, desired_elements, data_sets, 
file_name):

    # Initializes the dictionary of cells and the list of cell data

    cells_dictionary = dict()

    cell_dataList = []

    # Gets the cells which consist of the desired element

    for i in range(len(desired_elements)):

        print("Saves the mesh of", data_sets[i], "dataset\n")

        cells_dictionary[desired_elements[i]] = (
        original_mesh.get_cells_type(desired_elements[i]))

        # Gets the physical cell data for this element

        cell_dataPhysical = original_mesh.get_cell_data("gmsh:physical", 
        desired_elements[i])

        # Gets the geometric cell data for this element

        cell_dataGeometric = original_mesh.get_cell_data("gmsh:geometr"+
        "ical", desired_elements[i])

        # Adds the geometric cell data to the list of cell data

        cell_dataList.append(cell_dataGeometric)

        # Creates a mesh for this data set and saves it

        mesh_set = meshio.Mesh(points=original_mesh.points, cells={
        desired_elements[i]: cells_dictionary[desired_elements[i]]},
        cell_data={data_sets[i]: [cell_dataPhysical]})

        print(file_name+"_"+data_sets[i]+".xdmf")

        meshio.write(file_name+"_"+data_sets[i]+".xdmf", mesh_set)

    # Creates the mesh with all information, including geometric infor-
    # mation and saves it
    
    whole_mesh = meshio.Mesh(points=original_mesh.points, cells=
    cells_dictionary, cell_data={"whole_mesh": cell_dataList})

    meshio.write(file_name+".xdmf", whole_mesh)

########################################################################
#                           Physical groups                            #
########################################################################

# Defines a function to generate the dictionary of physical groups

def generate_physicalDictionary(entities_identifiers, entities_names, 
topological_dimension, generic_group=False):
    
    # Tests if there are the same number of names as the number of iden-
    # tifiers

    if len(entities_names)!=len(entities_identifiers):

        if len(entities_names)<len(entities_identifiers):

            raise ValueError("There are "+str(len(entities_names))+" n"+
            "ames for "+str(len(entities_identifiers))+" identifiers o"+
            "f topological dimension "+str(topological_dimension)+". T"+
            "he quantity of them must be equal.\n")
        
        elif len(entities_names)==0:

            raise ValueError("There are no names for the identifiers o"+
            "f topological dimension "+str(topological_dimension)+"\n")

    # Creates the dictionary entities for each physical group

    entities_dictionary = dict()

    # If a generic group is to be added, adds it with 0 tag

    if generic_group:

        entities_dictionary[0] = []

    # Proceeds to add the solicited physical groups

    for i in range(len(entities_names)):

        entities_dictionary[i+1] = []

    return entities_dictionary

# Defines a function to create physical groups from a list of names and
# from a dictionary of gmsh entities

def generate_physicalGroups(topological_dimension, groups_names, 
entities_dictionary, initial_groupNumber=1, add_genericGroup=False):
    
    group_type = 'volume'

    if topological_dimension==2:

        group_type = 'surface'

    # If a generic group is solicited, adds it

    if add_genericGroup and (0 in entities_dictionary.keys()):

        if len(entities_dictionary[0])>0:

            print("\nCreates the generic "+group_type+" physical group")

            gmsh.model.addPhysicalGroup(topological_dimension, 
            entities_dictionary[0], initial_groupNumber, name='Generic '
            +group_type)

            initial_groupNumber += 1

            gmsh.model.geo.synchronize()

    # Iterates through the surface physical groups

    for i in range(len(groups_names)):

        # Tests whether entities were really found

        if len(entities_dictionary[i+1])>0:

            print("\nCreates "+groups_names[i]+" "+group_type+" physic"+
            "al group")

            gmsh.model.addPhysicalGroup(topological_dimension, 
            entities_dictionary[i+1], initial_groupNumber, name=
            groups_names[i])

            initial_groupNumber += 1

        else:

            print("\n"+groups_names[i]+" "+group_type+" physical group"+
            " will not be created for no entities were found in it")

    gmsh.model.geo.synchronize()

    # Returns the number for the next physical group

    return initial_groupNumber

########################################################################
#                                 GMSH                                 #
########################################################################

# Defines a function to initialize GMSH and pre-process some needed in-
# formation

def gmsh_initialization(lc=0.5, surface_regionsExpressions=[], 
hexahedron_mesh=False, volume_regionsExpressions=[], 
surface_regionsNames=[], volume_regionsNames=[], tolerance_finders=1E-4):

    # Initializes the gmsh environment

    gmsh.initialize()

    ####################################################################
    #                     Surface physical groups                      #
    ####################################################################

    # Sets the list of identifier functions for surfaces

    surface_regionIdentifiers = []

    for i in range(len(surface_regionsExpressions)):

        # Uses the lambda function to set a driver, and uses the i=i 
        # snippet to set the i variable at the current iteration value

        surface_regionIdentifiers.append(lambda points_set , i=i: (
        region_finder.general_2Dboundary(points_set, 
        surface_regionsExpressions[i], tolerance=tolerance_finders)))

    # Creates the dictionary of surfaces for each physical groups

    dictionary_surfacesPhysGroups = generate_physicalDictionary(
    surface_regionIdentifiers, surface_regionsNames, 2)

    ####################################################################
    #                      Volume physical groups                      #
    ####################################################################

    # Sets the list of volume regions identifiers

    volume_regionIdentifiers = []

    for i in range(len(volume_regionsExpressions)):

        # Uses the lambda function to set a driver, and uses the i=i 
        # snippet to set the i variable at the current iteration value

        volume_regionIdentifiers.append(lambda points_set, i=i: (
        region_finder.general_3DEnclosure(points_set, 
        volume_regionsExpressions[i])))

    # Creates the dictionary of volume physical froups. Turns on the 
    # flag to add a generic physical group to account for the volumes
    # that are not inside any solicited physical group. Fenics, for e-
    # xample, requires the volumes to be all in physical groups

    dictionary_volumePhysGroups = generate_physicalDictionary(
    volume_regionIdentifiers, volume_regionsNames, 3, generic_group=
    True)

    ####################################################################
    #                        Geometric data list                       #
    ####################################################################

    # Initializes the list of geometric data that will be updated at 
    # each volume created

    geometric_data = [0, [[],[],[],[]], [[],[],[],[]], [[],[],[]], 
    dictionary_surfacesPhysGroups, surface_regionIdentifiers,
    dictionary_volumePhysGroups, volume_regionIdentifiers, 
    surface_regionsNames, volume_regionsNames, lc, hexahedron_mesh]

    return geometric_data

# Defines a function to finalize the gmsh program

def gmsh_finalize(mesh_topologicalDimension=3, mesh_polynomialOrder=1, 
geometric_data=[0, [[],[],[],[]], [[],[],[],[]], [[],[],[]], dict(), [], 
dict(), [], [], []], verbose=True, gmsh_version=2.2, file_name="mesh", 
file_directory=None, volume_elementType='tetra', surface_elementType=
'triangle', hexahedron_mesh=False, convert_to_xdmf=True):
    
    # Checks if the mesh is made of hexahedrons

    if hexahedron_mesh:

            volume_elementType = 'hexahedron'

            surface_elementType = 'quad'
    
    # Retrieves the geometric information

    dictionary_surfacesPhysGroups = geometric_data[4]

    dictionary_volumesPhysGroups = geometric_data[6]
    
    surface_regionsNames = geometric_data[8]
    
    volume_regionsNames = geometric_data[9]

    # Deletes the duplicated entities

    gmsh.model.geo.removeAllDuplicates()

    gmsh.model.geo.synchronize()

    ####################################################################
    #                          Physical groups                         #
    ####################################################################

    # Adds the volume physical groups

    n_physicalGroups = generate_physicalGroups(3, volume_regionsNames, 
    dictionary_volumesPhysGroups, add_genericGroup=True)

    # Adds the surface physical groups

    generate_physicalGroups(2, surface_regionsNames, 
    dictionary_surfacesPhysGroups, initial_groupNumber=n_physicalGroups)

    # Generates the mesh

    if mesh_topologicalDimension>0:

        gmsh.model.mesh.generate(mesh_topologicalDimension)

        # Sets the order of the polynomial

        gmsh.model.mesh.setOrder(mesh_polynomialOrder)

        # If no directory has been given to save the mesh in, creates 
        # one automatically

        if file_directory==None:

            file_directory = path_tools.get_parent_path_of_file(
            function_calls_to_retrocede=2)

        file_name = path_tools.verify_path(file_directory, file_name)

        # Sets the gmsh format version and saves the mesh to a file

        gmsh.option.setNumber("Mesh.MshFileVersion", gmsh_version)  

        gmsh.write(file_name+".msh")

        # Reads the saved gmsh mesh using 
        
        mesh_reading = None 

        if convert_to_xdmf:

            mesh_reading = meshio.read(file_name+".msh")

        # Verifies if the volume physical group is empty

        flag_volumeEmpty = verify_physicalGroupsEmptiness(
        dictionary_volumesPhysGroups)

        # Verifies if the surface physical group is empty

        flag_surfaceEmpty = verify_physicalGroupsEmptiness(
        dictionary_surfacesPhysGroups)

        if flag_volumeEmpty:

            print("\nWARNING: there is no volume physical group. No me"+
            "sh in .xdmf will be\nsaved.\n")

        elif flag_surfaceEmpty:

            print("\nWARNING: there is no surface physical group, alth"+
            "ough there are volume\nphysical groups. Volumetric mesh o"+
            "nly will be saved.\n")

            # Rewrites the mesh using the Mesh function from meshio

            if convert_to_xdmf:

                create_meshioMesh(mesh_reading, [volume_elementType], [
                "domain"], file_name)

        else:

            # Rewrites the mesh using the Mesh function from meshio

            if convert_to_xdmf:

                create_meshioMesh(mesh_reading, [volume_elementType, 
                surface_elementType], ["domain", "boundary"], file_name)

        # If the verbose flag is True, shows the result mesh

    if verbose:

        # Recovers the list of elements of the mesh (including points, 
        # lines, facets and mesh elements)

        entities = gmsh.model.getEntities()

        # Shows the information on the terminal

        mesh_data_retriever.get_meshInfo(entities)

        gmsh.fltk.run()

    gmsh.finalize()

# Defines a function to verify if there is at least one volume in physi-
# cal groups

def verify_physicalGroupsEmptiness(physical_groupDictionary):

    # Verifies the values

    for entities_list in physical_groupDictionary.values():

        if len(entities_list)>0:

            return False
        
    # If no list with at least one element has been found so far, there
    # is no entity in the physical group

    return True

########################################################################
#                         Transfinite meshing                          #
########################################################################

# Defines a function to retrieve transfinite and bias data when there 
# are more than 3 directions

def retrieve_transfiniteAndBiasData(transfinite_directions, 
set_ofDirections, bias_directions=dict()):

    # Sets a list of transfinite variables

    set_transfiniteVariables = []

    # Verifies whether there are indeed transfinite especifications

    if len(transfinite_directions)>0:

        # Verifies if there is the same number of variables and of 
        # transfinite directions

        if len(transfinite_directions)!=len(set_ofDirections):

            raise ValueError("The number of transfinite directions,",
            len(transfinite_directions), ", is different than the numb"+
            "er of transfinite variables,", len(set_ofDirections), "\n")

        for i in range(len(transfinite_directions)):

            set_transfiniteVariables.append(transfinite_directions[i])

    else:

        for i in range(len(set_ofDirections)):

            set_transfiniteVariables.append(0)

    # Recovers the biases

    set_biasVariables = []

    # Iterates through the directions

    if len(bias_directions.keys())>0:

        for direction in set_ofDirections:

            # Verifies if this direction is in the bias directions dic-
            # tionary

            if direction in bias_directions.keys():

                set_biasVariables.append(bias_directions[direction])

            else:

                raise KeyError("There is no", direction, "direction in"+
                " the bias directions dictionary.\n")

    else:

        for i in range(len(set_ofDirections)):

            set_biasVariables.append(1.0)
        
    return set_transfiniteVariables, set_biasVariables

# Defines a function to make a transfinite 3D mesh

def make_transfinite(lines_dictionary, surfaces_dictionary, 
volumes_dictionary, transfinite_sizingGroups, progress_factor=dict(),
hexahedron_mesh=False, color_RGB=[120,40,115]):
    
    # Gets the groups for progressed mesh (bias)

    progress_groups = list(progress_factor.keys())

    # Modifies lines to be transfinite entities

    for line in lines_dictionary:

        # Gets the line number

        line_number = 0

        try:

            line_number = int(line[1:])

        except NameError:
    
            print("The key of the lines dictionary must be lX, where X"+
            " is the number of the line.")
        
        for group in transfinite_sizingGroups:

            # Gets the sizing group

            sizing_group = transfinite_sizingGroups[group]

            # Checks if this sizing group is in the progress groups too

            if group in progress_groups:

                # Verifies if this line belongs to this sizing group

                if line_number in sizing_group[1]:

                    gmsh.model.geo.mesh.setTransfiniteCurve(
                    lines_dictionary["l"+str(line_number)], sizing_group[
                    0], "Progression", progress_factor[group])

                    break

            else:

                # Verifies if this line belongs to this sizing group

                if line_number in sizing_group[1]:

                    gmsh.model.geo.mesh.setTransfiniteCurve(
                    lines_dictionary["l"+str(line_number)], sizing_group[
                    0])

                    break

    # Updates surfaces and volumes

    for surface in surfaces_dictionary:

        gmsh.model.geo.mesh.setTransfiniteSurface(
        surfaces_dictionary[surface], "Left")

        # Recombines the mesh for using hexahedron mesh

        if hexahedron_mesh:

            gmsh.model.geo.mesh.setRecombine(2, surfaces_dictionary[
            surface])

    gmsh.model.geo.synchronize()

    for volume in volumes_dictionary:

        gmsh.model.geo.mesh.setTransfiniteVolume(volumes_dictionary[
        volume])

        gmsh.model.setColor([(3, volumes_dictionary[volume])], 
        *color_RGB)

    gmsh.model.geo.synchronize()

# Defines a function to programatically construct surfaces using a dic-
# tionary of curve loops as input. A optional argument is given, a list
# of loops that are meant to be surfaced using surface filling method

def surfaces_fromLoopsDict(loops_dictionary, surfaces_dictionary=dict(),
loops_surfaceFilling=[]):
    
    for loop in loops_dictionary:

        # Gets the number of the curve loop

        loop_number = 0

        try:

            loop_number = int(loop[4:])

        except NameError:
    
            print("The key of the curve loops dictionary must be loopX"+
            ", where X is the number of the loop.")

        # Verifies if the loop number is inside the list of surfaces to
        # be created using surface filling method

        if loop_number in loops_surfaceFilling:

            surfaces_dictionary["surface"+str(loop_number)] = gmsh.model.geo.addSurfaceFilling(
            [loops_dictionary[loop]])

            print("Loop number", loop_number, "Surface filling", 
            surfaces_dictionary["surface"+str(loop_number)])

        else:

            surfaces_dictionary["surface"+str(loop_number)] = gmsh.model.geo.addPlaneSurface(
            [loops_dictionary[loop]])

    # Returns the updated dictionary of surfaces

    return surfaces_dictionary

# Defines a function to get the coordinates of the points on the bounda-
# ry of a surface

def get_boudaryPointsSurface(surface_tag):

    # Initializes a list of coordinates of points

    points_coordinates = []

    # Gets the tags of the lines that bound this surface

    bound_lines = gmsh.model.getBoundary([(2,surface_tag)])

    # Iterates through the boundary lines

    for i in range(len(bound_lines)):

        # Gets the points that bound this line

        bound_points = gmsh.model.getBoundary([(1,bound_lines[i][1])])

        # Gets the first point and adds it to the list of coordinates

        points_coordinates.append(gmsh.model.getValue(0,bound_points[0][
        1],[]))

    # Returns the list of coordinates

    return points_coordinates

# Defines a function to get the boundary points of a line

def get_boundaryPointsLine(line_tag):

    return gmsh.model.getBoundary([(1,line_tag)], oriented=True)