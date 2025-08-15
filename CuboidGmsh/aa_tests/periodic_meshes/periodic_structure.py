import gmsh

import numpy as np

import meshio

import CuboidGmsh.tests.periodic_meshes.fiber_matrix_parameters as parameters_file

import CuboidGmsh.source.tool_box.mesh_data_retriever as mesh_data_retriever

import CuboidGmsh.source.tool_box.meshing_tools as mesh_tools

def generate_periodicMesh(file_name, flag_transfinite=1, verbose=
False):

    ####################################################################
    ####################################################################
    ##                        User's settings                         ##
    ####################################################################
    ####################################################################

    # Gets the parameters from the user-defined file

    (parameters_method, RVE_method, RVE_lengthX, RVE_lengthY,
    RVE_lengthZ, n_boxesX, n_boxesY, n_boxesZ, lc, n_transfinite, 
    volume_elementType, surface_elementType, material_phasesNames,
    volume_regionIdentifiers, volume_regionsNames, 
    surface_regionIdentifiers, surface_regionsNames) = parameters_file.generate_parameters(
    flag_transfinite=flag_transfinite)

    ####################################################################
    ####################################################################
    ##                          Calculations                          ##
    ####################################################################
    ####################################################################

    # Initialize GMSH

    gmsh.initialize()

    ####################################################################
    #                          RVE population                          #
    ####################################################################

    # Calculates the number of material phases

    n_materialPhases = len(material_phasesNames)

    # Initializes a dictionary of volumes for each physical group

    dictionary_volumesPhysGroups = dict()

    for i in range(1,n_materialPhases*(1+len(volume_regionIdentifiers))+
    1,1):

        dictionary_volumesPhysGroups[i] = []

    # Initializes a dictionary of surfaces for each physical group

    dictionary_surfacesPhysGroups = dict()

    for i in range(1,len(surface_regionIdentifiers)+1,1):

        dictionary_surfacesPhysGroups[i] = []

    # Adds the names of the regions to the vector of names of the phases.
    # Takes the care, nonetheless, to add the name of each phase to each
    # region

    volume_physicalGroupsNames = [i for i in material_phasesNames]

    for i in range(len(volume_regionIdentifiers)):

        for j in material_phasesNames:

            volume_physicalGroupsNames.append(volume_regionsNames[i]+'_'
            +j)

    # Iterates through the dimensions to generate each RVE that is
    # stacked into the whole ensemble

    # Iterates through the z direction

    for i in range(n_boxesZ):

        # Iterates through the y direction

        for j in range(n_boxesY):

            # Iterates through the x direction

            for k in range(n_boxesX):

                # Calculates the centroid of the bottom surface

                x_centroid = ((k+0.5)*RVE_lengthX)

                y_centroid = ((j+0.5)*RVE_lengthY)

                z_centroid = ((i+0.5)*RVE_lengthZ)

                # Creates the surfaces and the volumes of the RVE

                if flag_transfinite!=0:

                    (dictionary_volumesPhysGroups, 
                    dictionary_surfacesPhysGroups) = RVE_method(
                    x_centroid, y_centroid, z_centroid, RVE_lengthX, 
                    RVE_lengthY, RVE_lengthZ, parameters_method, 
                    dictionary_volumesPhysGroups, 
                    dictionary_surfacesPhysGroups, lc, 
                    volume_regionIdentifiers, surface_regionIdentifiers,
                    n_transfiniteCurves=n_transfinite)

                else:

                    (dictionary_volumesPhysGroups, 
                    dictionary_surfacesPhysGroups) = RVE_method(
                    x_centroid, y_centroid, z_centroid, RVE_lengthX, 
                    RVE_lengthY, RVE_lengthZ, parameters_method, 
                    dictionary_volumesPhysGroups, 
                    dictionary_surfacesPhysGroups, lc, 
                    volume_regionIdentifiers, surface_regionIdentifiers)

    ####################################################################
    #                 Geometry cleanning and stitching                 #
    ####################################################################

    # Removes the duplicate entities and synchronizes the geometry. The
    # removeAllDuplicates function uses boolean operations to stich eve-
    # rything together

    gmsh.model.geo.removeAllDuplicates()

    gmsh.model.geo.synchronize()
    
    ####################################################################
    #                          Physical groups                         #
    ####################################################################

    # Iterates through the volumetric physical groups

    n_volumetricPhysicalGroups = n_materialPhases*(1+len(
    volume_regionIdentifiers))

    for i in range(n_volumetricPhysicalGroups):

        print("\n\nCreates "+volume_physicalGroupsNames[i]+" volumetri"+
        "c physical group\n")

        # Adds the physical group of the (i-1)-th material phase

        gmsh.model.addPhysicalGroup(3, dictionary_volumesPhysGroups[(i+
        1)], i+1)

        gmsh.model.setPhysicalName(3, i+1, volume_physicalGroupsNames[i])

    # Iterates through the surface physical groups

    for i in range(len(surface_regionIdentifiers)):

        print("\n\nCreates "+surface_regionsNames[i]+" surface physica"+
        "l group\n")

        gmsh.model.addPhysicalGroup(2, dictionary_surfacesPhysGroups[i+1
        ], i+1+n_volumetricPhysicalGroups)

        gmsh.model.setPhysicalName(2, i+1+n_volumetricPhysicalGroups, 
        surface_regionsNames[i])

    ####################################################################
    #                               Mesh                               #
    ####################################################################

    # Synchronizes and generates the mesh

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)

    # Sets the gmsh format version and saves the mesh to a file

    gmsh.option.setNumber("Mesh.MshFileVersion",2.2)  

    gmsh.write(file_name+".msh")

    # Reads the saved gmsh mesh using meshio

    mesh_reading = meshio.read(file_name+".msh")

    # Rewrites the mesh using the Mesh function from meshio

    mesh_tools.create_meshioMesh(mesh_reading, [volume_elementType, 
    surface_elementType], ["domain", "boundary"], file_name)

    ####################################################################
    #                    Printing mesh's information                   #
    ####################################################################

    if verbose:

        # Recovers the list of elements of the mesh (including points, 
        # lines, facets and mesh elements)

        entities = gmsh.model.getEntities()

        # Shows the information on the terminal

        mesh_data_retriever.get_meshInfo(entities)

        gmsh.fltk.run()

    # Finalize GMSH

    gmsh.finalize()

########################################################################
#                               Testing                                #
########################################################################

generate_periodicMesh("micro_mesh", flag_transfinite=0, verbose=True)