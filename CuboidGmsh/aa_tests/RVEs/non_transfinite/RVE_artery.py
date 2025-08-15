import gmsh

import numpy as np

import CuboidGmsh.tests.RVEs.non_transfinite.elastin_fiber_creator as elastin

import CuboidGmsh.source.tool_box.meshing_tools as tools

import CuboidGmsh.source.tool_box.region_finder as region_finder

def RVE_elastinMicrostructure(x_centroid, y_centroid, z_centroid, 
RVE_lengthX, RVE_lengthY, RVE_lengthZ, parameters_method, 
dictionary_volumesPhysGroups, dictionary_surfacesPhysGroups, lc, 
volume_regionIdentifiers, surface_regionIdentifiers):

    ####################################################################
    #                     User-defined parameters                      #
    ####################################################################

    # Retrieves parameters of the arterial microstructure

    fiber_diameter = parameters_method[0]

    polar_angle = parameters_method[1]

    azimuthal_angle = parameters_method[2]

    collagen_thickness = parameters_method[3]

    lamella_thickness = parameters_method[4]

    # Calculates the intralamellar thickness by subtracting the lamella
    # thickness off of the RVE z length

    intralamellar_thickness = RVE_lengthZ-lamella_thickness

    # Retrieves the allowable distance between the tip of the elastin 
    # strut to the edge of the RVE

    clearance_strutsEdges = parameters_method[5]

    # Retrieves the number of struts along the x and the y axes

    n_strutsX = parameters_method[6]

    n_strutsY = parameters_method[7]

    ####################################################################
    #              Calculation - Authorized personel only              #
    ####################################################################
    
    # Creates a shifted centroid on a corner of the top of the bottom
    # collagen layer

    x_shiftedCentroid = x_centroid-(0.5*RVE_lengthX)

    y_shiftedCentroid = y_centroid-(0.5*RVE_lengthY)

    z_shiftedCentroid = z_centroid-(0.5*intralamellar_thickness)

    # Creates the dictionaries for points that belongs to the edges of
    # the boundary curves of the surfaces

    surfaces_points = dict()

    # To retrieves the curve loops of the elastin struts that belong to 
    # each surface of the collagen layers, creates lists of these enti-
    # ties. Retrieves the surfaces and the volume of the elastin struts 
    # that are buried within the collagen layer too

    bottom_collagenBottomLoopsList = []

    bottom_collagenTopLoopsList = []

    top_collagenBottomLoopsList = []

    top_collagenTopLoopsList = []

    bottom_collagenSurfaces = []

    top_collagenSurfaces = []

    muscular_surfaces = []

    bottom_lamellaSurfaces = []

    top_lamellaSurfaces = []

    elastin_volumes = []

    ####################################################################
    #                          Pre-processing                          #
    ####################################################################

    # Evaluates the projection of the elastin strut vector over the bot-
    # tom plane

    projection_x = (intralamellar_thickness*np.tan(azimuthal_angle)*
    np.cos(polar_angle))

    projection_y = (intralamellar_thickness*np.tan(azimuthal_angle)*
    np.sin(polar_angle))

    # Calculates the distance between centroids of elastin struts

    delta_elastinX = ((RVE_lengthX-fiber_diameter-(2*clearance_strutsEdges
    )-projection_x)/(n_strutsX-1))

    delta_elastinY = ((RVE_lengthY-fiber_diameter-(2*clearance_strutsEdges
    )-projection_y)/(n_strutsY-1))

    ####################################################################
    #                          Elastin struts                          #
    ####################################################################

    # Iterates through the elastin struts in the x direction

    for i in range(n_strutsX):

        # Iterates through the elastin struts in the y direction

        for j in range(n_strutsY):

            # Evaluates the centroid of the bottom surface of this elas-
            # tin strut. Corrects the centroid of the RVE so the system 
            # of coordinates is located in a corner over the top face of
            # the bottom collagen layer

            inferior_fiberCentroid = [(delta_elastinX*i)+
            clearance_strutsEdges+(0.5*fiber_diameter)+x_shiftedCentroid,
            (delta_elastinY*j)+clearance_strutsEdges+(0.5*fiber_diameter
            )+y_shiftedCentroid, z_shiftedCentroid]

            # Constructs the elastin strut and throws out the dictiona-
            # ries of curve loops and surfaces of this elastin strut

            (lines_dictionary, loops_dictionary, surfaces_dictionary,
            elastin_volume) = elastin.elastin_fiber(
            inferior_fiberCentroid, fiber_diameter, polar_angle, 
            azimuthal_angle, collagen_thickness, intralamellar_thickness, 
            lc)

            # Updates the list of volumes of elastin struts

            elastin_volumes.append(elastin_volume)

            # Updates the loops and surfaces

            bottom_collagenBottomLoopsList.append(loops_dictionary[
            "loop1"])

            bottom_lamellaSurfaces.append(surfaces_dictionary[
            "surface1"])

            bottom_collagenTopLoopsList.append(gmsh.model.geo.addCurveLoop([
            lines_dictionary["l9"], lines_dictionary["l10"], 
            lines_dictionary["l11"], lines_dictionary["l12"]]))

            top_collagenBottomLoopsList.append(gmsh.model.geo.addCurveLoop([
            lines_dictionary["l17"], lines_dictionary["l18"], 
            lines_dictionary["l19"], lines_dictionary["l20"]]))

            top_collagenTopLoopsList.append(loops_dictionary["loop14"])

            top_lamellaSurfaces.append(surfaces_dictionary["surface14"])

            for k in range(2,6):

                bottom_collagenSurfaces.append(surfaces_dictionary[
                "surface"+str(k)])

            for k in range(6,10):

                muscular_surfaces.append(surfaces_dictionary["surface"
                +str(k)])

            for k in range(10,14):

                top_collagenSurfaces.append(surfaces_dictionary[
                "surface"+str(k)])

    # As the elastin surfaces are not used for physical groups, a new
    # surface dictionary will be created now

    surfaces_dictionary = dict()

    ####################################################################
    #                      Bottom collagen layer                       #
    ####################################################################

    # Creates the points for the layer

    bottom_p11 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid, z_shiftedCentroid, lc)

    bottom_p12 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid, z_shiftedCentroid, lc)

    bottom_p13 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid, lc)

    bottom_p14 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid, lc)

    bottom_p21 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid, z_shiftedCentroid+collagen_thickness, lc)

    bottom_p22 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid, z_shiftedCentroid+collagen_thickness, lc)

    bottom_p23 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid+collagen_thickness, 
    lc)

    bottom_p24 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid+collagen_thickness, 
    lc)

    # Creates the lines

    bottom_l01 = gmsh.model.geo.addLine(bottom_p11, bottom_p12)

    bottom_l02 = gmsh.model.geo.addLine(bottom_p12, bottom_p13)

    bottom_l03 = gmsh.model.geo.addLine(bottom_p13, bottom_p14)

    bottom_l04 = gmsh.model.geo.addLine(bottom_p14, bottom_p11)

    bottom_l05 = gmsh.model.geo.addLine(bottom_p11, bottom_p21)

    bottom_l06 = gmsh.model.geo.addLine(bottom_p12, bottom_p22)

    bottom_l07 = gmsh.model.geo.addLine(bottom_p13, bottom_p23)

    bottom_l08 = gmsh.model.geo.addLine(bottom_p14, bottom_p24)

    bottom_l09 = gmsh.model.geo.addLine(bottom_p21, bottom_p22)

    bottom_l10 = gmsh.model.geo.addLine(bottom_p22, bottom_p23)

    bottom_l11 = gmsh.model.geo.addLine(bottom_p23, bottom_p24)

    bottom_l12 = gmsh.model.geo.addLine(bottom_p24, bottom_p21)

    # Creates the curve loops

    loop1 = gmsh.model.geo.addCurveLoop([bottom_l01, bottom_l02, 
    bottom_l03, bottom_l04])

    loop2 = gmsh.model.geo.addCurveLoop([bottom_l01, bottom_l06, 
    -bottom_l09, -bottom_l05])

    loop3 = gmsh.model.geo.addCurveLoop([bottom_l02, bottom_l07, 
    -bottom_l10, -bottom_l06])

    loop4 = gmsh.model.geo.addCurveLoop([bottom_l03, bottom_l08, 
    -bottom_l11, -bottom_l07])

    loop5 = gmsh.model.geo.addCurveLoop([bottom_l04, bottom_l05, 
    -bottom_l12, -bottom_l08])

    loop6 = gmsh.model.geo.addCurveLoop([bottom_l09, bottom_l10, 
    bottom_l11, bottom_l12])

    # Updates the list of loops to to add the loops of the collagen 
    # layer

    bottom_collagenBottomLoopsList = [loop1, 
    *bottom_collagenBottomLoopsList]

    bottom_collagenTopLoopsList = [loop6, 
    *bottom_collagenTopLoopsList]

    # Creates the surfaces of the bottom collagen layer

    bottom_surface1 = gmsh.model.geo.addPlaneSurface(
    bottom_collagenBottomLoopsList)

    bottom_surface2 = gmsh.model.geo.addPlaneSurface([loop2])

    bottom_surface3 = gmsh.model.geo.addPlaneSurface([loop3])

    bottom_surface4 = gmsh.model.geo.addPlaneSurface([loop4])

    bottom_surface5 = gmsh.model.geo.addPlaneSurface([loop5])

    bottom_surface6 = gmsh.model.geo.addPlaneSurface(
    bottom_collagenTopLoopsList)

    gmsh.model.geo.synchronize()

    # Adds the boundary points of the surfaces to their corresponding 
    # dictionary. Does this only for the lateral surfaces

    surfaces_points["surface1"] = tools.get_boudaryPointsSurface(
    bottom_surface2)

    surfaces_points["surface2"] = tools.get_boudaryPointsSurface(
    bottom_surface3)

    surfaces_points["surface3"] = tools.get_boudaryPointsSurface(
    bottom_surface4)

    surfaces_points["surface4"] = tools.get_boudaryPointsSurface(
    bottom_surface5)

    surfaces_dictionary["surface1"] = bottom_surface2

    surfaces_dictionary["surface2"] = bottom_surface3

    surfaces_dictionary["surface3"] = bottom_surface4

    surfaces_dictionary["surface4"] = bottom_surface5

    # Updates the list of surfaces of the bottom collagen layer

    bottom_collagenSurfaces = [bottom_surface1, bottom_surface2, 
    bottom_surface3, bottom_surface4, bottom_surface5, bottom_surface6, 
    *bottom_collagenSurfaces]

    # Creates the volume of the bottom collagen layer

    bottom_collagenAssembly = gmsh.model.geo.addSurfaceLoop(
    bottom_collagenSurfaces)

    bottom_collagenVolume = gmsh.model.geo.addVolume([
    bottom_collagenAssembly])

    gmsh.model.geo.synchronize()

    ####################################################################
    #                        Top collagen layer                        #
    ####################################################################

    # Creates the points for the layer

    top_p11 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid, z_shiftedCentroid+intralamellar_thickness-
    collagen_thickness, lc)

    top_p12 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid, z_shiftedCentroid+intralamellar_thickness-
    collagen_thickness, lc)

    top_p13 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid+
    intralamellar_thickness-collagen_thickness, lc)

    top_p14 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid+
    intralamellar_thickness-collagen_thickness, lc)

    top_p21 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid, z_shiftedCentroid+intralamellar_thickness, lc)

    top_p22 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid, z_shiftedCentroid+intralamellar_thickness, lc)

    top_p23 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid+
    intralamellar_thickness, lc)

    top_p24 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid+
    intralamellar_thickness, lc)

    # Creates the lines

    top_l01 = gmsh.model.geo.addLine(top_p11, top_p12)

    top_l02 = gmsh.model.geo.addLine(top_p12, top_p13)

    top_l03 = gmsh.model.geo.addLine(top_p13, top_p14)

    top_l04 = gmsh.model.geo.addLine(top_p14, top_p11)

    top_l05 = gmsh.model.geo.addLine(top_p11, top_p21)

    top_l06 = gmsh.model.geo.addLine(top_p12, top_p22)

    top_l07 = gmsh.model.geo.addLine(top_p13, top_p23)

    top_l08 = gmsh.model.geo.addLine(top_p14, top_p24)

    top_l09 = gmsh.model.geo.addLine(top_p21, top_p22)

    top_l10 = gmsh.model.geo.addLine(top_p22, top_p23)

    top_l11 = gmsh.model.geo.addLine(top_p23, top_p24)

    top_l12 = gmsh.model.geo.addLine(top_p24, top_p21)

    # Creates the curve loops

    loop1 = gmsh.model.geo.addCurveLoop([top_l01, top_l02, top_l03, 
    top_l04])

    loop2 = gmsh.model.geo.addCurveLoop([top_l01, top_l06, -top_l09, 
    -top_l05])

    loop3 = gmsh.model.geo.addCurveLoop([top_l02, top_l07, -top_l10, 
    -top_l06])

    loop4 = gmsh.model.geo.addCurveLoop([top_l03, top_l08, -top_l11, 
    -top_l07])

    loop5 = gmsh.model.geo.addCurveLoop([top_l04, top_l05, -top_l12, 
    -top_l08])

    loop6 = gmsh.model.geo.addCurveLoop([top_l09, top_l10, top_l11, 
    top_l12])

    # Updates the list of loops to to add the loops of the collagen 
    # layer

    top_collagenBottomLoopsList = [loop1, *top_collagenBottomLoopsList]

    top_collagenTopLoopsList = [loop6, *top_collagenTopLoopsList]

    # Creates the surfaces of the top collagen layer

    top_surface1 = gmsh.model.geo.addPlaneSurface(
    top_collagenBottomLoopsList)

    top_surface2 = gmsh.model.geo.addPlaneSurface([loop2])

    top_surface3 = gmsh.model.geo.addPlaneSurface([loop3])

    top_surface4 = gmsh.model.geo.addPlaneSurface([loop4])

    top_surface5 = gmsh.model.geo.addPlaneSurface([loop5])

    top_surface6 = gmsh.model.geo.addPlaneSurface(
    top_collagenTopLoopsList)

    gmsh.model.geo.synchronize()

    # Adds the boundary points of the surfaces to their corresponding 
    # dictionary. Does this only for the lateral surfaces

    surfaces_points["surface5"] = tools.get_boudaryPointsSurface(
    top_surface2)

    surfaces_points["surface6"] = tools.get_boudaryPointsSurface(
    top_surface3)

    surfaces_points["surface7"] = tools.get_boudaryPointsSurface(
    top_surface4)

    surfaces_points["surface8"] = tools.get_boudaryPointsSurface(
    top_surface5)

    surfaces_dictionary["surface5"] = top_surface2

    surfaces_dictionary["surface6"] = top_surface3

    surfaces_dictionary["surface7"] = top_surface4

    surfaces_dictionary["surface8"] = top_surface5

    # Updates the list of surfaces of the top collagen layer

    top_collagenSurfaces = [top_surface1, top_surface2, top_surface3, 
    top_surface4, top_surface5, top_surface6, *top_collagenSurfaces]

    # Creates the volume of the top collagen layer

    top_collagenAssembly = gmsh.model.geo.addSurfaceLoop(
    top_collagenSurfaces)

    top_collagenVolume = gmsh.model.geo.addVolume([
    top_collagenAssembly])

    gmsh.model.geo.synchronize()

    ####################################################################
    #                         Muscular filling                         #
    ####################################################################

    # Creates the lateral corner lines

    l5 = gmsh.model.geo.addLine(bottom_p21, top_p11)

    l6 = gmsh.model.geo.addLine(bottom_p22, top_p12)

    l7 = gmsh.model.geo.addLine(bottom_p23, top_p13)

    l8 = gmsh.model.geo.addLine(bottom_p24, top_p14)

    # Creates the lateral curve loops

    loop2 = gmsh.model.geo.addCurveLoop([bottom_l09, l6, -top_l01, -l5])

    loop3 = gmsh.model.geo.addCurveLoop([bottom_l10, l7, -top_l02, -l6])

    loop4 = gmsh.model.geo.addCurveLoop([bottom_l11, l8, -top_l03, -l7])

    loop5 = gmsh.model.geo.addCurveLoop([bottom_l12, l5, -top_l04, -l8])

    # Creates the lateral surfaces

    surface2 = gmsh.model.geo.addPlaneSurface([loop2])

    surface3 = gmsh.model.geo.addPlaneSurface([loop3])

    surface4 = gmsh.model.geo.addPlaneSurface([loop4])

    surface5 = gmsh.model.geo.addPlaneSurface([loop5])

    gmsh.model.geo.synchronize()

    # Adds the boundary points of the surfaces to their corresponding 
    # dictionary. Does this only for the lateral surfaces

    surfaces_points["surface9"] = tools.get_boudaryPointsSurface(
    surface2)

    surfaces_points["surface10"] = tools.get_boudaryPointsSurface(
    surface3)

    surfaces_points["surface11"] = tools.get_boudaryPointsSurface(
    surface4)

    surfaces_points["surface12"] = tools.get_boudaryPointsSurface(
    surface5)

    surfaces_dictionary["surface9"] = surface2

    surfaces_dictionary["surface10"] = surface3

    surfaces_dictionary["surface11"] = surface4

    surfaces_dictionary["surface12"] = surface5

    # Creates the volume

    assembly_volume = gmsh.model.geo.addSurfaceLoop(
    [bottom_surface6, surface2, surface3, surface4, surface5, 
    top_surface1, *muscular_surfaces])

    muscular_volume = gmsh.model.geo.addVolume([assembly_volume])

    gmsh.model.geo.synchronize()

    ####################################################################
    #                          Bottom lamella                          #
    ####################################################################

    # Halves the lamella thickness, so that half lamella is over and un-
    # der the inner microstructure

    lamella_thickness = 0.5*lamella_thickness

    # Adds the points to the bottom lamella

    bottom_p1 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid, z_shiftedCentroid-lamella_thickness, lc)

    bottom_p2 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid,z_shiftedCentroid-lamella_thickness, lc)

    bottom_p3 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid-lamella_thickness, 
    lc)

    bottom_p4 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid+RVE_lengthY, z_shiftedCentroid-lamella_thickness, 
    lc)

    # Adds the lines

    bottom_l1 = gmsh.model.geo.addLine(bottom_p1, bottom_p2)

    bottom_l2 = gmsh.model.geo.addLine(bottom_p2, bottom_p3)

    bottom_l3 = gmsh.model.geo.addLine(bottom_p3, bottom_p4)

    bottom_l4 = gmsh.model.geo.addLine(bottom_p4, bottom_p1)

    bottom_l5 = gmsh.model.geo.addLine(bottom_p1, bottom_p11)

    bottom_l6 = gmsh.model.geo.addLine(bottom_p2, bottom_p12)

    bottom_l7 = gmsh.model.geo.addLine(bottom_p3, bottom_p13)

    bottom_l8 = gmsh.model.geo.addLine(bottom_p4, bottom_p14)

    # Adds the surfaces

    loop1 = gmsh.model.geo.addCurveLoop([bottom_l1, bottom_l2, 
    bottom_l3, bottom_l4])

    loop2 = gmsh.model.geo.addCurveLoop([bottom_l1, bottom_l6, 
    -bottom_l01, -bottom_l5])

    loop3 = gmsh.model.geo.addCurveLoop([bottom_l2, bottom_l7, 
    -bottom_l02, -bottom_l6])

    loop4 = gmsh.model.geo.addCurveLoop([bottom_l3, bottom_l8, 
    -bottom_l03, -bottom_l7])

    loop5 = gmsh.model.geo.addCurveLoop([bottom_l4, bottom_l5, 
    -bottom_l04, -bottom_l8])

    bottom_lamellaSurface1 = gmsh.model.geo.addPlaneSurface([loop1])

    bottom_lamellaSurface2 = gmsh.model.geo.addPlaneSurface([loop2])

    bottom_lamellaSurface3 = gmsh.model.geo.addPlaneSurface([loop3])

    bottom_lamellaSurface4 = gmsh.model.geo.addPlaneSurface([loop4])

    bottom_lamellaSurface5 = gmsh.model.geo.addPlaneSurface([loop5])

    gmsh.model.geo.synchronize()

    # Adds the boundary points of the surfaces to their corresponding 
    # dictionary. Does this only for the lateral and bottom surfaces

    surfaces_points["surface13"] = tools.get_boudaryPointsSurface(
    bottom_lamellaSurface1)

    surfaces_points["surface14"] = tools.get_boudaryPointsSurface(
    bottom_lamellaSurface2)

    surfaces_points["surface15"] = tools.get_boudaryPointsSurface(
    bottom_lamellaSurface3)

    surfaces_points["surface16"] = tools.get_boudaryPointsSurface(
    bottom_lamellaSurface4)

    surfaces_points["surface17"] = tools.get_boudaryPointsSurface(
    bottom_lamellaSurface5)

    surfaces_dictionary["surface13"] = bottom_lamellaSurface1

    surfaces_dictionary["surface14"] = bottom_lamellaSurface2

    surfaces_dictionary["surface15"] = bottom_lamellaSurface3

    surfaces_dictionary["surface16"] = bottom_lamellaSurface4

    surfaces_dictionary["surface17"] = bottom_lamellaSurface5

    # Adds the volume

    assembly_volume = gmsh.model.geo.addSurfaceLoop([
    bottom_lamellaSurface1, bottom_lamellaSurface2, 
    bottom_lamellaSurface3, bottom_lamellaSurface4, 
    bottom_lamellaSurface5, bottom_surface1, *bottom_lamellaSurfaces])

    bottom_lamellaVolume = gmsh.model.geo.addVolume([assembly_volume])

    gmsh.model.geo.synchronize()

    ####################################################################
    #                           Top lamella                            #
    ####################################################################

    # Calculates the RVE height

    z_height = z_centroid+(0.5*RVE_lengthZ)

    # Adds the points to the top lamella

    top_p1 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid, z_height, lc)

    top_p2 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid, z_height, lc)

    top_p3 = gmsh.model.geo.addPoint(x_shiftedCentroid+RVE_lengthX, 
    y_shiftedCentroid+RVE_lengthY, z_height, lc)

    top_p4 = gmsh.model.geo.addPoint(x_shiftedCentroid, 
    y_shiftedCentroid+RVE_lengthY, z_height, lc)

    # Adds the lines

    top_l1 = gmsh.model.geo.addLine(top_p1, top_p2)

    top_l2 = gmsh.model.geo.addLine(top_p2, top_p3)

    top_l3 = gmsh.model.geo.addLine(top_p3, top_p4)

    top_l4 = gmsh.model.geo.addLine(top_p4, top_p1)

    top_l5 = gmsh.model.geo.addLine(top_p21, top_p1)

    top_l6 = gmsh.model.geo.addLine(top_p22, top_p2)

    top_l7 = gmsh.model.geo.addLine(top_p23, top_p3)

    top_l8 = gmsh.model.geo.addLine(top_p24, top_p4)

    # Adds the surfaces

    loop1 = gmsh.model.geo.addCurveLoop([top_l1, top_l2, top_l3,
    top_l4])

    loop2 = gmsh.model.geo.addCurveLoop([top_l1, -top_l6, -top_l09,
    top_l5])

    loop3 = gmsh.model.geo.addCurveLoop([top_l2, -top_l7, -top_l10,
    top_l6])

    loop4 = gmsh.model.geo.addCurveLoop([top_l3, -top_l8, -top_l11,
    top_l7])

    loop5 = gmsh.model.geo.addCurveLoop([top_l4, -top_l5, -top_l12,
    top_l8])

    top_lamellaSurface1 = gmsh.model.geo.addPlaneSurface([loop1])

    top_lamellaSurface2 = gmsh.model.geo.addPlaneSurface([loop2])

    top_lamellaSurface3 = gmsh.model.geo.addPlaneSurface([loop3])

    top_lamellaSurface4 = gmsh.model.geo.addPlaneSurface([loop4])

    top_lamellaSurface5 = gmsh.model.geo.addPlaneSurface([loop5])

    gmsh.model.geo.synchronize()

    # Adds the boundary points of the surfaces to their corresponding 
    # dictionary. Does this only for the lateral and bottom surfaces

    surfaces_points["surface18"] = tools.get_boudaryPointsSurface(
    top_lamellaSurface1)

    surfaces_points["surface19"] = tools.get_boudaryPointsSurface(
    top_lamellaSurface2)

    surfaces_points["surface20"] = tools.get_boudaryPointsSurface(
    top_lamellaSurface3)

    surfaces_points["surface21"] = tools.get_boudaryPointsSurface(
    top_lamellaSurface4)

    surfaces_points["surface22"] = tools.get_boudaryPointsSurface(
    top_lamellaSurface5)

    surfaces_dictionary["surface18"] = top_lamellaSurface1

    surfaces_dictionary["surface19"] = top_lamellaSurface2

    surfaces_dictionary["surface20"] = top_lamellaSurface3

    surfaces_dictionary["surface21"] = top_lamellaSurface4

    surfaces_dictionary["surface22"] = top_lamellaSurface5

    # Adds the volume

    assembly_volume = gmsh.model.geo.addSurfaceLoop([top_lamellaSurface1,
    top_lamellaSurface2, top_lamellaSurface3, top_lamellaSurface4,
    top_lamellaSurface5, top_surface6, *top_lamellaSurfaces])

    top_lamellaVolume = gmsh.model.geo.addVolume([assembly_volume])

    gmsh.model.geo.synchronize()
    
    ####################################################################
    ####################################################################
    ##                    Update of physical groups                   ##
    ####################################################################
    ####################################################################

    # Set the color of each volume of the major physical groups

    gmsh.model.setColor([(3,i) for i in elastin_volumes], 180, 4,
    38)

    gmsh.model.setColor([(3,muscular_volume)], 140, 28, 89)

    gmsh.model.setColor([(3,bottom_collagenVolume), (3,
    top_collagenVolume)], 99, 52, 141)

    gmsh.model.setColor([(3,bottom_lamellaVolume), (3,
    top_lamellaVolume)], 59, 76, 192)

    # Updates the list of volumes of each major physical group

    dictionary_volumesPhysGroups[1].extend(elastin_volumes)

    dictionary_volumesPhysGroups[2].append(muscular_volume)

    dictionary_volumesPhysGroups[3].extend([bottom_collagenVolume,
    top_collagenVolume])

    dictionary_volumesPhysGroups[4].extend([bottom_lamellaVolume,
    top_lamellaVolume])

    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    # Iterates through the surfaces

    for surface in surfaces_points:

        # Iterates through the surfaces identifiers to detect if they
        # belong to a surface region using the surface boundary points 
        # as criterion

        for i in range(len(surface_regionIdentifiers)):

            # Tests if the centroid belongs to this region

            if surface_regionIdentifiers[i](surfaces_points[surface]):
                
                # Adds this surface to the dictionary of physical surfa-
                # ces

                dictionary_surfacesPhysGroups[i+1].append(
                surfaces_dictionary[surface])

                # Breaks, because a surface cannot be part of more than
                # one physical group

                break

    gmsh.model.geo.synchronize()

    return dictionary_volumesPhysGroups, dictionary_surfacesPhysGroups

########################################################################
#                       Parameters verification                        #
########################################################################

def verify_parameters(parameters_method, RVE_lengthX, RVE_lengthY, 
RVE_lengthZ):

    # Retrieves parameters of the arterial microstructure

    fiber_diameter = parameters_method[0]

    polar_angle = parameters_method[1]

    azimuthal_angle = parameters_method[2]

    collagen_thickness = parameters_method[3]

    lamella_thickness = parameters_method[4]

    # Calculates the intralamellar thickness by subtracting the lamella
    # thickness off of the RVE z length

    intralamellar_thickness = RVE_lengthZ-lamella_thickness

    # Retrieves the allowable distance between the tip of the elastin 
    # strut to the edge of the RVE

    clearance_strutsEdges = parameters_method[5]

    # Retrieves the number of struts along the x and the y axes

    n_strutsX = parameters_method[6]

    n_strutsY = parameters_method[7]

    # Verifies whether the RVE lengths subtracted of the clearance lea-
    # ves any space for a single strut

    struct_projection = intralamellar_thickness*np.tan(azimuthal_angle)

    if ((struct_projection*np.cos(polar_angle))+fiber_diameter)>(
    RVE_lengthX-(2*clearance_strutsEdges)):
        
        raise Exception("\nThe strut clearance is too large, there is "+
        "no space for even a SINGLE elastin strut in the X direction.")

    if ((struct_projection*np.sin(polar_angle))+fiber_diameter)>(
    RVE_lengthY-(2*clearance_strutsEdges)):
        
        raise Exception("\nThe strut clearance is too large, there is "+
        "no space for even a SINGLE elastin strut in the Y direction.")

    if ((struct_projection*np.cos(polar_angle))+(fiber_diameter*
    n_strutsX))>(RVE_lengthX-(2*clearance_strutsEdges)):
        
        raise Exception("\nThe strut clearance is too large, there is "+
        "no space for "+str(n_strutsX)+" struts in the X direction.")

    if ((struct_projection*np.sin(polar_angle))+(fiber_diameter*
    n_strutsY))>(RVE_lengthY-(2*clearance_strutsEdges)):
        
        raise Exception("\nThe strut clearance is too large, there is "+
        "no space for "+str(n_strutsY)+" struts in the Y direction.")
    
    # Verifies the RVE height

    if lamella_thickness+(2*collagen_thickness)>=RVE_lengthZ:

        raise Exception("\nThe lamella and the collagen layer are too "+
        "thick, they occupy a total length equal or larger than the RV"+
        "E height")

########################################################################
#                               Testing                                #
########################################################################

def test_RVE():

    # Sets the centroid of the RVE

    x_centroid = 0.0

    y_centroid = 0.0

    z_centroid = 0.0

    # Defines the characteristic length of the mesh

    lc = 0.5

    # Defines parameters of the arterial microstructure

    fiber_diameter = 1.0

    polar_angle = 0.0

    azimuthal_angle = (19.0/180)*np.pi 

    collagen_thickness = 1.0

    lamella_thickness = 2.0

    # Defines the size of the RVE along the x axis (arterial circunfe-
    # rential) and along the y axis (arterial longitudinal)

    RVE_lengthX = 20.0

    RVE_lengthY = 15.0

    RVE_lengthZ = 10.0

    # Defines the allowable distance between the tip of the elastin 
    # strut to the edge of the RVE

    clearance_strutsEdges = 4.0

    # Defines the number of struts along the x and the y axes

    n_strutsX = 2

    n_strutsY = 2

    parameters_method = [fiber_diameter, polar_angle, azimuthal_angle,
    collagen_thickness, lamella_thickness, clearance_strutsEdges, 
    n_strutsX, n_strutsY]

    dictionary_volumesPhysGroups = dict()

    dictionary_volumesPhysGroups[1] = []

    dictionary_volumesPhysGroups[2] = []

    dictionary_volumesPhysGroups[3] = []

    dictionary_volumesPhysGroups[4] = []

    volume_regionIdentifiers = []

    # Sets the names for the volumetric physical groups

    volume_regionsNames = ["elastin_struts", "muscle", "collagen",
    "lamella"]
    
    # Defines a set of functions to identify if the created surfaces lie
    # within each physical surface. Hence, defines a function to each
    # physical surface. These functions get a set of points and verify
    # whether they belong to a region or not

    def surface_bottom(points_set):

        return region_finder.plane_boundary(points_set, [0.0, 0.0, -1.0],
        0.5*RVE_lengthZ)
    
    def surface_ahead(points_set):

        return region_finder.plane_boundary(points_set, [1.0, 0.0, 0.0],
        0.5*RVE_lengthX)
    
    def surface_larboard(points_set):

        return region_finder.plane_boundary(points_set, [0.0, 1.0, 0.0],
        0.5*RVE_lengthY)
    
    def surface_aft(points_set):

        return region_finder.plane_boundary(points_set, [-1.0, 0.0, 0.0],
        0.5*RVE_lengthX)
    
    def surface_starboard(points_set):

        return region_finder.plane_boundary(points_set, [0.0, -1.0, 0.0],
        0.5*RVE_lengthY)
    
    def surface_top(points_set):

        return region_finder.plane_boundary(points_set, [0.0, 0.0, 1.0],
        0.5*RVE_lengthZ)
    
    surface_regionIdentifiers = [surface_bottom, surface_ahead,
    surface_larboard, surface_aft, surface_starboard, surface_top]

    # Sets the names of the surface regions

    surface_regionsNames = ['bottom', 'ahead', 'larboard', 'aft', 'sta'+
    'rboard', 'top']

    dictionary_surfacesPhysGroups = tools.generate_physicalDictionary(
    surface_regionIdentifiers, surface_regionsNames, 2)

    # Sets the types of elements

    volume_elementType='tetra'
    
    surface_elementType= 'triangle'

    # Sets the mesh file name and the intended directory

    file_name = "artery_RVE"

    file_directory = "artery_RVE_meshes"

    # Initializes the gmsh object

    gmsh.initialize()

    dictionary_volumesPhysGroups, dictionary_surfacesPhysGroups = RVE_elastinMicrostructure(
    x_centroid, y_centroid, z_centroid, RVE_lengthX, RVE_lengthY, 
    RVE_lengthZ, parameters_method, dictionary_volumesPhysGroups, 
    dictionary_surfacesPhysGroups, lc, volume_regionIdentifiers, 
    surface_regionIdentifiers)

    ####################################################################
    #                         Post-processing                          #
    ####################################################################

    # Initializes the geometric data list

    geometric_data=[0, [[],[],[],[]], [[],[],[],[]], [[],[],[]], 
    dictionary_surfacesPhysGroups, [], dictionary_volumesPhysGroups, [], 
    surface_regionsNames, volume_regionsNames]

    # Saves the mesh

    tools.gmsh_finalize(geometric_data=geometric_data, file_name=
    file_name, file_directory=file_directory, volume_elementType=
    volume_elementType, surface_elementType=surface_elementType)

#test_RVE()