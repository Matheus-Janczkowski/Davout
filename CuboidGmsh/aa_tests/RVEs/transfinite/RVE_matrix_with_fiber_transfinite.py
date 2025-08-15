import gmsh

import numpy as np

import math

import CuboidGmsh.source.tool_box.meshing_tools as tools

# Defines a function that constructs the top and bottom surfaces of the
# RVE

def RVE_centralFiberInnerSquare4Quadrants(x_centroid, y_centroid, 
z_centroid, RVE_lengthX, RVE_lengthY, RVE_lengthZ, parameters_method, 
dictionary_volumesPhysGroups, dictionary_surfacesPhysGroups, lc, 
volume_regionIdentifiers, surface_regionIdentifiers, n_transfiniteCurves
=10):
    
    # Checks if no number of transfinite divisions has been given

    if isinstance(n_transfiniteCurves, int):

        n_base = n_transfiniteCurves+0

        n_transfiniteCurves = []

        # Puts the suggested value

        for i in range(61): 

            n_transfiniteCurves.append(n_base)

    # Takes the parameters of the method

    inner_squareSideSize = parameters_method[0]

    radius = parameters_method[1]

    # Defines the points for matrix section and the half size of the 
    # surface

    circle_interceptCoord = radius*0.5*math.sqrt(2)

    half_sizeX = 0.5*RVE_lengthX

    half_sizeY = 0.5*RVE_lengthY

    half_sizeZ = 0.5*RVE_lengthZ

    half_littleSquare = 0.5*inner_squareSideSize

    # Sets a dictionary for each one of the topologycal entities that a-
    # re liable to transfinite meshing, i.g. lines, surfaces and volumes

    lines_dictionary = dict()

    surfaces_dictionary = dict()

    volumes_dictionary = dict()

    # Sets a dictionary for the points that lie on the boundary of each
    # surfaces 

    surfaces_points = dict()

    ####################################################################
    ####################################################################
    ##                        Botttom surfaces                        ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Points                              #
    ####################################################################

    # Updates the z_centroid to the bottom surface of the RVE to make
    # easier the drawing of the RVE

    z_centroid -= 0.5*RVE_lengthZ

    # Defines the centroid

    p1 = gmsh.model.geo.addPoint(x_centroid, y_centroid, z_centroid, lc)

    # Defines the points for the inner square of the fiber

    p2 = gmsh.model.geo.addPoint(x_centroid+half_littleSquare,y_centroid
    -half_littleSquare, z_centroid, lc)

    p3 = gmsh.model.geo.addPoint(x_centroid+half_littleSquare,y_centroid
    +half_littleSquare, z_centroid, lc)

    p4 = gmsh.model.geo.addPoint(x_centroid-half_littleSquare,y_centroid
    +half_littleSquare, z_centroid, lc)

    p5 = gmsh.model.geo.addPoint(x_centroid-half_littleSquare,y_centroid
    -half_littleSquare, z_centroid, lc)

    # Defines the points for the fiber section (circle intercepts)

    p6 = gmsh.model.geo.addPoint(x_centroid+circle_interceptCoord, 
    y_centroid-circle_interceptCoord, z_centroid, lc)

    p7 = gmsh.model.geo.addPoint(x_centroid+circle_interceptCoord, 
    y_centroid+circle_interceptCoord, z_centroid, lc)

    p8 = gmsh.model.geo.addPoint(x_centroid-circle_interceptCoord, 
    y_centroid+circle_interceptCoord, z_centroid, lc)

    p9 = gmsh.model.geo.addPoint(x_centroid-circle_interceptCoord, 
    y_centroid-circle_interceptCoord, z_centroid, lc)

    # Defines the points for the outer perimeter

    p10 = gmsh.model.geo.addPoint(x_centroid+half_sizeX,y_centroid-
    half_sizeY, z_centroid, lc)

    p11 = gmsh.model.geo.addPoint(x_centroid+half_sizeX, y_centroid+
    half_sizeY, z_centroid, lc)

    p12 = gmsh.model.geo.addPoint(x_centroid-half_sizeX, y_centroid+
    half_sizeY, z_centroid, lc)

    p13 = gmsh.model.geo.addPoint(x_centroid-half_sizeX, y_centroid-
    half_sizeY, z_centroid, lc)
    
    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l1"] = gmsh.model.geo.addLine(p1, p2)

    lines_dictionary["l2"] = gmsh.model.geo.addLine(p2, p3)

    lines_dictionary["l3"] = gmsh.model.geo.addLine(p3, p1)

    lines_dictionary["l4"] = gmsh.model.geo.addLine(p3, p4)

    lines_dictionary["l5"] = gmsh.model.geo.addLine(p4, p1)

    lines_dictionary["l6"] = gmsh.model.geo.addLine(p4, p5)

    lines_dictionary["l7"] = gmsh.model.geo.addLine(p5, p1)

    lines_dictionary["l8"] = gmsh.model.geo.addLine(p5, p2)

    # Creates the lines for the fiber

    lines_dictionary["l9"] = gmsh.model.geo.addLine(p2, p6)

    lines_dictionary["l10"] = gmsh.model.geo.addCircleArc(p6, p1, p7)

    lines_dictionary["l11"] = gmsh.model.geo.addLine(p7, p3)

    lines_dictionary["l12"] = gmsh.model.geo.addCircleArc(p7, p1, p8)

    lines_dictionary["l13"] = gmsh.model.geo.addLine(p8, p4)

    lines_dictionary["l14"] = gmsh.model.geo.addCircleArc(p8, p1, p9)

    lines_dictionary["l15"] = gmsh.model.geo.addLine(p9, p5)

    lines_dictionary["l16"] = gmsh.model.geo.addCircleArc(p9, p1, p6)

    lines_dictionary["l17"] = gmsh.model.geo.addLine(p6, p10)

    lines_dictionary["l18"] = gmsh.model.geo.addLine(p10, p11)

    lines_dictionary["l19"] = gmsh.model.geo.addLine(p11, p7)

    lines_dictionary["l20"] = gmsh.model.geo.addLine(p11, p12)

    lines_dictionary["l21"] = gmsh.model.geo.addLine(p12, p8)

    lines_dictionary["l22"] = gmsh.model.geo.addLine(p12, p13)

    lines_dictionary["l23"] = gmsh.model.geo.addLine(p13, p9)

    lines_dictionary["l24"] = gmsh.model.geo.addLine(p13, p10)
    
    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    # Inner square

    loop1 = gmsh.model.geo.addCurveLoop([lines_dictionary["l1"], 
    lines_dictionary["l2"], lines_dictionary["l3"]])

    loop2 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l3"], 
    lines_dictionary["l4"], lines_dictionary["l5"]])

    loop3 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l5"], 
    lines_dictionary["l6"], lines_dictionary["l7"]])

    loop4 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l7"], 
    lines_dictionary["l8"], -lines_dictionary["l1"]])

    # Fiber

    loop5 = gmsh.model.geo.addCurveLoop([lines_dictionary["l9"], 
    lines_dictionary["l10"], lines_dictionary["l11"], -lines_dictionary[
    "l2"]])

    loop6 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l11"], 
    lines_dictionary["l12"], lines_dictionary["l13"], -lines_dictionary[
    "l4"]])

    loop7 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l13"], 
    lines_dictionary["l14"], lines_dictionary["l15"], -lines_dictionary[
    "l6"]])

    loop8 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l15"], 
    lines_dictionary["l16"], -lines_dictionary["l9"], -lines_dictionary[
    "l8"]])

    # Matrix

    loop9 = gmsh.model.geo.addCurveLoop([lines_dictionary["l17"], 
    lines_dictionary["l18"], lines_dictionary["l19"], -lines_dictionary[
    "l10"]])

    loop10 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l19"],
    lines_dictionary["l20"], lines_dictionary["l21"], -lines_dictionary[
    "l12"]])

    loop11 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l21"], 
    lines_dictionary["l22"], lines_dictionary["l23"], -lines_dictionary[
    "l14"]])

    loop12 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l23"], 
    lines_dictionary["l24"], -lines_dictionary["l17"], 
    -lines_dictionary["l16"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    # Inner square

    surfaces_dictionary["surface1"] = gmsh.model.geo.addPlaneSurface([
    loop1])

    surfaces_dictionary["surface2"] = gmsh.model.geo.addPlaneSurface([
    loop2])

    surfaces_dictionary["surface3"] = gmsh.model.geo.addPlaneSurface([
    loop3])

    surfaces_dictionary["surface4"] = gmsh.model.geo.addPlaneSurface([
    loop4])

    # Fiber

    surfaces_dictionary["surface5"] = gmsh.model.geo.addPlaneSurface([
    loop5])

    surfaces_dictionary["surface6"] = gmsh.model.geo.addPlaneSurface([
    loop6])

    surfaces_dictionary["surface7"] = gmsh.model.geo.addPlaneSurface([
    loop7])

    surfaces_dictionary["surface8"] = gmsh.model.geo.addPlaneSurface([
    loop8])

    # Matrix

    surfaces_dictionary["surface9"] = gmsh.model.geo.addPlaneSurface([
    loop9])

    surfaces_dictionary["surface10"] = gmsh.model.geo.addPlaneSurface([
    loop10])

    surfaces_dictionary["surface11"] = gmsh.model.geo.addPlaneSurface([
    loop11])

    surfaces_dictionary["surface12"] = gmsh.model.geo.addPlaneSurface([
    loop12])

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface1"] = [[]]

    surfaces_points["surface2"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface2"])

    surfaces_points["surface3"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface3"])

    surfaces_points["surface4"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface4"])

    surfaces_points["surface5"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface5"])

    surfaces_points["surface6"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface6"])

    surfaces_points["surface7"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface7"])

    surfaces_points["surface8"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface8"])

    surfaces_points["surface9"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface9"])

    surfaces_points["surface10"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface10"])

    surfaces_points["surface11"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface11"])

    surfaces_points["surface12"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface12"])
    
    ####################################################################
    ####################################################################
    ##                          Top surfaces                          ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Points                              #
    ####################################################################

    # Defines the centroid

    p14 = gmsh.model.geo.addPoint(x_centroid, y_centroid, z_centroid+
    RVE_lengthZ, lc)

    # Defines the points for the inner square of the fiber

    p15 = gmsh.model.geo.addPoint(x_centroid+half_littleSquare,y_centroid
    -half_littleSquare, z_centroid+RVE_lengthZ, lc)

    p16 = gmsh.model.geo.addPoint(x_centroid+half_littleSquare,y_centroid
    +half_littleSquare, z_centroid+RVE_lengthZ, lc)

    p17 = gmsh.model.geo.addPoint(x_centroid-half_littleSquare,y_centroid
    +half_littleSquare, z_centroid+RVE_lengthZ, lc)

    p18 = gmsh.model.geo.addPoint(x_centroid-half_littleSquare,y_centroid
    -half_littleSquare, z_centroid+RVE_lengthZ, lc)

    # Defines the points for the fiber section (circle intercepts)

    p19 = gmsh.model.geo.addPoint(x_centroid+circle_interceptCoord, 
    y_centroid-circle_interceptCoord, z_centroid+RVE_lengthZ, lc)

    p20 = gmsh.model.geo.addPoint(x_centroid+circle_interceptCoord, 
    y_centroid+circle_interceptCoord, z_centroid+RVE_lengthZ, lc)

    p21 = gmsh.model.geo.addPoint(x_centroid-circle_interceptCoord, 
    y_centroid+circle_interceptCoord, z_centroid+RVE_lengthZ, lc)

    p22 = gmsh.model.geo.addPoint(x_centroid-circle_interceptCoord, 
    y_centroid-circle_interceptCoord, z_centroid+RVE_lengthZ, lc)

    # Defines the points for the outer perimeter

    p23 = gmsh.model.geo.addPoint(x_centroid+half_sizeX,y_centroid-
    half_sizeY, z_centroid+RVE_lengthZ, lc)

    p24 = gmsh.model.geo.addPoint(x_centroid+half_sizeX, y_centroid+
    half_sizeY, z_centroid+RVE_lengthZ, lc)

    p25 = gmsh.model.geo.addPoint(x_centroid-half_sizeX, y_centroid+
    half_sizeY, z_centroid+RVE_lengthZ, lc)

    p26 = gmsh.model.geo.addPoint(x_centroid-half_sizeX, y_centroid-
    half_sizeY, z_centroid+RVE_lengthZ, lc)
    
    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l25"] = gmsh.model.geo.addLine(p14, p15)

    lines_dictionary["l26"] = gmsh.model.geo.addLine(p15, p16)

    lines_dictionary["l27"] = gmsh.model.geo.addLine(p16, p14)

    lines_dictionary["l28"] = gmsh.model.geo.addLine(p16, p17)

    lines_dictionary["l29"] = gmsh.model.geo.addLine(p17, p14)

    lines_dictionary["l30"] = gmsh.model.geo.addLine(p17, p18)

    lines_dictionary["l31"] = gmsh.model.geo.addLine(p18, p14)

    lines_dictionary["l32"] = gmsh.model.geo.addLine(p18, p15)

    # Creates the lines for the fiber

    lines_dictionary["l33"] = gmsh.model.geo.addLine(p15, p19)

    lines_dictionary["l34"] = gmsh.model.geo.addCircleArc(p19, p14, p20)

    lines_dictionary["l35"] = gmsh.model.geo.addLine(p20, p16)

    lines_dictionary["l36"] = gmsh.model.geo.addCircleArc(p20, p14, p21)

    lines_dictionary["l37"] = gmsh.model.geo.addLine(p21, p17)

    lines_dictionary["l38"] = gmsh.model.geo.addCircleArc(p21, p14, p22)

    lines_dictionary["l39"] = gmsh.model.geo.addLine(p22, p18)

    lines_dictionary["l40"] = gmsh.model.geo.addCircleArc(p22, p14, p19)

    lines_dictionary["l41"] = gmsh.model.geo.addLine(p19, p23)

    lines_dictionary["l42"] = gmsh.model.geo.addLine(p23, p24)

    lines_dictionary["l43"] = gmsh.model.geo.addLine(p24, p20)

    lines_dictionary["l44"] = gmsh.model.geo.addLine(p24, p25)

    lines_dictionary["l45"] = gmsh.model.geo.addLine(p25, p21)

    lines_dictionary["l46"] = gmsh.model.geo.addLine(p25, p26)

    lines_dictionary["l47"] = gmsh.model.geo.addLine(p26, p22)

    lines_dictionary["l48"] = gmsh.model.geo.addLine(p26, p23)
    
    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    # Inner square

    loop13 = gmsh.model.geo.addCurveLoop([lines_dictionary["l25"], 
    lines_dictionary["l26"], lines_dictionary["l27"]])

    loop14 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l27"], 
    lines_dictionary["l28"], lines_dictionary["l29"]])

    loop15 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l29"], 
    lines_dictionary["l30"], lines_dictionary["l31"]])

    loop16 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l31"], 
    lines_dictionary["l32"], -lines_dictionary["l25"]])

    # Fiber

    loop17 = gmsh.model.geo.addCurveLoop([lines_dictionary["l33"], 
    lines_dictionary["l34"], lines_dictionary["l35"], -lines_dictionary[
    "l26"]])

    loop18 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l35"], 
    lines_dictionary["l36"], lines_dictionary["l37"], -lines_dictionary[
    "l28"]])

    loop19 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l37"], 
    lines_dictionary["l38"], lines_dictionary["l39"], -lines_dictionary[
    "l30"]])

    loop20 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l39"], 
    lines_dictionary["l40"], -lines_dictionary["l33"], -lines_dictionary[
    "l32"]])

    # Matrix

    loop21 = gmsh.model.geo.addCurveLoop([lines_dictionary["l41"], 
    lines_dictionary["l42"], lines_dictionary["l43"], -lines_dictionary[
    "l34"]])

    loop22 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l43"], 
    lines_dictionary["l44"], lines_dictionary["l45"], -lines_dictionary[
    "l36"]])

    loop23 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l45"], 
    lines_dictionary["l46"], lines_dictionary["l47"], -lines_dictionary[
    "l38"]])

    loop24 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l47"], 
    lines_dictionary["l48"], -lines_dictionary["l41"], -lines_dictionary[
    "l40"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    # Inner square

    surfaces_dictionary["surface13"] = gmsh.model.geo.addPlaneSurface([
    loop13])

    surfaces_dictionary["surface14"] = gmsh.model.geo.addPlaneSurface([
    loop14])

    surfaces_dictionary["surface15"] = gmsh.model.geo.addPlaneSurface([
    loop15])

    surfaces_dictionary["surface16"] = gmsh.model.geo.addPlaneSurface([
    loop16])

    # Fiber

    surfaces_dictionary["surface17"] = gmsh.model.geo.addPlaneSurface([
    loop17])

    surfaces_dictionary["surface18"] = gmsh.model.geo.addPlaneSurface([
    loop18])

    surfaces_dictionary["surface19"] = gmsh.model.geo.addPlaneSurface([
    loop19])

    surfaces_dictionary["surface20"] = gmsh.model.geo.addPlaneSurface([
    loop20])

    # Matrix

    surfaces_dictionary["surface21"] = gmsh.model.geo.addPlaneSurface([
    loop21])

    surfaces_dictionary["surface22"] = gmsh.model.geo.addPlaneSurface([
    loop22])

    surfaces_dictionary["surface23"] = gmsh.model.geo.addPlaneSurface([
    loop23])

    surfaces_dictionary["surface24"] = gmsh.model.geo.addPlaneSurface([
    loop24])

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface13"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface13"])

    surfaces_points["surface14"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface14"])

    surfaces_points["surface15"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface15"])

    surfaces_points["surface16"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface17"])

    surfaces_points["surface17"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface17"])

    surfaces_points["surface18"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface18"])

    surfaces_points["surface19"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface19"])

    surfaces_points["surface20"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface20"])

    surfaces_points["surface21"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface21"])

    surfaces_points["surface22"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface22"])

    surfaces_points["surface23"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface23"])

    surfaces_points["surface24"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface24"])
    
    ####################################################################
    ####################################################################
    ##                      Inner square quarters                     ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l49"] = gmsh.model.geo.addLine(p14, p1)

    lines_dictionary["l50"] = gmsh.model.geo.addLine(p15, p2)

    lines_dictionary["l51"] = gmsh.model.geo.addLine(p16, p3)

    lines_dictionary["l52"] = gmsh.model.geo.addLine(p17, p4)

    lines_dictionary["l53"] = gmsh.model.geo.addLine(p18, p5)
    
    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    loop25 = gmsh.model.geo.addCurveLoop([lines_dictionary["l49"], 
    lines_dictionary["l1"], -lines_dictionary["l50"], -lines_dictionary[
    "l25"]])

    loop26 = gmsh.model.geo.addCurveLoop([lines_dictionary["l50"], 
    lines_dictionary["l2"], -lines_dictionary["l51"], -lines_dictionary[
    "l26"]])

    loop27 = gmsh.model.geo.addCurveLoop([lines_dictionary["l51"], 
    lines_dictionary["l3"], -lines_dictionary["l49"], -lines_dictionary[
    "l27"]])

    loop28 = gmsh.model.geo.addCurveLoop([lines_dictionary["l51"], 
    lines_dictionary["l4"], -lines_dictionary["l52"], -lines_dictionary[
    "l28"]])

    loop29 = gmsh.model.geo.addCurveLoop([lines_dictionary["l52"], 
    lines_dictionary["l5"], -lines_dictionary["l49"], -lines_dictionary[
    "l29"]])

    loop30 = gmsh.model.geo.addCurveLoop([lines_dictionary["l52"], 
    lines_dictionary["l6"], -lines_dictionary["l53"], -lines_dictionary[
    "l30"]])

    loop31 = gmsh.model.geo.addCurveLoop([lines_dictionary["l53"], 
    lines_dictionary["l7"], -lines_dictionary["l49"], -lines_dictionary[
    "l31"]])

    loop32 = gmsh.model.geo.addCurveLoop([lines_dictionary["l53"], 
    lines_dictionary["l8"], -lines_dictionary["l50"], -lines_dictionary[
    "l32"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    surfaces_dictionary["surface25"] = gmsh.model.geo.addPlaneSurface([
    loop25])

    surfaces_dictionary["surface26"] = gmsh.model.geo.addPlaneSurface([
    loop26])

    surfaces_dictionary["surface27"] = gmsh.model.geo.addPlaneSurface([
    loop27])

    surfaces_dictionary["surface28"] = gmsh.model.geo.addPlaneSurface([
    loop28])

    surfaces_dictionary["surface29"] = gmsh.model.geo.addPlaneSurface([
    loop29])

    surfaces_dictionary["surface30"] = gmsh.model.geo.addPlaneSurface([
    loop30])

    surfaces_dictionary["surface31"] = gmsh.model.geo.addPlaneSurface([
    loop31])

    surfaces_dictionary["surface32"] = gmsh.model.geo.addPlaneSurface([
    loop32])

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface25"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface25"])

    surfaces_points["surface26"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface26"])

    surfaces_points["surface27"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface27"])

    surfaces_points["surface28"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface28"])

    surfaces_points["surface29"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface29"])

    surfaces_points["surface30"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface30"])

    surfaces_points["surface31"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface32"])

    surfaces_points["surface32"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface32"])
    
    ####################################################################
    #                              Volumes                             #
    ####################################################################

    assembly_volume1 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface1"], 
    surfaces_dictionary["surface25"], surfaces_dictionary["surface26"], 
    surfaces_dictionary["surface27"], surfaces_dictionary["surface13"]])

    assembly_volume2 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface2"], 
    surfaces_dictionary["surface27"], surfaces_dictionary["surface28"], 
    surfaces_dictionary["surface29"], surfaces_dictionary["surface14"]])

    assembly_volume3 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface3"], 
    surfaces_dictionary["surface29"], surfaces_dictionary["surface30"], 
    surfaces_dictionary["surface31"], surfaces_dictionary["surface15"]])

    assembly_volume4 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface4"], 
    surfaces_dictionary["surface31"], surfaces_dictionary["surface32"], 
    surfaces_dictionary["surface25"], surfaces_dictionary["surface16"]])
    
    ####################################################################
    ####################################################################
    ##                      Cylindrical quarters                      ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l54"] = gmsh.model.geo.addLine(p19, p6)

    lines_dictionary["l55"] = gmsh.model.geo.addLine(p20, p7)

    lines_dictionary["l56"] = gmsh.model.geo.addLine(p21, p8)

    lines_dictionary["l57"] = gmsh.model.geo.addLine(p22, p9)
    
    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    loop33 = gmsh.model.geo.addCurveLoop([lines_dictionary["l50"], 
    lines_dictionary["l9"], -lines_dictionary["l54"], -lines_dictionary[
    "l33"]])

    loop34 = gmsh.model.geo.addCurveLoop([lines_dictionary["l54"], 
    lines_dictionary["l10"], -lines_dictionary["l55"], 
    -lines_dictionary["l34"]])

    loop35 = gmsh.model.geo.addCurveLoop([lines_dictionary["l55"], 
    lines_dictionary["l11"], -lines_dictionary["l51"], 
    -lines_dictionary["l35"]])

    loop36 = gmsh.model.geo.addCurveLoop([lines_dictionary["l55"], 
    lines_dictionary["l12"], -lines_dictionary["l56"], 
    -lines_dictionary["l36"]])

    loop37 = gmsh.model.geo.addCurveLoop([lines_dictionary["l56"], 
    lines_dictionary["l13"], -lines_dictionary["l52"], 
    -lines_dictionary["l37"]])

    loop38 = gmsh.model.geo.addCurveLoop([lines_dictionary["l56"], 
    lines_dictionary["l14"], -lines_dictionary["l57"], 
    -lines_dictionary["l38"]])

    loop39 = gmsh.model.geo.addCurveLoop([lines_dictionary["l57"], 
    lines_dictionary["l15"], -lines_dictionary["l53"], 
    -lines_dictionary["l39"]])

    loop40 = gmsh.model.geo.addCurveLoop([lines_dictionary["l57"], 
    lines_dictionary["l16"], -lines_dictionary["l54"], 
    -lines_dictionary["l40"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    surfaces_dictionary["surface33"] = gmsh.model.geo.addPlaneSurface([
    loop33])

    surfaces_dictionary["surface34"] = gmsh.model.geo.addSurfaceFilling(
    [loop34])

    surfaces_dictionary["surface35"] = gmsh.model.geo.addPlaneSurface([
    loop35])

    surfaces_dictionary["surface36"] = gmsh.model.geo.addSurfaceFilling(
    [loop36])

    surfaces_dictionary["surface37"] = gmsh.model.geo.addPlaneSurface([
    loop37])

    surfaces_dictionary["surface38"] = gmsh.model.geo.addSurfaceFilling(
    [loop38])

    surfaces_dictionary["surface39"] = gmsh.model.geo.addPlaneSurface([
    loop39])

    surfaces_dictionary["surface40"] = gmsh.model.geo.addSurfaceFilling(
    [loop40])

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface33"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface33"])

    surfaces_points["surface34"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface34"])

    surfaces_points["surface35"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface35"])

    surfaces_points["surface36"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface36"])

    surfaces_points["surface37"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface37"])

    surfaces_points["surface38"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface38"])

    surfaces_points["surface39"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface39"])

    surfaces_points["surface40"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface40"])
    
    ####################################################################
    #                              Volumes                             #
    ####################################################################

    assembly_volume5 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface5"], 
    surfaces_dictionary["surface26"], surfaces_dictionary["surface33"], 
    surfaces_dictionary["surface34"], surfaces_dictionary["surface35"], 
    surfaces_dictionary["surface17"]])

    assembly_volume6 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface6"], 
    surfaces_dictionary["surface28"], surfaces_dictionary["surface35"], 
    surfaces_dictionary["surface36"], surfaces_dictionary["surface37"], 
    surfaces_dictionary["surface18"]])

    assembly_volume7 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface7"], 
    surfaces_dictionary["surface30"], surfaces_dictionary["surface37"], 
    surfaces_dictionary["surface38"], surfaces_dictionary["surface39"], 
    surfaces_dictionary["surface19"]])

    assembly_volume8 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface8"], 
    surfaces_dictionary["surface32"], surfaces_dictionary["surface33"], 
    surfaces_dictionary["surface40"], surfaces_dictionary["surface39"], 
    surfaces_dictionary["surface20"]])
    
    ####################################################################
    ####################################################################
    ##                        Matrix quarters                         ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l58"] = gmsh.model.geo.addLine(p23, p10)

    lines_dictionary["l59"] = gmsh.model.geo.addLine(p24, p11)

    lines_dictionary["l60"] = gmsh.model.geo.addLine(p25, p12)

    lines_dictionary["l61"] = gmsh.model.geo.addLine(p26, p13)
    
    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    loop41 = gmsh.model.geo.addCurveLoop([lines_dictionary["l54"], 
    lines_dictionary["l17"], -lines_dictionary["l58"], -lines_dictionary[
    "l41"]])

    loop42 = gmsh.model.geo.addCurveLoop([lines_dictionary["l58"], 
    lines_dictionary["l18"], -lines_dictionary["l59"], -lines_dictionary[
    "l42"]])

    loop43 = gmsh.model.geo.addCurveLoop([lines_dictionary["l59"], 
    lines_dictionary["l19"], -lines_dictionary["l55"], -lines_dictionary[
    "l43"]])

    loop44 = gmsh.model.geo.addCurveLoop([lines_dictionary["l59"], 
    lines_dictionary["l20"], -lines_dictionary["l60"], -lines_dictionary[
    "l44"]])

    loop45 = gmsh.model.geo.addCurveLoop([lines_dictionary["l60"], 
    lines_dictionary["l21"], -lines_dictionary["l56"], -lines_dictionary[
    "l45"]])

    loop46 = gmsh.model.geo.addCurveLoop([lines_dictionary["l60"], 
    lines_dictionary["l22"], -lines_dictionary["l61"], -lines_dictionary[
    "l46"]])

    loop47 = gmsh.model.geo.addCurveLoop([lines_dictionary["l61"], 
    lines_dictionary["l23"], -lines_dictionary["l57"], -lines_dictionary[
    "l47"]])

    loop48 = gmsh.model.geo.addCurveLoop([lines_dictionary["l61"], 
    lines_dictionary["l24"], -lines_dictionary["l58"], -lines_dictionary[
    "l48"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    surfaces_dictionary["surface41"] = gmsh.model.geo.addPlaneSurface([
    loop41])

    surfaces_dictionary["surface42"] = gmsh.model.geo.addPlaneSurface([
    loop42])

    surfaces_dictionary["surface43"] = gmsh.model.geo.addPlaneSurface([
    loop43])

    surfaces_dictionary["surface44"] = gmsh.model.geo.addPlaneSurface([
    loop44])

    surfaces_dictionary["surface45"] = gmsh.model.geo.addPlaneSurface([
    loop45])

    surfaces_dictionary["surface46"] = gmsh.model.geo.addPlaneSurface([
    loop46])

    surfaces_dictionary["surface47"] = gmsh.model.geo.addPlaneSurface([
    loop47])

    surfaces_dictionary["surface48"] = gmsh.model.geo.addPlaneSurface([
    loop48])

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface41"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface41"])

    surfaces_points["surface42"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface42"])

    surfaces_points["surface43"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface43"])

    surfaces_points["surface44"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface44"])

    surfaces_points["surface45"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface45"])

    surfaces_points["surface46"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface46"])

    surfaces_points["surface47"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface47"])

    surfaces_points["surface48"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface48"])
    
    ####################################################################
    #                              Volumes                             #
    ####################################################################

    assembly_volume9 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface9"], 
    surfaces_dictionary["surface41"], surfaces_dictionary["surface42"], 
    surfaces_dictionary["surface43"], surfaces_dictionary["surface34"], 
    surfaces_dictionary["surface21"]])

    assembly_volume10 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface10"], 
    surfaces_dictionary["surface43"], surfaces_dictionary["surface44"], 
    surfaces_dictionary["surface45"], surfaces_dictionary["surface36"], 
    surfaces_dictionary["surface22"]])

    assembly_volume11 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface11"], 
    surfaces_dictionary["surface45"], surfaces_dictionary["surface46"], 
    surfaces_dictionary["surface47"], surfaces_dictionary["surface38"], 
    surfaces_dictionary["surface23"]])

    assembly_volume12 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface12"], 
    surfaces_dictionary["surface47"], surfaces_dictionary["surface48"], 
    surfaces_dictionary["surface41"], surfaces_dictionary["surface40"], 
    surfaces_dictionary["surface24"]])
    
    ####################################################################
    ####################################################################
    ##                            Assembly                            ##
    ####################################################################
    ####################################################################

    # Builds the volumes using the surfaces

    volumes_dictionary["volume1"] = gmsh.model.geo.addVolume([
    assembly_volume1])

    volumes_dictionary["volume2"] = gmsh.model.geo.addVolume([
    assembly_volume2])

    volumes_dictionary["volume3"] = gmsh.model.geo.addVolume([
    assembly_volume3])

    volumes_dictionary["volume4"] = gmsh.model.geo.addVolume([
    assembly_volume4])

    volumes_dictionary["volume5"] = gmsh.model.geo.addVolume([
    assembly_volume5])

    volumes_dictionary["volume6"] = gmsh.model.geo.addVolume([
    assembly_volume6])

    volumes_dictionary["volume7"] = gmsh.model.geo.addVolume([
    assembly_volume7])

    volumes_dictionary["volume8"] = gmsh.model.geo.addVolume([
    assembly_volume8])

    volumes_dictionary["volume9"] = gmsh.model.geo.addVolume([
    assembly_volume9])

    volumes_dictionary["volume10"] = gmsh.model.geo.addVolume([
    assembly_volume10])

    volumes_dictionary["volume11"] = gmsh.model.geo.addVolume([
    assembly_volume11])

    volumes_dictionary["volume12"] = gmsh.model.geo.addVolume([
    assembly_volume12])

    gmsh.model.geo.synchronize()

    ####################################################################
    ####################################################################
    ##                    Update of physical groups                   ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                               Fiber                              #
    ####################################################################

    list_volumesFiber = [volumes_dictionary["volume1"],
    volumes_dictionary["volume2"], volumes_dictionary["volume3"],
    volumes_dictionary["volume4"], volumes_dictionary["volume5"],
    volumes_dictionary["volume6"], volumes_dictionary["volume7"],
    volumes_dictionary["volume8"]]

    ####################################################################
    #                              Matrix                              #
    ####################################################################

    list_volumesMatrix = [volumes_dictionary["volume9"],
    volumes_dictionary["volume10"], volumes_dictionary["volume11"],
    volumes_dictionary["volume12"]]

    # Makes the z_centroid again at the real centroid

    z_centroid += 0.5*RVE_lengthZ

    # Tests whether the centroid of the RVE is within a region

    general_physicalGroup = True

    for i in range(len(volume_regionIdentifiers)):

        if volume_regionIdentifiers[i](x_centroid, y_centroid, 
        z_centroid):

            dictionary_volumesPhysGroups[(i*2)+3].extend(
            list_volumesFiber)

            dictionary_volumesPhysGroups[(i*2)+4].extend(
            list_volumesMatrix)

            # If the RVE is allocated in a specific region, changes the
            # overall physical group flag

            general_physicalGroup = False

            break

    if general_physicalGroup:

        # Updates the list of volumes of the fiber phase

        dictionary_volumesPhysGroups[1].extend(list_volumesFiber)

        # Updates the list of volumes of the matrix phase

        dictionary_volumesPhysGroups[2].extend(list_volumesMatrix)

        # Set the color of the fiber to the hot color

        gmsh.model.setColor([(3,i) for i in list_volumesFiber], 180, 4,
        38)

        # Set the color of the matrix to the cold color

        gmsh.model.setColor([(3,i) for i in list_volumesMatrix], 59, 76,
        192)

    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    # Iterates through the surfaces

    for surface in surfaces_points:

        # Iterates through the surfaces identifiers to detect if they
        # belong to a surface region using the surface centroid as cri-
        # terion

        for i in range(len(surface_regionIdentifiers)):

            # Tests if the centroid belongs to this region

            if surface_regionIdentifiers[i](*surfaces_points[surface]
            ):
                
                # Adds this surface to the dictionary of physical surfa-
                # ces

                dictionary_surfacesPhysGroups[i+1].append(
                surfaces_dictionary[surface])

                # Breaks, because a surface cannot be part of more than
                # one physical group

                break
            
    ####################################################################
    ####################################################################
    ##         Converts to transfinite to regularize the mesh         ##
    ####################################################################
    ####################################################################

    gmsh.model.geo.synchronize()

    # Sets to transfinite the lines and curves

    for i in range(1,62,1):

        gmsh.model.geo.mesh.setTransfiniteCurve(lines_dictionary["l"+
        str(i)],n_transfiniteCurves[i-1])

    # Sets to transfinite the surfaces

    for i in range(1,49,1):

        gmsh.model.geo.mesh.setTransfiniteSurface(surfaces_dictionary[
        "surface"+str(i)])

    for i in range(1,13,1):

        gmsh.model.mesh.setTransfiniteVolume(volumes_dictionary["volum"+
        "e"+str(i)])

    gmsh.model.geo.synchronize()

    return dictionary_volumesPhysGroups, dictionary_surfacesPhysGroups

# Defines a function that constructs the top and bottom surfaces of the
# RVE

def RVE_centralFiberWholeInnerSquare(x_centroid, y_centroid, z_centroid, 
RVE_lengthX, RVE_lengthY, RVE_lengthZ, parameters_method, 
dictionary_volumesPhysGroups, dictionary_surfacesPhysGroups, lc, 
volume_regionIdentifiers, surface_regionIdentifiers, n_transfiniteCurves
=10):
    
    # Checks if no number of transfinite divisions has been given

    if isinstance(n_transfiniteCurves, int):

        n_base = n_transfiniteCurves+0

        n_transfiniteCurves = []

        # Puts the suggested value

        for i in range(61): 

            n_transfiniteCurves.append(n_base)

    # Takes the parameters of the method

    inner_squareSideSize = parameters_method[0]

    radius = parameters_method[1]

    # Defines the points for matrix section and the half size of the 
    # surface

    circle_interceptCoord = radius*0.5*math.sqrt(2)

    half_sizeX = 0.5*RVE_lengthX

    half_sizeY = 0.5*RVE_lengthY

    half_sizeZ = 0.5*RVE_lengthZ

    half_littleSquare = 0.5*inner_squareSideSize

    # Sets a dictionary for each one of the topologycal entities that a-
    # re liable to transfinite meshing, i.g. lines, surfaces and volumes

    lines_dictionary = dict()

    surfaces_dictionary = dict()

    volumes_dictionary = dict()

    # Sets a dictionary for the centroids of the surfaces 

    surfaces_points = dict()

    ####################################################################
    ####################################################################
    ##                        Botttom surfaces                        ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Points                              #
    ####################################################################

    # Updates the z_centroid to the bottom surface of the RVE to make
    # easier the drawing of the RVE

    z_centroid -= 0.5*RVE_lengthZ

    # Defines the centroid

    p1 = gmsh.model.geo.addPoint(x_centroid, y_centroid, z_centroid, lc)

    # Defines the points for the inner square of the fiber

    p2 = gmsh.model.geo.addPoint(x_centroid+half_littleSquare,y_centroid
    -half_littleSquare, z_centroid, lc)

    p3 = gmsh.model.geo.addPoint(x_centroid+half_littleSquare,y_centroid
    +half_littleSquare, z_centroid, lc)

    p4 = gmsh.model.geo.addPoint(x_centroid-half_littleSquare,y_centroid
    +half_littleSquare, z_centroid, lc)

    p5 = gmsh.model.geo.addPoint(x_centroid-half_littleSquare,y_centroid
    -half_littleSquare, z_centroid, lc)

    # Defines the points for the fiber section (circle intercepts)

    p6 = gmsh.model.geo.addPoint(x_centroid+circle_interceptCoord, 
    y_centroid-circle_interceptCoord, z_centroid, lc)

    p7 = gmsh.model.geo.addPoint(x_centroid+circle_interceptCoord, 
    y_centroid+circle_interceptCoord, z_centroid, lc)

    p8 = gmsh.model.geo.addPoint(x_centroid-circle_interceptCoord, 
    y_centroid+circle_interceptCoord, z_centroid, lc)

    p9 = gmsh.model.geo.addPoint(x_centroid-circle_interceptCoord, 
    y_centroid-circle_interceptCoord, z_centroid, lc)

    # Defines the points for the outer perimeter

    p10 = gmsh.model.geo.addPoint(x_centroid+half_sizeX,y_centroid-
    half_sizeY, z_centroid, lc)

    p11 = gmsh.model.geo.addPoint(x_centroid+half_sizeX, y_centroid+
    half_sizeY, z_centroid, lc)

    p12 = gmsh.model.geo.addPoint(x_centroid-half_sizeX, y_centroid+
    half_sizeY, z_centroid, lc)

    p13 = gmsh.model.geo.addPoint(x_centroid-half_sizeX, y_centroid-
    half_sizeY, z_centroid, lc)
    
    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l2"] = gmsh.model.geo.addLine(p2, p3)

    lines_dictionary["l4"] = gmsh.model.geo.addLine(p3, p4)

    lines_dictionary["l6"] = gmsh.model.geo.addLine(p4, p5)

    lines_dictionary["l8"] = gmsh.model.geo.addLine(p5, p2)

    # Creates the lines for the fiber

    lines_dictionary["l9"] = gmsh.model.geo.addLine(p2, p6)

    lines_dictionary["l10"] = gmsh.model.geo.addCircleArc(p6, p1, p7)

    lines_dictionary["l11"] = gmsh.model.geo.addLine(p7, p3)

    lines_dictionary["l12"] = gmsh.model.geo.addCircleArc(p7, p1, p8)

    lines_dictionary["l13"] = gmsh.model.geo.addLine(p8, p4)

    lines_dictionary["l14"] = gmsh.model.geo.addCircleArc(p8, p1, p9)

    lines_dictionary["l15"] = gmsh.model.geo.addLine(p9, p5)

    lines_dictionary["l16"] = gmsh.model.geo.addCircleArc(p9, p1, p6)

    lines_dictionary["l17"] = gmsh.model.geo.addLine(p6, p10)

    lines_dictionary["l18"] = gmsh.model.geo.addLine(p10, p11)

    lines_dictionary["l19"] = gmsh.model.geo.addLine(p11, p7)

    lines_dictionary["l20"] = gmsh.model.geo.addLine(p11, p12)

    lines_dictionary["l21"] = gmsh.model.geo.addLine(p12, p8)

    lines_dictionary["l22"] = gmsh.model.geo.addLine(p12, p13)

    lines_dictionary["l23"] = gmsh.model.geo.addLine(p13, p9)

    lines_dictionary["l24"] = gmsh.model.geo.addLine(p13, p10)
    
    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    # Inner square

    loop1 = gmsh.model.geo.addCurveLoop([lines_dictionary["l2"], 
    lines_dictionary["l4"], lines_dictionary["l6"], lines_dictionary[
    "l8"]])

    # Fiber

    loop5 = gmsh.model.geo.addCurveLoop([lines_dictionary["l9"], 
    lines_dictionary["l10"], lines_dictionary["l11"], -lines_dictionary[
    "l2"]])

    loop6 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l11"], 
    lines_dictionary["l12"], lines_dictionary["l13"], -lines_dictionary[
    "l4"]])

    loop7 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l13"], 
    lines_dictionary["l14"], lines_dictionary["l15"], -lines_dictionary[
    "l6"]])

    loop8 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l15"], 
    lines_dictionary["l16"], -lines_dictionary["l9"], -lines_dictionary[
    "l8"]])

    # Matrix

    loop9 = gmsh.model.geo.addCurveLoop([lines_dictionary["l17"], 
    lines_dictionary["l18"], lines_dictionary["l19"], -lines_dictionary[
    "l10"]])

    loop10 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l19"],
    lines_dictionary["l20"], lines_dictionary["l21"], -lines_dictionary[
    "l12"]])

    loop11 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l21"], 
    lines_dictionary["l22"], lines_dictionary["l23"], -lines_dictionary[
    "l14"]])

    loop12 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l23"], 
    lines_dictionary["l24"], -lines_dictionary["l17"], 
    -lines_dictionary["l16"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    # Inner square

    surfaces_dictionary["surface1"] = gmsh.model.geo.addPlaneSurface([
    loop1])

    # Fiber

    surfaces_dictionary["surface5"] = gmsh.model.geo.addPlaneSurface([
    loop5])

    surfaces_dictionary["surface6"] = gmsh.model.geo.addPlaneSurface([
    loop6])

    surfaces_dictionary["surface7"] = gmsh.model.geo.addPlaneSurface([
    loop7])

    surfaces_dictionary["surface8"] = gmsh.model.geo.addPlaneSurface([
    loop8])

    # Matrix

    surfaces_dictionary["surface9"] = gmsh.model.geo.addPlaneSurface([
    loop9])

    surfaces_dictionary["surface10"] = gmsh.model.geo.addPlaneSurface([
    loop10])

    surfaces_dictionary["surface11"] = gmsh.model.geo.addPlaneSurface([
    loop11])

    surfaces_dictionary["surface12"] = gmsh.model.geo.addPlaneSurface([
    loop12])

    gmsh.model.geo.synchronize()

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface1"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface1"])

    surfaces_points["surface5"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface5"])

    surfaces_points["surface6"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface6"])

    surfaces_points["surface7"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface7"])

    surfaces_points["surface8"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface8"])

    surfaces_points["surface9"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface9"])

    surfaces_points["surface10"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface10"])

    surfaces_points["surface11"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface11"])

    surfaces_points["surface12"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface12"])
    
    ####################################################################
    ####################################################################
    ##                          Top surfaces                          ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Points                              #
    ####################################################################

    # Defines the centroid

    p14 = gmsh.model.geo.addPoint(x_centroid, y_centroid, z_centroid+
    RVE_lengthZ, lc)

    # Defines the points for the inner square of the fiber

    p15 = gmsh.model.geo.addPoint(x_centroid+half_littleSquare,y_centroid
    -half_littleSquare, z_centroid+RVE_lengthZ, lc)

    p16 = gmsh.model.geo.addPoint(x_centroid+half_littleSquare,y_centroid
    +half_littleSquare, z_centroid+RVE_lengthZ, lc)

    p17 = gmsh.model.geo.addPoint(x_centroid-half_littleSquare,y_centroid
    +half_littleSquare, z_centroid+RVE_lengthZ, lc)

    p18 = gmsh.model.geo.addPoint(x_centroid-half_littleSquare,y_centroid
    -half_littleSquare, z_centroid+RVE_lengthZ, lc)

    # Defines the points for the fiber section (circle intercepts)

    p19 = gmsh.model.geo.addPoint(x_centroid+circle_interceptCoord, 
    y_centroid-circle_interceptCoord, z_centroid+RVE_lengthZ, lc)

    p20 = gmsh.model.geo.addPoint(x_centroid+circle_interceptCoord, 
    y_centroid+circle_interceptCoord, z_centroid+RVE_lengthZ, lc)

    p21 = gmsh.model.geo.addPoint(x_centroid-circle_interceptCoord, 
    y_centroid+circle_interceptCoord, z_centroid+RVE_lengthZ, lc)

    p22 = gmsh.model.geo.addPoint(x_centroid-circle_interceptCoord, 
    y_centroid-circle_interceptCoord, z_centroid+RVE_lengthZ, lc)

    # Defines the points for the outer perimeter

    p23 = gmsh.model.geo.addPoint(x_centroid+half_sizeX,y_centroid-
    half_sizeY, z_centroid+RVE_lengthZ, lc)

    p24 = gmsh.model.geo.addPoint(x_centroid+half_sizeX, y_centroid+
    half_sizeY, z_centroid+RVE_lengthZ, lc)

    p25 = gmsh.model.geo.addPoint(x_centroid-half_sizeX, y_centroid+
    half_sizeY, z_centroid+RVE_lengthZ, lc)

    p26 = gmsh.model.geo.addPoint(x_centroid-half_sizeX, y_centroid-
    half_sizeY, z_centroid+RVE_lengthZ, lc)

    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l26"] = gmsh.model.geo.addLine(p15, p16)

    lines_dictionary["l28"] = gmsh.model.geo.addLine(p16, p17)

    lines_dictionary["l30"] = gmsh.model.geo.addLine(p17, p18)

    lines_dictionary["l32"] = gmsh.model.geo.addLine(p18, p15)

    # Creates the lines for the fiber

    lines_dictionary["l33"] = gmsh.model.geo.addLine(p15, p19)

    lines_dictionary["l34"] = gmsh.model.geo.addCircleArc(p19, p14, p20)

    lines_dictionary["l35"] = gmsh.model.geo.addLine(p20, p16)

    lines_dictionary["l36"] = gmsh.model.geo.addCircleArc(p20, p14, p21)

    lines_dictionary["l37"] = gmsh.model.geo.addLine(p21, p17)

    lines_dictionary["l38"] = gmsh.model.geo.addCircleArc(p21, p14, p22)

    lines_dictionary["l39"] = gmsh.model.geo.addLine(p22, p18)

    lines_dictionary["l40"] = gmsh.model.geo.addCircleArc(p22, p14, p19)

    lines_dictionary["l41"] = gmsh.model.geo.addLine(p19, p23)

    lines_dictionary["l42"] = gmsh.model.geo.addLine(p23, p24)

    lines_dictionary["l43"] = gmsh.model.geo.addLine(p24, p20)

    lines_dictionary["l44"] = gmsh.model.geo.addLine(p24, p25)

    lines_dictionary["l45"] = gmsh.model.geo.addLine(p25, p21)

    lines_dictionary["l46"] = gmsh.model.geo.addLine(p25, p26)

    lines_dictionary["l47"] = gmsh.model.geo.addLine(p26, p22)

    lines_dictionary["l48"] = gmsh.model.geo.addLine(p26, p23)

    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    # Inner square

    loop13 = gmsh.model.geo.addCurveLoop([lines_dictionary["l26"], 
    lines_dictionary["l28"], lines_dictionary["l30"], lines_dictionary[
    "l32"]])

    # Fiber

    loop17 = gmsh.model.geo.addCurveLoop([lines_dictionary["l33"], 
    lines_dictionary["l34"], lines_dictionary["l35"], -lines_dictionary[
    "l26"]])

    loop18 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l35"], 
    lines_dictionary["l36"], lines_dictionary["l37"], -lines_dictionary[
    "l28"]])

    loop19 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l37"], 
    lines_dictionary["l38"], lines_dictionary["l39"], -lines_dictionary[
    "l30"]])

    loop20 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l39"], 
    lines_dictionary["l40"], -lines_dictionary["l33"], -lines_dictionary[
    "l32"]])

    # Matrix

    loop21 = gmsh.model.geo.addCurveLoop([lines_dictionary["l41"], 
    lines_dictionary["l42"], lines_dictionary["l43"], -lines_dictionary[
    "l34"]])

    loop22 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l43"], 
    lines_dictionary["l44"], lines_dictionary["l45"], -lines_dictionary[
    "l36"]])

    loop23 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l45"], 
    lines_dictionary["l46"], lines_dictionary["l47"], -lines_dictionary[
    "l38"]])

    loop24 = gmsh.model.geo.addCurveLoop([-lines_dictionary["l47"], 
    lines_dictionary["l48"], -lines_dictionary["l41"], -lines_dictionary[
    "l40"]])

    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    # Inner square

    surfaces_dictionary["surface13"] = gmsh.model.geo.addPlaneSurface([
    loop13])

    # Fiber

    surfaces_dictionary["surface17"] = gmsh.model.geo.addPlaneSurface([
    loop17])

    surfaces_dictionary["surface18"] = gmsh.model.geo.addPlaneSurface([
    loop18])

    surfaces_dictionary["surface19"] = gmsh.model.geo.addPlaneSurface([
    loop19])

    surfaces_dictionary["surface20"] = gmsh.model.geo.addPlaneSurface([
    loop20])

    # Matrix

    surfaces_dictionary["surface21"] = gmsh.model.geo.addPlaneSurface([
    loop21])

    surfaces_dictionary["surface22"] = gmsh.model.geo.addPlaneSurface([
    loop22])

    surfaces_dictionary["surface23"] = gmsh.model.geo.addPlaneSurface([
    loop23])

    surfaces_dictionary["surface24"] = gmsh.model.geo.addPlaneSurface([
    loop24])

    gmsh.model.geo.synchronize()

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface13"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface13"])

    surfaces_points["surface17"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface17"])

    surfaces_points["surface18"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface18"])

    surfaces_points["surface19"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface19"])

    surfaces_points["surface20"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface20"])

    surfaces_points["surface21"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface21"])

    surfaces_points["surface22"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface22"])

    surfaces_points["surface23"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface23"])

    surfaces_points["surface24"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface24"])
    
    ####################################################################
    ####################################################################
    ##                      Inner square quarters                     ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l50"] = gmsh.model.geo.addLine(p15, p2)

    lines_dictionary["l51"] = gmsh.model.geo.addLine(p16, p3)

    lines_dictionary["l52"] = gmsh.model.geo.addLine(p17, p4)

    lines_dictionary["l53"] = gmsh.model.geo.addLine(p18, p5)

    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    loop26 = gmsh.model.geo.addCurveLoop([lines_dictionary["l50"], 
    lines_dictionary["l2"], -lines_dictionary["l51"], -lines_dictionary[
    "l26"]])

    loop28 = gmsh.model.geo.addCurveLoop([lines_dictionary["l51"], 
    lines_dictionary["l4"], -lines_dictionary["l52"], -lines_dictionary[
    "l28"]])

    loop30 = gmsh.model.geo.addCurveLoop([lines_dictionary["l52"], 
    lines_dictionary["l6"], -lines_dictionary["l53"], -lines_dictionary[
    "l30"]])

    loop32 = gmsh.model.geo.addCurveLoop([lines_dictionary["l53"], 
    lines_dictionary["l8"], -lines_dictionary["l50"], -lines_dictionary[
    "l32"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    surfaces_dictionary["surface26"] = gmsh.model.geo.addPlaneSurface([
    loop26])

    surfaces_dictionary["surface28"] = gmsh.model.geo.addPlaneSurface([
    loop28])

    surfaces_dictionary["surface30"] = gmsh.model.geo.addPlaneSurface([
    loop30])

    surfaces_dictionary["surface32"] = gmsh.model.geo.addPlaneSurface([
    loop32])

    gmsh.model.geo.synchronize()

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface26"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface26"])

    surfaces_points["surface28"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface28"])

    surfaces_points["surface30"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface30"])

    surfaces_points["surface32"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface32"])
    
    ####################################################################
    #                              Volumes                             #
    ####################################################################

    assembly_volume1 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface1"], 
    surfaces_dictionary["surface26"], surfaces_dictionary["surface28"], 
    surfaces_dictionary["surface30"], surfaces_dictionary["surface32"],
    surfaces_dictionary["surface13"]])
    
    ####################################################################
    ####################################################################
    ##                      Cylindrical quarters                      ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l54"] = gmsh.model.geo.addLine(p19, p6)

    lines_dictionary["l55"] = gmsh.model.geo.addLine(p20, p7)

    lines_dictionary["l56"] = gmsh.model.geo.addLine(p21, p8)

    lines_dictionary["l57"] = gmsh.model.geo.addLine(p22, p9)
    
    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    loop33 = gmsh.model.geo.addCurveLoop([lines_dictionary["l50"], 
    lines_dictionary["l9"], -lines_dictionary["l54"], -lines_dictionary[
    "l33"]])

    loop34 = gmsh.model.geo.addCurveLoop([lines_dictionary["l54"], 
    lines_dictionary["l10"], -lines_dictionary["l55"], 
    -lines_dictionary["l34"]])

    loop35 = gmsh.model.geo.addCurveLoop([lines_dictionary["l55"], 
    lines_dictionary["l11"], -lines_dictionary["l51"], 
    -lines_dictionary["l35"]])

    loop36 = gmsh.model.geo.addCurveLoop([lines_dictionary["l55"], 
    lines_dictionary["l12"], -lines_dictionary["l56"], 
    -lines_dictionary["l36"]])

    loop37 = gmsh.model.geo.addCurveLoop([lines_dictionary["l56"], 
    lines_dictionary["l13"], -lines_dictionary["l52"], 
    -lines_dictionary["l37"]])

    loop38 = gmsh.model.geo.addCurveLoop([lines_dictionary["l56"], 
    lines_dictionary["l14"], -lines_dictionary["l57"], 
    -lines_dictionary["l38"]])

    loop39 = gmsh.model.geo.addCurveLoop([lines_dictionary["l57"], 
    lines_dictionary["l15"], -lines_dictionary["l53"], 
    -lines_dictionary["l39"]])

    loop40 = gmsh.model.geo.addCurveLoop([lines_dictionary["l57"], 
    lines_dictionary["l16"], -lines_dictionary["l54"], 
    -lines_dictionary["l40"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    surfaces_dictionary["surface33"] = gmsh.model.geo.addPlaneSurface([
    loop33])

    surfaces_dictionary["surface34"] = gmsh.model.geo.addSurfaceFilling(
    [loop34])

    surfaces_dictionary["surface35"] = gmsh.model.geo.addPlaneSurface([
    loop35])

    surfaces_dictionary["surface36"] = gmsh.model.geo.addSurfaceFilling(
    [loop36])

    surfaces_dictionary["surface37"] = gmsh.model.geo.addPlaneSurface([
    loop37])

    surfaces_dictionary["surface38"] = gmsh.model.geo.addSurfaceFilling(
    [loop38])

    surfaces_dictionary["surface39"] = gmsh.model.geo.addPlaneSurface([
    loop39])

    surfaces_dictionary["surface40"] = gmsh.model.geo.addSurfaceFilling(
    [loop40])

    gmsh.model.geo.synchronize()

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface33"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface33"])

    surfaces_points["surface34"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface34"])

    surfaces_points["surface35"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface35"])

    surfaces_points["surface36"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface36"])

    surfaces_points["surface37"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface37"])

    surfaces_points["surface38"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface38"])

    surfaces_points["surface39"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface39"])

    surfaces_points["surface40"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface40"])
    
    ####################################################################
    #                              Volumes                             #
    ####################################################################

    assembly_volume5 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface5"], 
    surfaces_dictionary["surface26"], surfaces_dictionary["surface33"], 
    surfaces_dictionary["surface34"], surfaces_dictionary["surface35"], 
    surfaces_dictionary["surface17"]])

    assembly_volume6 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface6"], 
    surfaces_dictionary["surface28"], surfaces_dictionary["surface35"], 
    surfaces_dictionary["surface36"], surfaces_dictionary["surface37"], 
    surfaces_dictionary["surface18"]])

    assembly_volume7 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface7"], 
    surfaces_dictionary["surface30"], surfaces_dictionary["surface37"], 
    surfaces_dictionary["surface38"], surfaces_dictionary["surface39"], 
    surfaces_dictionary["surface19"]])

    assembly_volume8 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface8"], 
    surfaces_dictionary["surface32"], surfaces_dictionary["surface33"], 
    surfaces_dictionary["surface40"], surfaces_dictionary["surface39"], 
    surfaces_dictionary["surface20"]])
    
    ####################################################################
    ####################################################################
    ##                        Matrix quarters                         ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                              Lines                               #
    ####################################################################

    # Creates the lines for the inner square

    lines_dictionary["l58"] = gmsh.model.geo.addLine(p23, p10)

    lines_dictionary["l59"] = gmsh.model.geo.addLine(p24, p11)

    lines_dictionary["l60"] = gmsh.model.geo.addLine(p25, p12)

    lines_dictionary["l61"] = gmsh.model.geo.addLine(p26, p13)
    
    ####################################################################
    #                            Curve Loops                           #
    ####################################################################

    loop41 = gmsh.model.geo.addCurveLoop([lines_dictionary["l54"], 
    lines_dictionary["l17"], -lines_dictionary["l58"], -lines_dictionary[
    "l41"]])

    loop42 = gmsh.model.geo.addCurveLoop([lines_dictionary["l58"], 
    lines_dictionary["l18"], -lines_dictionary["l59"], -lines_dictionary[
    "l42"]])

    loop43 = gmsh.model.geo.addCurveLoop([lines_dictionary["l59"], 
    lines_dictionary["l19"], -lines_dictionary["l55"], -lines_dictionary[
    "l43"]])

    loop44 = gmsh.model.geo.addCurveLoop([lines_dictionary["l59"], 
    lines_dictionary["l20"], -lines_dictionary["l60"], -lines_dictionary[
    "l44"]])

    loop45 = gmsh.model.geo.addCurveLoop([lines_dictionary["l60"], 
    lines_dictionary["l21"], -lines_dictionary["l56"], -lines_dictionary[
    "l45"]])

    loop46 = gmsh.model.geo.addCurveLoop([lines_dictionary["l60"], 
    lines_dictionary["l22"], -lines_dictionary["l61"], -lines_dictionary[
    "l46"]])

    loop47 = gmsh.model.geo.addCurveLoop([lines_dictionary["l61"], 
    lines_dictionary["l23"], -lines_dictionary["l57"], -lines_dictionary[
    "l47"]])

    loop48 = gmsh.model.geo.addCurveLoop([lines_dictionary["l61"], 
    lines_dictionary["l24"], -lines_dictionary["l58"], -lines_dictionary[
    "l48"]])
    
    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    surfaces_dictionary["surface41"] = gmsh.model.geo.addPlaneSurface([
    loop41])

    surfaces_dictionary["surface42"] = gmsh.model.geo.addPlaneSurface([
    loop42])

    surfaces_dictionary["surface43"] = gmsh.model.geo.addPlaneSurface([
    loop43])

    surfaces_dictionary["surface44"] = gmsh.model.geo.addPlaneSurface([
    loop44])

    surfaces_dictionary["surface45"] = gmsh.model.geo.addPlaneSurface([
    loop45])

    surfaces_dictionary["surface46"] = gmsh.model.geo.addPlaneSurface([
    loop46])

    surfaces_dictionary["surface47"] = gmsh.model.geo.addPlaneSurface([
    loop47])

    surfaces_dictionary["surface48"] = gmsh.model.geo.addPlaneSurface([
    loop48])

    gmsh.model.geo.synchronize()

    # Adds the centroids of the surfaces to their corresponding dictio-
    # nary

    surfaces_points["surface41"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface41"])

    surfaces_points["surface42"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface42"])

    surfaces_points["surface43"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface43"])

    surfaces_points["surface44"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface44"])

    surfaces_points["surface45"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface45"])

    surfaces_points["surface46"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface46"])

    surfaces_points["surface47"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface47"])

    surfaces_points["surface48"] = tools.get_boudaryPointsSurface(
    surfaces_dictionary["surface48"])
    
    ####################################################################
    #                              Volumes                             #
    ####################################################################

    assembly_volume9 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface9"], 
    surfaces_dictionary["surface41"], surfaces_dictionary["surface42"], 
    surfaces_dictionary["surface43"], surfaces_dictionary["surface34"], 
    surfaces_dictionary["surface21"]])

    assembly_volume10 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface10"], 
    surfaces_dictionary["surface43"], surfaces_dictionary["surface44"], 
    surfaces_dictionary["surface45"], surfaces_dictionary["surface36"], 
    surfaces_dictionary["surface22"]])

    assembly_volume11 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface11"], 
    surfaces_dictionary["surface45"], surfaces_dictionary["surface46"], 
    surfaces_dictionary["surface47"], surfaces_dictionary["surface38"], 
    surfaces_dictionary["surface23"]])

    assembly_volume12 = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface12"], 
    surfaces_dictionary["surface47"], surfaces_dictionary["surface48"], 
    surfaces_dictionary["surface41"], surfaces_dictionary["surface40"], 
    surfaces_dictionary["surface24"]])
    
    ####################################################################
    ####################################################################
    ##                            Assembly                            ##
    ####################################################################
    ####################################################################

    # Builds the volumes using the surfaces

    volumes_dictionary["volume1"] = gmsh.model.geo.addVolume([
    assembly_volume1])

    volumes_dictionary["volume5"] = gmsh.model.geo.addVolume([
    assembly_volume5])

    volumes_dictionary["volume6"] = gmsh.model.geo.addVolume([
    assembly_volume6])

    volumes_dictionary["volume7"] = gmsh.model.geo.addVolume([
    assembly_volume7])

    volumes_dictionary["volume8"] = gmsh.model.geo.addVolume([
    assembly_volume8])

    volumes_dictionary["volume9"] = gmsh.model.geo.addVolume([
    assembly_volume9])

    volumes_dictionary["volume10"] = gmsh.model.geo.addVolume([
    assembly_volume10])

    volumes_dictionary["volume11"] = gmsh.model.geo.addVolume([
    assembly_volume11])

    volumes_dictionary["volume12"] = gmsh.model.geo.addVolume([
    assembly_volume12])

    gmsh.model.geo.synchronize()

    ####################################################################
    ####################################################################
    ##                    Update of physical groups                   ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                               Fiber                              #
    ####################################################################

    list_volumesFiber = [volumes_dictionary["volume1"],
    volumes_dictionary["volume5"], volumes_dictionary["volume6"], 
    volumes_dictionary["volume7"], volumes_dictionary["volume8"]]

    ####################################################################
    #                              Matrix                              #
    ####################################################################

    list_volumesMatrix = [volumes_dictionary["volume9"],
    volumes_dictionary["volume10"], volumes_dictionary["volume11"],
    volumes_dictionary["volume12"]]

    # Corrects the z coordinate of the RVE centroid for the RVE length

    z_centroid += 0.5*RVE_lengthZ

    # Tests whether the centroid of the RVE is within a region

    general_physicalGroup = True

    for i in range(len(volume_regionIdentifiers)):

        if volume_regionIdentifiers[i](x_centroid, y_centroid, 
        z_centroid):

            dictionary_volumesPhysGroups[(i*2)+3].extend(
            list_volumesFiber)

            dictionary_volumesPhysGroups[(i*2)+4].extend(
            list_volumesMatrix)

            # If the RVE is allocated in a specific region, changes the
            # overall physical group flag

            general_physicalGroup = False

            # Breaks, because a volume cannot be part of more than one
            # physical group

            break

    if general_physicalGroup:

        # Updates the list of volumes of the fiber phase

        dictionary_volumesPhysGroups[1].extend(list_volumesFiber)

        # Updates the list of volumes of the matrix phase

        dictionary_volumesPhysGroups[2].extend(list_volumesMatrix)

        # Set the color of the fiber to the hot color

        gmsh.model.setColor([(3,i) for i in list_volumesFiber], 180, 4,
        38)

        # Set the color of the matrix to the cold color

        gmsh.model.setColor([(3,i) for i in list_volumesMatrix], 59, 76,
        192)

    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    # Iterates through the surfaces

    for surface in surfaces_points:

        # Iterates through the surfaces identifiers to detect if they
        # belong to a surface region using the surface centroid as cri-
        # terion

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

    ####################################################################
    ####################################################################
    ##         Converts to transfinite to regularize the mesh         ##
    ####################################################################
    ####################################################################

    gmsh.model.geo.synchronize()

    # Sets to transfinite the lines and curves

    for i in range(1,62,1):

        if not (i in [1, 3, 5, 7, 25, 27, 29, 31, 49]):

            gmsh.model.geo.mesh.setTransfiniteCurve(lines_dictionary[
            "l"+str(i)],n_transfiniteCurves[i-1])

    # Sets to transfinite the surfaces

    for i in range(1,49,1):

        if not (i in [2, 3, 4, 14, 15, 16, 25, 27, 29, 31]):

            gmsh.model.geo.mesh.setTransfiniteSurface(surfaces_dictionary[
            "surface"+str(i)])

    for i in range(1,13,1):

        if not (i in [2, 3, 4]):

            gmsh.model.mesh.setTransfiniteVolume(volumes_dictionary["v"+
            "olume"+str(i)])

    gmsh.model.geo.synchronize()

    return dictionary_volumesPhysGroups, dictionary_surfacesPhysGroups

########################################################################
#                       Parameters verification                        #
########################################################################

def verify_parameters(parameters_method, RVE_lengthX, RVE_lengthY, 
RVE_lengthZ):

    # Retrieves parameters of the arterial microstructure

    inner_squareSideSize = parameters_method[0]

    radius = parameters_method[1]

    # Verifies whether the radius is lesser than the RVE sides

    if radius>(0.5*RVE_lengthX):
        
        raise Exception("\nThe radius of the fiber is too large, it do"+
        "es not fit into the RVE in the X direction")

    if radius>(0.5*RVE_lengthY):
        
        raise Exception("\nThe radius of the fiber is too large, it do"+
        "es not fit into the RVE in the Y direction")

    # Verifies the inner square

    if (0.5*np.sqrt(2.0)*inner_squareSideSize)>=radius:

        raise Exception("\nThe inner square of the fiber is too large,"+
        " it is equal or larger to the fiber itself")