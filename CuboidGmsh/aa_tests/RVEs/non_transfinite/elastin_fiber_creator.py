# Routine to store functions to create the elastin fibers in gmsh

import numpy as np

import gmsh

import CuboidGmsh.source.tool_box.geometric_tools as geo_tools

# Defines a function to create the elastin fiber in gmsh

def elastin_fiber(inferior_fiberCentroid, fiber_diameter, polar_angle, 
azimuthal_angle, collagen_thickness, intralamellar_thickness, lc):

    # Calculate the length of the principal axes using the projection of
    # a tilted cylinder over a plane z=z0
    #
    #   |               
    # b |        Imagine an ellipse here
    #   |
    #    ---------------------------------------
    #                      a

    a = fiber_diameter/np.cos(azimuthal_angle)

    b = fiber_diameter*1.0

    # Sets a dictionary for each point, line, curve loop, and surface

    points_dictionary = dict()

    lines_dictionary = dict()

    loops_dictionary = dict()

    surfaces_dictionary = dict()

    ####################################################################
    #                              Points                              #
    ####################################################################

    # Constructs a matrix of points coordinates for the lines connecting
    # the four ellipses

    points_matrix = [[], [], []]

    # Adds the points of the first section and the centroid of the el-
    # lipse

    points_matrix = add_edgePointsEllipse(points_matrix, 0.0, 0.0, 0.0, 
    a, b)

    # Adds the points of the second section

    shift = collagen_thickness*np.tan(azimuthal_angle)

    points_matrix = add_edgePointsEllipse(points_matrix, shift, 0.0, 
    collagen_thickness, a, b)

    # Adds the points of the third section

    shift = ((intralamellar_thickness-collagen_thickness)*np.tan(
    azimuthal_angle))

    points_matrix = add_edgePointsEllipse(points_matrix, shift, 0.0, 
    intralamellar_thickness-collagen_thickness, a, b)

    # Adds the points of the fourth section

    shift = intralamellar_thickness*np.tan(azimuthal_angle)

    points_matrix = add_edgePointsEllipse(points_matrix, shift, 0.0, 
    intralamellar_thickness, a, b)

    # Rotates these points using the rotation about the polar angle

    points_matrix = geo_tools.rotate_andTranslate(np.array(points_matrix
    ), polar_angle, *inferior_fiberCentroid)

    # Creates the points in gmsh

    points_dictionary["p10"] = gmsh.model.geo.addPoint(points_matrix[0,
    0], points_matrix[1,0], points_matrix[2,0], lc)

    points_dictionary["p11"] = gmsh.model.geo.addPoint(points_matrix[0,
    1], points_matrix[1,1], points_matrix[2,1], lc)

    points_dictionary["p12"] = gmsh.model.geo.addPoint(points_matrix[0,
    2], points_matrix[1,2], points_matrix[2,2], lc)

    points_dictionary["p13"] = gmsh.model.geo.addPoint(points_matrix[0,
    3], points_matrix[1,3], points_matrix[2,3], lc)

    points_dictionary["p14"] = gmsh.model.geo.addPoint(points_matrix[0,
    4], points_matrix[1,4], points_matrix[2,4], lc)

    points_dictionary["p20"] = gmsh.model.geo.addPoint(points_matrix[0,
    5], points_matrix[1,5], points_matrix[2,5], lc)

    points_dictionary["p21"] = gmsh.model.geo.addPoint(points_matrix[0,
    6], points_matrix[1,6], points_matrix[2,6], lc)

    points_dictionary["p22"] = gmsh.model.geo.addPoint(points_matrix[0,
    7], points_matrix[1,7], points_matrix[2,7], lc)

    points_dictionary["p23"] = gmsh.model.geo.addPoint(points_matrix[0,
    8], points_matrix[1,8], points_matrix[2,8], lc)

    points_dictionary["p24"] = gmsh.model.geo.addPoint(points_matrix[0,
    9], points_matrix[1,9], points_matrix[2,9], lc)

    points_dictionary["p30"] = gmsh.model.geo.addPoint(points_matrix[0,
    10], points_matrix[1,10], points_matrix[2,10], lc)

    points_dictionary["p31"] = gmsh.model.geo.addPoint(points_matrix[0,
    11], points_matrix[1,11], points_matrix[2,11], lc)

    points_dictionary["p32"] = gmsh.model.geo.addPoint(points_matrix[0,
    12], points_matrix[1,12], points_matrix[2,12], lc)

    points_dictionary["p33"] = gmsh.model.geo.addPoint(points_matrix[0,
    13], points_matrix[1,13],points_matrix[2,13], lc)

    points_dictionary["p34"] = gmsh.model.geo.addPoint(points_matrix[0,
    14], points_matrix[1,14], points_matrix[2,14], lc)

    points_dictionary["p40"] = gmsh.model.geo.addPoint(points_matrix[0,
    15], points_matrix[1,15], points_matrix[2,15], lc)

    points_dictionary["p41"] = gmsh.model.geo.addPoint(points_matrix[0,
    16], points_matrix[1,16], points_matrix[2,16], lc)

    points_dictionary["p42"] = gmsh.model.geo.addPoint(points_matrix[0,
    17], points_matrix[1,17], points_matrix[2,17], lc)

    points_dictionary["p43"] = gmsh.model.geo.addPoint(points_matrix[0,
    18], points_matrix[1,18], points_matrix[2,18], lc)

    points_dictionary["p44"] = gmsh.model.geo.addPoint(points_matrix[0,
    19], points_matrix[1,19], points_matrix[2,19], lc)

    ####################################################################
    #                              Curves                              #
    ####################################################################

    ### Inferior lamellar section

    x0 = inferior_fiberCentroid[0]*1.0

    y0 = inferior_fiberCentroid[1]*1.0

    z0 = inferior_fiberCentroid[2]*1.0

    gmsh.model.geo.addPoint(x0, y0, z0, lc)

    # Adds the ellipse quarters

    lines_dictionary["l1"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p11"], points_dictionary["p10"], 
    points_dictionary["p13"], points_dictionary["p12"])

    lines_dictionary["l2"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p12"], points_dictionary["p10"], 
    points_dictionary["p13"], points_dictionary["p13"])

    lines_dictionary["l3"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p13"], points_dictionary["p10"], 
    points_dictionary["p13"], points_dictionary["p14"])

    lines_dictionary["l4"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p14"], points_dictionary["p10"], 
    points_dictionary["p13"], points_dictionary["p11"])

    # Adds the lines connecting the ellipse quarters

    lines_dictionary["l5"] = gmsh.model.geo.addLine(points_dictionary[
    "p11"],points_dictionary["p21"])

    lines_dictionary["l6"] = gmsh.model.geo.addLine(points_dictionary[
    "p12"],points_dictionary["p22"])

    lines_dictionary["l7"] = gmsh.model.geo.addLine(points_dictionary[
    "p13"],points_dictionary["p23"])

    lines_dictionary["l8"] = gmsh.model.geo.addLine(points_dictionary[
    "p14"],points_dictionary["p24"])

    ### Inferior collagen top surface

    z0 = inferior_fiberCentroid[2]+collagen_thickness

    shift = collagen_thickness*np.tan(azimuthal_angle)

    x0 = inferior_fiberCentroid[0]+(shift*np.cos(polar_angle))

    y0 = inferior_fiberCentroid[1]+(shift*np.sin(polar_angle))

    # Adds the ellipse quarters

    lines_dictionary["l9"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p21"], points_dictionary["p20"], 
    points_dictionary["p23"], points_dictionary["p22"])

    lines_dictionary["l10"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p22"], points_dictionary["p20"], 
    points_dictionary["p23"], points_dictionary["p23"])

    lines_dictionary["l11"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p23"], points_dictionary["p20"], 
    points_dictionary["p23"], points_dictionary["p24"])

    lines_dictionary["l12"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p24"], points_dictionary["p20"], 
    points_dictionary["p23"], points_dictionary["p21"])

    # Adds the lines connecting the ellipse quarters

    lines_dictionary["l13"] = gmsh.model.geo.addLine(points_dictionary[
    "p21"],points_dictionary["p31"])

    lines_dictionary["l14"] = gmsh.model.geo.addLine(points_dictionary[
    "p22"],points_dictionary["p32"])

    lines_dictionary["l15"] = gmsh.model.geo.addLine(points_dictionary[
    "p23"],points_dictionary["p33"])

    lines_dictionary["l16"] = gmsh.model.geo.addLine(points_dictionary[
    "p24"],points_dictionary["p34"])

    ### superior collagen bottom surface

    z0 = (inferior_fiberCentroid[2]+intralamellar_thickness-
    collagen_thickness)

    shift = ((intralamellar_thickness-collagen_thickness)*np.tan(
    azimuthal_angle))

    x0 = inferior_fiberCentroid[0]+(shift*np.cos(polar_angle))

    y0 = inferior_fiberCentroid[1]+(shift*np.sin(polar_angle))

    # Adds the ellipse quarters

    lines_dictionary["l17"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p31"], points_dictionary["p30"], 
    points_dictionary["p33"], points_dictionary["p32"])

    lines_dictionary["l18"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p32"], points_dictionary["p30"], 
    points_dictionary["p33"], points_dictionary["p33"])

    lines_dictionary["l19"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p33"], points_dictionary["p30"], 
    points_dictionary["p33"], points_dictionary["p34"])

    lines_dictionary["l20"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p34"], points_dictionary["p30"], 
    points_dictionary["p33"], points_dictionary["p31"])

    # Adds the lines connecting the ellipse quarters

    lines_dictionary["l21"] = gmsh.model.geo.addLine(points_dictionary[
    "p31"],points_dictionary["p41"])

    lines_dictionary["l22"] = gmsh.model.geo.addLine(points_dictionary[
    "p32"],points_dictionary["p42"])

    lines_dictionary["l23"] = gmsh.model.geo.addLine(points_dictionary[
    "p33"],points_dictionary["p43"])

    lines_dictionary["l24"] = gmsh.model.geo.addLine(points_dictionary[
    "p34"],points_dictionary["p44"])

    ### superior collagen top surface

    z0 = inferior_fiberCentroid[2]+intralamellar_thickness

    shift = intralamellar_thickness*np.tan(azimuthal_angle)

    x0 = inferior_fiberCentroid[0]+(shift*np.cos(polar_angle))

    y0 = inferior_fiberCentroid[1]+(shift*np.sin(polar_angle))

    # Adds the ellipse quarters

    lines_dictionary["l25"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p41"], points_dictionary["p40"], 
    points_dictionary["p43"], points_dictionary["p42"])

    lines_dictionary["l26"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p42"], points_dictionary["p40"], 
    points_dictionary["p43"], points_dictionary["p43"])

    lines_dictionary["l27"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p43"], points_dictionary["p40"], 
    points_dictionary["p43"], points_dictionary["p44"])

    lines_dictionary["l28"] = gmsh.model.geo.addEllipseArc(
    points_dictionary["p44"], points_dictionary["p40"], 
    points_dictionary["p43"], points_dictionary["p41"])

    ####################################################################
    #                           Curve loops                            #
    ####################################################################

    # Adds the curve loops for the surfaces of the bottom layer of col-
    # lagen

    loops_dictionary["loop1"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l1"], lines_dictionary["l2"], lines_dictionary[
    "l3"], lines_dictionary["l4"]])

    loops_dictionary["loop2"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l1"], lines_dictionary["l6"], -lines_dictionary[
    "l9"], -lines_dictionary["l5"]])

    loops_dictionary["loop3"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l2"], lines_dictionary["l7"], -lines_dictionary[
    "l10"], -lines_dictionary["l6"]])

    loops_dictionary["loop4"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l3"], lines_dictionary["l8"], -lines_dictionary[
    "l11"], -lines_dictionary["l7"]])

    loops_dictionary["loop5"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l4"], lines_dictionary["l5"], -lines_dictionary[
    "l12"], -lines_dictionary["l8"]])

    # Adds the curve loops for the surfaces of the elastin surrounded by
    # muscle tissue

    loops_dictionary["loop6"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l9"], lines_dictionary["l14"], -lines_dictionary[
    "l17"], -lines_dictionary["l13"]])

    loops_dictionary["loop7"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l10"], lines_dictionary["l15"], -lines_dictionary[
    "l18"], -lines_dictionary["l14"]])

    loops_dictionary["loop8"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l11"], lines_dictionary["l16"], -lines_dictionary[
    "l19"], -lines_dictionary["l15"]])

    loops_dictionary["loop9"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l12"], lines_dictionary["l13"], -lines_dictionary[
    "l20"], -lines_dictionary["l16"]])

    # Adds the curve loops for the surfaces of the elastin surrounded by
    # collagen at the top collagen layer

    loops_dictionary["loop10"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l17"], lines_dictionary["l22"], -lines_dictionary[
    "l25"], -lines_dictionary["l21"]])

    loops_dictionary["loop11"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l18"], lines_dictionary["l23"], -lines_dictionary[
    "l26"], -lines_dictionary["l22"]])

    loops_dictionary["loop12"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l19"], lines_dictionary["l24"], -lines_dictionary[
    "l27"], -lines_dictionary["l23"]])

    loops_dictionary["loop13"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l20"], lines_dictionary["l21"], -lines_dictionary[
    "l28"], -lines_dictionary["l24"]])

    loops_dictionary["loop14"] = gmsh.model.geo.addCurveLoop([
    lines_dictionary["l25"], lines_dictionary["l26"], lines_dictionary[
    "l27"], lines_dictionary["l28"]])

    ####################################################################
    #                             Surfaces                             #
    ####################################################################

    # Adds the surfaces of the elastin fiber buried in the bottom colla-
    # gen fiber

    surfaces_dictionary["surface1"] = gmsh.model.geo.addPlaneSurface([
    loops_dictionary["loop1"]])

    for i in range(2,14):

        surfaces_dictionary["surface"+str(i)] = gmsh.model.geo.addSurfaceFilling(
        [loops_dictionary["loop"+str(i)]])

    surfaces_dictionary["surface14"] = gmsh.model.geo.addPlaneSurface([
    loops_dictionary["loop14"]])

    ####################################################################
    #                             Volumes                              #
    ####################################################################

    # Adds the volume of the elastin fiber

    assembly_volume = gmsh.model.geo.addSurfaceLoop([
    surfaces_dictionary["surface1"], surfaces_dictionary["surface2"], 
    surfaces_dictionary["surface3"], surfaces_dictionary["surface4"], 
    surfaces_dictionary["surface5"], surfaces_dictionary["surface6"],
    surfaces_dictionary["surface7"], surfaces_dictionary["surface8"],
    surfaces_dictionary["surface9"], surfaces_dictionary["surface10"],
    surfaces_dictionary["surface11"], surfaces_dictionary["surface12"],
    surfaces_dictionary["surface13"], surfaces_dictionary["surface14"]])

    elastin_volume = gmsh.model.geo.addVolume([assembly_volume])

    gmsh.model.geo.synchronize()

    return (lines_dictionary, loops_dictionary, surfaces_dictionary,
    elastin_volume)

# Defines a function to add the four points at the extremes of an inplane
# ellipse

def add_edgePointsEllipse(points_matrix, x0, y0, z0, a, b):

    # Adds the center

    points_matrix[0].append(x0)

    points_matrix[1].append(y0)

    points_matrix[2].append(z0)

    # Adds the first point

    points_matrix[0].append(x0-(0.5*a))

    points_matrix[1].append(y0)

    points_matrix[2].append(z0)

    # Adds the second point

    points_matrix[0].append(x0)

    points_matrix[1].append(y0-(0.5*b))

    points_matrix[2].append(z0)

    # Adds the third point

    points_matrix[0].append(x0+(0.5*a))

    points_matrix[1].append(y0)

    points_matrix[2].append(z0)

    # Adds the fourth point

    points_matrix[0].append(x0)

    points_matrix[1].append(y0+(0.5*b))

    points_matrix[2].append(z0)

    return points_matrix

########################################################################
#                               Testing                                #
########################################################################

def test_strut():

    # Defines the characteristic length of the mesh

    lc = 0.5

    # Defines parameters of the arterial microstructure

    fiber_diameter = 1.0

    polar_angle = (10/180)*np.pi 

    azimuthal_angle = (10.0/180)*np.pi 

    collagen_thickness = 1.0

    intralamellar_thickness = 5.0

    x_centroid = 10.1

    y_centroid = -5.4

    z_centroid = 25.0

    # Initializes the gmsh object

    gmsh.initialize()

    elastin_fiber([x_centroid, y_centroid, z_centroid], fiber_diameter, 
    polar_angle, azimuthal_angle, collagen_thickness, 
    intralamellar_thickness, lc)

    ####################################################################
    #                         Post-processing                          #
    ####################################################################

    # Deletes the duplicated entities

    gmsh.model.geo.removeAllDuplicates()

    gmsh.model.geo.synchronize()

    # Generates a 3D mesh

    gmsh.model.mesh.generate(3)

    gmsh.fltk.run()

    gmsh.finalize()

#test_strut()