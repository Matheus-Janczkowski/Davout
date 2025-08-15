import numpy as np

import CuboidGmsh.source.tool_box.region_finder as region_finder

import CuboidGmsh.tests.RVEs.transfinite.RVE_artery_transfinite as RVE_transfinite

import CuboidGmsh.tests.RVEs.non_transfinite.RVE_artery as RVE_nonTransfinite

# Defines a function to generate parameters for the periodic mesh

def generate_parameters(flag_transfinite=1):

    ####################################################################
    ####################################################################
    ##                        User's settings                         ##
    ####################################################################
    ####################################################################

    ####################################################################
    #                           Mesh sizing                            #
    ####################################################################

    # Defines the characteristic length

    lc = 0.5

    # Defines the base number of divisions in transfinite

    n_transfinite = 3

    ####################################################################
    #                        Geometry parameters                       #
    ####################################################################

    # Sets a list of names of each material phase

    material_phasesNames = ['elastin strut', 'muscular filing', ('coll'+
    'agen layer'), 'lamella']

    # Defines the number of parallelepipeds along the x, the y and the z 
    # directions

    n_boxesX = 1

    n_boxesY = 1

    n_boxesZ = 1
    
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

    ####################################################################
    #                   Volume region finder settings                  #
    ####################################################################

    # Defines a function to identify a group of RVEs within a region en-
    # closed by planes

    planes_normals = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]

    planes_biases = [2*RVE_lengthX, 2*RVE_lengthY, -RVE_lengthX,
    -RVE_lengthY, -RVE_lengthZ, n_boxesZ*RVE_lengthZ]

    def group_RVEs1(x, y, z):

        return region_finder.plane_enclosure(x, y, z, planes_normals,
        planes_biases)
    
    # Defines a vector of functions to identify each one a region of the
    # mesh

    volume_regionIdentifiers = []#[group_RVEs1]

    # Defines the names of the regions

    volume_regionsNames = []#['RVE']

    ####################################################################
    #                  Surface region finder settings                  #
    ####################################################################

    # Defines a set of functions to identify if the created surfaces lie
    # within each physical surface. Hence, defines a function to each
    # physical surface. These functions get a set of points and verify
    # whether they belong to a region or not

    def surface_bottom(points_set):

        return region_finder.plane_boundary(points_set, [0.0, 0.0, -1.0],
        0.0)
    
    def surface_ahead(points_set):

        return region_finder.plane_boundary(points_set, [1.0, 0.0, 0.0],
        n_boxesX*RVE_lengthX)
    
    def surface_larboard(points_set):

        return region_finder.plane_boundary(points_set, [0.0, 1.0, 0.0],
        n_boxesY*RVE_lengthY)
    
    def surface_aft(points_set):

        return region_finder.plane_boundary(points_set, [-1.0, 0.0, 0.0],
        0.0)
    
    def surface_starboard(points_set):

        return region_finder.plane_boundary(points_set, [0.0, -1.0, 0.0],
        0.0)
    
    def surface_top(points_set):

        return region_finder.plane_boundary(points_set, [0.0, 0.0, 1.0],
        n_boxesZ*RVE_lengthZ)
    
    surface_regionIdentifiers = [surface_bottom, surface_ahead,
    surface_larboard, surface_aft, surface_starboard, surface_top]

    # Sets the names of the surface regions

    surface_regionsNames = ['bottom', 'ahead', 'larboard', 'aft', 'sta'+
    'rboard', 'top']

    # Sets the type of element that is to be captured into the fenics
    # mesh. Both for volume and for surface

    volume_elementType = 'tetra'

    surface_elementType = 'triangle'

    ####################################################################
    #                        Parameters setting                        #
    ####################################################################

    # Sets the method to generate RVEs

    RVE_method = 0

    if flag_transfinite==1:

        raise Exception("\nTransfinite has not yet been defined for th"+
        "e artery RVE")

        RVE_method = RVE_transfinite.RVE_centralFiberWholeInnerSquare

    elif flag_transfinite==2:

        raise Exception("\nTransfinite has not yet been defined for th"+
        "e artery RVE")

        RVE_method = RVE_transfinite.RVE_centralFiberInnerSquare4Quadrants

    elif flag_transfinite==0:

        RVE_method = RVE_nonTransfinite.RVE_elastinMicrostructure

    ####################################################################
    #                      Parameters verification                     #
    ####################################################################

    # Assembles the vector of parameters for this method of RVE genera-
    # tion

    parameters_method = [fiber_diameter, polar_angle, azimuthal_angle,
    collagen_thickness, lamella_thickness, clearance_strutsEdges, 
    n_strutsX, n_strutsY]

    # Checks the parameters given by the user using a function from the
    # RVE generation file

    if flag_transfinite==1 or flag_transfinite==2:

        raise Exception("\nTransfinite has not yet been defined for th"+
        "e artery RVE")

    elif flag_transfinite==0:

        RVE_nonTransfinite.verify_parameters(parameters_method, 
        RVE_lengthX, RVE_lengthY, RVE_lengthZ)

    # Returns the parameters

    return (parameters_method, RVE_method, RVE_lengthX, RVE_lengthY,
    RVE_lengthZ, n_boxesX, n_boxesY, n_boxesZ, lc, n_transfinite, 
    volume_elementType, surface_elementType, material_phasesNames,
    volume_regionIdentifiers, volume_regionsNames, 
    surface_regionIdentifiers, surface_regionsNames)