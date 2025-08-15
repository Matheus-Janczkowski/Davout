import numpy as np

import CuboidGmsh.source.tool_box.region_finder as region_finder

import CuboidGmsh.tests.RVEs.transfinite.RVE_matrix_with_fiber_transfinite as RVE_transfinite

import CuboidGmsh.tests.RVEs.non_transfinite.RVE_matrix_with_fiber as RVE_nonTransfinite

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

    lc = 0.1

    # Defines the base number of divisions in transfinite

    n_transfinite = 3

    ####################################################################
    #                        Geometry parameters                       #
    ####################################################################

    # Sets a list of names of each material phase

    material_phasesNames = ['fiber', 'matrix']

    # The geometry is made out of a series of parallelepipeds stacked
    # together

    RVE_lengthX = 1.0#0.003161

    RVE_lengthY = 1.0#0.003161

    RVE_lengthZ = 1.5#0.003161

    # Defines the number of parallelepipeds along the x, the y and the z 
    # directions

    n_boxesX = 1

    n_boxesY = 1

    n_boxesZ = 1
    
    # Defines the parameters of the inner microstructure

    radius = 0.2#0.0005

    # A cylinder mesh to be structured must have a square inside. Hence, 
    # prescribe the side size of this square

    inner_squareSideSize = 2*radius/3 #1.0

    ####################################################################
    #                   Volume region finder settings                  #
    ####################################################################

    # Defines a function to identify a group of RVEs within a region en-
    # closed by planes

    planes_normals = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]

    center_RVEatZdirection = np.ceil(0.5*n_boxesZ)

    planes_biases = [2*RVE_lengthX, 2*RVE_lengthY, -RVE_lengthX,
    -RVE_lengthY, -(center_RVEatZdirection-1)*RVE_lengthZ, (RVE_lengthZ*
    center_RVEatZdirection)]

    def group_RVEs1(x, y, z):

        return region_finder.plane_enclosure(x, y, z, planes_normals,
        planes_biases)
    
    # Defines a vector of functions to identify each one a region of the
    # mesh

    volume_regionIdentifiers = [group_RVEs1]

    # Defines the names of the regions

    volume_regionsNames = ['RVE']

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

        RVE_method = RVE_transfinite.RVE_centralFiberWholeInnerSquare

    elif flag_transfinite==2:

        RVE_method = RVE_transfinite.RVE_centralFiberInnerSquare4Quadrants

    elif flag_transfinite==0:

        RVE_method = RVE_nonTransfinite.RVE_centralFiber

    ####################################################################
    #                      Parameters verification                     #
    ####################################################################

    # Assembles the vector of parameters for this method of RVE genera-
    # tion

    parameters_method = [inner_squareSideSize, radius]

    # Checks the parameters given by the user using a function from the
    # RVE generation file

    if flag_transfinite==1 or flag_transfinite==2:

        RVE_transfinite.verify_parameters(parameters_method, 
        RVE_lengthX, RVE_lengthY, RVE_lengthZ)

    elif flag_transfinite==0:

        RVE_nonTransfinite.verify_parameters(parameters_method, 
        RVE_lengthX, RVE_lengthY, RVE_lengthZ)

    # Returns the parameters

    return (parameters_method, RVE_method, RVE_lengthX, RVE_lengthY,
    RVE_lengthZ, n_boxesX, n_boxesY, n_boxesZ, lc, n_transfinite, 
    volume_elementType, surface_elementType, material_phasesNames,
    volume_regionIdentifiers, volume_regionsNames, 
    surface_regionIdentifiers, surface_regionsNames)