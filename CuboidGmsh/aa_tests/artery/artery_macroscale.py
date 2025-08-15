# Fist test with artery mesh

import numpy as np

import CuboidGmsh.source.solids.cuboid_cylinders as cuboid

import CuboidGmsh.source.tool_box.meshing_tools as tools

# Sets the inner arterial radius and the thickness of each layer

inner_radius = 10.0

intima_thickness = 0.2

media_thickness = 1.0

adventia_thickness = 0.7

# Sets a flag for a mesh of hexahedrons or not

hexahedron_mesh = False

# GAG transfinite discretization

gag_transfiniteCircumferential = 7

gag_transfiniteRadial = 3

gag_transfiniteAxial = 4

# Media sublayer radial discretization

media_transfiniteRadial = 2

# Layers transfinite discretization

layer_transfiniteCircumferential = 20

media_transfiniteRadial = 2

intima_transfiniteRadial = 2

adventia_transfiniteRadial = 4

layer_transfiniteAxial = 15

# Set the bias

layer_biasCircumferential = -1.1

layer_biasAxial = -1.2

# Sets the axial length of the artery section

axial_length = 15.0

# Sets the thickness, polar angle, and length of the GAG pool

GAG_thickness = 0.5

GAG_polarAngle = (10/180)*np.pi 

GAG_length = 0.7

# Sets the axial vector of the mesh

axis_vector = [1.0, 0.0, 0.0]

########################################################################
#                       Boundary surfaces setting                      #
########################################################################

# Sets the finder for the boundary surfaces

# YZ plane at x = 0

def back_expression(x, y, z):

    return x

# YZ plane at x = length

def front_expression(x, y, z):

    return x-axial_length

# XZ plane at y = 0

def right_expression(x, y, z):

    return y 

# XY plane at z = 0

def bottom_expression(x, y, z):

    return z 

# Internal surface of the artery wall

def internal_expression(x, y, z):

    return (z**2)+(y**2)-((inner_radius+intima_thickness)**2)

# Sets a list of expressions to find the surfaces at the boundaries

surface_regionsExpressions = [back_expression, front_expression, 
right_expression, bottom_expression, internal_expression]

# Sets the names of the surface regions

surface_regionsNames = ['back', 'front', 'right', 'bottom', 'internal '+
'wall']

########################################################################
#                        Volume regions setting                        #
########################################################################

# Sets an expression to find the GAG

def gag_expression(x, y, z, tolerance=1E-4):

    # Verifies the inner radius

    if ((y**2)+(z**2))<(((inner_radius+intima_thickness+(0.5*(
    media_thickness-GAG_thickness)))**2)-tolerance):

        return False 
    
    # Verifies the outer radius

    if ((y**2)+(z**2))>(((inner_radius+intima_thickness+media_thickness-
    (0.5*(media_thickness-GAG_thickness)))**2)+tolerance):

        return False 
    
    # Verifies the axial length

    if x>axial_length+tolerance:

        return False 
    
    if x<(axial_length-GAG_length-tolerance):

        return False 
    
    # Verifies the polar angle in the plane YZ

    polar_angle = 0.0

    if abs(y)<tolerance:

        if z>0.0:
        
            polar_angle = 0.5*np.pi

        else:
        
            polar_angle = 0.5*np.pi

    else:

        polar_angle = np.arctan(z/y)

    if (polar_angle>(0.5*np.pi+tolerance) or polar_angle<((0.5*np.pi)-
    GAG_polarAngle-tolerance)):

        return False 
    
    # If it passed over all the tests before, it lies in the GAG region

    return True

# Sets an expression to find the media layer

def media_expression(x, y, z, tolerance=1E-4):

    # Verifies the inner radius

    if ((y**2)+(z**2))<(((inner_radius+intima_thickness)**2)-tolerance):

        return False 
    
    # Verifies the outer radius

    if ((y**2)+(z**2))>(((inner_radius+intima_thickness+media_thickness
    )**2)+tolerance):

        return False 
    
    # Verifies the polar angle in the plane YZ

    polar_angle = 0.0

    if abs(y)<tolerance:

        if z>0.0:
        
            polar_angle = 0.5*np.pi

        else:
        
            polar_angle = 0.5*np.pi

    else:

        polar_angle = np.arctan(z/y)

    if ((polar_angle<(0.5*np.pi-tolerance) and polar_angle>((0.5*np.pi)+
    GAG_polarAngle-tolerance)) and (x>(axial_length-GAG_length+
    tolerance))):

        return False 
    
    # If it passed over all the tests before, it lies in the GAG region

    return True

# Sets an expression to find the intima layer

def intima_expression(x, y, z, tolerance=1E-4):

    # Verifies the inner radius

    if ((y**2)+(z**2))<((inner_radius**2)-tolerance):

        return False 
    
    # Verifies the outer radius

    if ((y**2)+(z**2))>(((inner_radius+intima_thickness)**2)+tolerance):

        return False 
    
    # If it passed over all the tests before, it lies in the GAG region

    return True

# Sets an expression to find the adventitia layer

def adventitia_expression(x, y, z, tolerance=1E-4):

    # Verifies the inner radius

    if ((y**2)+(z**2))<(((inner_radius+intima_thickness+media_thickness
    )**2)-tolerance):

        return False 
    
    # Verifies the outer radius

    if ((y**2)+(z**2))>(((inner_radius+intima_thickness+media_thickness+
    adventia_thickness)**2)+tolerance):

        return False 
    
    # If it passed over all the tests before, it lies in the GAG region

    return True

# Sets the list of volume expressions

volume_regionsExpressions = [gag_expression, media_expression, 
intima_expression, adventitia_expression]

# Sets the list of volume physical groups' names

volume_regionsNames = ['GAG', 'Media', 'Intima', 'Adventitia']

########################################################################
#                         GMSH initialization                          #
########################################################################

# Initializes the gmsh instance and constructs the data list

geometric_data = tools.gmsh_initialization(surface_regionsExpressions=
surface_regionsExpressions, volume_regionsExpressions=
volume_regionsExpressions, surface_regionsNames=surface_regionsNames,
volume_regionsNames=volume_regionsNames, hexahedron_mesh=hexahedron_mesh)

########################################################################
#                             Media layer                              #
########################################################################

# Creates the GAG

base_point = [axial_length-GAG_length*0.5, 0.0, inner_radius+
intima_thickness+(0.5*media_thickness)]

r_inner = inner_radius+intima_thickness+(0.5*(media_thickness-
GAG_thickness))

r_outer = inner_radius+intima_thickness+(0.5*(media_thickness+
GAG_thickness))

transfinite_directions = [gag_transfiniteAxial, gag_transfiniteRadial,
gag_transfiniteCircumferential]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, -GAG_polarAngle, shape_spin=0.5*
np.pi, transfinite_directions=transfinite_directions, geometric_data=
geometric_data)

# Adds the media layer

# Media layer - GAG sub layer

r_inner = inner_radius+intima_thickness+(0.5*(media_thickness-
GAG_thickness))

r_outer = inner_radius+intima_thickness+(0.5*(media_thickness+
GAG_thickness))

base_point = [0.5*(axial_length-GAG_length), 0.0, inner_radius+
intima_thickness+(0.5*media_thickness)]

transfinite_directionsMedia = [layer_transfiniteAxial, 
media_transfiniteRadial, layer_transfiniteCircumferential]

bias_directions = {"x": layer_biasAxial, "z": 
layer_biasCircumferential}

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, -GAG_polarAngle, 
shape_spin=0.5*np.pi, transfinite_directions=
transfinite_directionsMedia, bias_directions=bias_directions, 
geometric_data=geometric_data)

base_point = [0.5*(axial_length-GAG_length), inner_radius+
intima_thickness+(0.5*media_thickness), 0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, (0.5*np.pi)-
GAG_polarAngle, transfinite_directions=transfinite_directionsMedia, 
bias_directions=bias_directions, geometric_data=geometric_data)

base_point = [axial_length-(0.5*GAG_length), inner_radius+
intima_thickness+(0.5*media_thickness), 0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, (0.5*np.pi)-GAG_polarAngle, 
transfinite_directions=transfinite_directionsMedia, bias_directions=
bias_directions, geometric_data=geometric_data)

# Media layer - sublayer over the GAG sublayer

transfinite_directionsMedia = [layer_transfiniteAxial, 
media_transfiniteRadial, layer_transfiniteCircumferential]

r_inner = inner_radius+intima_thickness+(0.5*(media_thickness+
GAG_thickness))

r_outer = inner_radius+intima_thickness+media_thickness

base_point = [axial_length-GAG_length*0.5, 0.0, inner_radius+
intima_thickness+(0.75*(media_thickness-GAG_thickness))+GAG_thickness]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, -GAG_polarAngle, shape_spin=0.5*
np.pi, transfinite_directions=transfinite_directionsMedia, 
geometric_data=geometric_data)

base_point = [0.5*(axial_length-GAG_length), 0.0, inner_radius+
intima_thickness+(0.75*(media_thickness-GAG_thickness))+GAG_thickness]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, -GAG_polarAngle, 
shape_spin=0.5*np.pi, transfinite_directions=
transfinite_directionsMedia, bias_directions=bias_directions, 
geometric_data=geometric_data)

base_point = [0.5*(axial_length-GAG_length), inner_radius+
intima_thickness+(0.75*(media_thickness-GAG_thickness))+GAG_thickness, 
0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, (0.5*np.pi)-
GAG_polarAngle, transfinite_directions=transfinite_directionsMedia, 
bias_directions=bias_directions, geometric_data=geometric_data)

base_point = [axial_length-(0.5*GAG_length), inner_radius+
intima_thickness+(0.75*(media_thickness-GAG_thickness))+GAG_thickness, 
0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, (0.5*np.pi)-GAG_polarAngle, 
transfinite_directions=transfinite_directionsMedia, bias_directions=
bias_directions, geometric_data=geometric_data)

# Media layer - sublayer under the GAG sublayer

r_inner = inner_radius+intima_thickness

r_outer = inner_radius+intima_thickness+(0.5*(media_thickness
-GAG_thickness))

base_point = [axial_length-GAG_length*0.5, 0.0, inner_radius+
intima_thickness+(0.25*(media_thickness-GAG_thickness))]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, -GAG_polarAngle, shape_spin=0.5*
np.pi, transfinite_directions=transfinite_directionsMedia,geometric_data
=geometric_data)

base_point = [0.5*(axial_length-GAG_length), 0.0, inner_radius+
intima_thickness+(0.25*(media_thickness-GAG_thickness))]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, -GAG_polarAngle, 
shape_spin=0.5*np.pi, transfinite_directions=transfinite_directionsMedia, 
bias_directions=bias_directions, geometric_data=geometric_data)

base_point = [0.5*(axial_length-GAG_length), inner_radius+
intima_thickness+(0.25*(media_thickness-GAG_thickness)), 0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, (0.5*np.pi)-
GAG_polarAngle, transfinite_directions=transfinite_directionsMedia, 
bias_directions=bias_directions, geometric_data=geometric_data)

base_point = [axial_length-(0.5*GAG_length), inner_radius+
intima_thickness+(0.25*(media_thickness-GAG_thickness)), 0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, (0.5*np.pi)-GAG_polarAngle, 
transfinite_directions=transfinite_directionsMedia, bias_directions=
bias_directions, geometric_data=geometric_data)

########################################################################
#                             Intima layer                             #
########################################################################

transfinite_directionsIntima = [layer_transfiniteAxial, 
intima_transfiniteRadial, layer_transfiniteCircumferential]

base_point = [axial_length-GAG_length*0.5, 0.0, inner_radius+(0.5*
intima_thickness)]

r_inner = inner_radius*1.0

r_outer = inner_radius+intima_thickness

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, -GAG_polarAngle, shape_spin=0.5*
np.pi, transfinite_directions=transfinite_directionsIntima,
geometric_data=geometric_data)

base_point = [0.5*(axial_length-GAG_length), 0.0, inner_radius+(0.5*
intima_thickness)]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, -GAG_polarAngle, 
shape_spin=0.5*np.pi, transfinite_directions=
transfinite_directionsIntima, bias_directions=bias_directions, 
geometric_data=geometric_data)

base_point = [0.5*(axial_length-GAG_length), inner_radius+(0.5*
intima_thickness), 0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, (0.5*np.pi)-
GAG_polarAngle, transfinite_directions=transfinite_directionsIntima, 
bias_directions=bias_directions, geometric_data=geometric_data)

base_point = [axial_length-(0.5*GAG_length), inner_radius+(0.5*
intima_thickness), 0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, (0.5*np.pi)-GAG_polarAngle, 
transfinite_directions=transfinite_directionsIntima, bias_directions=
bias_directions, geometric_data=geometric_data)

########################################################################
#                           Adventitia layer                           #
########################################################################

transfinite_directionsAdventia = [layer_transfiniteAxial, 
adventia_transfiniteRadial, layer_transfiniteCircumferential]

base_point = [axial_length-GAG_length*0.5, 0.0, inner_radius+
intima_thickness+media_thickness+(0.5*adventia_thickness)]

r_inner = inner_radius+intima_thickness+media_thickness

r_outer = (inner_radius+intima_thickness+media_thickness+
adventia_thickness)

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, -GAG_polarAngle, shape_spin=0.5*
np.pi, transfinite_directions=transfinite_directionsAdventia,
geometric_data=geometric_data)

base_point = [0.5*(axial_length-GAG_length), 0.0, inner_radius+
intima_thickness+media_thickness+(0.5*adventia_thickness)]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, -GAG_polarAngle, 
shape_spin=0.5*np.pi, transfinite_directions=
transfinite_directionsAdventia, bias_directions=bias_directions, 
geometric_data=geometric_data)

base_point = [0.5*(axial_length-GAG_length), inner_radius+
intima_thickness+media_thickness+(0.5*adventia_thickness), 0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
axial_length-GAG_length, axis_vector, base_point, (0.5*np.pi)-
GAG_polarAngle, transfinite_directions=transfinite_directionsAdventia, 
bias_directions=bias_directions, geometric_data=geometric_data,)

base_point = [axial_length-(0.5*GAG_length), inner_radius+
intima_thickness+media_thickness+(0.5*adventia_thickness), 0.0]

geometric_data = cuboid.sector_hollowCylinder(r_inner, r_outer, 
GAG_length, axis_vector, base_point, (0.5*np.pi)-GAG_polarAngle, 
transfinite_directions=transfinite_directionsAdventia, bias_directions=
bias_directions, geometric_data=geometric_data)

########################################################################
#                          GMSH finalization                           #
########################################################################

# Finalizes the procedure and generates the mesh

tools.gmsh_finalize(geometric_data=geometric_data, hexahedron_mesh=
hexahedron_mesh)