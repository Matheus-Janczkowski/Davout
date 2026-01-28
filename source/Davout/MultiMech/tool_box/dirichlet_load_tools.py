# Routine to store functions to apply Dirichlet boundary conditions as
# loads

from dolfin import *

import ufl_legacy as ufl

import numpy as np

from copy import copy

from ..tool_box import numerical_tools

from ..tool_box import mesh_handling_tools as mesh_tools

# Defines a function to construct a Dirichlet boundary condition that a
# surface is rotated and translated

def SurfaceTranslationAndRotation(mesh_dataClass, boundary_physicalGroups, 
in_planeSpin=0.0, normal_toPlaneSpin=0.0, translation=[0.0, 0.0, 0.0], 
in_planeSpinDirection=[0.0, 0.0, 0.0], center_point=None, t=0.0, t_final=
1.0, parametric_load_curve="linear", rotation_x=None, rotation_y=None, 
rotation_z=None):
    
    original_translation = copy(translation)

    # Creates the time constants

    time_constant = Constant(t)

    maximum_time = Constant(t_final)
    
    # Verifies if the parametric load curve is a string and, then, uses
    # it to convert to an actual function

    if isinstance(parametric_load_curve, str):

        parametric_load_curve = numerical_tools.generate_loadingParametricCurves(
        parametric_load_curve)
    
    # Verifies if the physical group is a string and converts it

    boundary_physicalGroups, original_physicalGroup = mesh_tools.convert_physicalGroup(
    boundary_physicalGroups, mesh_dataClass, "boundary")

    # Verifies if the center point is None, then, proceeds to evaluate
    # the centroid of the region

    if center_point is None:

        center_point, area_inverse = mesh_tools.evaluate_centroidSurface(
        boundary_physicalGroups, mesh_dataClass)

    # Otherwise, verifies if it is a list

    elif not (isinstance(center_point, list)):

        raise TypeError("The center point of application of the follow"+
        "er torsion must be a list. The following was given though: "+
        str(center_point))

    # Checks whether the translation direction is a list
    
    if not isinstance(translation, list):

        raise TypeError("The 'translation' in SurfaceTranslationAndRot"+
        "ation is not a list. It must be a list with 3 elements, i.e. "+
        "a vector")   

    elif len(translation)!=3:

        raise ValueError("The 'translation' in SurfaceTranslationAndRo"+
        "tation is a list that has "+str(len(translation))+" elements."+
        " This list must have 3 elements, i.e. a vector in 3D space")
    
    else:

        # Creates the constant

        translation = Constant(translation)

    # Initializes the average normal vector

    average_normal = None

    # If the three rotation are to be used

    if (rotation_x is not None) or (rotation_y is not None) or (
    rotation_z is not None):
        
        # Verifies if all of them are not None

        if rotation_x is None:

            raise ValueError("'rotation_x' is None in 'SurfaceTranslat"+
            "ionAndRotation', but 'rotation_y' and 'rotation_z' aren't"+
            ". All rotations must be given if any is given.")

        elif rotation_y is None:

            raise ValueError("'rotation_y' is None in 'SurfaceTranslat"+
            "ionAndRotation', but 'rotation_x' and 'rotation_z' aren't"+
            ". All rotations must be given if any is given.")

        elif rotation_z is None:

            raise ValueError("'rotation_z' is None in 'SurfaceTranslat"+
            "ionAndRotation', but 'rotation_y' and 'rotation_x' aren't"+
            ". All rotations must be given if any is given.")
        
        # Populates the average vector with the rotation components. 
        # But converts first to radians

        average_normal = [numerical_tools.degrees_to_radians(rotation_x), 
        numerical_tools.degrees_to_radians(rotation_y), 
        numerical_tools.degrees_to_radians(rotation_z)]

        # Makes sure the in plane spin is null and that the normal to 
        # plane spin is equal to 1 to make the multiplication inside the
        # class consistent

        in_planeSpin = 0.0

        normal_toPlaneSpin = 1.0

        # Annuntiates the corrections

        print("\nTranslated and rotated surface Dirichlet BC:\nApplied"+
        " to the surface '"+str(original_physicalGroup)+"' with a tran"+
        "slation of "+str(original_translation)+"\nA rotation vector o"+
        "phi = "+str(average_normal)+"\n")

    # If the in plane spin direction and the normal to plane spin are to
    # be used

    else:

        (in_planeSpinDirection, in_planeSpin, normal_toPlaneSpin, 
        average_normal) = correct_spin_direction_and_get_norm(
        in_planeSpinDirection, in_planeSpin, normal_toPlaneSpin, 
        center_point, mesh_dataClass, boundary_physicalGroups, 
        original_physicalGroup, original_translation)

    # Defines a class to update the time constant and the traction am-
    # plitude

    R00 = Constant(0.0)

    R01 = Constant(0.0)

    R02 = Constant(0.0)

    R10 = Constant(0.0)

    R11 = Constant(0.0)

    R12 = Constant(0.0)

    R20 = Constant(0.0)

    R21 = Constant(0.0)

    R22 = Constant(0.0)

    t0 = Constant(0.0)

    t1 = Constant(0.0)

    t2 = Constant(0.0)

    class TimeUpdate:

        def __init__(self):
            
            self.t = time_constant

            self.R00 = R00

            self.R01 = R01

            self.R02 = R02

            self.R10 = R10

            self.R11 = R11

            self.R12 = R12

            self.R20 = R20

            self.R21 = R21

            self.R22 = R22

            self.t0 = t0

            self.t1 = t1

            self.t2 = t2

        # This class must have the update_load method to be called in 
        # the stepping algorithm

        def update_load(self, t):

            # Updates the time constant

            self.t.assign(Constant(t))

            # Evaluates the load using the parametric curve

            load_value = float(parametric_load_curve(self.t/maximum_time
            ))

            # Updates the translation vector

            new_translation = [load_value*component for component in (
            original_translation)]

            # Updates the vectors

            new_inPlaneVector = [(in_planeSpin*load_value*component
            ) for component in in_planeSpinDirection]

            new_normalVector = [(normal_toPlaneSpin*load_value*component
            ) for component in average_normal]

            # Evaluates the rotation matrices

            rotation_normalAxis = numerical_tools.rotation_tensorEulerRodrigues(
            new_normalVector)

            # Then, the matrix about the inplane direction

            rotation_inPlane = numerical_tools.rotation_tensorEulerRodrigues(
            new_inPlaneVector)

            # Multiplies the two matrices into one and substracts the 
            # identity matrix because what we really want is the displa-
            # cement

            total_matrix = (np.dot(rotation_inPlane, rotation_normalAxis
            )-np.eye(3))

            # Constructs the displacement field and interpolates it

            self.R00.assign(Constant(total_matrix[0,0]))

            self.R01.assign(Constant(total_matrix[0,1]))

            self.R02.assign(Constant(total_matrix[0,2]))

            self.R10.assign(Constant(total_matrix[1,0]))

            self.R11.assign(Constant(total_matrix[1,1]))

            self.R12.assign(Constant(total_matrix[1,2]))

            self.R20.assign(Constant(total_matrix[2,0]))

            self.R21.assign(Constant(total_matrix[2,1]))

            self.R22.assign(Constant(total_matrix[2,2]))

            self.t0.assign(Constant(new_translation[0]))

            self.t1.assign(Constant(new_translation[1]))

            self.t2.assign(Constant(new_translation[2]))

            # Annuntiates the corrections

            print("\nTranslation and rotation of surface Dirichlet BC "+
            "at physical group '"+str(original_physicalGroup)+"':\n"+
            str(load_value*100)+"% of the final boundary condition's v"+
            "alue is applied using the parametric load curve\nRotates "+
            "and evaluates the displacement by using a transformation "+
            "matrix of\n"+str(total_matrix)+"\nand translates by a vec"+
            "tor of "+str(new_translation)+"\n")

    # Instantiates the loading class that was built and returns it as 
    # well as the traction vector object

    time_class = TimeUpdate()

    # Uses Expression instead of UserExpression to avoid bugs

    translated_rotatedDisplacement = Expression(("(R00*(x[0]-xc0))+(R0"+
    "1*(x[1]-xc1))+(R02*(x[2]-xc2))+t0", "(R10*(x[0]-xc0))+(R11*(x[1]-"+
    "xc1))+(R12*(x[2]-xc2))+t1", "(R20*(x[0]-xc0))+(R21*(x[1]-xc1))+(R"+
    "22*(x[2]-xc2))+t2"), R00=R00, R01=R01, R02=R02, R10=R10, R11=R11, 
    R12=R12, R20=R20, R21=R21, R22=R22, t0=t0, t1=t1, t2=t2, xc0=
    center_point[0], xc1=center_point[1], xc2=center_point[2], degree=1)

    return translated_rotatedDisplacement, time_class

# Defines a function to compute the corrected in plane spin direction 
# and the average normal vector

def correct_spin_direction_and_get_norm(in_planeSpinDirection, 
in_planeSpin, normal_toPlaneSpin, center_point, mesh_dataClass,
boundary_physicalGroups, original_physicalGroup, original_translation):
    
    # If parallelization is chosen, raises an error

    if mesh_dataClass.comm is not None:

        raise NotImplementedError("'SurfaceTranslationAndRotation' is "+
        "selected with parallelization. This is available only if 'rot"+
        "ation_x', 'rotation_y', and 'rotation_z' are selected, instea"+
        "d of in_planeSpin and normal_toPlaneSpin")

    # Checks whether the in plane spin direction is a list
    
    if not isinstance(in_planeSpinDirection, list):

        raise TypeError("The 'in_planeSpinDirection' in SurfaceTransla"+
        "tionAndRotation is not a list. It must be a list with 3 eleme"+
        "nts, i.e. a vector")   

    elif len(in_planeSpinDirection)!=3:

        raise ValueError("The 'in_planeSpinDirection' in SurfaceTransl"+
        "ationAndRotation is a list that has "+str(len(
        in_planeSpinDirection))+" elements. This list must have 3 elem"+
        "ents, i.e. a vector in 3D space") 
    
    elif in_planeSpinDirection!=[0.0, 0.0, 0.0]:

        if in_planeSpin==0.0:

            raise ValueError("The in_planeSpinDirection is different t"+
            "han zero, "+str(in_planeSpinDirection)+", but no in_plane"+
            "Spin angle has been given")
        
        in_planeNorm = 1.0/np.sqrt((in_planeSpinDirection[0]**2)+(
        in_planeSpinDirection[1]**2)+(in_planeSpinDirection[2]**2))

        in_planeSpinDirection = [in_planeNorm*a for a in in_planeSpinDirection]
    
    # Checks whether in_plane spin is a float

    if isinstance(in_planeSpin, int):

        in_planeSpin = (in_planeSpin/180)*np.pi
    
    elif not isinstance(in_planeSpin, float):

        raise TypeError("The 'in_planeSpin' in SurfaceTranslationAndRo"+
        "tation is not a float, "+str(in_planeSpin)+". Provide it in d"+
        "egrees") 
    
    else:

        # Converts from degrees to radians

        in_planeSpin = (in_planeSpin/180)*np.pi
    
    # Checks whether the normal plane spin is a float

    if isinstance(normal_toPlaneSpin, int):

        normal_toPlaneSpin = (normal_toPlaneSpin/180)*np.pi
    
    elif not isinstance(normal_toPlaneSpin, float):

        raise TypeError("The 'normal_toPlaneSpin' in SurfaceTranslatio"+
        "nAndRotation is not a float, "+str(normal_toPlaneSpin)+". Pro"+
        "vide it in degrees")   
    
    else:

        # Converts from degrees to radians

        normal_toPlaneSpin = (normal_toPlaneSpin/180)*np.pi

    # Gets the node closest to the center point

    center_nodeNumber, center_point = mesh_tools.find_nodeClosestToPoint(
    mesh_dataClass, center_point, None, None)

    # Sets the centroid vector as constant

    centroid_point = Constant(center_point)

    # Sets the relative position vector on the surface

    relative_position = mesh_dataClass.x-centroid_point
    
    # Looks for the nodes coordinates that are on elements adjacent to 
    # the center point

    adjacent_centerNodes, adjacent_centerNodesCoord = (
    mesh_tools.find_nodesOnSurfaceAroundNode(mesh_dataClass, 
    boundary_physicalGroups, node_number=center_nodeNumber))

    # Evaluates the distance between the center point and the adjacent 
    # nodes

    adjacent_distances = []

    for node_coordinate in adjacent_centerNodesCoord:

        adjacent_distances.append(np.sqrt(((node_coordinate[0]-
        center_point[0])**2)+((node_coordinate[1]-center_point[1])**2)+
        ((node_coordinate[2]-center_point[2])**2)))

    # Finds the greatest distance between the center node and its imme-
    # diately close nodes

    adjacent_distance = max(adjacent_distances)

    # Defines a scalar parameter to integrate the normal only over the 
    # elements that are adjacent to the center point, where the torsion
    # axis passes through. It throws zero when the point is out of these
    # elements and 1 otherwise

    adjacency_parameter = ufl.conditional(gt(dot(relative_position,
    relative_position), adjacent_distance**2), 0.0, 1.0)

    # Gets the normal vector in the referential configuration

    normal_vector = mesh_dataClass.n

    # Evaluates the average normal vector of the surface integrating 
    # over the elements adjacent to the node where the center of the 
    # torsion moment is applied

    average_normal = [float(assemble(adjacency_parameter*normal_vector[0
    ]*mesh_dataClass.ds(boundary_physicalGroups))), float(assemble(
    adjacency_parameter*normal_vector[1]*mesh_dataClass.ds(
    boundary_physicalGroups))), float(assemble(adjacency_parameter*
    normal_vector[2]*mesh_dataClass.ds(boundary_physicalGroups)))]

    # Normalizes this average vector

    norm_vector = (1.0/np.sqrt((average_normal[0]**2)+(average_normal[1
    ]**2)+(average_normal[2]**2)))

    average_normal = [norm_vector*average_normal[0], (norm_vector*
    average_normal[1]), norm_vector*average_normal[2]]

    # Evaluates the inner product of the average normal vector with the
    # in-plane spin direction

    dot_inplaceSpinAverageNormal = ((average_normal[0]*
    in_planeSpinDirection[0])+(average_normal[1]*in_planeSpinDirection[1
    ])+(average_normal[2]*in_planeSpinDirection[2]))

    # Verifies if the in-plane spin direction is out of the surface os-
    # cular plane

    if dot_inplaceSpinAverageNormal>1E-4:

        print("WARNING: the in_planeSpinDirection provided to the Surf"+
        "aceTranslationAndRotation is not contained in the plane of th"+
        "e surface '"+str(original_physicalGroup)+"'. It was projected"+
        " onto the surface though\n")

        # Projects the in-plane spin direction and normalizes the outcome

        projected_inplaneSpinDirection = [in_planeSpinDirection[0]-(
        average_normal[0]*dot_inplaceSpinAverageNormal), 
        in_planeSpinDirection[1]-(average_normal[1]*
        dot_inplaceSpinAverageNormal), in_planeSpinDirection[2]-(
        average_normal[2]*dot_inplaceSpinAverageNormal)]

        projected_norm = np.sqrt((projected_inplaneSpinDirection[0]**2)+(
        projected_inplaneSpinDirection[1]**2)+(
        projected_inplaneSpinDirection[2]**2))

        if projected_norm>1E-5:

            projected_norm = 1/projected_norm

            in_planeSpinDirection = [(projected_norm*
            projected_inplaneSpinDirection[0]), (projected_norm*
            projected_inplaneSpinDirection[1]), (projected_norm*
            projected_inplaneSpinDirection[2])]

        else:

            raise ValueError("The in_planeSpinDirection given to the S"+
            "urfaceTranslationAndRotation is perpendicular to the surf"+
            "ace '"+str(original_physicalGroup)+"' plane. Thus, it can"+
            "not be used to rotate this surface about a vector contain"+
            "ed in this plane")

    # Annuntiates the corrections

    print("\nTranslated and rotated surface Dirichlet BC:\nApplied to "+
    "the surface '"+str(original_physicalGroup)+"' with a translation "+
    "of "+str(original_translation)+"\nA in-plane spin of "+str(
    (in_planeSpin/np.pi)*180)+" degrees about a direction of "+str(
    in_planeSpinDirection)+"\nA normal-to-plane spin of "+str(
    normal_toPlaneSpin)+" degrees\n")

    return (in_planeSpinDirection, in_planeSpin, normal_toPlaneSpin, 
    average_normal)