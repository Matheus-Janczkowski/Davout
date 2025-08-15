# Routine to store tools to create loads

from dolfin import *

import ufl_legacy as ufl

import numpy as np

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import tensor_tools

from ..tool_box import numerical_tools

########################################################################
#                          Referential forces                          #
########################################################################

# Defines a function to construct a uniform traction on a surface in the
# referential configuration

def UniformReferentialTraction(amplitude_tractionX, amplitude_tractionY,
amplitude_tractionZ, physical_group, t=0.0, t_final=1.0, 
parametric_load_curve="linear"):
    
    # Verifies if the parametric load curve is a string and, then, uses
    # it to convert to an actual function

    if isinstance(parametric_load_curve, str):

        parametric_load_curve = numerical_tools.generate_loadingParametricCurves(
        parametric_load_curve)

    # Creates the time constants

    time_constant = Constant(t)

    maximum_time = Constant(t_final)

    # Evaluates the traction vector at the referential configuration

    T = Constant([0.0, 0.0, 0.0])

    # Creates the updating class

    class TimeUpdate:

        def __init__(self):
            
            self.t = time_constant

            self.T = T

        def update_load(self, t):

            self.t.assign(Constant(t))

            # Evaluates the load curve

            load_value = float(parametric_load_curve(time_constant/
            maximum_time))

            # Updates the traction vector

            self.T.assign(Constant([load_value*amplitude_tractionX, 
            load_value*amplitude_tractionY, (load_value*
            amplitude_tractionZ)]))

            # Annuntiates the corrections

            print("\nUniform referential traction at physical group '"+
            str(physical_group)+"':\n"+str(load_value*100)+"% of the "+
            "final load is applied using the parametric load curve\n")

    time_update = TimeUpdate()

    # Returns the traction and the time constant

    return T, time_update

# Defines a function to construct a torsion in the referential configu-
# ration

def NormalReferentialTorsion(mesh_dataClass, amplitude_torsion, 
physical_group, center_point=None, influence_radius=None, t=0.0, t_final=
1.0, parametric_load_curve="linear", no_parasiticForcesCorrection=True):
    
    # Verifies if the parametric load curve is a string and, then, uses
    # it to convert to an actual function

    if isinstance(parametric_load_curve, str):

        parametric_load_curve = numerical_tools.generate_loadingParametricCurves(
        parametric_load_curve)
    
    # Verifies if the physical group is a string and converts it

    physical_group, original_physicalGroup = mesh_tools.convert_physicalGroup(
    physical_group, mesh_dataClass, "boundary")

    # Verifies if the center point is None, then, proceeds to evaluate
    # the centroid of the region

    area_inverse = 0.0

    if center_point is None:

        center_point, area_inverse = mesh_tools.evaluate_centroidSurface(
        physical_group, mesh_dataClass)

    # Otherwise, verifies if it is a list

    elif not (isinstance(center_point, list)):

        raise TypeError("The center point of application of the follow"+
        "er torsion must be a list. The following was given though: "+
        str(center_point))

    else:

        area_inverse = (1.0/float(assemble(1.0*mesh_dataClass.ds(
        physical_group))))

    # Gets the node closest to the center point

    center_nodeNumber, center_point = mesh_tools.find_nodeClosestToPoint(
    mesh_dataClass, center_point, None, None)

    # Sets the centroid vector as constant

    centroid_point = Constant(center_point)

    # Sets the relative position vector on the surface

    relative_position = mesh_dataClass.x-centroid_point

    # Sets the localization parameter as a scalar value that is 1 inside
    # a circle of influence of the traction around the center point where 
    # the torsion axis passes through, and 0 everywhere else. If the in-
    # fluence radius is None, this parameter will be 1 everywhere

    localization_parameter = Constant(1.0)

    if not (influence_radius is None):

        # Checks if it is a float

        if not (isinstance(influence_radius, int) or isinstance(
        influence_radius, float)):
            
            raise TypeError("The influence radius of the NormalFollowe"+
            "rTorsion must be a float number, whereas the given value "+
            "is "+str(influence_radius))
        
        localization_parameter = ufl.conditional(gt(dot(
        relative_position, relative_position), influence_radius**2), 0.0, 
        1.0)
    
    # Looks for the nodes coordinates that are on elements adjacent to 
    # the center point

    adjacent_centerNodes, adjacent_centerNodesCoord = (
    mesh_tools.find_nodesOnSurfaceAroundNode(mesh_dataClass, 
    physical_group, node_number=center_nodeNumber))

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
    ]*mesh_dataClass.ds(physical_group))), float(assemble(
    adjacency_parameter*normal_vector[1]*mesh_dataClass.ds(
    physical_group))), float(assemble(adjacency_parameter*normal_vector[
    2]*mesh_dataClass.ds(physical_group)))]

    # Normalizes this average vector

    norm_vector = (1.0/np.sqrt((average_normal[0]**2)+(average_normal[1
    ]**2)+(average_normal[2]**2)))

    average_normalVector = Constant([norm_vector*average_normal[0], (
    norm_vector*average_normal[1]), norm_vector*average_normal[2]])

    # Creates the time constants

    time_constant = Constant(t)

    maximum_time = Constant(t_final)

    # Gets the traction direction as the cross product of the relative
    # position

    traction_direction = cross(normal_vector, relative_position)

    # Gets the norm of this vector. Multiplies by the magnitude of the 
    # relative position vector squared to get a radially increasing 
    # traction profile

    norm_traction = (sqrt(dot(relative_position, relative_position))/
    ufl.conditional(gt(sqrt(dot(traction_direction, traction_direction)
    ), 1E-5), sqrt(dot(traction_direction, traction_direction)), 1.0))

    # Initializes the calculation of the traction amplitude

    traction_amplitude = Constant(amplitude_torsion)

    # Creates the traction vector. Multiplies by the traction amplitude,
    # which is FOR NOW given by the amplitude torsion itself; this value
    # will be later corrected to assure that the traction field is inde-
    # ed producing the desired moment in the normal direction in the re-
    # ferential configuration

    T_unscaled = (traction_amplitude*norm_traction*traction_direction*
    localization_parameter)

    # Creates the traction vector at the principal directions to balance
    # out parasitic forces. This traction is not scaled by the deformed
    # to referential configurations update because the parasitic forces
    # are already integrated in the reference configuration

    T_parasiticBalance = Constant([0.0, 0.0, 0.0])

    # Gets the real moment done by this traction in the direction of the 
    # average normal vector

    real_momentTrial = float(assemble(dot(cross(relative_position, 
    T_unscaled), average_normalVector)*mesh_dataClass.ds(physical_group)
    ))

    # Gets the signal and the absolute value of the moment

    moment_signal = np.sign(real_momentTrial)

    if moment_signal==0.0:

        moment_signal = 1.0

    moment_absoluteValue = np.abs(real_momentTrial)

    # If the moment is too close to zero, gets it as the tolerance

    moment_absoluteValue = max(moment_absoluteValue, 1E-5)

    # Reconstructs the moment

    real_moment = moment_signal*moment_absoluteValue

    # Updates the traction amplitude

    correction_factor = amplitude_torsion/real_moment

    traction_amplitude.assign(traction_amplitude*correction_factor)

    # Gets the parasitic values of traction in each one of the directions 
    # of the space. Parasitic forces are resultant forces that are not 
    # intended and must be balanced

    parasitic_forceX = float(assemble(T_unscaled[0]*mesh_dataClass.ds(
    physical_group)))

    parasitic_forceY = float(assemble(T_unscaled[1]*mesh_dataClass.ds(
    physical_group)))

    parasitic_forceZ = float(assemble(T_unscaled[2]*mesh_dataClass.ds(
    physical_group)))
    
    if not no_parasiticForcesCorrection:

        # Adds this parasitic values to the correction

        T_parasiticBalance.assign(Constant((area_inverse*
        parasitic_forceX, area_inverse*parasitic_forceY, area_inverse*
        parasitic_forceZ)))

    # Annuntiates the corrections

    print("\nNormal referential torsion:\nThe real moment applied to t"+
    "he surface '"+str(original_physicalGroup)+"' by the current tract"+
    "ion\nfield is "+str(real_moment)+".\nThe traction amplitude is co"+
    "rrected to "+str(traction_amplitude.values()[0])+"\nA correction "+
    "factor of "+str(correction_factor)+" was used\nThe current tracti"+
    "on field induces a parasitic force of\nx direction: "+str(
    parasitic_forceX)+"\ny direction: "+str(parasitic_forceY)+"\nz dir"+
    "ection: "+str(parasitic_forceZ)+"\n"+str(float(
    parametric_load_curve(time_constant/maximum_time)*100))+"% "+
    "of the final load is applied using the parametric load cu"+
    "rve\n")
    
    if not no_parasiticForcesCorrection:
        
        print("All of the parasitic forces are balanced out by opposit"+
        "e uniform tractions\n")

    # Scales the traction vector to give the actual moment

    T = (parametric_load_curve(time_constant/maximum_time)*(T_unscaled-
    T_parasiticBalance))

    # Defines a class to update the time constant and the traction am-
    # plitude

    class TimeUpdate:

        def __init__(self):
            
            self.t = time_constant

        # This class must have the update_load method to be called in 
        # the stepping algorithm

        def update_load(self, t):

            # Updates the time constant

            self.t.assign(Constant(t))

            # Annuntiates the corrections

            print("\nNormal referential torsion at physical group '"+
            str(original_physicalGroup)+"':\n"+str(float(
            parametric_load_curve(self.t/maximum_time)*100))+"% of the"+
            " final load is applied using the parametric load curve\n")

    # Instantiates the loading class that was built and returns it as 
    # well as the traction vector object

    time_class = TimeUpdate()

    return T, time_class

########################################################################
#                           Follower forces                            #
########################################################################

# Defines a function to construct a normal uniform traction in a surfa-
# ce, this means the traction vector is always perpendicular to the sur-
# face. The user can set how the amplitude of the function is interpola-
# ted between the initial and final time points; the default value is a
# linear curve

def NormalUniformFollowerTraction(field, field_numerical, mesh_dataClass,
amplitude_traction, t=0.0, t_final=1.0, parametric_load_curve="linear"):
    
    # Verifies if the parametric load curve is a string and, then, uses
    # it to convert to an actual function

    if isinstance(parametric_load_curve, str):

        parametric_load_curve = numerical_tools.generate_loadingParametricCurves(
        parametric_load_curve)

    # Creates the time constants

    time_constant = Constant(t)

    maximum_time = Constant(t_final)

    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(field)+I

    J = det(F)

    # Evaluates the traction vector at the referential configuration

    T = (parametric_load_curve(time_constant/maximum_time)*
    amplitude_traction*J*((inv(F)).T)*mesh_dataClass.n)

    # Returns the traction and the time constant

    return T, time_constant

# Defines a function to construct a torsion on a surface. The center 
# point, where the axis of torsion intersect the surface, is optional,
# if not given, the centroid of the surface will be used automatically.
# The torsion is applied as a shearing traction inside a circular re-
# gion, whose center is the center point; the radius can be given or e-
# valuated by the minimum distance to the edge if not supplied

def NormalFollowerTorsion(field, field_numerical, mesh_dataClass, 
amplitude_torsion, physical_group, center_point=None, influence_radius=
None, t=0.0, t_final=1.0, parametric_load_curve="linear", 
no_parasiticForcesCorrection=False):
    
    # Verifies if the parametric load curve is a string and, then, uses
    # it to convert to an actual function

    if isinstance(parametric_load_curve, str):

        parametric_load_curve = numerical_tools.generate_loadingParametricCurves(
        parametric_load_curve)
    
    # Verifies if the physical group is a string and converts it

    physical_group, original_physicalGroup = mesh_tools.convert_physicalGroup(
    physical_group, mesh_dataClass, "boundary")

    # Verifies if the center point is None, then, proceeds to evaluate
    # the centroid of the region

    area_inverse = 0.0

    if center_point is None:

        center_point, area_inverse = mesh_tools.evaluate_centroidSurface(
        physical_group, mesh_dataClass)

    # Otherwise, verifies if it is a list

    elif not (isinstance(center_point, list)):

        raise TypeError("The center point of application of the follow"+
        "er torsion must be a list. The following was given though: "+
        str(center_point))

    else:

        area_inverse = (1.0/float(assemble(1.0*mesh_dataClass.ds(
        physical_group))))

    # Gets the node closest to the center point

    center_nodeNumber, center_point = mesh_tools.find_nodeClosestToPoint(
    mesh_dataClass, center_point, None, None)

    # Sets the centroid vector as constant

    centroid_point = Constant(center_point)

    # Sets the relative position vector on the surface

    relative_position = mesh_dataClass.x-centroid_point

    # Sets the localization parameter as a scalar value that is 1 inside
    # a circle of influence of the traction around the center point where 
    # the torsion axis passes through, and 0 everywhere else. If the in-
    # fluence radius is None, this parameter will be 1 everywhere

    localization_parameter = Constant(1.0)

    if not (influence_radius is None):

        # Checks if it is a float

        if not (isinstance(influence_radius, int) or isinstance(
        influence_radius, float)):
            
            raise TypeError("The influence radius of the NormalFollowe"+
            "rTorsion must be a float number, whereas the given value "+
            "is "+str(influence_radius))
        
        localization_parameter = ufl.conditional(gt(dot(
        relative_position, relative_position), influence_radius**2), 0.0, 
        1.0)
    
    # Looks for the nodes coordinates that are on elements adjacent to 
    # the center point

    adjacent_centerNodes, adjacent_centerNodesCoord = (
    mesh_tools.find_nodesOnSurfaceAroundNode(mesh_dataClass, 
    physical_group, node_number=center_nodeNumber))

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

    # Sets the average normal vector as constant

    average_normalVector = Constant([0.0, 0.0, 0.0])

    # Creates the time constants

    time_constant = Constant(t)

    maximum_time = Constant(t_final)
    
    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(field)+I

    J = det(F)

    # Gets the normal vector in the deformed configuration (but it does
    # not have to be unitary because, later, the whole vector will be 
    # rescaled)

    normal_vector = (inv(F).T)*mesh_dataClass.n

    # Gets the relative displacement (subtracting the displacement at 
    # the center point)

    centroid_displacement = field_numerical(Point(center_point))

    relative_displacement = as_vector([field[0]-centroid_displacement[0],
    field[1]-centroid_displacement[1], field[2]-centroid_displacement[2]
    ])

    # Gets the traction direction as the cross product of the relative
    # position translated by the displacement with the normal vector

    displaced_relativePosition = relative_position+relative_displacement

    traction_direction = cross(normal_vector, displaced_relativePosition)

    # Gets the norm of this vector. Multiplies by the magnitude of the 
    # relative position vector squared and divided by the influence ra-
    # dius squared to get a radially increasing traction profile

    norm_traction = (sqrt(dot(displaced_relativePosition, 
    displaced_relativePosition))/ufl.conditional(gt(sqrt(dot(
    traction_direction, traction_direction)), 1E-5), sqrt(dot(
    traction_direction, traction_direction)), 1.0))

    # Initializes the calculation of the traction amplitude

    traction_amplitude = Constant(amplitude_torsion)

    # Creates the traction vector. Multiplies by the traction amplitude,
    # which is FOR NOW given by the amplitude torsion itself; this value
    # will be later corrected at each loading step to assure that the 
    # traction field is indeed producing the desired moment in the nor-
    # mal direction. Multiplies also by a correction factor to account
    # for the pull back of the traction vector to the referential confi-
    # guration

    T_unscaled = (traction_amplitude*norm_traction*traction_direction*J*
    sqrt(dot(normal_vector,normal_vector))*localization_parameter)

    # Creates the traction vector at the principal directions to balance
    # out parasitic forces. This traction is not scaled by the deformed
    # to referential configurations update because the parasitic forces
    # are already integrated in the reference configuration

    T_parasiticBalance = Constant([0.0, 0.0, 0.0])

    # Scales the traction vector to give the actual moment

    T = (parametric_load_curve(time_constant/maximum_time)*(T_unscaled-
    T_parasiticBalance))

    # Defines a class to update the time constant and the traction am-
    # plitude

    class TimeUpdate:

        def __init__(self):
            
            self.t = time_constant

            self.average_normalVector = average_normalVector

            self.traction_amplitude = traction_amplitude

            self.T_parasiticBalance = T_parasiticBalance

        # This class must have the update_load method to be called in 
        # the stepping algorithm

        def update_load(self, t):

            # Updates the time constant

            self.t.assign(Constant(t))

            # Evaluates the average normal vector of the surface inte-
            # grating over the elements adjacent to the node where the
            # center of the torsion moment is applied

            average_normal = [float(assemble(adjacency_parameter*
            normal_vector[0]*mesh_dataClass.ds(physical_group))), float(
            assemble(adjacency_parameter*normal_vector[1]*
            mesh_dataClass.ds(physical_group))), float(assemble(
            adjacency_parameter*normal_vector[2]*mesh_dataClass.ds(
            physical_group)))]

            # Normalizes this average vector

            norm_vector = (1.0/np.sqrt((average_normal[0]**2)+(
            average_normal[1]**2)+(average_normal[2]**2)))

            average_normal = [norm_vector*average_normal[0], (norm_vector
            *average_normal[1]), norm_vector*average_normal[2]]

            self.average_normalVector.assign(Constant(average_normal))

            # Gets the real moment done by this traction in the direc-
            # tion of the average normal vector

            real_momentTrial = float(assemble(dot(cross(
            displaced_relativePosition, T_unscaled), 
            self.average_normalVector)*mesh_dataClass.ds(physical_group)
            ))

            # Gets the signal and the absolute value of the moment

            moment_signal = np.sign(real_momentTrial)

            if moment_signal==0.0:

                moment_signal = 1.0

            moment_absoluteValue = np.abs(real_momentTrial)

            # If the moment is too close to zero, gets it as the tole-
            # rance

            moment_absoluteValue = max(moment_absoluteValue, 1E-5)

            # Reconstructs the moment

            real_moment = moment_signal*moment_absoluteValue

            # Updates the traction amplitude

            correction_factor = amplitude_torsion/real_moment

            self.traction_amplitude.assign(self.traction_amplitude*
            correction_factor)

            # Gets the parasitic values of traction in each one of the 
            # directions of the space. Parasitic forces are resultant 
            # forces that are not intended and must be balanced

            parasitic_forceX = float(assemble(T_unscaled[0]*
            mesh_dataClass.ds(physical_group)))

            parasitic_forceY = float(assemble(T_unscaled[1]*
            mesh_dataClass.ds(physical_group)))

            parasitic_forceZ = float(assemble(T_unscaled[2]*
            mesh_dataClass.ds(physical_group)))
            
            if not no_parasiticForcesCorrection:

                # Adds this parasitic values to the correction

                self.T_parasiticBalance.assign(Constant((area_inverse*
                parasitic_forceX, area_inverse*parasitic_forceY, 
                area_inverse*parasitic_forceZ)))

            # Annuntiates the corrections

            print("\nNormal follower torsion:\nThe real moment applied"+
            " to the surface '"+str(original_physicalGroup)+"' by the "+
            "current traction\nfield is "+str(real_moment)+".\nThe tr"+
            "action amplitude is corrected to "+str(
            self.traction_amplitude.values()[0])+"\nA correction facto"+
            "r of "+str(correction_factor)+" was used\nThe current tra"+
            "ction field induces a parasitic force of\nx direction: "+
            str(parasitic_forceX)+"\ny direction: "+str(parasitic_forceY
            )+"\nz direction: "+str(parasitic_forceZ)+"\n"+str(
            float(parametric_load_curve(self.t/maximum_time)*100))+"% "+
            "of the final load is applied using the parametric load cu"+
            "rve\n")
            
            if not no_parasiticForcesCorrection:
                
                print("All of the parasitic forces are balanced out by"+
                " opposite uniform tractions\n")

    # Instantiates the loading class that was built and returns it as 
    # well as the traction vector object

    time_class = TimeUpdate()

    return T, time_class

# Defines a function to construct a bending moment on a surface. A point
# must be supplied as well as a axis vector. The moment vector will be 
# aligned with this axis vector and will pass through the given point. 
# The amplitude is the correct bending moment at the deformed configura-
# tion

def NormalFollowerMoment(field, field_numerical, mesh_dataClass, 
amplitude_bendingMoment, physical_group, bending_axis=None, center_point=
None, influence_radius=None, final_referencePoint=None, t=0.0, t_final=
1.0, parametric_load_curve="linear", no_parasiticForcesCorrection=False):
    
    # Verifies if the parametric load curve is a string and, then, uses
    # it to convert to an actual function

    if isinstance(parametric_load_curve, str):

        parametric_load_curve = numerical_tools.generate_loadingParametricCurves(
        parametric_load_curve)
    
    # Verifies if the physical group is a string and converts it

    physical_group, original_physicalGroup = mesh_tools.convert_physicalGroup(
    physical_group, mesh_dataClass, "boundary")

    # Verifies if the center point is None, then, proceeds to evaluate
    # the centroid of the region

    area_inverse = 0.0

    if center_point is None:

        center_point, area_inverse = mesh_tools.evaluate_centroidSurface(
        physical_group, mesh_dataClass)

    # Otherwise, verifies if it is a list

    elif not (isinstance(center_point, list)):

        raise TypeError("The center point of application of the follow"+
        "er bending moment must be a list. The following was given tho"+
        "ugh: "+str(center_point))

    else:

        area_inverse = (1.0/float(assemble(1.0*mesh_dataClass.ds(
        physical_group))))

    # Checks if the direction vector and if the final point were not gi-
    # ven

    if (bending_axis is None) and (final_referencePoint is None):

        raise ValueError("In NormalFollowerMoment, the bending axis is"+
        " None and also is the final_referencePoint. Hence, it's not p"+
        "ossible to determine the direction of the bending axis")
    
    elif not (bending_axis is None):

        # Checks if the bending axis is a list

        if not isinstance(bending_axis, list):

            raise TypeError("The bending_axis must be a list to set th"+
            "e bending axis in NormalFollowerMoment")
        
    else:

        # Checks if the final reference point is a list

        if not isinstance(final_referencePoint, list):

            raise TypeError("The final_referencePoint must be a list t"+
            "o set the bending direction in NormalFollowerMoment")

        # Gets the bending direction subtracting the center point from
        # the final reference point

        bending_axis = [final_referencePoint[0]-center_point[0],
        final_referencePoint[1]-center_point[1], (final_referencePoint[2
        ]-center_point[2])]

    # Gets the node closest to the center point

    center_nodeNumber, center_point = mesh_tools.find_nodeClosestToPoint(
    mesh_dataClass, center_point, None, None)

    # Sets the centroid vector as constant

    centroid_point = Constant(center_point)

    # Creates the time constants

    time_constant = Constant(t)

    maximum_time = Constant(t_final)
    
    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(field)+I

    J = det(F)

    # Gets the normal vector in the deformed configuration (but it does
    # not have to be unitary because, later, the whole vector will be 
    # rescaled)

    normal_vector = (inv(F).T)*mesh_dataClass.n

    # Sets the direction vector as a Constant and pushes it forward to
    # the deformed configuration

    bending_axis = F*Constant(bending_axis)

    # With the bending axis, creates the projection tensor to project 
    # the relative position onto a plane given by the bending axis

    projection_tensor = tensor_tools.projection_tensor(bending_axis)

    # Sets the relative position vector on the surface

    relative_position = mesh_dataClass.x-centroid_point

    # Gets the relative displacement (subtracting the displacement at 
    # the center point)

    centroid_displacement = field_numerical(Point(center_point))

    relative_displacement = as_vector([field[0]-centroid_displacement[0],
    field[1]-centroid_displacement[1], field[2]-centroid_displacement[2]
    ])

    # Translates the relative position with the relative displacement

    displaced_relativePosition = relative_position+relative_displacement

    # Sets the projected relative position

    radial_distance = projection_tensor*displaced_relativePosition

    # Sets the direction of the traction

    traction_direction = cross(bending_axis, radial_distance)

    # Initializes the calculation of the traction amplitude

    traction_amplitude = Constant(amplitude_bendingMoment)

    # Sets the localization parameter as a scalar value that is 1 inside
    # a circle of influence of the traction around the center point where 
    # the torsion axis passes through, and 0 everywhere else. If the in-
    # fluence radius is None, this parameter will be 1 everywhere

    localization_parameter = Constant(1.0)

    if not (influence_radius is None):

        # Checks if it is a float

        if not (isinstance(influence_radius, int) or isinstance(
        influence_radius, float)):
            
            raise TypeError("The influence radius of the NormalFollowe"+
            "rMoment must be a float number, whereas the given value i"+
            "s "+str(influence_radius))
        
        localization_parameter = ufl.conditional(gt(dot(
        relative_position, relative_position), influence_radius**2), 0.0, 
        1.0)

    # Constructs the unscaled traction profile as a linearly increasing
    # traction from the bending axis outwards. Multiplies also by a cor-
    # rection factor to account for the pull back of the traction vector 
    # to the referential configuration

    T_unscaled = (traction_amplitude*(sqrt(dot(radial_distance, 
    radial_distance))/ufl.conditional(gt(sqrt(dot(traction_direction, 
    traction_direction)), 1E-5), sqrt(dot(traction_direction, 
    traction_direction)), 1.0))*traction_direction*J*sqrt(dot(
    normal_vector, normal_vector))*localization_parameter)

    # Creates the traction vector at the principal directions to balance
    # out parasitic forces. This traction is not scaled by the deformed
    # to referential configurations update because the parasitic forces
    # are already integrated in the reference configuration

    T_parasiticBalance = Constant([0.0, 0.0, 0.0])

    # Scales the traction vector to give the actual moment

    T = (parametric_load_curve(time_constant/maximum_time)*(T_unscaled-
    T_parasiticBalance))

    # Defines a class to update the time constant and the traction am-
    # plitude

    class TimeUpdate:

        def __init__(self):
            
            self.t = time_constant

            self.traction_amplitude = traction_amplitude

            self.T_parasiticBalance = T_parasiticBalance

        # This class must have the update_load method to be called in 
        # the stepping algorithm

        def update_load(self, t):

            # Updates the time constant

            self.t.assign(Constant(t))

            # Gets the real moment done by this traction in the direc-
            # tion in the deformed configuration

            real_localMoment = cross(radial_distance, T_unscaled)

            real_momentTrial = float(np.sqrt(assemble(dot(
            real_localMoment, real_localMoment)*mesh_dataClass.ds(
            physical_group))))

            # Gets the signal and the absolute value of the moment

            moment_signal = np.sign(real_momentTrial)

            if moment_signal==0.0:

                moment_signal = 1.0

            moment_absoluteValue = np.abs(real_momentTrial)

            # If the moment is too close to zero, gets it as the tole-
            # rance

            moment_absoluteValue = max(moment_absoluteValue, 1E-5)

            # Reconstructs the moment

            real_moment = moment_signal*moment_absoluteValue

            # Updates the traction amplitude

            correction_factor = amplitude_bendingMoment/real_moment

            self.traction_amplitude.assign(self.traction_amplitude*
            correction_factor)

            # Gets the parasitic values of traction in each one of the 
            # directions of the space. Parasitic forces are resultant 
            # forces that are not intended and must be balanced

            parasitic_forceX = float(assemble(T_unscaled[0]*
            mesh_dataClass.ds(physical_group)))

            parasitic_forceY = float(assemble(T_unscaled[1]*
            mesh_dataClass.ds(physical_group)))

            parasitic_forceZ = float(assemble(T_unscaled[2]*
            mesh_dataClass.ds(physical_group)))

            if not no_parasiticForcesCorrection:

                # Adds this parasitic values to the correction

                self.T_parasiticBalance.assign(Constant((area_inverse*
                parasitic_forceX, area_inverse*parasitic_forceY, 
                area_inverse*parasitic_forceZ)))

            # Annuntiates the corrections

            print("\nNormal follower bending moment:\nThe real moment "+
            "applied to the surface '"+str(original_physicalGroup)+"' "+
            "by the current traction\nfield is "+str(real_moment)+".\n"+
            "The traction amplitude is corrected to "+str(
            self.traction_amplitude.values()[0])+"\nA correction facto"+
            "r of "+str(correction_factor)+" was used\nThe current tra"+
            "ction field induces a parasitic force of\nx direction: "+
            str(parasitic_forceX)+"\ny direction: "+str(parasitic_forceY
            )+"\nz direction: "+str(parasitic_forceZ)+"\n"+str(
            float(parametric_load_curve(self.t/maximum_time)*100))+"% "+
            "of the final load is applied using the parametric load cu"+
            "rve\n")
            
            if not no_parasiticForcesCorrection:
                
                print("All of the parasitic forces are balanced out by"+
                " opposite uniform tractions\n")

    # Instantiates the loading class that was built and returns it as 
    # well as the traction vector object

    time_class = TimeUpdate()

    return T, time_class