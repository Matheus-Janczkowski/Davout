# Routine to store methods to generate solids topologically equal to 
# cubes

import gmsh 

import numpy as np

import copy

from ..tool_box import geometric_tools as geo

########################################################################
#                      Whole cuboid manufacturing                      #
########################################################################

# Defines a function to create a cuboid from points to volume using 
# points supplied by the user

def make_cuboid(corner_coordinatesOriginal, lines_instructionsOriginal=
dict(), surfaces_instructions=["surface filling", "surface filling", 
"surface filling", "surface filling", "surface filling", 
"surface filling"], transfinite_directions=[], bias_directions=dict(), 
rotation_vector=[0.0, 0.0, 0.0], translation_vector=[0.0, 0.0, 0.0], 
geometric_data=[0, [[],[],[],[]], [[],[],[],[]], [[],[],[]], dict(), [], 
dict(), [], [], [], 0.5, False], verbose=False, 
explicit_volume_physical_group_name=None, 
explicit_surface_physical_group_name=None):
    
    # If the transfinite directions are equal to zero, it means they are
    # not to be transfinite

    if len(transfinite_directions)>0:

        if transfinite_directions[0]==0:

            transfinite_directions = []
    
    # Recovers the information from the input list

    (volume, whole_setOfPoints, whole_setOfLines, whole_setOfSurfaces, 
    dictionary_surfacesPhysGroups, surface_identifiers,
    dictionary_volumesPhysGroups, volume_identifiers, 
    surface_regionsNames, volume_regionsNames, lc, hexahedron_mesh
    ) = geometric_data
    
    # Transforms the corner coordinates into a numpy array

    corner_coordinates = np.array(copy.deepcopy(
    corner_coordinatesOriginal))

    # Rotates and translates these points

    corner_coordinates = geo.rotate_andTranslateEulerRodrigues(
    corner_coordinates, rotation_vector, *translation_vector)

    # Copies the instructions for the lines

    lines_instructions = copy.deepcopy(lines_instructionsOriginal)

    # Verifies the lines instructions and look for points there to rota-
    # te aswell

    for instruction in lines_instructions:

        # Verifies if this instruction is not a line

        if (lines_instructions[instruction][0]!="line" and (
        lines_instructions[instruction][-1]!="do not rotate")):

            # Rotates the auxiliary points

            lines_instructions[instruction][1] = geo.rotate_andTranslateEulerRodrigues(
            lines_instructions[instruction][1], rotation_vector, 
            *translation_vector)

    # Use these coordinates to create points

    points_list, whole_setOfPoints = create_cuboidPoints(
    corner_coordinates, whole_setOfPoints=whole_setOfPoints,
    lc=lc)

    # Creates the lines

    whole_setOfLines = create_cuboidLines(points_list, 
    lines_instructions=lines_instructions, transfinite_directions=
    transfinite_directions, bias_directions=bias_directions,
    whole_setOfLines=whole_setOfLines, verbose=verbose)

    # Creates the surfaces and the volume

    flag_transfinite = False

    if len(transfinite_directions)>0:

        flag_transfinite = True

    volume, whole_setOfSurfaces, surfaces_cornersDict = create_cuboidFromLines(
    points_list, whole_setOfLines, whole_setOfSurfaces=
    whole_setOfSurfaces, surface_instruction=surfaces_instructions, 
    flag_transfinite=flag_transfinite, hexahedron_mesh=hexahedron_mesh,
    verbose=verbose)

    # Verifies whether any of these surfaces belong to a boundary

    if explicit_surface_physical_group_name is not None:

        if not isinstance(explicit_surface_physical_group_name, dict):

            raise TypeError("'explicit_surface_physical_group_name' is"+
            " "+str(explicit_surface_physical_group_name)+", it must b"+
            "e a dictionary")
        
        # Iterates through the surfaces of the volume

        for surface_tag, surface_points in surfaces_cornersDict.items():

            # If the local number of the surface is one of the requested

            if surface_points[1] in explicit_surface_physical_group_name:

                # Iterates through the available surface names

                for i in range(len(surface_regionsNames)):

                    if explicit_surface_physical_group_name[
                    surface_points[1]]==surface_regionsNames[i]:

                        # Adds the surface tag to the dictionary of phy-
                        # sical surfaces

                        dictionary_surfacesPhysGroups[i+1].append(
                        surface_tag)

                        break

    else:

        for surface_tag, surface_points in surfaces_cornersDict.items():

            for i in range(len(surface_identifiers)):

                if surface_identifiers[i](surface_points[0]):

                    # Adds the surface tag to the dictionary of physical 
                    # surfaces

                    dictionary_surfacesPhysGroups[i+1].append(surface_tag)

                    break

    # Verifies whether this volume belongs to a physical group

    corner_coordinates = np.transpose(corner_coordinates)

    flag_identified = False

    # Verifies if a explicit physical group is given

    if explicit_volume_physical_group_name is not None:

        for i in range(len(volume_regionsNames)):

            if explicit_volume_physical_group_name==volume_regionsNames[
            i]:

                dictionary_volumesPhysGroups[i+1].append(volume)

                flag_identified = True 

                # Colors the volume

                color = color_interpolation(i+1, len(volume_regionsNames))

                gmsh.model.setColor([(3,volume)], *color)

                break 

    if not flag_identified:

        for i in range(len(volume_identifiers)):

            # If the volume identifier returns true, the volume is inde-
            # ed in the region

            if volume_identifiers[i](corner_coordinates):

                dictionary_volumesPhysGroups[i+1].append(volume)

                flag_identified = True

                # Colors the volume

                color = color_interpolation(i+1, len(volume_identifiers))

                gmsh.model.setColor([(3,volume)], *color)

                break

    # If this volume is not part of a physical group, turns this volume 
    # into the generic physical group

    if not flag_identified:

        dictionary_volumesPhysGroups[0].append(volume)

        # Colors the volume

        color = color_interpolation(0, len(volume_identifiers))

        gmsh.model.setColor([(3,volume)], *color)

    if verbose:

        print(dictionary_volumesPhysGroups, "\n")

    # Puts the output variables inside a output list

    geometric_data = [volume, whole_setOfPoints, whole_setOfLines, 
    whole_setOfSurfaces, dictionary_surfacesPhysGroups, 
    surface_identifiers, dictionary_volumesPhysGroups, 
    volume_identifiers, surface_regionsNames, volume_regionsNames, lc,
    hexahedron_mesh]

    return geometric_data

########################################################################
#                     Volume and surfaces building                     #
########################################################################

# Defines a function to generate a cuboid using lines (take care with 
# the order of the lines). The surface instruction is a list of strings,
# Where each element must be either "plane surface" or "surface filling"

def create_cuboidFromLines(points_list, whole_setOfLines, 
surface_instruction=["plane surface", "plane surface", "plane surface",
"plane surface", "plane surface", "plane surface"], flag_transfinite=
False, hexahedron_mesh=False, whole_setOfSurfaces=[[],[],[]], verbose=
False):
    
    # Retrieves the points and adds them to a dictionary of points

    p1, p2, p3, p4, p5, p6, p7, p8 = points_list

    preliminary_points = dict()

    preliminary_points[1] = p1

    preliminary_points[2] = p2

    preliminary_points[3] = p3

    preliminary_points[4] = p4

    preliminary_points[5] = p5

    preliminary_points[6] = p6

    preliminary_points[7] = p7

    preliminary_points[8] = p8

    #print("\nThe dictionary of points is initialized as:", preliminary_points, "\n")

    # Sets a dictionary that tells the opposite point to the x point 
    # looking from the y surface. The key is the surface index, and the
    # list is the opposite point given the index of the current point

    opposite_point = dict()

    opposite_point[1] = [5, 6, 7, 8, 1, 2, 3, 4]

    opposite_point[6] = [5, 6, 7, 8, 1, 2, 3, 4]
    
    opposite_point[2] = [4, 3, 2, 1, 8, 7, 6, 5]
    
    opposite_point[4] = [4, 3, 2, 1, 8, 7, 6, 5]
    
    opposite_point[3] = [2, 1, 4, 3, 6, 5, 8, 7]
    
    opposite_point[5] = [2, 1, 4, 3, 6, 5, 8, 7]

    # Sets a dictionary of points cross opposite, i.e. they are opposite
    # using two surfaces that are perpendicular to the surface of inte-
    # rest

    cross_oppositePoint = dict()

    cross_oppositePoint[1] = [3, 4, 1, 2, 7, 8, 5, 6]

    cross_oppositePoint[2] = [6, 5, 8, 7, 2, 1, 4, 3]

    cross_oppositePoint[3] = [8, 7, 6, 5, 4, 3, 2, 1]

    cross_oppositePoint[4] = [6, 5, 8, 7, 2, 1, 4, 3]

    cross_oppositePoint[5] = [8, 7, 6, 5, 4, 3, 2, 1]

    cross_oppositePoint[6] = [3, 4, 1, 2, 7, 8, 5, 6]

    # Sets a dictionary that tells the opposite surface

    surface_opposite = dict()

    surface_opposite[1] = 6

    surface_opposite[2] = 4

    surface_opposite[3] = 5

    surface_opposite[4] = 2

    surface_opposite[5] = 3

    surface_opposite[6] = 1

    ####################################################################
    #                           Curve loops                            #
    ####################################################################
    
    # Initializes a dictionary of curve loops

    loops = dict()

    # Lower XY plane

    loops[1] = [[4,1], [1,2], [2,3], [3,4]]

    # Front YZ plane

    loops[2] = [[1,5], [5,6], [6,2], [2,1]]

    # Front XZ plane

    loops[3] = [[3,7], [7,6], [6,2], [2,3]]

    # Back YZ plane

    loops[4] = [[4,8], [8,7], [7,3], [3,4]]

    # Back XZ plane

    loops[5] = [[4,8], [8,5], [5,1], [1,4]]

    # Upper XY plane

    loops[6] = [[8,5], [5,6], [6,7], [7,8]]#
    
    # Sets a dictionary for the corners for these surfaces

    surface_corners = dict()

    for loop in loops:

        # Sets the corners for this surface

        surface_corners[loop] = [point[0] for point in loops[loop]]

    ####################################################################
    #                            Surfaces                              #
    ####################################################################

    # Initializes a dictionary of surfaces

    surfaces = dict()

    # Initializes a counter of created surfaces

    n_createdSurfaces = 0

    # Initializes a list of loops that were already created

    repeated_loops = []

    # Initializes a flag to inform whether the dictionary of points has
    # already been altered

    flag_dictionaryAlteration = False

    points_dictionary = dict()

    # Iterates through the curve loops to find surfaces that were already
    # created

    for loop in loops:

        # Iterates through the whole set of surfaces

        for i in range(len(whole_setOfSurfaces[0])):

            # Checks if the corner points of this surface are equal to 
            # any other already made

            current_points = [preliminary_points[point
            ] for point in surface_corners[loop]]

            prior_points = whole_setOfSurfaces[0][i]

            same_listResult = is_theSameList(current_points, prior_points)

            if same_listResult and not flag_dictionaryAlteration:

                # Gets the surface and the local enumeration of the 
                # points of the already made surface

                old_surfaceID = whole_setOfSurfaces[2][i][0]

                old_surfaceLocalPoints = whole_setOfSurfaces[2][i][1]
                
                # Updates the list of loops that are repeated and upda-
                # tes the dictionary of surfaces. This surface will be
                # reordered to the opposite surface of the old surface

                repeated_loops.append(surface_opposite[old_surfaceID])

                if verbose:

                    print("The surface", loop, ", tag:", 
                    whole_setOfSurfaces[1][i], ", is already created w"+
                    "ith local number:", repeated_loops[-1])

                surfaces[repeated_loops[-1]] = whole_setOfSurfaces[1][i]

                # Gets the the local numbering of this surface

                opposite_surfaceLocalPoints = [opposite_point[
                old_surfaceID][point-1] for point in (
                old_surfaceLocalPoints)]

                # Gets the keys at the points dictionary for the points
                # of the corners of the old surface

                keys_oldSurfaceCorners = get_keys(preliminary_points,
                prior_points)

                # Gets the local enumeration of the points opposite to 
                # the points on the corners of the old surface

                keys_oppositeSurfaceCorners = [opposite_point[loop][
                point-1] for point in (keys_oldSurfaceCorners)]

                # Gets the points opposite to the old surface

                opposite_surfacePoints = [preliminary_points[point
                ] for point in keys_oppositeSurfaceCorners]
                
                # Updates the dictionary of points to make the order of 
                # the surface corners consistent to the mesh of the ad-
                # jacent volumes

                points_dictionary = reordinate_dictionary(
                preliminary_points, old_surfaceLocalPoints, 
                opposite_surfaceLocalPoints, opposite_surfacePoints, 
                prior_points)

                if verbose:

                    print("Points dictionary first update:", 
                    points_dictionary)

                # Updates the flag that informs if the points dictionary
                # has been altered

                flag_dictionaryAlteration = True

                #points_dictionary = reordinate_points(
                #preliminary_points, opposite_point, current_points, 
                #prior_points, surface_corners[loop], points_list, loop,
                #verbose=verbose)

                break

            elif same_listResult and flag_dictionaryAlteration:

                # Gets the local indices for these points in the new 
                # dictionary

                local_numbering = get_keys(points_dictionary, 
                prior_points)

                # Looks for the surface local number

                for surface_number, corner_points in surface_corners.items():

                    # If the order is not the same, but the numbers are, 
                    # this means that it is the right side, but the face
                    # is wrong, thus, the solid must be rotated around 
                    # the normal vector of the first encountered shared 
                    # surface

                    if is_theSameList(corner_points, local_numbering):

                        # Verifies if the orientation is opposite

                        """flag_reversed = True

                        for j in range(4):

                            if corner_points[j]!=opposite_point[
                            repeated_loops[0]][local_numbering[j]-1]:
                                
                                flag_reversed = False

                                print(opposite_point[
                            repeated_loops[0]][local_numbering[j]-1])#break"""
                                
                        flag_reversed = test_permutationNeed(
                        local_numbering, corner_points)

                        if verbose:

                            print("Surface ", surface_number, "; tag:", 
                            whole_setOfSurfaces[1][i], "; current poin"+
                            "ts:", local_numbering, "; corner points: ", 
                            corner_points, "; flag to reverse:", 
                            flag_reversed)
                
                        # Updates the list of loops that are repeated and 
                        # updates the dictionary of surfaces

                        repeated_loops.append(surface_number)

                        # If the surfaces must be reversed (rotated pi),
                        # reflects the dictionary of points

                        if flag_reversed:

                            # Gets the first shared surface to keep it as 
                            # reference

                            initial_surface = repeated_loops[0]

                            # Reverses the surfaces that were already 
                            # found, except the first one and the oppo-
                            # site to it

                            for j in range(1,len(repeated_loops)):

                                reversed_surface = surface_opposite[
                                repeated_loops[j]]

                                if reversed_surface!=initial_surface:

                                    repeated_loops[j] = reversed_surface

                            # Cross reverses the dictionary of points u-
                            # sing the dictionary of cross opposite 
                            # points

                            for j in range(4):

                                # Gets the point

                                original_point = local_numbering[j]

                                # Gets its cross opposite

                                cross_opposite = cross_oppositePoint[
                                initial_surface][original_point-1]

                                # Gets the points' tags

                                original_tag = points_dictionary[
                                original_point]

                                cross_tag = points_dictionary[
                                cross_opposite]

                                # Swaps them

                                points_dictionary[original_point] = (
                                cross_tag)

                                points_dictionary[cross_opposite] = (
                                original_tag)

                        surfaces[repeated_loops[-1]] = (
                        whole_setOfSurfaces[1][i])

                        break

    # Initializes a dictionary to tell the old numbering of a surface 
    # after the rotations 

    old_surface_numbering = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

    # If no common surface has been found, the preliminary points dic-
    # tionary is indeed the right one

    if not flag_dictionaryAlteration:

        points_dictionary = preliminary_points

    # Otherwise, verifies how the surface numbering changed

    else:

        # Gets the new indexes of the points

        points_new_local_numbering = get_keys(points_dictionary, [
        preliminary_points[i+1] for i in range(8)])

        # Iterates through the surface corners

        for surface_number, corners_local_numbers in surface_corners.items():

            # Gets the new local numbers

            new_corner_enumeration = [points_new_local_numbering[corner-1
            ] for corner in corners_local_numbers]

            # Iterates through the original corners again to find the 
            # matching sequence of points

            for new_surface_number, corners_numbers in surface_corners.items():

                if is_theSameList(new_corner_enumeration, corners_numbers
                ):
                    
                    old_surface_numbering[new_surface_number] = (
                    surface_number)

                    break

        if verbose:

            print("The surface local numbering was switched to: "+str(
            old_surface_numbering))

    if verbose:

        print("Repeated loops:", repeated_loops)

        print("Points dictionary:", points_dictionary)

    # Initializes a dictionary of points at the corners of each surface

    surfaces_cornersDict = dict()

    # Iterates for the loops again to create the ones which are not re-
    # peated

    for loop in loops:

        # If it were not created, create it

        if not (loop in repeated_loops):

            n_createdSurfaces += 1

            surface_tag = 0

            # Creates the curve loop

            lines_list = []

            loop_points = loops[loop]

            for pair_ofPoints in loop_points:

                for j in range(len(whole_setOfLines[0])):

                    if (whole_setOfLines[0][j]==points_dictionary[
                    pair_ofPoints[0]] and whole_setOfLines[1][j]==(
                    points_dictionary[pair_ofPoints[1]])):

                        lines_list.append(whole_setOfLines[2][j])

                        break

                    elif (whole_setOfLines[1][j]==points_dictionary[
                    pair_ofPoints[0]] and whole_setOfLines[0][j]==(
                    points_dictionary[pair_ofPoints[1]])):

                        lines_list.append(-whole_setOfLines[2][j])

                        break

            if verbose:
                
                print("Surface", loop, "has the lines:", lines_list, 
                "due to the points:", loop_points)

            curve_loop = gmsh.model.geo.addCurveLoop(lines_list)

            # Creates the surface using the given method

            if surface_instruction[loop-1]=="plane surface":

                surface_tag = gmsh.model.geo.addPlaneSurface([curve_loop
                ])

            elif surface_instruction[loop-1]=="surface filling":

                surface_tag = gmsh.model.geo.addSurfaceFilling(
                [curve_loop])

            else:

                raise NameError('Instructions for surface building can'+
                ' be either "plane surface" or "surface filling"\n')
            
            # Updates the surface tag and recovers the surface corners
            
            surfaces[loop] = surface_tag 

            corner_points = [points_dictionary[point] for point in (
            surface_corners[loop])]

            # Adds these corners to the surfaces corner dictionary

            # Gets the coordinates of the corners

            corners_coordinates = []

            for corner in corner_points:

                corners_coordinates.append(gmsh.model.getValue(0,corner,
                []))

            surfaces_cornersDict[surface_tag] = [corners_coordinates, 
            old_surface_numbering[loop]]

            #print("Surface", surface_tag, " has the corner points:",
            #corner_points, "for local corners:", surface_corners[loop], 
            #"and curve loop", loop, ":", lines_list)

            whole_setOfSurfaces[0].append(corner_points)

            whole_setOfSurfaces[1].append(surface_tag)

            # Updates the local list of points in the local system of 
            # points enumeration and the local number of the surface

            whole_setOfSurfaces[2].append([loop, surface_corners[loop]])

            # Checks wheter transfinite is being asked for

            if flag_transfinite:

                # Sets this surfaces as transfinite

                gmsh.model.geo.mesh.setTransfiniteSurface(surface_tag,
                cornerTags=corner_points, arrangement="Left")

                # Recombines the mesh for using hexahedron mesh

                if hexahedron_mesh:

                    gmsh.model.geo.mesh.setRecombine(2, surface_tag)

    if verbose:

        print("Surfaces:", surfaces, "\n")

    # If no surface was created, through an error, for two volumes are
    # occupying the same space

    if n_createdSurfaces==0:

        raise ReferenceError("All the surfaces "+str(list(
        surfaces.keys()))+" for this volume have already been created,"+
        " thus, it is ill-posed, for another volume has the same surfa"+
        "ces.")

    gmsh.model.geo.synchronize()
        
    ####################################################################
    #                              Volumes                             #
    ####################################################################

    # Creates the surface loop, then the volume

    assembly = gmsh.model.geo.addSurfaceLoop(list(surfaces.values()))

    volume = gmsh.model.geo.addVolume([assembly])

    if verbose:

        print("Creates volume", volume, "\n\n")

    gmsh.model.geo.synchronize()
        
    ####################################################################
    #                            Transfinite                           #
    ####################################################################

    # Checks wheter transfinite is being asked for

    if flag_transfinite:

        # Sets the volume as transfinite
        
        gmsh.model.geo.mesh.setTransfiniteVolume(volume, cornerTags=[
        points_dictionary[4], points_dictionary[1], points_dictionary[2
        ], points_dictionary[3], points_dictionary[8], 
        points_dictionary[5], points_dictionary[6], points_dictionary[7]
        ])

        gmsh.model.geo.synchronize()
    
    # Returns the volume and the whole set of surfaces

    return (volume, whole_setOfSurfaces, surfaces_cornersDict)

# Defines a function to verify if a list has the exact same element as
# another, despite the order being different. The second list is the 
# true list

def is_theSameList(list1, true_list):

    # Iterates through the elements of the first list

    for element_1 in list1:

        # Verifies if the element is not in the second list

        if not (element_1 in true_list):

            return False

    return True

# Defines a function to get the keys for a list of values

def get_keys(dictionary, values):

    keys = []

    for value in values:

        for key, compared_value in dictionary.items():

            if compared_value==value:

                keys.append(key)

                break 

    return keys

########################################################################
#                            Lines building                            #
########################################################################

# Defines a function to create the twelve lines of a cuboid using the 
# given instructions and further info. The whole set of lines is a list
# of lists: the first two are the initial and end point tags, respecti-
# vely; the third list is for the line tag, and the fourth for the 
# transfinite and bias information

def create_cuboidLines(points_list, lines_instructions=dict(), 
whole_setOfLines=[[], [], [], []], transfinite_directions=[0,0,0], 
bias_directions=dict(), verbose=False):
    
    # Creates a dictionary of opposite lines

    opposite_lines = dict()

    opposite_lines[1] = [3, 5, 7]

    opposite_lines[2] = [4, 6, 8]

    opposite_lines[3] = [1, 5, 7]

    opposite_lines[4] = [2, 6, 8]

    opposite_lines[5] = [1, 3, 7]

    opposite_lines[6] = [2, 4, 8]

    opposite_lines[7] = [1, 3, 5]

    opposite_lines[8] = [2, 4, 6]

    opposite_lines[9] = [10, 11, 12]

    opposite_lines[10] = [9, 11, 12]

    opposite_lines[11] = [9, 10, 12]

    opposite_lines[12] = [9, 10, 11]

    # Makes the list of points for each line

    lines_points = [[0, 1, transfinite_directions[1], "y"], [2, 1, 
    transfinite_directions[0], "x"], [3, 2, transfinite_directions[1
    ], "y"], [3, 0, transfinite_directions[0], "x"], [4, 5, 
    transfinite_directions[1], "y"], [6, 5, transfinite_directions[0], 
    "x"], [7, 6, transfinite_directions[1], "y"], [7, 4, 
    transfinite_directions[0], "x"], [0, 4, transfinite_directions[2],
    "z"], [1, 5, transfinite_directions[2], "z"], [2, 6, 
    transfinite_directions[2], "z"], [3, 7, transfinite_directions[2], 
    "z"]]

    # Initializes a list of lines that have already been created

    repeated_lines = []

    # Iterates through the lines of the cuboid

    for i in range(12):

        # First of all, verifies if this line has already been created

        for j in range(len(whole_setOfLines[0])):

            # If the tags of both points are equal, do not create this 
            # line again

            if (whole_setOfLines[0][j]==points_list[lines_points[i][0]]
            and whole_setOfLines[1][j]==points_list[lines_points[i][1]]):
                
                # Updates the list of already created lines
                                                   
                repeated_lines.append(i)

                # Gets the current transfinite information

                current_transfiniteInfo = whole_setOfLines[3][j]

                # Gets the opposite lines

                local_oppositeLines = opposite_lines[i+1]

                # Iterates through them to deliver the transfinite in-
                # formation, for all opposite lines must have the same 
                # transfinite discretization

                for opposite_line in local_oppositeLines:

                    lines_points[opposite_line-1][2] = (
                    current_transfiniteInfo[0])

                break

            # If the boundary points are the same but in different order

            elif (whole_setOfLines[0][j]==points_list[lines_points[i][1]
            ] and whole_setOfLines[1][j]==points_list[lines_points[i][0]
            ]):
                
                # Updates the list of already created lines
                                                   
                repeated_lines.append(i)

                # Gets the current transfinite information

                current_transfiniteInfo = whole_setOfLines[3][j]

                # Gets the opposite lines

                local_oppositeLines = opposite_lines[i+1]

                # Iterates through them to deliver the transfinite in-
                # formation, for all opposite lines must have the same 
                # transfinite discretization

                for opposite_line in local_oppositeLines:

                    lines_points[opposite_line-1][2] = (
                    current_transfiniteInfo[0])

                break

    # Iterates through the lines to create them 

    for i in range(12):

        if not (i in repeated_lines):

            # Initializes the new line tag

            line_tag = 0

            # Verifies if this line is in the lines instructions dictio-
            # nary

            if (i+1) in lines_instructions.keys():

                # Gets the type of the line - straight, circle arc, el-
                # liptic arc, or spline

                line_type = lines_instructions[i+1][0]

                # Tests if it is a straight line

                if line_type=="line":

                    line_tag = gmsh.model.geo.addLine(points_list[
                    lines_points[i][0]], points_list[lines_points[i][1]])

                # Tests if it is a circle arc

                elif line_type=="circle arc":

                    # Creates the center point

                    center_point = gmsh.model.geo.addPoint(
                    lines_instructions[i+1][1][0][0], lines_instructions[
                    i+1][1][1][0], lines_instructions[i+1][1][2][0])

                    line_tag = gmsh.model.geo.addCircleArc(points_list[
                    lines_points[i][0]], center_point, points_list[
                    lines_points[i][1]])

                # Tests if it is an elliptic arc

                elif line_type=="elliptic arc":

                    # Creates the center point

                    center_point = gmsh.model.geo.addPoint(
                    lines_instructions[i+1][1][0][0], lines_instructions[
                    i+1][1][1][0], lines_instructions[i+1][1][2][0])

                    # Creates the major point

                    major_point = gmsh.model.geo.addPoint(
                    lines_instructions[i+1][1][0][1], lines_instructions[
                    i+1][1][1][1], lines_instructions[i+1][1][2][1])

                    # Adds the ellipse points

                    ellipse_points = [points_list[lines_points[i][0]], 
                    center_point, major_point, points_list[lines_points[
                    i][1]]]

                    # Creates the elliptic arc

                    line_tag = gmsh.model.geo.addEllipseArc(
                    *ellipse_points)

                # Tests if it is a spline

                elif line_type=="spline":

                    # Adds the initial point

                    spline_points = [points_list[lines_points[i][0]]]

                    # Adds the middle points

                    for j in range(len(lines_instructions[i+1][1][0])):

                        spline_point = gmsh.model.geo.addPoint(
                        lines_instructions[i+1][1][0][j], 
                        lines_instructions[i+1][1][1][j],
                        lines_instructions[i+1][1][2][j])

                        spline_points.append(spline_point)

                    # Adds the end point

                    spline_points.append(points_list[lines_points[i][1]])

                    # Creates the spline

                    line_tag = gmsh.model.geo.addSpline(spline_points)

            else:

                # If it is not, the default function is to create a
                # straight line

                line_tag = gmsh.model.geo.addLine(points_list[
                lines_points[i][0]], points_list[lines_points[i][1]])

            # Updates the line tag to the whole set of lines

            whole_setOfLines[0].append(points_list[lines_points[i][0]])

            whole_setOfLines[1].append(points_list[lines_points[i][1]])

            whole_setOfLines[2].append(line_tag)

            whole_setOfLines[3].append(lines_points[i][2:4])

            # Makes the line transfinite if necessary

            if sum(transfinite_directions)>0:

                # Verifies if transfinite values are supplied for all 
                # directions

                if len(transfinite_directions)<3:

                    raise ValueError("If transfinite is required, the "+
                    "number of divisions must be supplied for all dire"+
                    "ction: x, y, and z.")
                
                # Does transfinite meshing upon the curve

                if verbose:

                    print("Makes line", line_tag, "with bias", 
                    bias_directions, "using axis", lines_points[i][3])

                make_lineTransfinite(line_tag, lines_points[i][2], 
                lines_points[i][3], bias_directions)

    gmsh.model.geo.synchronize()

    # Returns the list of lines

    return whole_setOfLines

# Defines a function to make the line transfinite

def make_lineTransfinite(line, transfinite_nDivisions, bias_axis, 
bias_directions):
    
    # Sets lines in the x direction as transfinite

    if bias_axis in bias_directions.keys():

        # Verifies if the bias values is a float, then the default bias
        # method is progression

        bias_value = bias_directions[bias_axis]

        if isinstance(bias_value, float) or isinstance(bias_value, int):

            gmsh.model.geo.mesh.setTransfiniteCurve(line, 
            transfinite_nDivisions, "Progression", bias_value)

        # If it is a list, the method is passed

        elif isinstance(bias_value, list):

            # Verifies if it has a length of 2

            if len(bias_value)!=2:

                raise IndexError("The bias value is a list, but it mus"+
                "t have a length of two: the first element is the name"+
                " of the method, either 'Progression' or 'Bump'; and t"+
                "he second element is the bias factor, a float. The cu"+
                "rrent bias value is: "+str(bias_value))

            # Verifies if the first value is either 'Progression' or 
            # 'Bump'

            elif not (bias_value[0]=="Progression" or (bias_value[0]==
            "Bump")):

                raise NameError("The bias value is a list, the first v"+
                "alue must be the name of the method, either 'Progress"+
                "ion' or 'Bump'. Currently, it is: "+str(bias_value[0]))
            
            elif not (isinstance(bias_value[1], float) or isinstance(
            bias_value[1], int)):
                
                raise TypeError("The bias value is a list, the second "+
                "value of this list must be the bias factor, either a "+
                "float or an integer. Currently, it is: "+str(
                bias_value[1]))
            
            gmsh.model.geo.mesh.setTransfiniteCurve(line, 
            transfinite_nDivisions, bias_value[0], bias_value[1])

        else:

            raise TypeError("The bias value can be either a float or i"+
            "nteger to tell the bias factor directly; or it can be a l"+
            "ist with the name of the bias method ('Progression' or 'B"+
            "ump'), and the value of the bias factor. Currently, it is"+
            ": "+str(bias_value))

    else:

        gmsh.model.geo.mesh.setTransfiniteCurve(line, 
        transfinite_nDivisions)

########################################################################
#                            Points building                           #
########################################################################

# Defines a function to create the points of a cuboid volume. The cuboid
# points is a numpy array, where the rows are the coordinates x, y, and
# z, respectively. The whole_setOfPoints is a list of lists, where the
# lists are x, y, and z coordinates, respectively; the fourth list has
# the tags of the points

def create_cuboidPoints(cuboid_points, whole_setOfPoints=[[],[],[],[]], 
tolerance=1E-8, lc=0.1):
    
    # Counts the number of points in the whole set of already created 
    # points

    n_oldPoints = len(whole_setOfPoints[0])

    # Creates a list of points for this new cuboid

    new_points = []
    
    # Iterates through the cuboid points, whose points are intended to 
    # be added

    for i in range(cuboid_points.shape[1]):

        # Iterates through the whole set of points to check if this 
        # point has already been added

        flag_found = False

        for j in range(n_oldPoints):

            # Evaluates the distance between those two points

            dx = cuboid_points[0,i]-whole_setOfPoints[0][j]

            dy = cuboid_points[1,i]-whole_setOfPoints[1][j]

            dz = cuboid_points[2,i]-whole_setOfPoints[2][j]

            distance = np.sqrt((dx*dx)+(dy*dy)+(dz*dz))

            # If the distance is less than the tolerance, the point is
            # considered equal to the one already created

            if distance<tolerance:

                # Makes this new point equal to the old one

                new_points.append(whole_setOfPoints[3][j])

                # Updates the flag for this point

                flag_found = True

        # If the point was not found, create it

        if not flag_found:

            point_tag = gmsh.model.geo.addPoint(cuboid_points[0,i
            ], cuboid_points[1,i], cuboid_points[2,i],lc)

            new_points.append(point_tag)

            # Updates the array of points

            whole_setOfPoints[0].append(cuboid_points[0,i])

            whole_setOfPoints[1].append(cuboid_points[1,i])

            whole_setOfPoints[2].append(cuboid_points[2,i])

            whole_setOfPoints[3].append(point_tag)

            # Updates the number of points in the whole set

            n_oldPoints += 1

    gmsh.model.geo.synchronize()

    # Returns the cuboid points dictionary and the list with the whole
    # set of points

    return new_points, whole_setOfPoints

########################################################################
#                    Entities' ordering permutation                    #
########################################################################

# Cuboid must be connected in a characteristic order, so that the trans-
# finite mesh is consistent throughout the whole domain

# Defines a function to test reordinations of a vector of indices that 
# is isotropic, i.e., if the test vector is one of these permutations 
# of the true vector, there is no need to change orientation

def test_permutationNeed(test_vector, true_vector):

    # Get the indexes of the test vector in the true vector

    indexes_inTest = indexes_betweenLists(test_vector, true_vector)

    # Defines a list of indexes that is isotropic

    isotropic_permutations = [[0,1,2,3], [2,3,0,1], [0,3,2,1], [2,1,0,3]]

    # Test whether the indexes are one of these permutations

    for permutation in isotropic_permutations:

        if indexes_inTest==permutation:

            return False 
        
    # If it is not one of these permutations, the surfaces must be rota-
    # ted indeed
        
    return True

# Defines a function to get the indexes of the element of a list in ano-
# ther list

def indexes_betweenLists(list1, true_list):

    # Initializes a list of indexes

    indexes_list = []

    # Iterates through the elements of the old list

    for element_true in true_list:

        for i in range(len(list1)):

            if element_true==list1[i]:

                indexes_list.append(i)

                break 

    return indexes_list

# Defines a function to reordinate the points dictionary 

def reordinate_dictionary(points_dictionary, local_numbering, 
opposite_numbering, local_newPoints, opposite_newPoints):
    
    #print("\n", points_dictionary)

    # If the points lists are the same, returns the original dictionary

    if local_newPoints==opposite_newPoints:

        print("The points on both surfaces have the same ordering\n")

        return points_dictionary
    
    # Initializes the new dictionary

    new_dictionary = dict()
    
    # Converts the points from the common surface

    for i in range(len(local_numbering)):

        # Gets the direct correspondent in the same surface

        new_dictionary[local_numbering[i]] = local_newPoints[i]
    
    # Converts the points from the opposite surface

    for i in range(len(opposite_numbering)):

        # Gets the direct correspondent in the same surface

        new_dictionary[opposite_numbering[i]] = opposite_newPoints[i]

    #print(new_dictionary, "\n")

    return new_dictionary

# Defines a function to reordinate the points

def reordinate_points(points_dictionary, opposite_points, old_pointsList,
new_pointsList, local_numbering, points_list, surface_index, verbose=
False):
    
    if verbose:
    
        print("\nOld points:", old_pointsList)

        print("New points:", new_pointsList)

        print("Local numbering:", local_numbering)

    # If the points lists are the same, returns the original dictionary

    if old_pointsList==new_pointsList:

        return points_dictionary
    
    # Initializes the new dictionary

    new_dictionary = dict()
    
    # Converts the points

    for i in range(len(local_numbering)):

        # Gets the direct correspondent in the same surface

        new_dictionary[local_numbering[i]] = new_pointsList[i]

        for j in range(len(local_numbering)):

            if points_list[local_numbering[j]-1]==new_pointsList[i]:

                # Gets the correspondent in the opposite surface

                new_dictionary[opposite_points[surface_index][
                local_numbering[i]-1]] = points_dictionary[
                opposite_points[surface_index][local_numbering[j]-1]]

                break

    return new_dictionary

# Defines a function to reorder a list given a list of indexes

def reordinate_list(list, indexes_list):

    new_list = []

    for index in indexes_list:

        new_list.append(list[index])

    return new_list

########################################################################
#                               Coloring                               #
########################################################################

# Defines a function to interpolate colors using an integer input from 0
# to n, where n is the maximum given integer

def color_interpolation(input, maximum_input, initial_RGBColor=[0,0,255], 
final_RGBColor=[255,0,0]):
    
    # If the maximum input is null or the input is null, returns white

    if maximum_input==0 or input==0:

        return [255,255,255]
    
    # Gets the normalized input

    input = input/maximum_input

    # Interpolates the colors

    color = [0,0,0]

    for i in range(3):

        color[i] = int(np.ceil((((1-input)*initial_RGBColor[i])+(input*
        final_RGBColor[i]))))

    # Returns the color

    return color

########################################################################
#                                Testing                               #
########################################################################

def test_reordenation():

    list = [1,2,3,4,5,6,7,8,9,10]

    true_list = [7, 5, 2, 10, 1, 9, 4, 6, 3, 8]

    print("Original list:", list, "\n")

    print("True list:", true_list, "\n")
    
    indexes_list = indexes_betweenLists(list, true_list)

    new_list = reordinate_list(list, indexes_list)

    print("Indexes:", indexes_list)

    print("New list:", new_list, "\n")

def test_cuboid():

    gmsh.initialize()

    # Defines the characteristic length of the mesh

    lc = 0.5

    x = 15

    y = 9

    z = 10

    center_point = [[0.5*x], [0.5*y], [0.5*z]]

    # The ellipse points are the center and the major axis respectively

    ellipse_points = [[0.5*x, 0.5*x], [0.5*y, -x], [z,z]]

    spline_points = [[0.3*x, 0.6*x], [1.3*y, 0.8*y], [1.1*z, 0.6*z]]

    lines_instructions = dict()

    lines_instructions[5] = ["circle arc", center_point]

    lines_instructions[7] = ["elliptic arc", ellipse_points]

    lines_instructions[6] = ["spline", spline_points]

    # Sets the transfinite directions (x, y, and z)

    transfinite_directions = [4,5,6]

    # Sets the bias

    bias_directions = dict()

    bias_directions["x"] = 1
    
    bias_directions["y"] = 1.5
    
    bias_directions["z"] = 1.2

    # Sets the corner points

    cuboid_points = np.array([[x, x, 0.0, 0.0, x, x, 0.0, 0.0], [0.0, y,
    y, 0.0, 0.0, y, y, 0.0], [0.0, 0.0, 0.0, 0.0, z, z, z, z]])

    points_list, whole_setOfPoints = create_cuboidPoints(cuboid_points)

    print(whole_setOfPoints)

    points_list, whole_setOfPoints = create_cuboidPoints(cuboid_points,
    whole_setOfPoints=whole_setOfPoints)

    print(whole_setOfPoints)

    whole_setOfLines = create_cuboidLines(points_list, lines_instructions=
    lines_instructions, transfinite_directions=transfinite_directions,
    bias_directions=bias_directions)

    print(whole_setOfLines)

    whole_setOfLines = create_cuboidLines(points_list, lines_instructions=
    lines_instructions)

    print(whole_setOfLines)

    # Sets the instructions for the surfaces ("plane surface" or "surfa-
    # ce filling")

    surface_instruction = ["plane surface", "plane surface", 
    "plane surface", "plane surface", "plane surface", "plane surface"]

    # Creates the cuboid

    volume, whole_setOfSurfaces, surfaces_cornersDict = create_cuboidFromLines(
    points_list, whole_setOfLines, surface_instruction=surface_instruction, 
    flag_transfinite=True, hexahedron_mesh=False)

    print(whole_setOfSurfaces)

    #volume, whole_setOfSurfaces, surfaces_list = create_cuboidFromLines(
    #points_list, whole_setOfLines, surface_instruction=surface_instruction, 
    #flag_transfinite=True, hexahedron_mesh=False, whole_setOfSurfaces=
    #whole_setOfSurfaces)

    #print(whole_setOfSurfaces)

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

def test_makeCuboid():

    gmsh.initialize()

    # Defines the characteristic length of the mesh

    lc = 1.0

    x = 15

    y = 9

    z = 10

    center_point = [[0.5*x], [0.5*y], [0.5*z]]

    # The ellipse points are the center and the major axis respectively

    ellipse_points = [[0.5*x, 0.5*x], [0.5*y, -x], [z,z]]

    spline_points = [[0.3*x, 0.6*x], [1.3*y, 0.8*y], [1.1*z, 0.6*z]]

    lines_instructions = dict()

    lines_instructions[5] = ["circle arc", center_point]

    lines_instructions[7] = ["elliptic arc", ellipse_points]

    lines_instructions[6] = ["spline", spline_points]

    # Sets the transfinite directions (x, y, and z)

    transfinite_directions = [4,5,6]

    # Sets the bias

    bias_directions = dict()

    bias_directions["x"] = 1
    
    bias_directions["y"] = 1.5
    
    bias_directions["z"] = 1.2

    # Sets the corner points

    cuboid_points = [[x, x, 0.0, 0.0, x, x, 0.0, 0.0], [0.0, y, y, 0.0, 
    0.0, y, y, 0.0], [0.0, 0.0, 0.0, 0.0, z, z, z, z]]

    make_cuboid(cuboid_points, lc=lc, transfinite_directions=
    transfinite_directions, bias_directions=bias_directions, 
    lines_instructionsOriginal=lines_instructions, rotation_vector=[0.0, 0.0, 
    0.5*np.pi], translation_vector=[5.0, 6.0, 7.0])

    # Deletes the duplicated entities

    gmsh.model.geo.removeAllDuplicates()

    gmsh.model.geo.synchronize()

    # Generates a 3D mesh

    gmsh.model.mesh.generate(3)

    gmsh.fltk.run()

    gmsh.finalize()

def test_twoCuboids():

    gmsh.initialize()

    # Defines the characteristic length of the mesh

    lc = 1.0

    x = 15

    y = 9

    z = 10

    center_point = [[x], [0.5*y], [0.8*z]]

    # The ellipse points are the center and the major axis respectively

    r = np.sqrt(((0.2*z)**2)+((0.5*y)**2))

    print(r/z)

    #ellipse_points = [[0.0, 0.0], [0.5*y, 0.5*y], [0.8*z,(0.8*z)+r]]

    center_point2 = [[0.0], [0.5*y], [0.8*z]]

    lines_instructions = dict()

    lines_instructions[5] = ["circle arc", center_point]

    lines_instructions[7] = ["circle arc", center_point2]

    # Sets the transfinite directions (x, y, and z)

    transfinite_directions = [4,5,6]

    # Sets the bias

    bias_directions = dict()

    bias_directions["x"] = 1
    
    bias_directions["y"] = 1.5
    
    bias_directions["z"] = 1.2

    # Sets the corner points

    cuboid_points = [[x, x, 0.0, 0.0, x, x, 0.0, 0.0], [0.0, y, y, 0.0, 
    0.0, y, y, 0.0], [0.0, 0.0, 0.0, 0.0, z, z, z, z]]

    surfaces_instructions = ["plane surface", "plane surface", 
    "plane surface", "plane surface", "plane surface", "surface filling"]

    volume1, whole_setOfPoints, whole_setOfLines, whole_setOfSurfaces, surfaces_list = make_cuboid(
    cuboid_points, lc=lc, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions, lines_instructionsOriginal=lines_instructions, 
    rotation_vector=[0.0, 0.0, 0.5*np.pi], translation_vector=[5.0, 6.0, 
    7.0], surfaces_instructions=surfaces_instructions)
    
    volume2, whole_setOfPoints, whole_setOfLines, whole_setOfSurfaces, surfaces_list = make_cuboid(
    cuboid_points, lc=lc, transfinite_directions=transfinite_directions, 
    bias_directions=bias_directions, lines_instructionsOriginal=lines_instructions, 
    rotation_vector=[0.0, 0.0, 0.5*np.pi], translation_vector=[5.0, 21.0, 
    7.0], surfaces_instructions=surfaces_instructions, whole_setOfPoints=
    whole_setOfPoints, whole_setOfLines=whole_setOfLines, 
    whole_setOfSurfaces=whole_setOfSurfaces)

    # Deletes the duplicated entities

    gmsh.model.geo.removeAllDuplicates()

    gmsh.model.geo.synchronize()

    # Generates a 3D mesh

    gmsh.model.mesh.generate(3)

    gmsh.fltk.run()

    gmsh.finalize()

#test_cuboid()

#test_makeCuboid()

#test_twoCuboids()

#test_reordenation()