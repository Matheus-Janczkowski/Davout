# Routine to plot perspective lines coming out ot vanishing points

import numpy as np

from ..tool_box.collage_classes import ColorMiscellany

# Defines a function to plot perspective lines from a list of vanishing
# points. Each point has a corresponding dictionary

def perspective_lines_from_vanishing_points(axes_object, 
vanishing_points_list, thickness, depth_order, x_min, x_max, y_min, 
y_max, verbose=False):
    
    # Instantiates the color class

    colors_class = ColorMiscellany()

    # Iterates through the list of vanishing points

    for index_point, point_dict in enumerate(vanishing_points_list):

        # Sets a list of necessary keys

        necessary_keys = ["coordinates", "rays central direction", "an"+
        "gle amplitude", "number of rays"]

        # Sets a list of optional keys

        optional_keys = ["color", "thickness"]

        # Verifies if it is a dictionary

        if not isinstance(point_dict, dict):

            obligatory_keys = "Obligatory:"

            for key in necessary_keys:

                obligatory_keys += "\n'"+str(key)+"'"

            opt_keys = "\n\nOptional:"

            for key in optional_keys:

                opt_keys += "\n'"+str(key)+"'"

            raise TypeError("The "+str(index_point+1)+"-th vanishing p"+
            "oint is not a dictionary. It must be a dictionary with th"+
            "e following keys:\n"+obligatory_keys+opt_keys)

        # Verifies if any of the keys if not there

        for key in necessary_keys:

            if not (key in point_dict):

                keys = ""

                for key in necessary_keys:

                    keys += "\n'"+str(key)+"'"

                opt_keys = "\n\nOptional:"

                for key in optional_keys:

                    opt_keys += "\n'"+str(key)+"'"

                raise ValueError("The key '"+str(key)+"' was not found"+
                " in the "+str(index_point+1)+"-th vanishing point. Ch"+
                "eck the necessary keys for creating perspective lines"+
                "/rays:"+keys+opt_keys)
            
        # Verifies if the color was given

        color = 'red'

        if "color" in point_dict:

            color = colors_class(point_dict["color"])
            
        # Verifies if the thickness was given

        if "thickness" in point_dict:

            thickness = point_dict["thickness"]

            # Verifies if it is a number

            if (not isinstance(thickness, float)) and (not isinstance(
            thickness, int)):
                
                raise TypeError("The 'thickness' key provided to creat"+
                "e the "+str(index_point+1)+"-th vanishing point is no"+
                "t a float neither an integer. Currently, it is: "+str(
                thickness))
            
        # Gets the vanishing points coordinates

        coordinates = point_dict["coordinates"]

        # Verifies if the coordinates is a list

        if (not isinstance(coordinates, list)) or (len(coordinates)!=2):

            raise TypeError("The 'coordinates' key provided to create "+
            "the "+str(index_point+1)+"-th vanishing point is not a li"+
            "st with 2 elements within, such as [x, y]. Currently, it "+
            "is: "+str(coordinates))
        
        # Transforms the coordinates vector to a numpy array

        coordinates = np.array(coordinates)
        
        # Gets the rays central direction

        rays_central_direction = point_dict["rays central direction"]

        # Verifies if it is a list

        if (not isinstance(rays_central_direction, list)) or (len(
        rays_central_direction)!=2):

            raise TypeError("The 'rays central direction' key provided"+
            " to create the "+str(index_point+1)+"-th vanishing point "+
            "is not a list with 2 elements within, such as [x, y]. Cur"+
            "rently it is: "+str(rays_central_direction))
        
        # Transforms the ray central direction to a numpy array, and
        # normalizes it, then scales by the maximum ray length

        rays_central_direction = np.array(rays_central_direction)

        rays_central_direction = (1/np.linalg.norm(rays_central_direction
        )*rays_central_direction)
        
        # Gets the angle amplitude

        angle_amplitude = point_dict["angle amplitude"]

        # Verifies if it is a number

        if (not isinstance(angle_amplitude, float)) and (not isinstance(
        angle_amplitude, int)):
            
            raise TypeError("The 'angle amplitude' key provided to cre"+
            "ate the "+str(index_point+1)+"-th vanishing point is not "+
            "a float neither an integer. Currently, it is: "+str(
            angle_amplitude))
        
        # Gets the number of rays

        number_of_rays = point_dict["number of rays"]

        # Verifies if it is an integer

        if (not isinstance(number_of_rays, int)) or number_of_rays<2:
            
            raise TypeError("The 'number of rays' key provided to crea"+
            "te the "+str(index_point+1)+"-th vanishing point is not a"+
            "n integer or is less than 2. Currently, it is: "+str(
            number_of_rays))
        
        # Iterates through the number of points. But forces an odd number
        # of points to have a ray over the central direction

        half_number_of_points = int(np.floor(number_of_rays*0.5))+1

        for i in range(half_number_of_points):

            ############################################################
            #                         Right ray                        #
            ############################################################

            # Calculates the angle to this ray, and converts to radians

            ray_angle = angle_amplitude*(i/(half_number_of_points-1))*(
            np.pi/180)

            # Calculates the rotation matrix for the angle to the right

            R = np.array([[np.cos(ray_angle), -np.sin(ray_angle)], [
            np.sin(ray_angle),  np.cos(ray_angle)]])

            # Gets the initial and final points on the boundary

            initial_point, final_point = clip_lines(R@(
            rays_central_direction), coordinates, x_min, x_max, y_min,
            y_max)

            # Plots the right ray

            if initial_point is not None:

                axes_object.plot([initial_point[0], final_point[0]], [
                initial_point[1], final_point[1]], color=color, 
                linewidth=thickness, zorder=depth_order, clip_on=True)

            ############################################################
            #                         Left ray                         #
            ############################################################

            # Calculates the rotation matrix for the angle to the left

            R = np.array([[np.cos(-ray_angle), -np.sin(-ray_angle)], [
            np.sin(-ray_angle),  np.cos(-ray_angle)]])

            # Skips the left ray if the angle is 0

            if abs(ray_angle)>1E-7:

                # Gets the initial and final points on the boundary

                initial_point, final_point = clip_lines(R@(
                rays_central_direction), coordinates, x_min, x_max, 
                y_min, y_max)

                # Plots the left ray

                if initial_point is not None:

                    axes_object.plot([initial_point[0], final_point[0]], 
                    [initial_point[1], final_point[1]], color=color, 
                    linewidth=thickness, zorder=depth_order, clip_on=
                    True)

    # Returns the plotted object

    if verbose:

        print("Finishes printing the perspective lines\n")

    return axes_object

# Defines a function to clip lines overextending the canvas

def clip_lines(point_direction, origin, x_min, x_max, y_min, y_max, 
tolerance=1E-8, throw_error=False):

    # Makes sure the direction is unitary

    point_direction = ((1/np.linalg.norm(point_direction))*
    point_direction)

    # Initializes a list of facet normals

    normals = [np.array([0.0, -1.0]), np.array([1.0, 0.0]), np.array([
    0.0, 1.0]), np.array([-1.0, 0.0])]

    # Initializes a list of points in each one of the facets

    points_in_plane = [np.array([0.5*(x_min+x_max), y_min]), np.array([
    x_max, 0.5*(y_min+y_max)]), np.array([0.5*(x_min+x_max), y_max]), 
    np.array([x_min, 0.5*(y_min+y_max)])]

    # Initializes a list of step lengths to cross each facet from the o-
    # rigin point given the direction

    alphas = []

    # Initializes a list of points

    cross_points = []

    # Iterates through the facets

    for normal, point_in_plane in zip(normals, points_in_plane):

        # Gets the shortest path to the plane

        shortest_path = get_closest_point_in_plane(origin, normal, 
        point_in_plane)

        # Evaluates the step length of the direction to break into the
        # plane by making the sum of this direction with the normal vec-
        # tor orthogonal to the latter

        alpha = -(np.dot(shortest_path, shortest_path)/(np.dot(
        point_direction, shortest_path)+1E-9))

        # If the step length is negative, discard it

        if alpha>0.0:

            # Gets the candidate to a cross point

            cross_point = origin+(alpha*point_direction)

            # If the components are within bounds, it is a valid cross 
            # point

            if ((cross_point[0]>=(x_min-tolerance) and cross_point[0]<=(
            x_max+tolerance)) and (cross_point[1]>=(y_min-tolerance
            ) and cross_point[1]<=(y_max+tolerance))):
                
                # Verifies if the list of alphas is empty

                if len(alphas)==0:

                    # Appends it

                    alphas.append(alpha)

                    cross_points.append(cross_point)

                # Otherwise, verifies if the alpha is smaller than the
                # previous one

                elif alpha<alphas[0]:

                    alphas = [alpha, alphas[0]]

                    cross_points = [cross_point, cross_points[0]]

                else:

                    alphas.append(alpha)

                    cross_points.append(cross_point)

    # If only one valid alpha has been found, returns the origin point 
    # as well

    if len(alphas)==1:

        cross_points = [origin, cross_points[0]]

    # Otherwise, something wrong happened
    
    elif len(alphas)!=2:

        if throw_error:

            raise ValueError("It was impossible to find a point crossi"+
            "ng the box defined by "+str(x_min)+" <= x <= "+str(x_max)+
            " and "+str(y_min)+" <= y <= "+str(y_max)+" from the origi"+
            "n point "+str(origin)+" using the direction "+str(
            point_direction)) 
        
        return None, None

    return cross_points[0], cross_points[1]

# Defines a function to calculate the vector of a point to the closest
# point in a plane

def get_closest_point_in_plane(origin, outward_plane_normal, plane_point):

    # Makes sure the plane normal is unitary

    outward_plane_normal = ((1/np.linalg.norm(outward_plane_normal))*
    outward_plane_normal)

    # Gets the vector from the origin to the point in plane

    vector = plane_point-origin

    # Calculates the inner product with the plane normal, and projects
    # onto the direction of the normal

    shortest_direction = (np.dot(vector, outward_plane_normal)*
    outward_plane_normal)

    # Return the shortest direction. Multiplies by -1 since the normal
    # points outwardly

    return -shortest_direction