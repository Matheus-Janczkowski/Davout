# Routine to store conversion functionalities between coordinate systems

import numpy as np

# Defines a function to convert from cartesian to cylindrical coordina-
# tes

def cartesian_to_cylindrical_coordinates(x, y, z):

    # Tests if x is null

    if abs(x)<1E-6:

        if y<0:

            return 1.5*np.pi, abs(y), z 
        
        else:

            return 0.5*np.pi, y, z 
        
    # Tests if it is the first or second quadrant

    if y>=0:

        # Tests if it is the first quadrant

        if x>0:

            return np.arctan(y/x), np.sqrt((x*x)+(y*y)), z

        # Otherwise, it is the second quadrant

        else:

            return np.arctan(y/x)+np.pi, np.sqrt((x*x)+(y*y)), z 
        
    # Tests if it is the third or fourth

    else:

        # Tests if it is the fourth quadrant

        if x>0:

            return np.arctan(y/x)+(2*np.pi), np.sqrt((x*x)+(y*y)), z

        # Otherwise, it is the third quadrant

        else:

            return np.arctan(y/x)+np.pi, np.sqrt((x*x)+(y*y)), z 