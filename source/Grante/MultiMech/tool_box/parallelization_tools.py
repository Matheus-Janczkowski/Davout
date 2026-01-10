# Routine to store methods to help with parallelized tasks

from dolfin import *

########################################################################
#                               Printing                               #
########################################################################

# Defines a function to print with just the first processor

def mpi_print(comm, *args, **kwargs):

    if comm is None or (MPI.rank(comm)==0):

        print(*args, **kwargs, flush=True)

########################################################################
#                            File generation                           #
########################################################################

# Defines a function to create a xdmf file with or without the communi-
# cator object

def mpi_xdmf_file(comm, filename):

    if comm is None:

        return XDMFFile(filename)
    
    else:

        return XDMFFile(comm, filename)

########################################################################
#                          Function execution                          #
########################################################################
    
# Defines a function to execute another function at the first processor
# only

def mpi_execute_function(comm, function, *positional_arguments, 
**keyword_arguments):
    
    if (comm is None) or MPI.rank(comm) == 0:

        return function(*positional_arguments, **keyword_arguments)
    
########################################################################
#                              FEM fields                              #
########################################################################

# Defines a function to evaluate a field at a point
    
def mpi_evaluate_field_at_point(comm, field_function, point):

    # Evaluates at the point

    point_value = field_function(point)

    # Verifies if the communicator object is not None, then, sums the
    # value, in between processors, since only one processor owns this
    # point

    if comm is not None:

        point_value = MPI.sum(comm, point_value)

    return point_value
