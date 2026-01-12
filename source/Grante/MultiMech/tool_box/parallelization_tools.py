# Routine to store methods to help with parallelized tasks

from dolfin import *  

########################################################################
#                             Communicator                             #
########################################################################                             

# Defines a function to create the communicator object, comm, to commu-
# nicate between processors

def mpi_create_comm(automatic_comm_generation, comm=None):

    if automatic_comm_generation and (comm is None):

        comm = MPI.comm_world

    return comm

########################################################################
#                               Barrier                                #
########################################################################

# Defines a function to create a barrier to synchronize all processors

def mpi_barrier(comm):

    if comm is not None:

        MPI.barrier(comm)

########################################################################
#                               Printing                               #
########################################################################

# Defines a function to print with just the first processor

def mpi_print(comm, *args, **kwargs):

    if (comm is None) or (MPI.rank(comm)==0):

        print(*args, **kwargs, flush=True)

########################################################################
#                            File generation                           #
########################################################################

# Defines a function to create a xdmf file with or without the communi-
# cator object

def mpi_xdmf_file(comm, filename, add_termination=False):

    # Adds the termination

    if add_termination:

        filename += ".xdmf"

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
    
    # If no parallel execution is required, the comm object is None. 
    # Then, the execution is as normal
    
    if comm is None:

        return function(*positional_arguments, **keyword_arguments)
    
    # Otherwise, the processor must be evaluated

    rank = MPI.rank(comm)

    # Initializes a result object and computes it only if the first pro-
    # cessor is asking for it

    result = None

    if rank==0:

        result = function(*positional_arguments, **keyword_arguments)

    # Broadcasts result to all processors, so all of them have the same
    # result

    result = comm.bcast(result, root=0)

    return result
    
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
