# Routine to store tools to create loads

from dolfin import *

from ..tool_box import numerical_tools

########################################################################
#                          Referential forces                          #
########################################################################

# Defines a function to construct a uniform body force on a domain in 
# the referential configuration

def UniformReferentialBodyForce(amplitude_bodyX, amplitude_bodyY,
amplitude_bodyZ, physical_group, t=0.0, t_final=1.0, 
parametric_load_curve="linear"):
    
    # Verifies if the parametric load curve is a string and, then, uses
    # it to convert to an actual function

    if isinstance(parametric_load_curve, str):

        parametric_load_curve = numerical_tools.generate_loadingParametricCurves(
        parametric_load_curve)

    # Creates the time constants

    time_constant = Constant(t)

    maximum_time = Constant(t_final)

    # Evaluates the body force vector at the referential configuration

    B = Constant([0.0, 0.0, 0.0])

    class TimeUpdate:

        def __init__(self):
            
            self.t = time_constant

            self.B = B

        # This class must have the update_load method to be called in 
        # the stepping algorithm

        def update_load(self, t):

            # Updates the time constant

            self.t.assign(Constant(t))

            # Evaluates the parametric load curve

            load_value = parametric_load_curve(self.t/maximum_time)

            # Updates the vector

            self.B.assign(Constant([load_value*amplitude_bodyX, 
            load_value*amplitude_bodyY, load_value*amplitude_bodyZ]))

            # Annuntiates the corrections

            print("\nUniform referential body force at physical group "+
            "'"+str(physical_group)+"':\n"+str(float(load_value*100))+
            "% of the final load is applied using the parametric load "+
            "curve\n")

    time_update = TimeUpdate()

    # Returns the body force vector and the time constant

    return B, time_update

########################################################################
#                           Follower forces                            #
########################################################################