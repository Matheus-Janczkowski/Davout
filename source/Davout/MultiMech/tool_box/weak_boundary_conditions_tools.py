# Routine to store methods to generate boundary conditions weakly impo-
# sed using Lagrange multipliers

from dolfin import *

import numpy as np

from .numerical_tools import generate_loadingParametricCurves

# Defines a function to weakly impose the volumetric average of the dis-
# placement gradient

def VolumetricAverageOfDisplacementGradient(mesh_dataClass, 
elements_dictionary, average_gradient, parametric_load_curve="linear",
t_final=1.0):
    
    # Verifies if the parametric load curve is a string and, then, uses
    # it to convert to an actual function

    if isinstance(parametric_load_curve, str):

        parametric_load_curve = generate_loadingParametricCurves(
        parametric_load_curve)
    
    # Verifies if the average gradient is a list

    if ((not isinstance(average_gradient, list)) or len(average_gradient
    )!=3):
        
        raise ValueError("'average_gradient' in 'VolumetricAverageOfDi"+
        "splacementGradient' must be a list in the format [[du1/dX1, d"+
        "u1/dX2, du1/dX3], [du2/dX1, du2/dX2, du2/dX3], [du3/dX1, du3/"+
        "dX2, du3/dX3]]. Currently, it is not a list or does not have "+
        "the right format:\n"+str(average_gradient))
    
    for i, row in enumerate(average_gradient):

        if ((not isinstance(row, list)) or len(row)!=3):
            
            raise ValueError("'average_gradient' in 'VolumetricAverage"+
            "OfDisplacementGradient' must be a list in the format [[du"+
            "1/dX1, du1/dX2, du1/dX3], [du2/dX1, du2/dX2, du2/dX3], [d"+
            "u3/dX1, du3/dX2, du3/dX3]]. Currently, the "+str(i+1)+"-t"+
            "h row is not a list or does not have the right format:\n"+
            str(row))
        
    # Creates a FEniCS constant for the average gradient

    current_average_gradient = Constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0
    ], [0.0, 0.0, 0.0]])

    final_average_gradient = np.array(average_gradient)
        
    # Creates a class for the load to be updated in time

    class TimeUpdate:

        def __init__(self, t_final, current_average_gradient, 
        final_average_gradient, parametric_load_curve):

            self.maximum_time = t_final

            self.current_average_gradient = current_average_gradient
            
            self.final_average_gradient = final_average_gradient

            self.parametric_load_curve = parametric_load_curve

        # Defines a function to update the average gradient

        def update_load(self, t):

            self.current_average_gradient.assign(Constant(
            self.parametric_load_curve(t/self.maximum_time)*
            self.final_average_gradient))

    # Instantiates the time updating class

    time_update = TimeUpdate(t_final, current_average_gradient, 
    final_average_gradient, parametric_load_curve)
    
    # Adds a finite element for the Lagrange multiplier to enforce the
    # constraint on the volumetric average of the displacement gradient

    elements_dictionary["Lagrange multiplier displacement gradient ave"+
    "rage"] = {"field type": "tensor", "interpolation function": "Real", 
    "polynomial degree": 0}

    # Defines a class for the variational form

    class VariationalFormAverageOfDisplacementGradient:

        def __init__(self, average_gradient):
            
            self.average_gradient = average_gradient

            # Evaluates the volume of the domain

            self.volume_inverse = 1.0/assemble(1.0*mesh_dataClass.dx)

        # Defines a function to generate the corresponding portion of 
        # the variational form

        def variational_form_update(self, functional_data_class):

            return ((inner(functional_data_class.solution_fields["Lagra"+
            "nge multiplier displacement gradient average"], 
            self.volume_inverse*grad(
            functional_data_class.variation_fields["Displacement"]))*
            mesh_dataClass.dx)+(inner(
            functional_data_class.variation_fields["Lagrange multiplie"+
            "r displacement gradient average"], self.volume_inverse*(
            grad(functional_data_class.solution_fields["Displacement"])
            -self.average_gradient))*mesh_dataClass.dx))
        
    # Returns the class and the dictionary of finite elements

    return (VariationalFormAverageOfDisplacementGradient(
    current_average_gradient), time_update, elements_dictionary)