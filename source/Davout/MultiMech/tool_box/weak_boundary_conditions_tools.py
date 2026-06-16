# Routine to store methods to generate boundary conditions weakly impo-
# sed using Lagrange multipliers

from dolfin import *

# Defines a function to weakly impose the volumetric average of the dis-
# placement gradient

def VolumetricAverageOfDisplacementGradient(mesh_dataClass, 
elements_dictionary, average_gradient):
    
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
    
    # Adds a finite element for the Lagrange multiplier to enforce the
    # constraint on the volumetric average of the displacement gradient

    elements_dictionary["Lagrange multiplier displacement gradient ave"+
    "rage"] = {"field type": "tensor", "interpolation function": "Real", 
    "polynomial degree": 0}

    # Defines a class for the variational form

    class VariationalFormAverageOfDisplacementGradient:

        def __init__(self, average_gradient):
            
            self.average_gradient = Constant(average_gradient)

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

    return (VariationalFormAverageOfDisplacementGradient(average_gradient
    ), elements_dictionary)