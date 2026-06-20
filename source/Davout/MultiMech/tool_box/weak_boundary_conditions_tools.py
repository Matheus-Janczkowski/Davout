# Routine to store methods to generate boundary conditions weakly impo-
# sed using Lagrange multipliers

from dolfin import *

import numpy as np

from scipy.linalg import sqrtm, logm

from .numerical_tools import generate_loadingParametricCurves, rotation_tensorEulerRodrigues

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

    if ((not isinstance(average_gradient, list)) or len(
    average_gradient)!=3):
        
        raise ValueError("'average_gradient' in 'VolumetricAverage"+
        "OfDisplacementGradient' must be a list in the format [[du"+
        "1/dX1, du1/dX2, du1/dX3], [du2/dX1, du2/dX2, du2/dX3], [d"+
        "u3/dX1, du3/dX2, du3/dX3]]. Currently, it is not a list o"+
        "r does not have the right format:\n"+str(average_gradient))
    
    for i, row in enumerate(average_gradient):

        if ((not isinstance(row, list)) or len(row)!=3):
            
            raise ValueError("'average_gradient' in 'VolumetricAve"+
            "rageOfDisplacementGradient' must be a list in the for"+
            "mat [[du1/dX1, du1/dX2, du1/dX3], [du2/dX1, du2/dX2, "+
            "du2/dX3], [du3/dX1, du3/dX2, du3/dX3]]. Currently, th"+
            "e "+str(i+1)+"-th row is not a list or does not have "+
            "the right format:\n"+str(row))
        
    # Saves the final average gradient to be enforced

    final_average_gradient = np.array(average_gradient)

    # Gets the corresponding average deformation gradient

    deformation_gradient = final_average_gradient+np.eye(3)

    # Gets the polar decomposition of the deformation gradient

    stretch_tensor = sqrtm(deformation_gradient.T@deformation_gradient)

    rotation_skew_tensor = logm(deformation_gradient@np.linalg.inv(
    stretch_tensor))

    rotation_axis = np.array([-rotation_skew_tensor[1,2], 
    rotation_skew_tensor[0,2], -rotation_skew_tensor[0,1]])
        
    # Creates a FEniCS constant for the average gradient

    current_average_gradient = Constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0
    ], [0.0, 0.0, 0.0]])
        
    # Creates a class for the load to be updated in time

    class TimeUpdate:

        def __init__(self, t_final, current_average_gradient, 
        stretch_tensor, rotation_axis, parametric_load_curve):

            self.maximum_time = t_final

            self.current_average_gradient = current_average_gradient
            
            self.stretch_tensor = stretch_tensor

            self.rotation_axis = rotation_axis

            self.parametric_load_curve = parametric_load_curve

        # Defines a function to update the average gradient

        def update_load(self, t):

            load_interpolation = self.parametric_load_curve(t/
            self.maximum_time)
            
            # Gets the interpolated stretch tensor and the rotation axis

            new_stretch_tensor = ((np.eye(3)*(1.0-load_interpolation))+(
            self.stretch_tensor*load_interpolation))

            new_rotation_axis = self.rotation_axis*load_interpolation

            # Gets the new rotation tensor

            new_rotation_tensor = rotation_tensorEulerRodrigues(
            new_rotation_axis.tolist(), return_numpy_array=True)

            # Gets the corresponding right Cauchy-Green deformation ten-
            # sor

            F = new_rotation_tensor@new_stretch_tensor

            stretches = np.linalg.eigvals(new_stretch_tensor)

            # Gets the corresponding displacement gradient

            new_value = F-np.eye(3)

            print("Updates the imposed displacement gradient to:\n"+str(
            new_value)+",\nthe corresponding stretches are:\n"+str(
            stretches)+"\n")

            self.current_average_gradient.assign(Constant(new_value))

    # Instantiates the time updating class

    time_update = TimeUpdate(t_final, current_average_gradient, 
    stretch_tensor, rotation_axis, parametric_load_curve)
    
    # Adds a finite element for the Lagrange multiplier to enforce the
    # constraint on the volumetric average of the displacement gradient

    elements_dictionary["Lagrange multiplier displacement gradient ave"+
    "rage"] = {"field type": "tensor", "interpolation function": "Real", 
    "polynomial degree": 0}

    # Defines a class for the variational form

    class VariationalFormAverageOfDisplacementGradient:

        def __init__(self, average_gradient):
            
            self.average_gradient = average_gradient

        # Defines a function to generate the corresponding portion of 
        # the variational form

        def variational_form_update(self, functional_data_class):

            return ((inner(functional_data_class.variation_fields["Lag"+
            "range multiplier displacement gradient average"], 
            self.average_gradient-grad(
            functional_data_class.solution_fields["Displacement"]))
            *mesh_dataClass.dx)-(inner(
            functional_data_class.solution_fields["Lagrange multiplier"+
            " displacement gradient average"], grad(
            functional_data_class.variation_fields["Displacement"]))*
            mesh_dataClass.dx))
        
    # Returns the class and the dictionary of finite elements

    return (VariationalFormAverageOfDisplacementGradient(
    current_average_gradient), time_update, elements_dictionary)