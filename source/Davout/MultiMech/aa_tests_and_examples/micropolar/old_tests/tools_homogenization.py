# Homogenization tools module

from dolfin import *
import numpy as np

def get_micro_cell_volume(dx):
    RVE_volume = assemble(1*dx(3)) + assemble(1*dx(4))
    return RVE_volume

def get_homogenized(variable, dx):
    # Calculate the volume of the Representative Volume Element (RVE)
    RVE_volume = get_micro_cell_volume(dx)

    # Initialize the homogenized deformation gradient tensor
    Homogenized_variable = np.zeros(3) 

    for m in range(3): 
        # Compute homogenized components
        Homogenized_variable[m] = (
            (1 / RVE_volume) * assemble(variable[m] * dx(3)) + 
            (1 / RVE_volume) * assemble(variable[m] * dx(4))
        )

    return Homogenized_variable 

def get_homogenized_gradient(grad_variable, dx):
    # Calculate the volume of the Representative Volume Element (RVE)
    RVE_volume = get_micro_cell_volume(dx)

    # Initialize the homogenized deformation gradient tensor
    Homogenized_gradient = np.zeros((3, 3)) 

    # Loop through tensor components
    for m in range(3): 
        for n in range(3):
            # Compute homogenized components
            Homogenized_gradient[m, n] = (
                (1 / RVE_volume) * assemble(grad_variable[m, n] * dx(3)) + 
                (1 / RVE_volume) * assemble(grad_variable[m, n] * dx(4))
            )

    return Homogenized_gradient 

