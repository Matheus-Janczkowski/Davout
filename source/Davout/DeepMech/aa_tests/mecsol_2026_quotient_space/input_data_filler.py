# Routine to store a method to fill in the load steps into the binary 
# file of the input data

import numpy as np

from scipy.linalg import sqrtm, logm

from .....Davout.PythonicUtilities.path_tools import get_parent_path_of_file

from .....Davout.MultiMech.tool_box.numerical_tools import rotation_tensorEulerRodrigues

def fill_load_steps(input_data_file, displacement_data_file, 
new_input_file, results_path, initial_time, final_time):

    # Loads the input data

    input_data = np.load(results_path+"//"+input_data_file)

    # Loads the displacement matrix

    displacement_matrix = np.load(results_path+"//"+
    displacement_data_file)

    # If the displacement file and the input data have the same number
    # of rows

    if displacement_matrix.shape[0]==input_data.shape[0]:

        print("The displacement matrix and the input data have the sam"+
        "e number of rows, "+str(displacement_matrix.shape[0])+". Noth"+
        "ing has to be done\n")

    # Verifies if the displacement file has n times more data samples
    # than the input data

    elif displacement_matrix.shape[0]%input_data.shape[0]==0:

        # Gets the number of load steps

        n_load_steps = int(displacement_matrix.shape[0]/
        input_data.shape[0])

        # Initializes a list of samples

        new_input_list = []

        # Iterates through the current samples

        for i in range(input_data.shape[0]):

            # Gets the sample input data

            sample_data = input_data[i,:]

            # Recovers the displacement gradient

            displacement_gradient = np.array([[sample_data[2], 
            sample_data[3], sample_data[4]], [sample_data[5], 
            sample_data[6], sample_data[7]], [sample_data[8], 
            sample_data[9], sample_data[10]]])

            # Gets the corresponding deformation gradient

            deformation_gradient = displacement_gradient+np.eye(3)

            # Gets the polar decomposition of the deformation gradient

            stretch_tensor = sqrtm(deformation_gradient.T@
            deformation_gradient)

            rotation_skew_tensor = logm(deformation_gradient@
            np.linalg.inv(stretch_tensor))

            rotation_axis = np.array([-rotation_skew_tensor[1,2], 
            rotation_skew_tensor[0,2], -rotation_skew_tensor[0,1]])

            # Iterates through the load steps

            for load_step in range(n_load_steps):

                # Assumes a linear interpolation of the load data 

                load_interpolation = initial_time+((final_time-
                initial_time)*(load_step/(n_load_steps-1)))
            
                # Gets the interpolated stretch tensor and the rotation 
                # axis

                new_stretch_tensor = ((np.eye(3)*(1.0-load_interpolation
                ))+(stretch_tensor*load_interpolation))

                new_rotation_axis = rotation_axis*load_interpolation

                # Gets the new rotation tensor

                new_rotation_tensor = rotation_tensorEulerRodrigues(
                new_rotation_axis.tolist(), return_numpy_array=True)

                # Gets the corresponding deformation gradient

                F = new_rotation_tensor@new_stretch_tensor

                # Gets the corresponding displacement gradient

                new_disp_grad = F-np.eye(3)

                # Adds this to the input data

                new_input_list.append([sample_data[0], sample_data[1],
                new_disp_grad[0,0], new_disp_grad[0,1], new_disp_grad[0,
                2], new_disp_grad[1,0], new_disp_grad[1,1], 
                new_disp_grad[1,2], new_disp_grad[2,0], new_disp_grad[2,
                1], new_disp_grad[2,2]])

        # Saves the new input matrix

        new_input_list = np.array(new_input_list)

        print("The new input data matrix has shape "+str(
        new_input_list.shape)+", whereas it had "+str(input_data.shape[0
        ])+" rows before.\nThe displacement matrix, on the other hand,"+
        " has "+str(displacement_matrix.shape[0])+"\n")

        np.save(results_path+"//"+new_input_file, new_input_list)

    else:

        raise IndexError("The displacement matrix has "+str(
        displacement_matrix.shape[0])+" rows, whereas the input data h"+
        "as "+str(input_data.shape[0])+" rows. They are not equal, nor"+
        "are the first a multiple of the second\n")

# Testing block

if __name__=="__main__":

    results_path = get_parent_path_of_file()+"//results"

    input_data_file = "00_successful_data_matrix.npy" 
    
    displacement_data_file = "00_succesful_displacement_matrix.npy"

    new_input_file = "00_successful_complete_data_matrix.npy"

    fill_load_steps(input_data_file, displacement_data_file, 
    new_input_file, results_path, 0.2, 1.0)