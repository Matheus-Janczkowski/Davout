# Routine to store methods to apply Neumann boundary conditions on sur-
# faces

import tensorflow as tf

from .tensorflow_utilities import convert_object_to_tensor

########################################################################
#                            Traction vector                           #
########################################################################

# Defines a class to store methods to build a traction vector on a sur-
# face

class TractionVectorOnSurface:

    def __init__(self, surface_mesh_data, traction_information, 
    physical_group_name, n_realizations, mesh_realizations_common_info):
        
        # Gets the number of batched BVP instances

        self.n_realizations = n_realizations
        
        # Saves the given information

        self.surface_mesh_data = surface_mesh_data

        self.mesh_realizations_common_info = mesh_realizations_common_info

        # Initializes the traction vector

        traction_vector = []

        # Sets a flag for batching mode

        self.batching_number = False

        # Verifies if the traction information provided by the user has
        # the keys for each traction component

        dimensions_names = ["X", "Y", "Z"]

        for dimension in dimensions_names:

            # Verifies the key

            amplitude_key = "amplitude_traction"+str(dimension)

            if not (amplitude_key in traction_information):

                raise KeyError("The information dictionary provided to"+
                " set 'TractionVectorOnSurface' in surface '"+str(
                physical_group_name)+"' does not have the key '"+
                amplitude_key+"'. The given dictionary is: "+str(
                traction_information))
            
            else:

                # Gets the component value

                component_value = traction_information[amplitude_key]

                # Verifies if component value is a list

                if isinstance(component_value, list):

                    # Verifies if a batching number has been already gi-
                    # ven

                    if self.batching_number:
                        
                        # Verifies if it is equal to the number of ele-
                        # ments in the current list 

                        if self.batching_number!=len(component_value):

                            raise IndexError("Each component of the pr"+
                            "escribed traction vector is meant to be a"+
                            " list, i.e. multiple BVP instances are to"+
                            " be created. But the previous component l"+
                            "ists had size of "+str(self.batching_number
                            )+", whereas the list corresponding to the"+
                            " component '"+str(amplitude_key)+"' has s"+
                            "ize of "+str(len(component_value)))
                
                    # If it was not created yet, saves the length of the gi-
                    # ven list

                    else:

                        self.batching_number = len(component_value)

                traction_vector.append(component_value)
        
        # Converts traction vector to a tensor

        self.traction_vector = convert_object_to_tensor(traction_vector,
        mesh_realizations_common_info.float_dtype)

        # If batching mode is on, unravels the tensor

        if self.batching_number:

            # Verifies if the number of realizations given is consistent
            # witht the global number of realizations

            if self.batching_number!=n_realizations:

                raise ValueError("The number of realizations of tracti"+
                "ons in 'TractionVectorOnSurface' is "+str(
                self.batching_number)+" whereas the global number of r"+
                "ealizations is "+str(n_realizations)+". They must be "+
                "the same")

            # The current tensor is [3, n_realizations], but the indices
            # must be permutated to make it [n_realizations, 3]

            self.traction_vector = tf.transpose(self.traction_vector, 
            perm=[1,0])

            # Then reshapes it to be fit latter into the space of 
            # [n_realizations, n_elements, n_quadrature_points, 3]

            self.traction_vector = tf.reshape(self.traction_vector, [
            self.n_realizations, 1, 1, 3])

        else:

            # Then reshapes it to be fit latter into the space of 
            # [n_realizations, n_elements, n_quadrature_points, 3]

            self.traction_vector = tf.reshape(self.traction_vector, [1, 
            1, 1, 3])
        
        # Calls the method to build the traction tensor if required

        self.traction_tensor = tf.Variable(tf.broadcast_to(
        self.traction_vector, [self.n_realizations, 
        self.mesh_realizations_common_info.number_elements, 
        self.mesh_realizations_common_info.number_quadrature_points, 3]))
        
    # Defines a function to build the traction [n_elements, 
    # n_quadrature_points, 3] with the first ever given vector of para-
    # meters

    @tf.function
    def compute_traction(self, vector_of_parameters):

        # Gets the number of elements and the number of quadrature points
        # to create the traction tensor

        self.traction_tensor.assign(tf.broadcast_to(self.traction_vector, 
        [self.n_realizations, 
        self.mesh_realizations_common_info.number_elements, 
        self.mesh_realizations_common_info.number_quadrature_points, 3]))

########################################################################
#                       Prescribed stress tensor                       #
########################################################################

# Defines a class to store methods to build a traction vector on a sur-
# face, given a prescribed stress tensor on a boundary

class FirstPiolaKirchhoffOnSurface:

    def __init__(self, surface_mesh_data, 
    prescribed_first_piola_kirchhoff_info, physical_group_name, 
    n_realizations, mesh_realizations_common_info):
        
        # Gets the number of batched BVP instances

        self.n_realizations = n_realizations
        
        # Saves the given information

        self.surface_mesh_data = surface_mesh_data

        self.mesh_realizations_common_info = mesh_realizations_common_info

        # Initializes the list of prescribed first Piola-Kirchhoff stress
        # tensor

        prescribed_first_piola_kirchhoff = []

        # Sets a flag for batching mode

        self.batching_number = False

        # Iterates through the first index of the tensor

        for i in range(3):

            # Adds this row

            prescribed_first_piola_kirchhoff.append([])

            # Iterates through the second index

            for j in range(3):

                # Verifies if this index is present on the information
                # dictionary

                index_key = "P"+str(i+1)+str(j+1)

                if not (index_key in (
                prescribed_first_piola_kirchhoff_info)):
                    
                    raise KeyError("The information dictionary provide"+
                    "d to set 'FirstPiolaKirchhoffOnSurface' in surfac"+
                    "e '"+str(physical_group_name)+"' does not have th"+
                    "e key '"+index_key+"'. The given dictionary is: "+
                    str(prescribed_first_piola_kirchhoff_info))
                
                # Gets the value

                component_value = prescribed_first_piola_kirchhoff_info[
                index_key]

                # If the value is a list, activates batching mode

                if isinstance(component_value, list):

                    # Verifies if a batching number has been already gi-
                    # ven

                    if self.batching_number:

                        # Verifies if it is equal to the number of ele-
                        # ments in the current list 

                        if self.batching_number!=len(component_value):

                            raise IndexError("Each component of the pr"+
                            "escribed first Piola-Kirchhoff stress ten"+
                            "sor is meant to be a list, i.e. multiple "+
                            "BVP instances are to be created. But the "+
                            "previous component lists had size of "+str(
                            self.batching_number)+", whereas the list "+
                            "corresponding to the component '"+index_key+
                            "' has size of "+str(len(component_value)))
                    
                    # If it was not created yet, saves the length of the
                    # given list

                    else:

                        self.batching_number = len(component_value)
                
                # Updates the list using this index
                
                prescribed_first_piola_kirchhoff[-1].append(
                component_value)
        
        # Converts prescribed_first_piola_kirchhoff list of lists to 
        # a tensor

        self.prescribed_first_piola_kirchhoff = convert_object_to_tensor(
        prescribed_first_piola_kirchhoff, 
        mesh_realizations_common_info.float_dtype)

        # If batching mode is on, unravels the tensor

        if self.batching_number:

            # Verifies if the number of realizations given is consistent
            # witht the global number of realizations

            if self.batching_number!=n_realizations:

                raise ValueError("The number of realizations of tracti"+
                "ons in 'FirstPiolaKirchhoffOnSurface' is "+str(
                self.batching_number)+" whereas the global number of r"+
                "ealizations is "+str(n_realizations)+". They must be "+
                "the same")

            # The current tensor is [3,3,n_realizations], but the indi-
            # ces must be permutated to make it [n_realizations, 3, 3]

            self.prescribed_first_piola_kirchhoff = tf.transpose(
            self.prescribed_first_piola_kirchhoff, perm=[2,0,1])

            # Defines the proper function to compute the traction as the
            # one that contracts the surface normal with the multiple P
            # tensors

            self.appropriate_computation = self.compute_traction_for_batched_value

        # Otherwise, just mounts it

        else:

            # Defines the proper function to compute the traction as the
            # one that broadcasts a single P to multiple BVPs

            self.appropriate_computation = self.compute_traction_for_single_value
        
        # Calls the method to build the traction tensor

        self.traction_tensor = tf.Variable(self.appropriate_computation(
        ))
        
    # Defines a function to build the traction [n_realizations, n_ele-
    # ments, n_quadrature_points, 3] with the first ever given vector of 
    # parameters

    @tf.function
    def compute_traction(self, vector_of_parameters):

        # Computes the chosen function

        self.traction_tensor.assign(self.appropriate_computation())
    
    # Defines a function to compute the traction where a single instance
    # of prescribed first Piola-Kirchhoff is used for all realizations 
    # of the BVP

    @tf.function
    def compute_traction_for_single_value(self):

        # Contracts the first Piola-Kirchhoff stress tensor with the ten-
        # sor of normal vectors of the mesh. But broadcasts it. The re-
        # sult is [n_realizations, n_elements, n_quadrature_points, n_
        # dimensions]

        return tf.broadcast_to(tf.expand_dims(tf.einsum('ij,eqj->eqi', 
        self.prescribed_first_piola_kirchhoff, 
        self.surface_mesh_data.normal_vector), axis=0), [
        self.n_realizations, 
        *self.mesh_realizations_common_info.dofs_per_element.shape])
    
    # Defines a function to compute the traction where multiple instan-
    # ces of prescribed first Piola-Kirchhoff were given. Each corres-
    # ponding to a realization of the BVP

    @tf.function
    def compute_traction_for_batched_value(self, vector_of_parameters):

        # Contracts the first Piola-Kirchhoff stress tensor with the ten-
        # sor of normal vectors of the mesh. But broadcasts it. The re-
        # sult is [n_realizations, n_elements, n_quadrature_points, n_
        # dimensions]

        return tf.einsum('pij,eqj->peqi', 
        self.prescribed_first_piola_kirchhoff, 
        self.surface_mesh_data.normal_vector)