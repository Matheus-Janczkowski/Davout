# Routine to store methods for homogenization

from dolfin import *

import numpy as np

from ..tool_box import variational_tools

from ..tool_box import functional_tools

from ..tool_box import tensor_tools

from ..tool_box.parallelization_tools import mpi_execute_function

from ...PythonicUtilities import file_handling_tools as file_tools

from ...PythonicUtilities import programming_tools

from ...PythonicUtilities import recursion_tools

# Defines a function to homogenize a generic field

def homogenize_genericField(field, homogenized_fieldList, time, 
inverse_volume, dx, subdomain, file_name, comm):
    
    # Gets the dimensionality of the field

    dimensionality = 0

    try:

        dimensionality = field.value_shape()

    # If the value shape could not be retrieved, it might be a ufl
    # object, such as a gradient

    except:

        dimensionality = field.ufl_shape

    # Initializes the homogenized value

    homogenized_value = 0.0

    # If the field is a scalar

    if len(dimensionality)==0:

        if isinstance(subdomain, int) or isinstance(subdomain, tuple):

            homogenized_value += inverse_volume*assemble(field*dx(
            subdomain))

        else:

            homogenized_value += inverse_volume*assemble(field*dx)

    # If the field is not scalar, iterates through the dimensions

    else:

        # Makes the homogenized value a list with the format of the 
        # field to be homogenized, e.g. vectors, tensors and so forth

        homogenized_value = np.zeros(dimensionality)

        # Get the possible combinations of indexes

        indexes_combinations = recursion_tools.get_indexesCombinations(
        dimensionality)

        if isinstance(subdomain, int):

            # Iterates through the combinations

            for index_combination in indexes_combinations:

                homogenized_value[*index_combination] = (inverse_volume*
                assemble(field[*index_combination]*dx(subdomain)))
                
        elif isinstance(subdomain, tuple):

            # Iterates through the combinations

            for index_combination in indexes_combinations:

                for subsubdomain in subdomain:

                    homogenized_value[*index_combination] += (
                    inverse_volume*assemble(field[*index_combination]*
                    dx(subsubdomain)))

        else:

            # Iterates through the combinations

            for index_combination in indexes_combinations:

                homogenized_value[*index_combination] = (inverse_volume*
                assemble(field[*index_combination]*dx))

        # Gets the homogenized value back to a list

        homogenized_value = homogenized_value.tolist()

    # Updates the homogenized field

    homogenized_fieldList.append([time, homogenized_value])

    # Saves the homogenized field to a txt file

    mpi_execute_function(comm, file_tools.list_toTxt, 
    homogenized_fieldList, file_name, add_extension=False)

    return homogenized_fieldList

########################################################################
#                        Stress homogenization                         #
########################################################################

# Defines a function to homogenize a stress tensor given by a constitu-
# tive relation

def homogenize_stressTensor(field, constitutive_model, stress_name,
stress_method, homogenized_tensorList, time, inverse_volume, dx, 
homogenization_subdomain, file_name, physical_groupsList, 
physical_groupsNamesToTags, fields_namesDict, required_fieldsNames,
comm):
    
    # Converts the homogenization subdomain to the physical groups tags

    converted_homogenizationSubdomain = []

    if homogenization_subdomain=="":

        converted_homogenizationSubdomain = [""]

    elif isinstance(homogenization_subdomain, tuple):

        # Iterates though the elements of the tuple

        for sub in homogenization_subdomain:

            # Checks if the domain is a string

            if isinstance(sub, str):

                converted_homogenizationSubdomain.append(
                variational_tools.verify_physicalGroups(sub, 
                physical_groupsList, physical_groupsNamesToTags=
                physical_groupsNamesToTags))

            else:

                converted_homogenizationSubdomain.append(sub)

    else:

        # Checks if the domain is a string

        if isinstance(homogenization_subdomain, str):

            converted_homogenizationSubdomain.append(variational_tools.verify_physicalGroups(
            homogenization_subdomain, physical_groupsList, 
            physical_groupsNamesToTags=physical_groupsNamesToTags))

        else:

            converted_homogenizationSubdomain.append(
            homogenization_subdomain)
    
    # Initializes the homogenized tensor

    homogenized_tensor = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 
    0.0]]

    tensor_indexes = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [
    2,1], [2,2]]
    
    # Verifies if the domain is homogeneous. If the constitutive model 
    # is a dictionary, the domain in heterogeneous

    if isinstance(constitutive_model, dict):

        # If the domain is heterogeneous, the stress field must be pro-
        # jected for each subdomain

        for local_subdomain, local_constitutiveModel in constitutive_model.items():

            # Gets the fields for this constitutive model

            retrieved_fields = functional_tools.select_fields(field, 
            required_fieldsNames[local_subdomain], fields_namesDict)

            # Gets the stress tensor field

            stress_field = programming_tools.get_result(
            getattr(local_constitutiveModel, stress_method)(
            retrieved_fields), stress_name)

            # Verifies if more than one physical group is given for 
            # the same constitutive model

            if isinstance(local_subdomain, tuple):

                # Iterates though the elements of the tuple

                for sub in local_subdomain:

                    # Checks if the domain is a string

                    if isinstance(sub, str):

                        sub = variational_tools.verify_physicalGroups(
                        sub, physical_groupsList, 
                        physical_groupsNamesToTags=
                        physical_groupsNamesToTags)

                    # Adds the contribution of this domain if it is one 
                    # of the subdomains to be used in he homogenization

                    if (sub in converted_homogenizationSubdomain) or (
                    converted_homogenizationSubdomain==[""]):

                        for index in tensor_indexes:

                            homogenized_tensor[index[0]][index[1]] += (
                            inverse_volume*assemble(stress_field[*index
                            ]*dx(sub)))

            else:

                # Checks if the domain is a string

                if isinstance(local_subdomain, str):

                    local_subdomain = variational_tools.verify_physicalGroups(
                    local_subdomain, physical_groupsList, 
                    physical_groupsNamesToTags= 
                    physical_groupsNamesToTags)

                # Adds the contribution of this domain if it is one 
                # of the subdomains to be used in he homogenization

                if ((local_subdomain in converted_homogenizationSubdomain
                ) or (converted_homogenizationSubdomain==[""])):

                    for index in tensor_indexes:

                        homogenized_tensor[index[0]][index[1]] += (
                        inverse_volume*assemble(stress_field[*index]*
                        dx(local_subdomain)))

    else:

        # Gets the fields for this constitutive model

        retrieved_fields = functional_tools.select_fields(field, 
        required_fieldsNames, fields_namesDict)

        # Gets the stress tensor field

        stress_field = programming_tools.get_result(getattr(
        constitutive_model, stress_method)(retrieved_fields), 
        stress_name)

        # Adds the contribution of this domain

        if converted_homogenizationSubdomain==[""]:

            for index in tensor_indexes:

                homogenized_tensor[index[0]][index[1]] += (
                inverse_volume*assemble(stress_field[*index]*dx))

        else:

            for domain in converted_homogenizationSubdomain:

                for index in tensor_indexes:

                    homogenized_tensor[index[0]][index[1]] += (
                    inverse_volume*assemble(stress_field[*index]*dx(
                    domain)))

    # Adds the homogenized tensor to the list

    homogenized_tensorList.append([time, homogenized_tensor])

    # Saves the homogenized field to a txt file

    mpi_execute_function(comm, file_tools.list_toTxt, 
    homogenized_tensorList, file_name, add_extension=False)

    return homogenized_tensorList

# Defines a function to homogenize the couple first Piola-Kirchhoff 
# stress tensor

def homogenize_coupleFirstPiola(field, constitutive_model, 
homogenized_tensorList, time, position_vector, inverse_volume, dx, 
homogenization_subdomain, file_name, physical_groupsList, 
physical_groupsNamesToTags, fields_namesDict, required_fieldsNames, comm):
    
    # Converts the homogenization subdomain to the physical groups tags

    converted_homogenizationSubdomain = []

    if homogenization_subdomain=="":

        converted_homogenizationSubdomain = [""]

    elif isinstance(homogenization_subdomain, tuple):

        # Iterates though the elements of the tuple

        for sub in homogenization_subdomain:

            # Checks if the domain is a string

            if isinstance(sub, str):

                converted_homogenizationSubdomain.append(
                variational_tools.verify_physicalGroups(sub, 
                physical_groupsList, physical_groupsNamesToTags=
                physical_groupsNamesToTags))

            else:

                converted_homogenizationSubdomain.append(sub)

    else:

        # Checks if the domain is a string

        if isinstance(homogenization_subdomain, str):

            converted_homogenizationSubdomain.append(variational_tools.verify_physicalGroups(
            homogenization_subdomain, physical_groupsList, 
            physical_groupsNamesToTags=physical_groupsNamesToTags))

        else:

            converted_homogenizationSubdomain.append(
            homogenization_subdomain)
    
    # Initializes the homogenized tensor

    homogenized_tensor = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 
    0.0]]

    tensor_indexes = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [
    2,1], [2,2]]
    
    # Verifies if the domain is homogeneous. If the constitutive model 
    # is a dictionary, the domain in heterogeneous

    if isinstance(constitutive_model, dict):

        # If the domain is heterogeneous, the stress field must be pro-
        # jected for each subdomain

        for local_subdomain, local_constitutiveModel in constitutive_model.items():

            # Gets the fields for this constitutive model

            retrieved_fields = functional_tools.select_fields(field, 
            required_fieldsNames[local_subdomain], fields_namesDict)

            # Gets the Kirchhoff stress tensor

            kirchhoff_stressField = programming_tools.get_result(
            getattr(local_constitutiveModel, "kirchhoff_stress")(
            retrieved_fields), "kirchhoff")

            # Gets the couple stress tensor field

            couple_stressField = programming_tools.get_result(
            getattr(local_constitutiveModel, "first_piolaStress")(
            retrieved_fields), "couple_first_piola_kirchhoff")

            # Gets the stress field

            stress_field = (couple_stressField+tensor_tools.tensor_productAxialByPosition(
            kirchhoff_stressField, position_vector))

            # Verifies if more than one physical group is given for 
            # the same constitutive model

            if isinstance(local_subdomain, tuple):

                # Iterates though the elements of the tuple

                for sub in local_subdomain:

                    # Checks if the domain is a string

                    if isinstance(sub, str):

                        sub = variational_tools.verify_physicalGroups(
                        sub, physical_groupsList, 
                        physical_groupsNamesToTags=
                        physical_groupsNamesToTags)

                    # Adds the contribution of this domain if it is one 
                    # of the subdomains to be used in he homogenization

                    if (sub in converted_homogenizationSubdomain) or (
                    converted_homogenizationSubdomain==[""]):

                        for index in tensor_indexes:

                            homogenized_tensor[index[0]][index[1]] += (
                            inverse_volume*assemble(stress_field[*index
                            ]*dx(sub)))

            else:

                # Checks if the domain is a string

                if isinstance(local_subdomain, str):

                    local_subdomain = variational_tools.verify_physicalGroups(
                    local_subdomain, physical_groupsList, 
                    physical_groupsNamesToTags= 
                    physical_groupsNamesToTags)

                # Adds the contribution of this domain if it is one 
                # of the subdomains to be used in he homogenization

                if ((local_subdomain in converted_homogenizationSubdomain
                ) or (converted_homogenizationSubdomain==[""])):

                    for index in tensor_indexes:

                        homogenized_tensor[index[0]][index[1]] += (
                        inverse_volume*assemble(stress_field[*index]*
                        dx(local_subdomain)))

    else:

        # Gets the fields for this constitutive model

        retrieved_fields = functional_tools.select_fields(field, 
        required_fieldsNames, fields_namesDict)

        # Gets the Kirchhoff stress tensor

        kirchhoff_stressField = programming_tools.get_result(getattr(
        constitutive_model, "kirchhoff_stress")(retrieved_fields), "ki"+
        "rchhoff")

        # Gets the couple stress tensor field

        couple_stressField = programming_tools.get_result(getattr(
        constitutive_model, "first_piolaStress")(retrieved_fields), "c"+
        "ouple_first_piola_kirchhoff")

        # Gets the stress field

        stress_field = (couple_stressField+tensor_tools.tensor_productAxialByPosition(
        kirchhoff_stressField, position_vector))

        # Adds the contribution of this domain

        if converted_homogenizationSubdomain==[""]:

            for index in tensor_indexes:

                homogenized_tensor[index[0]][index[1]] += (
                inverse_volume*assemble(stress_field[*index]*dx))

        else:

            for domain in converted_homogenizationSubdomain:

                for index in tensor_indexes:

                    homogenized_tensor[index[0]][index[1]] += (
                    inverse_volume*assemble(stress_field[*index]*dx(
                    domain)))

    # Adds the homogenized tensor to the list

    homogenized_tensorList.append([time, homogenized_tensor])

    # Saves the homogenized field to a txt file

    mpi_execute_function(comm, file_tools.list_toTxt, 
    homogenized_tensorList, file_name, add_extension=False)

    return homogenized_tensorList

# Defines a function to evaluate the anisotropic tensor of the commuta-
# tivity between the second Piola-Kirchhoff stress tensor and the right
# Cauchy-Green stress tensor

def commutativity_tensorHomogenizedSandC(field, constitutive_model, 
stress_name, stress_method, homogenized_tensorList, time, inverse_volume, 
dx, homogenization_subdomain, file_name, physical_groupsList, 
physical_groupsNamesToTags, fields_namesDict, required_fieldsNames):
    
    pass