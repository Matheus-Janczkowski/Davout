# Routine to store some methods to help evaluate constitutive models. It
# includes stress tensors transformations and derivation. Cauchy and Mi-
# cropolar continua are considered

from dolfin import *

import ufl_legacy as ufl

from copy import deepcopy

import numpy as np

from ..tool_box import tensor_tools

from ..tool_box import programming_tools

from ..tool_box import functional_tools

from ..tool_box import variational_tools

from ..tool_box import file_handling_tools as file_tools

from ..tool_box import plotting_tools

# Defines the indices for Einstein summation notation

i, j, k, l = ufl.indices(4)

########################################################################
#                     Field checking and retrieving                    #
########################################################################

# Defines a function to check if the dictionary of material parameters 
# has all the keys

def check_materialDictionary(dictionary, required_keys):

    for key in required_keys:

        if not (key in dictionary):

            raise KeyError("The key '"+str(key)+"' was not found in th"+
            "e dictionary of material parameters. See: "+str(
            dictionary.keys()))

# Defines a function to retrieve from the constitutive tools the neces-
# sary fields' names

def get_constitutiveModelFields(constitutive_models):

    # If the constitutive_models is a dictionary

    if isinstance(constitutive_models, dict):

        # Initializes a dcitionary of required fields' names

        required_fieldsNames = dict()

        # Iterates through the constitutive models (classes)

        for subdomain, constitutive_class in constitutive_models.items():

            required_names = []

            try:

                # Gets the required names

                required_names = constitutive_class.required_fieldsNames

            except:

                raise AttributeError("The constitutive model in the su"+
                "bdomain given by "+str(subdomain)+" does not have the"+
                " variable (attribute) of required fields' names (requ"+
                "ired_fieldsNames). This constitutive model's class mu"+
                "st be corrected")

            # Adds the required names to the dictionary

            required_fieldsNames[subdomain] = required_names

        # Returns the list of required fields' names

        return required_fieldsNames

    # If the constitutive models variable is not a dictionary, just take
    # the names

    else:

        required_names = []

        try:

            # Gets the required names

            required_names = constitutive_models.required_fieldsNames

        except AttributeError:

            raise AttributeError("The constitutive model in the subdom"+
            "ain given by "+str(subdomain)+" does not have the variabl"+
            "e (attribute) of required fields' names (required_fieldsN"+
            "ames). This constitutive model's class must be corrected")

        except:

            raise TypeError("The constitutive model "+str(
            constitutive_models)+" is not a class. Select a proper con"+
            "stitutive class")

        return required_names

########################################################################
#                    Cauchy-continuum stress tensors                   #
########################################################################

# Defines a function to evaluate the second Piola-Kirchhoff stress ten-
# sor from the derivative of the Helmholtz potential w.r.t. the right
# Cauchy-Green strain tensor

def S_fromDPsiDC(helmholtz_potential, u):

    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(u)+I

    # Evaluates the right Cauchy-Green strain tensor, C. Makes C a 
    # variable to differentiate the Helmholtz potential with respect 
    # to C

    C = (F.T)*F
    
    C = variable(C)  

    # Evaluates the Helmholtz potential

    W = helmholtz_potential(C)

    # Evaluates the second Piola-Kirchhoff stress tensor differenti-
    # ating the potential w.r.t. C

    S = 2*diff(W,C)

    return S

# Defines a function to evaluate the Cauchy stress tensor as the push 
# forward of the second Piola-Kirchhoff stress tensor

def push_forwardS(S, u):

    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(u)+I

    # Evaluates the determinant of the deformation gradient

    J = ufl.det(F)

    # Pushes forward to the deformed configuration

    sigma = (1.0/J)*F*S*F.T

    return sigma

# Defines a function to transform the Cauchy stress to the second Piola-
# Kirchhoff stress

def S_fromCauchy(sigma, u):

    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(u)+I

    # Evaluates the determinant of the deformation gradient

    J = ufl.det(F)

    # Uses the pull back operation

    S = J*inv(F)*sigma*(inv(F).T)

    return S

# Defines a function to transform the Cauchy stress to the first Piola-
# Kirchhoff stress

def P_fromCauchy(sigma, u):

    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(u)+I

    # Evaluates the determinant of the deformation gradient

    J = ufl.det(F)

    # Uses the Piola transformation

    P = J*sigma*(inv(F).T)

    return P

# Defines a function to transform the Cauchy stress to the Kirchhoff 
# stress tensor

def tau_fromCauchy(sigma, u):

    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(u)+I

    # Evaluates the determinant of the deformation gradient

    J = ufl.det(F)

    # Uses the Piola transformation

    tau = J*sigma

    return tau

# Defines a function to transform the first Piola-Kirchhoff stress from
# the second one

def P_fromS(S, u):

    # Evaluates the deformation gradient

    I = Identity(3)

    F = grad(u)+I

    # Transforms S to P

    P = F*S

    return P

########################################################################
#                     Micropolar curvature tensor                      #
########################################################################

# Defines the micropolar curvature tensor

def micropolar_curvatureTensor(phi):

    # Defines the permutation tensor for 3D euclidean space

    perm = ufl.PermutationSymbol(3)
    
    # Computes the micro-rotation tensor Rbar
     
    R_bar = tensor_tools.rotation_tensorEulerRodrigues(phi)
    
    # Computes the gradient of each component of R_bar to get a third 
    # order tensor, grad_Rbar

    grad_Rbar = grad(R_bar)

    # Constructs the third order curvature tensor K_third

    K_third = as_tensor(R_bar[j,i]*grad_Rbar[j,k,l], (i,k,l))

    # Contracts with the permutation tensor to form K_second

    K_second = 0.5*as_tensor(perm[i,j,k]*K_third[k,j,l], (i,l))

    return K_second

########################################################################
########################################################################
##                          Post-processing                           ##
########################################################################
########################################################################

########################################################################
#                       Saving of stress measures                      #
########################################################################

# Defines a function to get, project and save a stress field

def save_stressField(output_object, field, time, flag_parentMeshReuse,
stress_solutionPlotNames, stress_name, stress_method, fields_namesDict):

    # If the flag to reuse parent mesh information is true, just save 
    # the information given in the output object

    if flag_parentMeshReuse:

        output_object.parent_toChildMeshResult.rename(
        *stress_solutionPlotNames)

        output_object.result.write(
        output_object.parent_toChildMeshResult, time)

        return output_object
    
    # Verifies if the output object has the attribute with the names of 
    # the required fields

    if not hasattr(output_object, "required_fieldsNames"):

        raise AttributeError("The class of data for the post-process o"+
        "f saving the stress field does not have the attribute 'requir"+
        "ed_fieldsNames'. This class must have it")

    # Verifies if the domain is homogeneous

    if isinstance(output_object.constitutive_model, dict):

        # Initializes a list of pairs of constitutive models and inte-
        # gration domain

        integration_pairs = []

        # If the domain is heterogeneous, the stress field must be pro-
        # jected for each subdomain

        for subdomain, local_constitutiveModel in output_object.constitutive_model.items():

            # Gets the fields for this constitutive model

            retrieved_fields = functional_tools.select_fields(field, 
            output_object.required_fieldsNames[subdomain], 
            fields_namesDict)

            # Gets the stress field

            stress_field = programming_tools.get_result(
            programming_tools.get_attribute(local_constitutiveModel, 
            stress_method, "The constitutive model\n"+str(
            local_constitutiveModel)+"\ndoes not have the attribute '"+
            str(stress_method)+"', thus the stress field cannot be upd"+
            "ated")(retrieved_fields), stress_name)

            # Verifies if more than one physical group is given for the
            # same constitutive model

            if isinstance(subdomain, tuple):

                # Iterates though the elements of the tuple

                for sub in subdomain:

                    # Converts the subdomain to an integer tag

                    sub = variational_tools.verify_physicalGroups(sub, 
                    output_object.physical_groupsList, 
                    output_object.physical_groupsNamesToTags,
                    throw_error=False)

                    # Checks if this subdomain is in the domain physical
                    # groups 

                    if sub in output_object.physical_groupsList:

                        # Adds this pair of constitutive model and inte-
                        # gration domain to the list of such pairs

                        integration_pairs.append([stress_field, sub])

            else:

                # Converts the subdomain to an integer tag

                subdomain = variational_tools.verify_physicalGroups(
                subdomain, output_object.physical_groupsList, 
                output_object.physical_groupsNamesToTags, throw_error=
                False)

                # Checks if this subdomain is in the domain physical
                # groups 

                if subdomain in output_object.physical_groupsList:

                    # Adds this pair of constitutive model and integra-
                    # tion domain to the list of such pairs

                    integration_pairs.append([stress_field, subdomain])

        # Projects this piecewise continuous field of stress into a FE 
        # space

        stress_fieldFunction = variational_tools.project_piecewiseField(
        integration_pairs, output_object.dx, output_object.W, 
        output_object.physical_groupsList, 
        output_object.physical_groupsNamesToTags, solution_names=
        stress_solutionPlotNames)

        # Saves the field into the sharable result with a submesh

        output_object.parent_toChildMeshResult = stress_fieldFunction

        # Writes the field to the file

        output_object.result.write(stress_fieldFunction, time)

    else:

        # Gets the fields for this constitutive model

        retrieved_fields = functional_tools.select_fields(field, 
        output_object.required_fieldsNames, fields_namesDict)

        # Gets the stress field

        stress_field = programming_tools.get_result(
        programming_tools.get_attribute(output_object.constitutive_model, 
        stress_method, "The constitutive model\n"+str(
        output_object.constitutive_model)+"\ndoes not have the attribu"+
        "te '"+str(stress_method)+"', thus the stress field cannot be "+
        "updated")(field), stress_name)

        # Projects the stress into a function

        stress_fieldFunction = project(stress_field, output_object.W)

        stress_fieldFunction.rename(*stress_solutionPlotNames)

        # Saves the field into the sharable result with a submesh

        output_object.parent_toChildMeshResult = stress_fieldFunction

        # Writes the field to the file

        output_object.result.write(stress_fieldFunction, time)

    return output_object

# Defines a function to project the Cauchy stress field and get the 
# pressure from it at a point

def save_pressureAtPoint(output_object, field, time, stress_name, 
stress_method, fields_namesDict):
    
    # Verifies if the output object has the attribute with the names of 
    # the required fields

    if not hasattr(output_object, "required_fieldsNames"):

        raise AttributeError("The class of data for the post-process o"+
        "f saving the pressure field at a point does not have the attr"+
        "ibute 'required_fieldsNames'. This class must have it")

    # Verifies if the domain is homogeneous

    if isinstance(output_object.constitutive_model, dict):

        # Initializes a list of pairs of constitutive models and inte-
        # gration domain

        integration_pairs = []

        # If the domain is heterogeneous, the stress field must be pro-
        # jected for each subdomain

        for subdomain, local_constitutiveModel in output_object.constitutive_model.items():

            # Gets the fields for this constitutive model

            retrieved_fields = functional_tools.select_fields(field, 
            output_object.required_fieldsNames[subdomain], 
            fields_namesDict)

            # Gets the stress field

            stress_field = programming_tools.get_result(
            programming_tools.get_attribute(local_constitutiveModel, 
            stress_method, "The constitutive model\n"+str(
            local_constitutiveModel)+"\ndoes not have the attribute '"+
            str(stress_method)+"', thus the pressure at a point cannot"+
            " be updated")(retrieved_fields), stress_name)

            # Verifies if more than one physical group is given for the
            # same constitutive model

            if isinstance(subdomain, tuple):

                # Iterates though the elements of the tuple

                for sub in subdomain:

                    # Converts the subdomain to an integer tag

                    sub = variational_tools.verify_physicalGroups(sub, 
                    output_object.physical_groupsList, 
                    output_object.physical_groupsNamesToTags,
                    throw_error=False)

                    # Checks if this subdomain is in the domain physical
                    # groups 

                    if sub in output_object.physical_groupsList:

                        # Adds this pair of constitutive model and inte-
                        # gration domain to the list of such pairs. Gets
                        # the trace divided by 3 to get the pressure

                        integration_pairs.append([(1/3)*tr(stress_field
                        ), sub])

            else:

                # Converts the subdomain to an integer tag

                subdomain = variational_tools.verify_physicalGroups(
                subdomain, output_object.physical_groupsList, 
                output_object.physical_groupsNamesToTags, throw_error=
                False)

                # Checks if this subdomain is in the domain physical
                # groups 

                if subdomain in output_object.physical_groupsList:

                    # Adds this pair of constitutive model and integra-
                    # tion domain to the list of such pairs. Gets the 
                    # trace divided by 3 to get the pressure

                    integration_pairs.append([(1/3)*tr(stress_field), 
                    subdomain])

        # Projects this piecewise continuous field of stress into a FE 
        # space

        pressure_fieldFunction = variational_tools.project_piecewiseField(
        integration_pairs, output_object.dx, output_object.W, 
        output_object.physical_groupsList, 
        output_object.physical_groupsNamesToTags)

        # Updates the pressure by evaluating it the field at a point

        output_object.result.append([time, pressure_fieldFunction(Point(
        output_object.point_coordinates))])

    else:

        # Gets the fields for this constitutive model

        retrieved_fields = functional_tools.select_fields(field, 
        output_object.required_fieldsNames, fields_namesDict)

        # Gets the stress field

        stress_field = programming_tools.get_result(
        programming_tools.get_attribute(output_object.constitutive_model, 
        stress_method, "The constitutive model\n"+str(
        output_object.constitutive_model)+"\ndoes not have the attribu"+
        "te '"+str(stress_method)+"', thus the pressure at a point can"+
        "not be updated")(retrieved_fields), stress_name)

        # Projects the stress into a function taking the trace to get 
        # the pressure

        pressure_fieldFunction = project((1/3)*tr(stress_field), 
        output_object.W)

        # Updates the pressure by evaluating it the field at a point

        output_object.result.append([time, pressure_fieldFunction(Point(
        output_object.point_coordinates))])

    # Saves the pressure at a point to a txt file

    file_tools.list_toTxt(output_object.result, output_object.file_name, 
    add_extension=True)

    # If it is to plot the data

    if output_object.flag_plotting:

        plotting_tools.plane_plot(output_object.file_name+".pdf", data=
        output_object.result,  x_label=r"$t$", y_label=r"$p$", title=
        r"pressure at $x="+str(output_object.point_coordinates[0])+",\;"+
        "y="+str(output_object.point_coordinates[1])+",\;z="+str(
        output_object.point_coordinates[2])+"$", highlight_points=True)

    return output_object

# Defines a function to get the traction field on the referential confi-
# guration

def save_referentialTraction(output_object, field, time, stress_name, 
stress_method, fields_namesDict):
    
    # Verifies if the output object has the attribute with the names of 
    # the required fields

    if not hasattr(output_object, "required_fieldsNames"):

        raise AttributeError("The class of data for the post-process o"+
        "f saving the referential traction field at a point does not h"+
        "ave the attribute 'required_fieldsNames'. This class must hav"+
        "e it")

    # Verifies if the domain is homogeneous

    if isinstance(output_object.constitutive_model, dict):

        # Initializes a list of pairs of constitutive models and inte-
        # gration domain

        integration_pairs = []

        # If the domain is heterogeneous, the stress field must be pro-
        # jected for each subdomain

        for subdomain, local_constitutiveModel in output_object.constitutive_model.items():

            # Gets the fields for this constitutive model

            retrieved_fields = functional_tools.select_fields(field, 
            output_object.required_fieldsNames[subdomain], 
            fields_namesDict)

            # Gets the stress field

            stress_field = programming_tools.get_result(
            programming_tools.get_attribute(local_constitutiveModel, 
            stress_method, "The constitutive model\n"+str(
            local_constitutiveModel)+"\ndoes not have the attribute '"+
            str(stress_method)+"', thus the referential traction canno"+
            "t be updated")(retrieved_fields), stress_name)

            # Verifies if more than one physical group is given for the
            # same constitutive model

            if isinstance(subdomain, tuple):

                # Iterates though the elements of the tuple

                for sub in subdomain:

                    # Converts the subdomain to an integer tag

                    sub = variational_tools.verify_physicalGroups(sub, 
                    output_object.physical_groupsList, 
                    output_object.physical_groupsNamesToTags,
                    throw_error=False)

                    # Checks if this subdomain is in the domain physical
                    # groups 

                    if sub in output_object.physical_groupsList:

                        # Adds this pair of constitutive model and inte-
                        # gration domain to the list of such pairs. Gets
                        # the trace divided by 3 to get the pressure

                        integration_pairs.append([stress_field*
                        output_object.referential_normal, sub])

            else:

                # Converts the subdomain to an integer tag

                subdomain = variational_tools.verify_physicalGroups(
                subdomain, output_object.physical_groupsList, 
                output_object.physical_groupsNamesToTags, throw_error=
                False)

                # Checks if this subdomain is in the domain physical
                # groups 

                if subdomain in output_object.physical_groupsList:

                    # Adds this pair of constitutive model and integra-
                    # tion domain to the list of such pairs. Gets the 
                    # trace divided by 3 to get the pressure

                    integration_pairs.append([stress_field*
                    output_object.referential_normal, subdomain])

        # Projects this piecewise continuous field of traction into a FE 
        # space

        traction_fieldFunction = variational_tools.project_overBoundary(
        integration_pairs, output_object.ds, output_object.W, 
        output_object.physical_groupsList, 
        output_object.physical_groupsNamesToTags, solution_names=["tra"+
        "ction", "DNS"])

        # Writes the field to the file

        output_object.result.write(traction_fieldFunction, time)

    else:

        # Gets the fields for this constitutive model

        retrieved_fields = functional_tools.select_fields(field, 
        output_object.required_fieldsNames, fields_namesDict)

        # Gets the stress field

        stress_field = programming_tools.get_result(
        programming_tools.get_attribute(output_object.constitutive_model, 
        stress_method, "The constitutive model\n"+str(
        output_object.constitutive_model)+"\ndoes not have the attribu"+
        "te '"+str(stress_method)+"', thus the referential traction ca"+
        "nnot be updated")(retrieved_fields), stress_name)

        # Projects the stress into a function taking the trace to get 
        # the pressure

        traction_fieldFunction = variational_tools.project_overBoundary(
        [[dot(stress_field,output_object.referential_normal), ""]], 
        output_object.ds, output_object.W, 
        output_object.physical_groupsList, 
        output_object.physical_groupsNamesToTags, solution_names=["tra"+
        "ction", "DNS"])

        # Writes the field to the file

        output_object.result.write(traction_fieldFunction, time)

    return output_object

########################################################################
#                      Saving of elasticity tensor                     #
########################################################################

# Defines a function to save the first elasticity tensor (dP/dF)

def save_elasticityTensor(output_object, field, time, tensor_method,
tensor_name, fields_namesDict):
    
    # Verifies if the output object has the attribute with the names of 
    # the required fields

    if not hasattr(output_object, "required_fieldsNames"):

        raise AttributeError("The class of data for the post-process o"+
        "f saving the pressure field at a point does not have the attr"+
        "ibute 'required_fieldsNames'. This class must have it")
    
    # Initializes the tensor as a list in Voigt notation (1111, 1112, 
    # 1113, 1121, 1122, 1123, 1131, 1132, 1133, 1211, 1212...)

    tensor_voigt = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    full_tensor = np.zeros((3,3,3,3))

    # Verifies if the domain is homogeneous

    if isinstance(output_object.constitutive_model, dict):

        # Initializes a list of pairs of constitutive models and inte-
        # gration domain

        integration_pairs = []

        # If the domain is heterogeneous, the stress field must be pro-
        # jected for each subdomain

        for subdomain, local_constitutiveModel in output_object.constitutive_model.items():

            # Gets the fields for this constitutive model

            retrieved_fields = functional_tools.select_fields(field, 
            output_object.required_fieldsNames[subdomain], 
            fields_namesDict)

            # Gets the elasticity tensor field

            dStress_dStrain = programming_tools.get_result(
            programming_tools.get_attribute(local_constitutiveModel, 
            tensor_method, "The constitutive model\n"+str(
            local_constitutiveModel)+"\ndoes not have the attribute '"+
            str(tensor_method)+"', thus the elasticity tensor cannot b"+
            "e evaluated")(retrieved_fields), tensor_name)

            # Verifies if more than one physical group is given for the
            # same constitutive model

            if isinstance(subdomain, tuple):

                # Iterates though the elements of the tuple

                for sub in subdomain:

                    # Converts the subdomain to an integer tag

                    sub = variational_tools.verify_physicalGroups(sub, 
                    output_object.physical_groupsList, 
                    output_object.physical_groupsNamesToTags,
                    throw_error=False)

                    # Checks if this subdomain is in the domain physical
                    # groups 

                    if sub in output_object.physical_groupsList:

                        # Adds this pair of constitutive model and inte-
                        # gration domain to the list of such pairs. Gets
                        # the trace divided by 3 to get the pressure

                        integration_pairs.append([dStress_dStrain, sub])

            else:

                # Converts the subdomain to an integer tag

                subdomain = variational_tools.verify_physicalGroups(
                subdomain, output_object.physical_groupsList, 
                output_object.physical_groupsNamesToTags, throw_error=
                False)

                # Checks if this subdomain is in the domain physical
                # groups 

                if subdomain in output_object.physical_groupsList:

                    # Adds this pair of constitutive model and integra-
                    # tion domain to the list of such pairs. Gets the 
                    # trace divided by 3 to get the pressure

                    integration_pairs.append([dStress_dStrain, 
                    subdomain])

        # Iterates through the components

        for i in range(9):

            for j in range(9):

                # Gets the indices from the Voigt notation

                indices = output_object.voigt_notation[(i,j)]

                # Initializes the integration pairs for this index

                index_integrationPairs = []

                for pair in integration_pairs:

                    index_integrationPairs.append([pair[0][indices[0], 
                    indices[1], indices[2], indices[3]], pair[1]])

                # Projects the component into a FE space

                elasticity_tensorFunction = variational_tools.project_piecewiseField(
                index_integrationPairs, output_object.dx, 
                output_object.W, output_object.physical_groupsList, 
                output_object.physical_groupsNamesToTags)

                # Stores the component

                tensor_voigt[i][j] = elasticity_tensorFunction(Point(
                output_object.point_coordinates))

    else:

        # Gets the fields for this constitutive model

        retrieved_fields = functional_tools.select_fields(field, 
        output_object.required_fieldsNames, fields_namesDict)

        # Gets the elasticity tensor field

        elasticity_tensor = programming_tools.get_result(
        programming_tools.get_attribute(output_object.constitutive_model, 
        tensor_method, "The constitutive model\n"+str(
        output_object.constitutive_model)+"\ndoes not have the attribu"+
        "te '"+str(tensor_method)+"', thus the elasticity tensor canno"+
        "t be evaluated")(retrieved_fields), tensor_name)

        # Iterates through the components

        #"""
        for i in range(9):

            for j in range(9):

                # Gets the indices from the Voigt notation

                indices = output_object.voigt_notation[(i,j)]

                # Projects the component

                elasticity_tensorFunction = project(elasticity_tensor[
                indices[0], indices[1], indices[2], indices[3]], 
                output_object.W)

                # Stores the component

                tensor_voigt[i][j] = elasticity_tensorFunction(Point(
                output_object.point_coordinates))#"""
        
        """for i in range(3):

            for j in range(3):

                for k in range(3):

                    for l in range(3):

                        full_tensor[i,j,k,l] = project(elasticity_tensor[
                        i,j,k,l], output_object.W)(Point(
                        output_object.point_coordinates))"""

    # Updates the tensor as a list

    output_object.result.append([time, tensor_voigt])

    #output_object.result.append([time, full_tensor.tolist()])

    # Saves the elasticity tensor at a point to a txt file

    file_tools.list_toTxt(output_object.result, output_object.file_name, 
    add_extension=True)

    # If it is to plot the data

    if output_object.flag_plotting:

        # Gets the title and the basic name of the file

        base_file_name = ""

        title = ""

        if tensor_name=="first_elasticity_tensor":

            title = ("$\\frac{\\partial\\boldsymbol{P}}{\\partial\\bol"+
            "dsymbol{F}}$\n")

            base_file_name = "first_elasticity_tensor_dP_dF"

        elif tensor_name=="second_elasticity_tensor":

            title = ("$\\frac{\\partial\\boldsymbol{S}}{\\partial\\bol"+
            "dsymbol{C}}$\n")

            base_file_name = "second_elasticity_tensor_dS_dC"

        elif tensor_name=="third_elasticity_tensor":

            title = ("$\\frac{\\partial\\boldsymbol{\\sigma}}{\\partia"+
            "l\\boldsymbol{b}}$\n")

            base_file_name = "third_elasticity_tensor_dsigma_db"

        # Gets the optional arguments

        scaling_function = output_object.optional_arguments["scaling f"+
        "unction"]

        scaling_functionAdditionalParams = output_object.optional_arguments[
        "scaling function additional parameters"]

        color_map = output_object.optional_arguments["color map"]

        max_ticksColorBar = output_object.optional_arguments["maximum "+
        "ticks on color bar"]

        # Sets the scientific notation flag

        flag_scientificNotation = True

        if scaling_function=="logarithmic filter":

            flag_scientificNotation = False

        # Gets the ticks values from the Voigt notation

        x_ticksLabels = {}

        y_ticksLabels = {}

        for i in range(9):

            index = output_object.voigt_notation[(i,0)]

            x_ticksLabels[i+1] = str(index[0]+1)+str(index[1]+1)

        for i in range(9):

            index = output_object.voigt_notation[(i,0)]

            y_ticksLabels[9-i] = str(index[0]+1)+str(index[1]+1)

        # Plots the elasticity tensor

        plotting_tools.plot_matrix(deepcopy(output_object.result), 
        output_object.parent_path, base_file_name, include_time=True, 
        scaling_function=scaling_function, color_map=color_map, title=
        title, flag_scientificNotation=flag_scientificNotation, 
        scaling_functionAdditionalParams=
        scaling_functionAdditionalParams, max_ticksColorBar=
        max_ticksColorBar, x_grid=[3.5, 6.5], y_grid=[3.5, 6.5],
        element_size=23, x_ticksLabels=x_ticksLabels, y_ticksLabels=
        y_ticksLabels)

    return output_object