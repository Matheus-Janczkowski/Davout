# Routine to store some methods to post-process solution inside the 
# pseudotime stepping methods

from dolfin import *

from ..tool_box import variational_tools

from ..tool_box import homogenization_tools

from ..tool_box import constitutive_tools

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import numerical_tools

from ..tool_box import read_write_tools

from ...PythonicUtilities import path_tools

from ...PythonicUtilities import file_handling_tools as file_tools

from ...PythonicUtilities.dictionary_tools import get_first_key_from_value

########################################################################
#                      Post-processing tools list                      #
########################################################################

########################################################################
########################################################################
##                            Field saving                            ##
########################################################################
########################################################################

# Defines a function to initialize and save a field

def initialize_fieldSaving(data, direct_codeData, submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    intermediate_saving = data[2]

    readable_xdmf_flag = data[3]

    visualization_copy = data[4]

    field_name = data[5]

    # Gets the functional data class

    functional_data_class = direct_codeData[0]

    mesh_data_class = direct_codeData[1]

    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)
    
    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, file_name):

            # Defines a flag for intermediate saving to allow visualiza-
            # tion during solution

            self.intermediate_saving = intermediate_saving 

            # Defines a flag to save the xdmf file as a readable file. 
            # This allows the saved file to be read afterwards as a 
            # function in fenics

            self.readable_xdmf_flag = readable_xdmf_flag

            # Sets a counter of solution steps

            self.solution_steps = 0

            # Creates the file to have all the time steps. If a readable
            # xdmf file is to be use, creates a None object first, as the
            # write_field_to_xdmf will take control of everything

            if self.readable_xdmf_flag:

                self.result = None 

                # Creates an instance of the FunctionalData class to 
                # store solution and function spaces

                self.functional_data_dict = functional_data_class

                self.mesh_data_class = mesh_data_class

                self.file_name = file_name

                # Sets the information for visualization copies, since
                # the readable xdmf files might not appear in some com-
                # puters

                self.visualization_copy = visualization_copy

                self.visualization_copy_file = None

                # Saves into the class the information to build the mock
                # functional data class for the visualization copy

                self.field_type = functional_data_class.elements_dictionary_copy[
                field_name]["field type"]

                self.interpolation_function = functional_data_class.elements_dictionary_copy[
                field_name]["interpolation function"]

                self.polynomial_degree = functional_data_class.elements_dictionary_copy[
                field_name]["polynomial degree"]

            else:
            
                self.result = XDMFFile(file_name+".xdmf")

                # Createa a dummy functional data class

                self.functional_data_dict = None

                self.mesh_data_class = None

            # Sets the result. It can be either a single file or a set 
            # of files when the solution is to be independently saved

            if self.intermediate_saving:

                # Saves the base name

                self.base_fileName = file_name

    # Initializes the output object

    output_object = OutputObject(file_name)

    return output_object

# Defines a function to update the file with the field

def update_fieldSaving(output_object, field, field_number, time, 
fields_namesDict):
    
    print("Updates the saving of the "+str(field_number)+" field\n")

    # Gets the name of the field

    field_name = get_first_key_from_value(fields_namesDict, field_number)

    # Verifies if each load step must be saved in a separate file to al-
    # low visualization during simulation

    if output_object.intermediate_saving:

        # Updates the counter of solution steps

        output_object.solution_steps += 1

        # Adds a new file to this time step

        file_name = output_object.base_fileName+"_"+str(
        output_object.solution_steps)+".xdmf"

        current_result = XDMFFile(file_name)

        # If the problem has a single field

        if field_number==-1:

            # Verifies if a readable xdmf is to be used

            if output_object.readable_xdmf_flag:

                # Writes the field to the main file using the proper me-
                # thod for writing a readable xdmf

                writing_result = read_write_tools.write_field_to_xdmf(
                output_object.functional_data_dict, time=time, 
                field_name=field_name, visualization_copy=
                output_object.visualization_copy, close_file=False,
                file=output_object.result, time_step=
                output_object.solution_steps-1, explicit_file_name=
                output_object.file_name, visualization_copy_file=
                output_object.visualization_copy_file, 
                code_given_mesh_data_class=output_object.mesh_data_class,
                field_type=output_object.field_type, interpolation_function=
                output_object.interpolation_function, polynomial_degree=
                output_object.polynomial_degree)

                # Separates the readable file off the visualization copy
                # file, if there is any

                if output_object.visualization_copy:

                    output_object.result = writing_result[0]

                    output_object.visualization_copy_file = (
                    writing_result[1])

                else:

                    output_object.result = writing_result

            else:

                # Writes the field to the main file

                output_object.result.write(field, time)

            # Writes the field to the extra file

            current_result.write(field, time)
        
        # If the problem has multiple fields

        else:

            # Verifies if a readable xdmf is to be used

            if output_object.readable_xdmf_flag:

                # Writes the field to the main file using the proper me-
                # thod for writing a readable xdmf

                writing_result = read_write_tools.write_field_to_xdmf(
                output_object.functional_data_dict, time=time, 
                field_name=field_name, visualization_copy=
                output_object.visualization_copy, close_file=False,
                file=output_object.result, time_step=
                output_object.solution_steps-1, explicit_file_name=
                output_object.file_name, visualization_copy_file=
                output_object.visualization_copy_file, 
                code_given_mesh_data_class=output_object.mesh_data_class,
                field_type=output_object.field_type, interpolation_function=
                output_object.interpolation_function, polynomial_degree=
                output_object.polynomial_degree)

                # Separates the readable file off the visualization copy
                # file, if there is any

                if output_object.visualization_copy:

                    output_object.result = writing_result[0]

                    output_object.visualization_copy_file = (
                    writing_result[1])

                else:

                    output_object.result = writing_result

            else:
                
                output_object.result.write(field[field_number], time)

            # Writes the field to the extra file

            current_result.write(field[field_number], time)

        current_result.close()

        # Returns the output class

        return output_object

    else:

        # If the problem has a single field

        if field_number==-1:

            # Verifies if a readable xdmf is to be used

            if output_object.readable_xdmf_flag:

                # Writes the field to the main file using the proper me-
                # thod for writing a readable xdmf

                writing_result = read_write_tools.write_field_to_xdmf(
                output_object.functional_data_dict, time=time, 
                field_name=field_name, visualization_copy=
                output_object.visualization_copy, close_file=False,
                file=output_object.result, time_step=
                output_object.solution_steps-1, explicit_file_name=
                output_object.file_name, visualization_copy_file=
                output_object.visualization_copy_file, 
                code_given_mesh_data_class=output_object.mesh_data_class,
                field_type=output_object.field_type, interpolation_function=
                output_object.interpolation_function, polynomial_degree=
                output_object.polynomial_degree)

                # Separates the readable file off the visualization copy
                # file, if there is any

                if output_object.visualization_copy:

                    output_object.result = writing_result[0]

                    output_object.visualization_copy_file = (
                    writing_result[1])

                else:

                    output_object.result = writing_result

            else:
                
                output_object.result.write(field, time)

            return output_object
        
        # If the problem has multiple fields

        else:

            # Verifies if a readable xdmf is to be used

            if output_object.readable_xdmf_flag:

                # Writes the field to the main file using the proper me-
                # thod for writing a readable xdmf

                writing_result = read_write_tools.write_field_to_xdmf(
                output_object.functional_data_dict, time=time, 
                field_name=field_name, visualization_copy=
                output_object.visualization_copy, close_file=False,
                file=output_object.result, time_step=
                output_object.solution_steps-1, explicit_file_name=
                output_object.file_name, visualization_copy_file=
                output_object.visualization_copy_file, 
                code_given_mesh_data_class=output_object.mesh_data_class,
                field_type=output_object.field_type, interpolation_function=
                output_object.interpolation_function, polynomial_degree=
                output_object.polynomial_degree)

                # Separates the readable file off the visualization copy
                # file, if there is any

                if output_object.visualization_copy:

                    output_object.result = writing_result[0]

                    output_object.visualization_copy_file = (
                    writing_result[1])

                else:

                    output_object.result = writing_result

            else:
                
                output_object.result.write(field[field_number], time)

            return output_object

########################################################################
#                          Cauchy stress field                         #
########################################################################

# Defines a function to initialize the Cauchy stress field file

def initialize_cauchyStressSaving(data, direct_codeData, submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the polynomial degree of the interpolation function

    polynomial_degree = data[2]

    # Gets the mesh, the constitutive model, and the volume integrator
    # from the data directly provided by the code

    mesh = direct_codeData[0]

    constitutive_model = direct_codeData[1]

    dx = direct_codeData[2]

    physical_groupsList = direct_codeData[3] 
    
    physical_groupsNamesToTags = direct_codeData[4]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = TensorFunctionSpace(mesh, "DG", 0)

    else:

        W = TensorFunctionSpace(mesh, "CG", polynomial_degree)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = XDMFFile(file_name)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags, 
        parent_toChildMeshResult):
            
            self.result = file 

            self.W = W 

            self.constitutive_model = constitutive_model

            self.dx = dx 

            self.physical_groupsList = physical_groupsList 

            self.physical_groupsNamesToTags = physical_groupsNamesToTags
            
            # Defines a sharable result between a parent mesh and a sub-
            # mesh

            self.parent_toChildMeshResult = parent_toChildMeshResult

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = OutputObject(file, W, constitutive_model, dx, 
    physical_groupsList, physical_groupsNamesToTags, 0.0)

    return output_object

# Defines a function to update the Cauchy stress field

def update_cauchyStressSaving(output_object, field, field_number, time, 
fields_namesDict, flag_parentMeshReuse=False):
    
    print("Updates the saving of the Cauchy stress field\n")
    
    return constitutive_tools.save_stressField(output_object, field, 
    time, flag_parentMeshReuse, ["Cauchy stress", "stress"], "cauchy", 
    "cauchy_stress", fields_namesDict)

########################################################################
#                      Couple Cauchy stress field                      #
########################################################################

# Defines a function to initialize the couple Cauchy stress field file

def initialize_coupleCauchyStressSaving(data, direct_codeData, 
submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the polynomial degree of the interpolation function

    polynomial_degree = data[2]

    # Gets the mesh, the constitutive model, and the volume integrator
    # from the data directly provided by the code

    mesh = direct_codeData[0]

    constitutive_model = direct_codeData[1]

    dx = direct_codeData[2]

    physical_groupsList = direct_codeData[3] 
    
    physical_groupsNamesToTags = direct_codeData[4]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = TensorFunctionSpace(mesh, "DG", 0)

    else:

        W = TensorFunctionSpace(mesh, "CG", polynomial_degree)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = XDMFFile(file_name)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the couple stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags,
        parent_toChildMeshResult):
            
            self.result = file 

            self.W = W 

            self.constitutive_model = constitutive_model

            self.dx = dx 

            self.physical_groupsList = physical_groupsList 

            self.physical_groupsNamesToTags = physical_groupsNamesToTags
            
            # Defines a sharable result between a parent mesh and a sub-
            # mesh

            self.parent_toChildMeshResult = parent_toChildMeshResult

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = OutputObject(file, W, constitutive_model, dx, 
    physical_groupsList, physical_groupsNamesToTags, 0.0)

    return output_object

# Defines a function to update the couple Cauchy stress field

def update_coupleCauchyStressSaving(output_object, field, field_number, 
time, fields_namesDict, flag_parentMeshReuse=False):
    
    print("Updates the saving of the couple Cauchy stress field\n")

    return constitutive_tools.save_stressField(output_object, field, 
    time, flag_parentMeshReuse, ["Couple Cauchy stress", "stress"], "c"+
    "ouple_cauchy", "cauchy_stress", fields_namesDict)

########################################################################
#                  First Piola-Kirchhoff stress field                  #
########################################################################

# Defines a function to initialize the first Piola-Kirchhoff stress 
# field file

def initialize_firstPiolaStressSaving(data, direct_codeData, 
submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the polynomial degree of the interpolation function

    polynomial_degree = data[2]

    # Gets the mesh, the constitutive model, and the volume integrator
    # from the data directly provided by the code

    mesh = direct_codeData[0]

    constitutive_model = direct_codeData[1]

    dx = direct_codeData[2]

    physical_groupsList = direct_codeData[3] 
    
    physical_groupsNamesToTags = direct_codeData[4]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = TensorFunctionSpace(mesh, "DG", 0)

    else:

        W = TensorFunctionSpace(mesh, "CG", polynomial_degree)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = XDMFFile(file_name)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags, 
        parent_toChildMeshResult):
            
            self.result = file 

            self.W = W 

            self.constitutive_model = constitutive_model

            self.dx = dx 

            self.physical_groupsList = physical_groupsList 

            self.physical_groupsNamesToTags = physical_groupsNamesToTags
            
            # Defines a sharable result between a parent mesh and a sub-
            # mesh

            self.parent_toChildMeshResult = parent_toChildMeshResult

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = OutputObject(file, W, constitutive_model, dx, 
    physical_groupsList, physical_groupsNamesToTags, 0.0)

    return output_object

# Defines a function to update the first Piola-Kirchhoff stress field

def update_firstPiolaStressSaving(output_object, field, field_number, 
time, fields_namesDict, flag_parentMeshReuse=False):
    
    print("Updates the saving of the first Piola-Kirchhoff stress fiel"+
    "d\n")
    
    return constitutive_tools.save_stressField(output_object, field, 
    time, flag_parentMeshReuse, ["First Piola-Kirchhoff stress", "stre"+
    "ss"], "first_piola_kirchhoff", "first_piolaStress", 
    fields_namesDict)

########################################################################
#              Couple first Piola-Kirchhoff stress field               #
########################################################################

# Defines a function to initialize the couple first Piola-Kirchhoff 
# stress field file

def initialize_coupleFirstPiolaStressSaving(data, direct_codeData, 
submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the polynomial degree of the interpolation function

    polynomial_degree = data[2]

    # Gets the mesh, the constitutive model, and the volume integrator
    # from the data directly provided by the code

    mesh = direct_codeData[0]

    constitutive_model = direct_codeData[1]

    dx = direct_codeData[2]

    physical_groupsList = direct_codeData[3] 
    
    physical_groupsNamesToTags = direct_codeData[4]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = TensorFunctionSpace(mesh, "DG", 0)

    else:

        W = TensorFunctionSpace(mesh, "CG", polynomial_degree)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = XDMFFile(file_name)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags, 
        parent_toChildMeshResult):
            
            self.result = file 

            self.W = W 

            self.constitutive_model = constitutive_model

            self.dx = dx 

            self.physical_groupsList = physical_groupsList 

            self.physical_groupsNamesToTags = physical_groupsNamesToTags
            
            # Defines a sharable result between a parent mesh and a sub-
            # mesh

            self.parent_toChildMeshResult = parent_toChildMeshResult

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = OutputObject(file, W, constitutive_model, dx, 
    physical_groupsList, physical_groupsNamesToTags, 0.0)

    return output_object

# Defines a function to update the couple first Piola-Kirchhoff stress 
# field

def update_coupleFirstPiolaStressSaving(output_object, field, 
field_number, time, fields_namesDict, flag_parentMeshReuse=False):
    
    print("Updates the saving of the couple first Piola-Kirchhoff stre"+
    "ss field\n")
    
    return constitutive_tools.save_stressField(output_object, field, 
    time, flag_parentMeshReuse, ["Couple first Piola-Kirchhoff stress", 
    "stress"], "couple_first_piola_kirchhoff", "first_piolaStress", 
    fields_namesDict)

########################################################################
#                            Traction fields                           #
########################################################################

# Defines a function to initialize the traction field file

def initialize_tractionSaving(data, direct_codeData, submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the polynomial degree of the interpolation function

    polynomial_degree = data[2]

    # Gets the mesh, the constitutive model, and the surface integrator
    # from the data directly provided by the code

    mesh = direct_codeData[0]

    constitutive_model = direct_codeData[1]

    ds = direct_codeData[2]

    physical_groupsList = direct_codeData[3] 
    
    physical_groupsNamesToTags = direct_codeData[4]

    referential_normal = direct_codeData[5]

    # Creates the function space for the traction as a vector function 
    # space

    W = 0.0

    if polynomial_degree==0:

        W = VectorFunctionSpace(mesh, "DG", 0)

    else:

        W = VectorFunctionSpace(mesh, "CG", polynomial_degree)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = XDMFFile(file_name)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, ds, 
        physical_groupsList, physical_groupsNamesToTags, 
        referential_normal):

            self.W = W 

            self.constitutive_model = constitutive_model

            self.ds = ds 

            self.result = file

            self.physical_groupsList = physical_groupsList 

            self.physical_groupsNamesToTags = physical_groupsNamesToTags

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

            self.referential_normal = referential_normal

    output_object = OutputObject(file, W, constitutive_model, ds, 
    physical_groupsList, physical_groupsNamesToTags, referential_normal)

    return output_object

# Defines a function to update the pressure field

def update_referentialTractionSaving(output_object, field, field_number, 
time, fields_namesDict):
    
    print("Updates the saving of referential traction field\n")
    
    return constitutive_tools.save_referentialTraction(output_object, 
    field, time, "first_piola_kirchhoff", "first_piolaStress", 
    fields_namesDict)

########################################################################
########################################################################
##                          Fields at points                          ##
########################################################################
########################################################################

# Defines a function to initialize the pressure field file

def initialize_pressureAtPointSaving(data, direct_codeData, submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    n_digits = 3

    if len(data)==6:

        n_digits = data[5]

    # Gets the polynomial degree of the interpolation function

    polynomial_degree = data[2]

    # Gets the mesh, the constitutive model, and the volume integrator
    # from the data directly provided by the code

    mesh = direct_codeData[0]

    constitutive_model = direct_codeData[1]

    dx = direct_codeData[2]

    physical_groupsList = direct_codeData[3] 
    
    physical_groupsNamesToTags = direct_codeData[4]

    # Gets the coordinates of the point and already finds the closest 
    # node to it

    point_coordinates = mesh_tools.find_nodeClosestToPoint(mesh, data[3],
    None, None)[1]

    # Gets the flag for plotting or not

    flag_plotting = data[4]

    # Creates the function space for the pressure as a scalar

    W = 0.0

    if polynomial_degree==0:

        W = FunctionSpace(mesh, "DG", 0)

    else:

        W = FunctionSpace(mesh, "CG", polynomial_degree)

    # Verifies if an extension has been added to the file name

    if len(file_name)>4:

        if file_name[-4:len(file_name)]==".txt":

            file_name = file_name[0:-4]

    # Initializes the list of pressure values along the loading steps

    pressure_list = []

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file_name, parent_path, W, constitutive_model, 
        dx, physical_groupsList, physical_groupsNamesToTags, n_digits,
        parent_toChildMeshResult, point_coordinates, pressure_list,
        flag_plotting):

            self.W = W 

            self.constitutive_model = constitutive_model

            self.dx = dx 

            self.file_name = file_name

            self.parent_path = parent_path

            self.physical_groupsList = physical_groupsList 

            self.physical_groupsNamesToTags = physical_groupsNamesToTags

            self.point_coordinates = point_coordinates

            self.result = pressure_list

            self.flag_plotting = flag_plotting

            self.digits = n_digits

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = OutputObject(file_name, parent_path, W, 
    constitutive_model, dx, physical_groupsList, 
    physical_groupsNamesToTags, n_digits, 0.0, point_coordinates, 
    pressure_list, flag_plotting)

    return output_object

# Defines a function to update the pressure field

def update_pressureAtPointSaving(output_object, field, field_number, time, 
fields_namesDict):
    
    print("Updates the saving of the pressure at point "+str(
    output_object.point_coordinates)+"\n")
    
    return constitutive_tools.save_pressureAtPoint(output_object, field, 
    time, "cauchy", "cauchy_stress", fields_namesDict, digits=
    output_object.digits)

########################################################################
########################################################################
##                          Mesh properties                           ##
########################################################################
########################################################################

# Defines a function to initialize the file to store the mesh volume a-
# long iterations

def initialize_mesh_volume(data, direct_codeData, submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the volume integrator from the data directly provided by the 
    # code

    dx = direct_codeData[0]

    # Gets the initial volume

    initial_volume = assemble(1.0*dx)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Verifies if an extension has been added to the file name

    if len(file_name)>4:

        if file_name[-4:len(file_name)]==".txt":

            file_name = file_name[0:-4]

    # Initializes the list of values of the ratio of the new volume to
    # the reference volume along the loading steps

    volume_ratio_list = []

    # Assembles the file and the function space into a class

    class OutputObject:

        def __init__(self, file_name, dx, volume_ratio_list, 
        initial_volume):

            self.dx = dx 

            self.file_name = file_name

            self.result = volume_ratio_list

            self.initial_volume = initial_volume

    output_object = OutputObject(file_name, dx, volume_ratio_list, 
    initial_volume)

    return output_object

# Defines a function to update the mesh volume

def update_mesh_volume(output_object, field, field_number, time, 
fields_namesDict):
    
    # Verifies if the field is the displacement field

    field_name = get_first_key_from_value(fields_namesDict, field_number)

    if field_name!="Displacement":

        raise NameError("The field name is '"+str(field_name)+"', but "+
        "the field required to evaluate the 'SaveMeshVolumeRatioToRefe"+
        "renceVolume' post-process must be 'Displacement'")
    
    print("Updates the saving of the ratio of the meshe's volume to th"+
    "e initial volume of the mesh\n")

    # Gets the jacobian

    I = Identity(3)

    F = grad(field[field_number])+I 

    J = det(F)
    
    # Calculates the new mesh volume

    new_volume = assemble(J*output_object.dx) 

    # Gets the ratio of the new volume to the reference

    ratio = new_volume/output_object.initial_volume

    # Appends this result to the output class

    output_object.result.append([time, ratio])

    # Saves the list of volume ratios in a txt file

    file_tools.list_toTxt(output_object.result, output_object.file_name, 
    add_extension=True)
    
    return output_object

########################################################################
#                            Homogenization                            #
########################################################################

# Defines a function to initialize the homogenization of the field

def initialize_fieldHomogenization(data, direct_codeData, submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the subdomain to integrate

    subdomain = data[2]

    # Gets the integration measure

    dx = direct_codeData[0]

    physical_groupsList = direct_codeData[1] 
    
    physical_groupsNamesToTags = direct_codeData[2]

    # Evaluates the volume of the domain

    volume = 0.0

    # If the solution comes from a submesh, there can be no domain

    if submesh_flag:

        if (isinstance(subdomain, int) or isinstance(subdomain, tuple)
        or isinstance(subdomain, list)):
            
            raise ValueError("This solution comes from a submesh and t"+
            "he subdomain "+str(subdomain)+" is solicited. Subdomains "+
            "cannot be used in fields from submeshes, for theses meshe"+
            "s do not have physical groups.")

    # If a physical group of the mesh is given or a tuple of physical 
    # groups

    if isinstance(subdomain, list):

        # Checks for any elements being strings

        for i in range(len(subdomain)):
            
            if isinstance(subdomain[i], str):

                subdomain[i] = variational_tools.verify_physicalGroups(
                subdomain[i], physical_groupsList, 
                physical_groupsNamesToTags=physical_groupsNamesToTags)

        # Converts to tuple

        subdomain = tuple(subdomain)

    if isinstance(subdomain, int):

        volume = assemble(1*dx(subdomain))

    elif isinstance(subdomain, tuple):

        for sub in subdomain:

            volume += assemble(1*dx(sub))

    # Otherwise, integrates over the whole domain to get the volume

    elif isinstance(subdomain, str):

        if len(subdomain)==0:

            volume = assemble(1*dx)

        else:

            subdomain = variational_tools.verify_physicalGroups(
            subdomain, physical_groupsList, physical_groupsNamesToTags=
            physical_groupsNamesToTags)

            volume = assemble(1*dx(subdomain))

    # Initializes the homogenized field list

    homogenized_fieldList = []

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, homogenized_fieldList, inverse_volume, dx, 
        subdomain, file_name):
            
            self.result = homogenized_fieldList

            self.inverse_volume = inverse_volume

            self.dx = dx 

            self.subdomain = subdomain 

            self.file_name = file_name

    output_object = OutputObject(homogenized_fieldList, (1.0/volume), dx, 
    subdomain, file_name)

    return output_object

# Defines a function to update the homogenized field

def update_fieldHomogenization(output_object, field, field_number, time,
fields_namesDict):
    
    print("Updates the homogenization of the "+str(field_number)+" fie"+
    "ld\n")

    # If the problem has a single field

    if field_number==-1:

        # Homogenizes the field and updates the list of homogenized field
        # along time

        output_object.result = homogenization_tools.homogenize_genericField(
        field, output_object.result, time, output_object.inverse_volume, 
        output_object.dx, output_object.subdomain, 
        output_object.file_name)

        return output_object

    # If the problem has multiple fields

    else:

        # Homogenizes the field and updates the list of homogenized field
        # along time

        output_object.result = homogenization_tools.homogenize_genericField(
        field[field_number], output_object.result, time, 
        output_object.inverse_volume, output_object.dx, 
        output_object.subdomain, output_object.file_name)

        return output_object

# Defines a function to initialize the homogenization of the gradient of 
# a field

def initialize_gradientFieldHomogenization(data, direct_codeData, 
submesh_flag):

    output = initialize_fieldHomogenization(data, direct_codeData, 
    submesh_flag)

    return output

# Defines a function to update the homogenization of the gradient of a 
# field

def update_gradientFieldHomogenization(output_object, field, 
field_number, time, fields_namesDict):
    
    print("Updates the homogenization of the gradient of the "+str(
    field_number)+" field\n")

    # Gets the gradient of the field

    grad_field = 0.0

    # If the problem has only one field

    if field_number==-1:

        grad_field = grad(field)

    # If the problem has multiple fields

    else:

        grad_field = grad(field[field_number])

    # Gets the homogenization of the gradient. Sets the field number to
    # -1, for a single field is sent downstream

    output_object = update_fieldHomogenization(output_object, 
    grad_field, -1, time, fields_namesDict)

    return output_object

# Defines a function to initialize the homogenized value of the first 
# Piola-Kirchhof

def initialize_firstPiolaHomogenization(data, direct_codeData, 
submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the subdomain to integrate

    subdomain = data[2]

    # Gets the integration measure

    dx = direct_codeData[0]

    physical_groupsList = direct_codeData[1] 
    
    physical_groupsNamesToTags = direct_codeData[2]

    constitutive_model = direct_codeData[3]

    # Evaluates the volume of the domain

    volume = 0.0

    # If the solution comes from a submesh, there can be no domain

    if submesh_flag:

        if (isinstance(subdomain, int) or isinstance(subdomain, tuple)
        or isinstance(subdomain, list)):
            
            raise ValueError("This solution comes from a submesh and t"+
            "he subdomain "+str(subdomain)+" is solicited. Subdomains "+
            "cannot be used in fields from submeshes, for theses meshe"+
            "s do not have physical groups.")

    # If a physical group of the mesh is given or a tuple of physical 
    # groups

    if isinstance(subdomain, list):

        subdomain = tuple(subdomain)

    if isinstance(subdomain, int):

        volume = assemble(1*dx(subdomain))

    elif isinstance(subdomain, tuple):

        for sub in subdomain:

            if isinstance(sub, str):

                volume += assemble(1*dx(variational_tools.verify_physicalGroups(
                sub, physical_groupsList, physical_groupsNamesToTags=
                physical_groupsNamesToTags)))

            else:

                volume += assemble(1*dx(sub))

    # Otherwise, integrates over the whole domain to get the volume

    elif isinstance(subdomain, str):

        if len(subdomain)==0:

            volume = assemble(1*dx)

        else:

            volume = assemble(1*dx(variational_tools.verify_physicalGroups(
            subdomain, physical_groupsList, physical_groupsNamesToTags=
            physical_groupsNamesToTags)))

    # Initializes the homogenized field list

    homogenized_firstPiolaList = []

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, homogenized_firstPiolaList, inverse_volume, dx, 
        subdomain, file_name, constitutive_model, physical_groupsList,
        physical_groupsNamesToTags):
            
            self.result = homogenized_firstPiolaList

            self.inverse_volume = inverse_volume

            self.dx = dx 

            self.subdomain = subdomain 

            self.file_name = file_name

            self.constitutive_model = constitutive_model

            self.physical_groupsList = physical_groupsList
    
            self.physical_groupsNamesToTags = physical_groupsNamesToTags

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = OutputObject(homogenized_firstPiolaList, (1.0/volume
    ), dx, subdomain, file_name, constitutive_model, physical_groupsList,
    physical_groupsNamesToTags)

    return output_object

# Defines a function to update the homogenization of the first Piola-
# Kirchhof

def update_firstPiolaHomogenization(output_object, field, field_number, 
time, fields_namesDict):
    
    print("Updates the homogenization of the first Piola-Kirchhoff str"+
    "ess field\n")

    output_object.result = homogenization_tools.homogenize_stressTensor(
    field, output_object.constitutive_model, "first_piola_kirchhoff", 
    "first_piolaStress", output_object.result, time, 
    output_object.inverse_volume, output_object.dx, 
    output_object.subdomain,output_object.file_name, 
    output_object.physical_groupsList, 
    output_object.physical_groupsNamesToTags, fields_namesDict, 
    output_object.required_fieldsNames)

    return output_object

# Defines a function to initialize the homogenized value of the first 
# couple Piola-Kirchhof

def initialize_coupleFirstPiolaHomogenization(data, direct_codeData, 
submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the subdomain to integrate

    subdomain = data[2]

    # Gets the integration measure

    dx = direct_codeData[0]

    physical_groupsList = direct_codeData[1] 
    
    physical_groupsNamesToTags = direct_codeData[2]

    constitutive_model = direct_codeData[3]

    position_vector = direct_codeData[4]

    # Evaluates the volume of the domain

    volume = 0.0

    # If the solution comes from a submesh, there can be no domain

    if submesh_flag:

        if (isinstance(subdomain, int) or isinstance(subdomain, tuple)
        or isinstance(subdomain, list)):
            
            raise ValueError("This solution comes from a submesh and t"+
            "he subdomain "+str(subdomain)+" is solicited. Subdomains "+
            "cannot be used in fields from submeshes, for theses meshe"+
            "s do not have physical groups.")

    # If a physical group of the mesh is given or a tuple of physical 
    # groups

    if isinstance(subdomain, list):

        subdomain = tuple(subdomain)

    if isinstance(subdomain, int):

        volume = assemble(1*dx(subdomain))

    elif isinstance(subdomain, tuple):

        for sub in subdomain:

            if isinstance(sub, str):

                volume += assemble(1*dx(variational_tools.verify_physicalGroups(
                sub, physical_groupsList, physical_groupsNamesToTags=
                physical_groupsNamesToTags)))

            else:

                volume += assemble(1*dx(sub))

    # Otherwise, integrates over the whole domain to get the volume

    elif isinstance(subdomain, str):

        if len(subdomain)==0:

            volume = assemble(1*dx)

        else:

            volume = assemble(1*dx(variational_tools.verify_physicalGroups(
            subdomain, physical_groupsList, physical_groupsNamesToTags=
            physical_groupsNamesToTags)))

    # Initializes the homogenized field list

    homogenized_firstPiolaList = []

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, homogenized_firstPiolaList, inverse_volume, dx, 
        subdomain, file_name, constitutive_model, physical_groupsList,
        physical_groupsNamesToTags, position_vector):
            
            self.result = homogenized_firstPiolaList

            self.inverse_volume = inverse_volume

            self.dx = dx 

            self.position_vector = position_vector

            self.subdomain = subdomain 

            self.file_name = file_name

            self.constitutive_model = constitutive_model

            self.physical_groupsList = physical_groupsList
    
            self.physical_groupsNamesToTags = physical_groupsNamesToTags

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = OutputObject(homogenized_firstPiolaList, (1.0/volume
    ), dx, subdomain, file_name, constitutive_model, physical_groupsList,
    physical_groupsNamesToTags, position_vector)

    return output_object

# Defines a function to update the homogenization of the couple first 
# Piola-Kirchhof

def update_coupleFirstPiolaHomogenization(output_object, field, 
field_number, time, fields_namesDict):
    
    print("Updates the homogenization of the couple first Piola-Kirchh"+
    "off stress field\n")

    """output_object.result = homogenization_tools.homogenize_stressTensor(
    field, output_object.constitutive_model, "couple_first_piola_kirch"+
    "hoff", "first_piolaStress", output_object.result, time, 
    output_object.inverse_volume, output_object.dx, 
    output_object.subdomain,output_object.file_name, 
    output_object.physical_groupsList, 
    output_object.physical_groupsNamesToTags, fields_namesDict, 
    output_object.required_fieldsNames)"""

    output_object.result = homogenization_tools.homogenize_coupleFirstPiola(
    field, output_object.constitutive_model, output_object.result, time, 
    output_object.position_vector, output_object.inverse_volume, 
    output_object.dx, output_object.subdomain, output_object.file_name, 
    output_object.physical_groupsList, 
    output_object.physical_groupsNamesToTags, fields_namesDict, 
    output_object.required_fieldsNames)

    return output_object

# Defines a function to initialize the homogenized value of the Cauchy
# stress over the reference configuration

def initialize_cauchyHomogenization(data, direct_codeData, submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the subdomain to integrate

    subdomain = data[2]

    # Gets the integration measure

    dx = direct_codeData[0]

    physical_groupsList = direct_codeData[1] 
    
    physical_groupsNamesToTags = direct_codeData[2]

    constitutive_model = direct_codeData[3]

    # Evaluates the volume of the domain

    volume = 0.0

    # If the solution comes from a submesh, there can be no domain

    if submesh_flag:

        if (isinstance(subdomain, int) or isinstance(subdomain, tuple)
        or isinstance(subdomain, list)):
            
            raise ValueError("This solution comes from a submesh and t"+
            "he subdomain "+str(subdomain)+" is solicited. Subdomains "+
            "cannot be used in fields from submeshes, for theses meshe"+
            "s do not have physical groups.")

    # If a physical group of the mesh is given or a tuple of physical 
    # groups

    if isinstance(subdomain, list):

        subdomain = tuple(subdomain)

    if isinstance(subdomain, int):

        volume = assemble(1*dx(subdomain))

    elif isinstance(subdomain, tuple):

        for sub in subdomain:

            if isinstance(sub, str):

                volume += assemble(1*dx(variational_tools.verify_physicalGroups(
                sub, physical_groupsList, physical_groupsNamesToTags=
                physical_groupsNamesToTags)))

            else:

                volume += assemble(1*dx(sub))

    # Otherwise, integrates over the whole domain to get the volume

    elif isinstance(subdomain, str):

        if len(subdomain)==0:

            volume = assemble(1*dx)

        else:

            volume = assemble(1*dx(variational_tools.verify_physicalGroups(
            subdomain, physical_groupsList, physical_groupsNamesToTags=
            physical_groupsNamesToTags)))

    # Initializes the homogenized field list

    homogenized_cauchyList = []

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, homogenized_cauchyList, inverse_volume, dx, 
        subdomain, file_name, constitutive_model, physical_groupsList,
        physical_groupsNamesToTags):
            
            self.result = homogenized_cauchyList

            self.inverse_volume = inverse_volume

            self.dx = dx 

            self.subdomain = subdomain 

            self.file_name = file_name

            self.constitutive_model = constitutive_model

            self.physical_groupsList = physical_groupsList
    
            self.physical_groupsNamesToTags = physical_groupsNamesToTags

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = OutputObject(homogenized_cauchyList, (1.0/volume
    ), dx, subdomain, file_name, constitutive_model, physical_groupsList,
    physical_groupsNamesToTags)

    return output_object

# Defines a function to update the homogenization of the first Piola-
# Kirchhof

def update_cauchyHomogenization(output_object, field, field_number, 
time, fields_namesDict):
    
    print("Updates the homogenization of the Cauchy stress field\n")

    output_object.result = homogenization_tools.homogenize_stressTensor(
    field, output_object.constitutive_model, "cauchy", "cauchy_stress", 
    output_object.result, time, output_object.inverse_volume, 
    output_object.dx, output_object.subdomain, output_object.file_name, 
    output_object.physical_groupsList, 
    output_object.physical_groupsNamesToTags, fields_namesDict, 
    output_object.required_fieldsNames)

    return output_object

# Defines a function to initialize the homogenized value of the couple
# Cauchy stress

def initialize_coupleCauchyHomogenization(data, direct_codeData, 
submesh_flag):
    
    return initialize_cauchyHomogenization(data, direct_codeData, 
    submesh_flag)

# Defines a function to update the homogenization of the couple Cauchy
# stress

def update_coupleCauchyHomogenization(output_object, field, field_number, 
time, fields_namesDict):
    
    print("Updates the homogenization of the couple Cauchy stress fiel"
    "d\n")

    output_object.result = homogenization_tools.homogenize_stressTensor(
    field, output_object.constitutive_model, "couple_cauchy", "cauchy_"+
    "stress", output_object.result, time, output_object.inverse_volume, 
    output_object.dx, output_object.subdomain, output_object.file_name, 
    output_object.physical_groupsList, 
    output_object.physical_groupsNamesToTags, fields_namesDict, 
    output_object.required_fieldsNames)

    return output_object

########################################################################
#                          Elasticity tensors                          #
########################################################################

# Defines a function to initialize the first elasticity tensor

def initialize_firstElasticityTensor(data, direct_codeData, 
submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the polynomial degree of the interpolation function

    polynomial_degree = data[2]

    # Gets the mesh, the constitutive model, and the volume integrator
    # from the data directly provided by the code

    mesh = direct_codeData[0]

    constitutive_model = direct_codeData[1]

    dx = direct_codeData[2]

    physical_groupsList = direct_codeData[3] 
    
    physical_groupsNamesToTags = direct_codeData[4]

    # Gets the coordinates of the point and already finds the closest 
    # node to it

    point_coordinates = mesh_tools.find_nodeClosestToPoint(mesh, data[3],
    None, None)[1]

    # Gets the flag for plotting or not

    flag_plotting = data[4]

    # Gets the Voigt notation

    indices = None 

    # Tests if the Voigt notation is the conventional one

    if data[5]=="conventional":

        # Iterates through the indices to generate the notation 1111,
        # 1122, 1133, 1112, 1123, 1113, 2211, 2222...

        indices = [[0,0], [1,1], [2,2], [0,1], [1,2], [0,2], [1,0], [2,1
        ], [2,0]]

    # Tests if the Voig notation is the Paraview/Fenics one, called na-
    # tural

    elif data[5]=="natural":

        # Iterates through the indices to generate the notation 1111,
        # 1112, 1113, 1121, 1122, 1123, 1131, 1132, 1133, 1211...

        indices = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1
        ], [2,2]]

    else:

        raise TypeError("The Voigt notation information for the evalua"+
        "tion of elasticity tensors must be 'conventional' or 'natural"+
        "'. The given value is: "+str(data[5])+".\n'conventional' is 1"+
        "111, 1122, 1133, 1112, 1123, 1113, 2211, 2222... whereas 'nat"+
        "ural' is 1111, 1112, 1113, 1121, 1122, 1123, 1131, 1132, 1133"+
        ", 1211...")

    # Initializes the Voigt notation object as dictionary of tuple keys 
    # and lists as values

    voigt_notation = {}

    # Populates the Voigt dictionary

    for i in range(9):

        for j in range(9):

            voigt_notation[(i,j)] = [indices[i][0], indices[i][1],
            indices[j][0], indices[j][1]]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = FunctionSpace(mesh, "DG", 0)

    else:

        W = FunctionSpace(mesh, "CG", polynomial_degree)

    # Gets the optional arguments for the plot of the tensor, and 
    # checks them against the default ones

    optional_arguments = data[6]

    if optional_arguments=="default":

        optional_arguments = None

    optional_arguments = numerical_tools.check_additionalParameters(
    optional_arguments, {"scaling function": "logarithmic filter",
    "scaling function additional parameters": {"alpha": 3}, "color"+
    " map": "blue orange green white purple brown pink", "maximum "+
    "ticks on color bar": 24})

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Verifies if an extension has been added to the file name

    if len(file_name)>4:

        if file_name[-4:len(file_name)]==".txt":

            file_name = file_name[0:-4]

    # Initializes the list of elasticity  values along the loading steps

    elasticity_tensorList = []

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file_name, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags, 
        parent_toChildMeshResult, point_coordinates, 
        elasticity_tensorList, flag_plotting, voigt_notation, 
        parent_path, optional_arguments):

            self.W = W 

            self.constitutive_model = constitutive_model

            self.dx = dx 

            self.file_name = file_name

            self.physical_groupsList = physical_groupsList 

            self.physical_groupsNamesToTags = physical_groupsNamesToTags

            self.point_coordinates = point_coordinates

            self.result = elasticity_tensorList

            self.flag_plotting = flag_plotting

            self.voigt_notation = voigt_notation

            self.parent_path = parent_path

            self.optional_arguments = optional_arguments

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

    output_object = output_object = OutputObject(file_name, W, 
    constitutive_model, dx, physical_groupsList, 
    physical_groupsNamesToTags, 0.0, point_coordinates, 
    elasticity_tensorList, flag_plotting, voigt_notation, parent_path,
    optional_arguments)

    return output_object

# Defines a function to update the first elasticity tensor

def update_firstElasticityTensor(output_object, field, field_number, 
time, fields_namesDict, flag_parentMeshReuse=False):
    
    print("Updates the saving of the first elasticity tensor\n")

    return constitutive_tools.save_elasticityTensor(output_object,
    field, time, "first_elasticityTensor", "first_elasticity_tensor",
    fields_namesDict)

# Defines a function to initialize the second elasticity tensor

def initialize_secondElasticityTensor(data, direct_codeData, 
submesh_flag):

    return initialize_firstElasticityTensor(data, direct_codeData, 
    submesh_flag)

# Defines a function to update the second elasticity tensor

def update_secondElasticityTensor(output_object, field, field_number, 
time, fields_namesDict, flag_parentMeshReuse=False):
    
    print("Updates the saving of the second elasticity tensor\n")

    return constitutive_tools.save_elasticityTensor(output_object, 
    field, time, "second_elasticityTensor", "second_elasticity_tensor",
    fields_namesDict)

# Defines a function to initialize the third elasticity tensor

def initialize_thirdElasticityTensor(data, direct_codeData, 
submesh_flag):

    return initialize_firstElasticityTensor(data, direct_codeData, 
    submesh_flag)

# Defines a function to update the third elasticity tensor

def update_thirdElasticityTensor(output_object, field, field_number, 
time, fields_namesDict, flag_parentMeshReuse=False):
    
    print("Updates the saving of the third elasticity tensor\n")

    return constitutive_tools.save_elasticityTensor(output_object, 
    field, time, "third_elasticityTensor", "third_elasticity_tensor",
    fields_namesDict)