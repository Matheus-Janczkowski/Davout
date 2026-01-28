# Routine to store some methods to post-process solution inside the 
# pseudotime stepping methods

from dolfin import *

from ..tool_box import variational_tools

from ..tool_box import homogenization_tools

from ..tool_box import constitutive_tools

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import numerical_tools

from ..tool_box import read_write_tools

from ..tool_box.parallelization_tools import mpi_xdmf_file, mpi_print, mpi_execute_function

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

    comm_object = mesh_data_class.comm

    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)
    
    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, file_name):

            # Saves the comm object

            self.comm_object = comm_object

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
            
                self.result = mpi_xdmf_file(comm_object, file_name+".xdmf")

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
    
    mpi_print(output_object.comm_object, "Updates the saving of the "+
    str(field_number)+" field\n")

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

        current_result = mpi_xdmf_file(output_object.comm_object, 
        file_name, add_termination=True)

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
                output_object.polynomial_degree, comm_object=
                output_object.comm_object)

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
                output_object.polynomial_degree, comm_object=
                output_object.comm_object)

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
                output_object.polynomial_degree, comm_object=
                output_object.comm_object)

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
                output_object.polynomial_degree, comm_object=
                output_object.comm_object)

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

    comm_object = direct_codeData[5]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = TensorFunctionSpace(mesh, "DG", 0)

    else:

        W = TensorFunctionSpace(mesh, "CG", polynomial_degree)
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = mpi_xdmf_file(comm_object, file_name, add_termination=True)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags, 
        parent_toChildMeshResult):

            # Saves the comm object

            self.comm_object = comm_object
            
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
    
    mpi_print(output_object.comm_object, "Updates the saving of the Ca"+
    "uchy stress field\n")
    
    # Verifies if there is a pressure field. A correction to the Cauchy 
    # stress tensor must be added to this case, since the constitutive 
    # model does not give the spherical part of the stress in incompres-
    # sible hyperelasticity

    pressure_correction = Constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
    0.0, 0.0, 0.0]])

    if "Pressure" in fields_namesDict:

        mpi_print(output_object.comm_object, "Adds the correction of t"+
        "he pressure to the saving of the Cauchy stress field\n")

        # Gets the pressure field

        p_field = field[fields_namesDict["Pressure"]]

        # Updates the correction tensor in the spatial configuration

        pressure_correction = p_field*Identity(3)
    
    return constitutive_tools.save_stressField(output_object, field, 
    time, flag_parentMeshReuse, ["Cauchy stress", "stress"], "cauchy", 
    "cauchy_stress", fields_namesDict, pressure_correction=
    pressure_correction)

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

    comm_object = direct_codeData[5]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = TensorFunctionSpace(mesh, "DG", 0)

    else:

        W = TensorFunctionSpace(mesh, "CG", polynomial_degree)
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = mpi_xdmf_file(comm_object, file_name, add_termination=True)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the couple stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags,
        parent_toChildMeshResult):

            # Saves the comm object

            self.comm_object = comm_object
            
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
    
    mpi_print(output_object.comm_object, "Updates the saving of the co"+
    "uple Cauchy stress field\n")

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

    comm_object = direct_codeData[5]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = TensorFunctionSpace(mesh, "DG", 0)

    else:

        W = TensorFunctionSpace(mesh, "CG", polynomial_degree)
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = mpi_xdmf_file(comm_object, file_name, add_termination=True)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags, 
        parent_toChildMeshResult):

            # Saves the comm object

            self.comm_object = comm_object
            
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
    
    mpi_print(output_object.comm_object, "Updates the saving of the fi"+
    "rst Piola-Kirchhoff stress field\n")
    
    # Verifies if the field is the displacement field. Uses the max 
    # function since in single field simulations, the field number is -1

    field_name = get_first_key_from_value(fields_namesDict, max(
    field_number, 0))

    if field_name!="Displacement":

        message = "Field name <-> Field number"

        for key, value in fields_namesDict.items():

            message += "\n"+str(key)+" <-> "+str(value)

        raise NameError("The field name is '"+str(field_name)+"', but "+
        "the field required to evaluate the 'SaveFirstPiolaStressField"+
        "' post-process must be 'Displacement'. The dictionary of fiel"+
        "ds names has the following keys:\n"+message+"\n\nThe asked nu"+
        "mber was "+str(field_number))

    # If there is a single field, field number will be -1

    u_field = None

    if field_number==-1:

        u_field = field

    else:

        u_field = field[field_number]
    
    # Verifies if there is a pressure field. A correction to the first
    # Piola-Kirchhoff stress tensor must be added to this case, since 
    # the constitutive model does not give the spherical part of the
    # stress in incompressible hyperelasticity

    pressure_correction = Constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
    0.0, 0.0, 0.0]])

    if "Pressure" in fields_namesDict:

        mpi_print(output_object.comm_object, "Adds the correction of t"+
        "he pressure to the saving of the first Piola-Kirchhoff stress"+
        " field\n")

        # Gets the deformation gradient and the jacobian

        I = Identity(3)

        F = grad(u_field)+I

        J = det(F)

        # Gets the pressure field

        p_field = field[fields_namesDict["Pressure"]]

        # Updates the correction tensor

        pressure_correction = p_field*J*inv(F.T)
    
    return constitutive_tools.save_stressField(output_object, field, 
    time, flag_parentMeshReuse, ["First Piola-Kirchhoff stress", "stre"+
    "ss"], "first_piola_kirchhoff", "first_piolaStress", 
    fields_namesDict, pressure_correction=pressure_correction)

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

    comm_object = direct_codeData[5]

    # Creates the function space for the stress as a tensor

    W = 0.0

    if polynomial_degree==0:

        W = TensorFunctionSpace(mesh, "DG", 0)

    else:

        W = TensorFunctionSpace(mesh, "CG", polynomial_degree)
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the file

    file = mpi_xdmf_file(comm_object, file_name, add_termination=True)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, dx, 
        physical_groupsList, physical_groupsNamesToTags, 
        parent_toChildMeshResult):

            # Saves the comm object

            self.comm_object = comm_object
            
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
    
    mpi_print(output_object.comm_object, "Updates the saving of the co"+
    "uple first Piola-Kirchhoff stress field\n")
    
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

    mesh_data_class = direct_codeData[0]

    constitutive_model = direct_codeData[1]

    physical_groupsList = direct_codeData[2] 
    
    physical_groupsNamesToTags = mesh_data_class.domain_physicalGroupsNameToTag

    referential_normal = mesh_data_class.n

    comm_object = mesh_data_class.comm

    # Creates the function space for the traction as a vector function 
    # space

    W = 0.0

    if polynomial_degree==0:

        W = VectorFunctionSpace(mesh_data_class.mesh, "DG", 0)

    else:

        W = VectorFunctionSpace(mesh_data_class.mesh, "CG", 
        polynomial_degree)
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # As a boundary physical group can encompass multiple volumetric 
    # physical groups, a new mesh function must be created to break the-
    # se physical groups down. A new ds measure must also be created

    new_boundary_mesh_function, new_boundary_dict = mesh_tools.break_boundary_physical_groups(
    mesh_data_class, insert_domain_tags=True)

    # Creates the new ds measure

    new_ds = Measure("ds", domain=mesh_data_class.mesh, subdomain_data=
    new_boundary_mesh_function)

    # Initializes the file

    file = mpi_xdmf_file(comm_object, file_name, add_termination=True)

    # Assembles the file and the function space into a class. This post-
    # process does have a variable that can be shared with a submesh, 
    # and it is the stress field

    class OutputObject:

        def __init__(self, file, W, constitutive_model, new_ds, 
        physical_groupsList, physical_groupsNamesToTags, 
        referential_normal, new_boundary_dict):

            # Saves the comm object

            self.comm_object = comm_object

            self.W = W 

            self.constitutive_model = constitutive_model

            self.ds = new_ds 

            self.result = file

            self.physical_groupsList = physical_groupsList 

            self.physical_groupsNamesToTags = physical_groupsNamesToTags

            # Gets the names of the fields that are actually necessary
            # to the evaluation of stress

            self.required_fieldsNames = constitutive_tools.get_constitutiveModelFields(
            self.constitutive_model)

            self.referential_normal = referential_normal

            self.new_boundary_dict = new_boundary_dict

    output_object = OutputObject(file, W, constitutive_model, new_ds, 
    physical_groupsList, physical_groupsNamesToTags, referential_normal,
    new_boundary_dict)

    return output_object

# Defines a function to update the pressure field

def update_referentialTractionSaving(output_object, field, field_number, 
time, fields_namesDict):
    
    mpi_print(output_object.comm_object, "Updates the saving of refere"+
    "ntial traction field\n")

    # Verifies if the field is the displacement field. Uses the max 
    # function since in single field simulations, the field number is -1

    field_name = get_first_key_from_value(fields_namesDict, max(
    field_number, 0))

    if field_name!="Displacement":

        message = "Field name <-> Field number"

        for key, value in fields_namesDict.items():

            message += "\n"+str(key)+" <-> "+str(value)

        raise NameError("The field name is '"+str(field_name)+"', but "+
        "the field required to evaluate the 'SaveReferentialTractionFi"+
        "eld' post-process must be 'Displacement'. The dictionary of f"+
        "elds names has the following keys:\n"+message+"\n\nThe asked "+
        "number was "+str(field_number))

    # If there is a single field, field number will be -1

    u_field = None

    if field_number==-1:

        u_field = field

    else:

        u_field = field[field_number]
    
    # Verifies if there is a pressure field. A correction to the first
    # Piola-Kirchhoff stress tensor must be added to this case, since 
    # the constitutive model does not give the spherical part of the
    # stress in incompressible hyperelasticity

    pressure_correction = Constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
    0.0, 0.0, 0.0]])

    if "Pressure" in fields_namesDict:

        mpi_print(output_object.comm_object, "Adds the correction of t"+
        "he pressure to the saving of the referential traction field\n")

        # Gets the deformation gradient and the jacobian

        I = Identity(3)

        F = grad(u_field)+I

        J = det(F)

        # Gets the pressure field

        p_field = field[fields_namesDict["Pressure"]]

        # Updates the correction tensor

        pressure_correction = p_field*J*inv(F.T)
    
    return constitutive_tools.save_referentialTraction(output_object, 
    field, time, "first_piola_kirchhoff", "first_piolaStress", 
    fields_namesDict, pressure_correction=pressure_correction)

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

    comm_object = direct_codeData[5]

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

            # Saves the comm object

            self.comm_object = comm_object

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
    
    mpi_print(output_object.comm_object, "Updates the saving of the pr"+
    "essure at point "+str(output_object.point_coordinates)+"\n")
    
    return constitutive_tools.save_pressureAtPoint(output_object, field, 
    time, "cauchy", "cauchy_stress", fields_namesDict, 
    output_object.comm_object, digits=output_object.digits)

# Defines a function to initialize the file to store thhe strain energy
# of the mesh

def initialize_strain_energy(data, direct_codeData, submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    # Gets the mesh data class and the constitutive model

    mesh_data_class = direct_codeData[0]

    constitutive_model = direct_codeData[1]
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the list of values of the strain energy

    strain_energy_list = []

    # Assembles the file and the function space into a class

    class OutputObject:

        def __init__(self, file_name, mesh_data_class, 
        constitutive_model, strain_energy_list):

            # Saves the comm object

            self.comm_object = mesh_data_class.comm

            self.mesh_data_class = mesh_data_class

            self.file_name = file_name

            self.result = strain_energy_list

            self.constitutive_model = constitutive_model

    output_object = OutputObject(file_name, mesh_data_class, 
    constitutive_model, strain_energy_list)

    return output_object

# Defines a function to update the strain energy

def update_strain_energy(output_object, field, field_number, time, 
fields_namesDict):
    
    # Verifies if the field is the displacement field. Uses the max 
    # function since in single field simulations, the field number is -1

    field_name = get_first_key_from_value(fields_namesDict, max(
    field_number, 0))

    if field_name!="Displacement":

        message = "Field name <-> Field number"

        for key, value in fields_namesDict.items():

            message += "\n"+str(key)+" <-> "+str(value)

        raise NameError("The field name is '"+str(field_name)+"', but "+
        "the field required to evaluate the 'SaveStrainEnergy' post-pr"+
        "ocess must be 'Displacement'. The dictionary of fields names "+
        "has the following keys:\n"+message+"\n\nThe asked number was "+
        str(field_number))
    
    mpi_print(output_object.comm_object, "Updates the saving of the st"+
    "rain energy over the whole mesh\n")

    # Gets the jacobian

    I = Identity(3)

    F = None

    # If there is a single field, field number will be -1

    if field_number==-1:

        F = grad(field)+I 

    else:

        F = grad(field[field_number])+I 

    # Calculates the right Cauchy-Green strain tensor

    C = (F.T)*F

    # Initializes the strain energy value

    strain_energy_value = 0.0
    
    # Verifies if the constitutive model is a dictionary

    if isinstance(output_object.constitutive_model, dict):
        
        # Iterates through the physical groups

        for physical_group, local_constitutive_model in (
        output_object.constitutive_model.items()):
            
            # If the physical group is a tuple

            if isinstance(physical_group, tuple):

                # Iterates through it

                for local_physical_group in physical_group:

                    # Verifies if this physical group is a true physical
                    # group

                    if not (local_physical_group in (
                    output_object.mesh_data_class.domain_physicalGroupsNameToTag)):
                        
                        raise ValueError("The phyical group '"+str(
                        local_physical_group)+"' is not a proper physi"+
                        "cal group in the mesh. It was given at the di"+
                        "ctionary of constitutive models. Check the pr"+
                        "oper physical groups names:\n"+str(list(
                        output_object.mesh_data_class.domain_physicalGroupsNameToTag.keys())))

                    # Adds the contribution of the strain energy of this
                    # physical group

                    strain_energy_value += assemble(
                    local_constitutive_model.strain_energy(C)*
                    output_object.mesh_data_class.dx(
                    output_object.mesh_data_class.domain_physicalGroupsNameToTag[
                    local_physical_group]))

            # Otherwise

            else:

                # Verifies if this physical group is a true physical
                # group

                if not (physical_group in (
                output_object.mesh_data_class.domain_physicalGroupsNameToTag)):
                    
                    raise ValueError("The phyical group '"+str(
                    physical_group)+"' is not a proper physical group "+
                    "in the mesh. It was given at the dictionary of co"+
                    "nstitutive models. Check the proper physical grou"+
                    "ps names:\n"+str(list(
                    output_object.mesh_data_class.domain_physicalGroupsNameToTag.keys())))

                # Adds the contribution of the strain energy of this
                # physical group

                strain_energy_value += assemble(
                local_constitutive_model.strain_energy(C)*
                output_object.mesh_data_class.dx(
                output_object.mesh_data_class.domain_physicalGroupsNameToTag[
                physical_group]))

    # Otherwise, computes for the whole mesh at once

    else:

        strain_energy_value += assemble(
        output_object.constitutive_model.strain_energy(C)*
        output_object.mesh_data_class.dx)

    # Appends this result to the output class

    output_object.result.append([time, strain_energy_value])

    # Saves the list of volume ratios in a txt file

    mpi_execute_function(output_object.comm_object, 
    file_tools.list_toTxt, output_object.result, output_object.file_name, 
    add_extension=True)
    
    return output_object

# Defines a function to initialize the file to store forces and moments 
# on a surface

def initialize_forces_moments_on_surface(data, direct_codeData, 
submesh_flag):

    # Gets the directory and the name of the file

    parent_path = data[0]

    file_name = data[1]

    surface_physical_group = data[2]

    # Gets the mesh data class and the constitutive model

    mesh_data_class = direct_codeData[0]

    constitutive_model = direct_codeData[1]
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Initializes the list of values of the forces and moments in the 
    # following order: fx, fy, fz, mx, my, mz

    forces_moments_list = []

    # As a boundary physical group can encompass multiple volumetric 
    # physical groups, a new mesh function must be created to break the-
    # se physical groups down. A new ds measure must also be created

    new_boundary_mesh_function, new_boundary_dict = mesh_tools.break_boundary_physical_groups(
    mesh_data_class)

    # Creates the new ds measure

    new_ds = Measure("ds", domain=mesh_data_class.mesh, subdomain_data=
    new_boundary_mesh_function)

    # Calculates the centroid of the surface

    surface_centroid, area_inverse = mesh_tools.evaluate_centroidSurface(
    surface_physical_group, mesh_data_class)

    # Transforms the centroid into a constant

    surface_centroid = Constant(surface_centroid)

    # Gets the surface position vector by translating the mesh position
    # vector by the surface centroid

    surface_position_vector = mesh_data_class.x-surface_centroid

    # Assembles the file and the function space into a class

    class OutputObject:

        def __init__(self, file_name, mesh_data_class, 
        constitutive_model, forces_moments_list, new_ds, 
        new_boundary_physical_groups_dict, surface_physical_group,
        surface_position_vector, area_inverse, surface_centroid):

            # Saves the comm object

            self.comm_object = mesh_data_class.comm

            self.mesh_data_class = mesh_data_class

            self.file_name = file_name

            self.result = forces_moments_list

            self.constitutive_model = constitutive_model

            self.new_ds = new_ds

            self.new_boundary_physical_groups_dict = new_boundary_physical_groups_dict

            self.surface_physical_group = surface_physical_group

            self.surface_position_vector = surface_position_vector

            self.surface_area = 1/area_inverse

            self.surface_centroid = surface_centroid

    output_object = OutputObject(file_name, mesh_data_class, 
    constitutive_model, forces_moments_list, new_ds, new_boundary_dict,
    surface_physical_group, surface_position_vector, area_inverse,
    surface_centroid)

    return output_object

# Defines a function to update the forces and moments on a surface

def update_forces_moments_on_surface(output_object, field, field_number, 
time, fields_namesDict):
    
    # Verifies if the field is the displacement field. Uses the max 
    # function since in single field simulations, the field number is -1

    field_name = get_first_key_from_value(fields_namesDict, max(
    field_number, 0))

    if field_name!="Displacement":

        message = "Field name <-> Field number"

        for key, value in fields_namesDict.items():

            message += "\n"+str(key)+" <-> "+str(value)

        raise NameError("The field name is '"+str(field_name)+"', but "+
        "the field required to evaluate the 'SaveForcesMomentsOnSurfac"+
        "e' post-process must be 'Displacement'. The dictionary of fie"+
        "lds names has the following keys:\n"+message+"\n\nThe asked n"+
        "umber was "+str(field_number))
    
    mpi_print(output_object.mesh_data_class.comm, "Updates the saving "+
    "of the forces and moments on the surface' "+str(
    output_object.surface_physical_group)+"'. This surface has an area"+
    " of "+str(output_object.surface_area)+"\nThe centroid of this sur"+
    "face is at "+str(output_object.surface_centroid.values())+"\n")

    # If there is a single field, field number will be -1

    u_field = None

    if field_number==-1:

        u_field = field

    else:

        u_field = field[field_number]

    # Verifies if there is a pressure field. A correction to the first
    # Piola-Kirchhoff stress tensor must be added to this case, since 
    # the constitutive model does not give the spherical part of the
    # stress in incompressible hyperelasticity

    pressure_correction = Constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [
    0.0, 0.0, 0.0]])

    if "Pressure" in fields_namesDict:

        mpi_print(output_object.mesh_data_class.comm, "Adds the correc"+
        "tion of the pressure to the saving of forces and moments on s"+
        "urface "+str(output_object.surface_physical_group)+"\n")

        # Gets the deformation gradient and the jacobian

        I = Identity(3)

        F = grad(u_field)+I

        J = det(F)

        # Gets the pressure field

        p_field = field[fields_namesDict["Pressure"]]

        # Updates the correction tensor

        pressure_correction = p_field*J*inv(F.T)

    # Initializes the forces and moments components

    forces = [0.0, 0.0, 0.0]

    moments = [0.0, 0.0, 0.0]

    # Gets the volumetric physical groups attached to the surface

    physical_groups_attached = output_object.new_boundary_physical_groups_dict[
    output_object.surface_physical_group]
    
    # Verifies if the constitutive model is a dictionary

    if isinstance(output_object.constitutive_model, dict):
        
        # Iterates through the physical groups

        for physical_group, local_constitutive_model in (
        output_object.constitutive_model.items()):
            
            # If the physical group is a tuple

            if isinstance(physical_group, tuple):

                # Iterates through it

                for local_physical_group in physical_group:

                    # Verifies if this physical group is a true physical
                    # group

                    if not (local_physical_group in (
                    output_object.mesh_data_class.domain_physicalGroupsNameToTag)):
                        
                        raise ValueError("The phyical group '"+str(
                        local_physical_group)+"' is not a proper physi"+
                        "cal group in the mesh. It was given at the di"+
                        "ctionary of constitutive models. Check the pr"+
                        "oper physical groups names:\n"+str(list(
                        output_object.mesh_data_class.domain_physicalGroupsNameToTag.keys())))
                    
                    # Verifies if this group is attached to the surface

                    if local_physical_group in physical_groups_attached:

                        # Gets the resulting force

                        surface_force = ((local_constitutive_model.first_piolaStress(
                        u_field)+pressure_correction)*output_object.mesh_data_class.n)

                        # Assembles the three components

                        forces[0] += assemble(surface_force[0]*
                        output_object.new_ds(physical_groups_attached[
                        local_physical_group]))

                        forces[1] += assemble(surface_force[1]*
                        output_object.new_ds(physical_groups_attached[
                        local_physical_group]))

                        forces[2] += assemble(surface_force[2]*
                        output_object.new_ds(physical_groups_attached[
                        local_physical_group]))

                        # Gets the resulting moment

                        surface_moment = cross(
                        output_object.surface_position_vector, (
                        local_constitutive_model.first_piolaStress(
                        u_field)+pressure_correction)*
                        output_object.mesh_data_class.n)

                        # Assembles the three components

                        moments[0] += assemble(surface_moment[0]*
                        output_object.new_ds(physical_groups_attached[
                        local_physical_group]))

                        moments[1] += assemble(surface_moment[1]*
                        output_object.new_ds(physical_groups_attached[
                        local_physical_group]))

                        moments[2] += assemble(surface_moment[2]*
                        output_object.new_ds(physical_groups_attached[
                        local_physical_group]))

            # Otherwise

            else:

                # Verifies if this physical group is a true physical
                # group

                if not (physical_group in (
                output_object.mesh_data_class.domain_physicalGroupsNameToTag)):
                    
                    raise ValueError("The phyical group '"+str(
                    physical_group)+"' is not a proper physical group "+
                    "in the mesh. It was given at the dictionary of co"+
                    "nstitutive models. Check the proper physical grou"+
                    "ps names:\n"+str(list(
                    output_object.mesh_data_class.domain_physicalGroupsNameToTag.keys())))

                # Verifies if this group is attached to the surface

                if physical_group in physical_groups_attached:

                    # Gets the resulting force

                    surface_force = ((local_constitutive_model.first_piolaStress(
                    u_field)+pressure_correction)*output_object.mesh_data_class.n)

                    # Assembles the three components

                    forces[0] += assemble(surface_force[0]*
                    output_object.new_ds(physical_groups_attached[
                    physical_group]))

                    forces[1] += assemble(surface_force[1]*
                    output_object.new_ds(physical_groups_attached[
                    physical_group]))

                    forces[2] += assemble(surface_force[2]*
                    output_object.new_ds(physical_groups_attached[
                    physical_group]))

                    # Gets the resulting moment

                    surface_moment = cross(
                    output_object.surface_position_vector, (
                    local_constitutive_model.first_piolaStress(
                    u_field)+pressure_correction)
                    *output_object.mesh_data_class.n)

                    # Assembles the three components

                    moments[0] += assemble(surface_moment[0]*
                    output_object.new_ds(physical_groups_attached[
                    physical_group]))

                    moments[1] += assemble(surface_moment[1]*
                    output_object.new_ds(physical_groups_attached[
                    physical_group]))

                    moments[2] += assemble(surface_moment[2]*
                    output_object.new_ds(physical_groups_attached[
                    physical_group]))

    # Otherwise, computes for the whole mesh at once

    else:

        # Gets the resulting force

        surface_force = ((output_object.constitutive_model.first_piolaStress(
        u_field)+pressure_correction)*output_object.mesh_data_class.n)

        # Gets the resulting moment

        surface_moment = cross(output_object.surface_position_vector, (
        output_object.constitutive_model.first_piolaStress(u_field)+
        pressure_correction)*output_object.mesh_data_class.n)

        # Iterates through the physical groups attached to this surface

        for integration_tag in physical_groups_attached.values():

            # Assembles the three components

            forces[0] += assemble(surface_force[0]*output_object.new_ds(
            integration_tag))

            forces[1] += assemble(surface_force[1]*output_object.new_ds(
            integration_tag))

            forces[2] += assemble(surface_force[2]*output_object.new_ds(
            integration_tag))

            # Assembles the three components

            moments[0] += assemble(surface_moment[0]*output_object.new_ds(
            integration_tag))

            moments[1] += assemble(surface_moment[1]*output_object.new_ds(
            integration_tag))

            moments[2] += assemble(surface_moment[2]*output_object.new_ds(
            integration_tag))

    # Appends this result to the output class

    output_object.result.append([time, [forces[0], forces[1], forces[2],
    moments[0], moments[1], moments[2]]])

    # Saves the list of volume ratios in a txt file

    mpi_execute_function(output_object.comm_object, 
    file_tools.list_toTxt, output_object.result, output_object.file_name, 
    add_extension=True)
    
    return output_object

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

    comm_object = direct_codeData[1]

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

            # Saves the comm object

            self.comm_object = comm_object

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
    
    # Verifies if the field is the displacement field. Uses the max 
    # function since in single field simulations, the field number is -1

    field_name = get_first_key_from_value(fields_namesDict, max(
    field_number, 0))

    if field_name!="Displacement":

        message = "Field name <-> Field number"

        for key, value in fields_namesDict.items():

            message += "\n"+str(key)+" <-> "+str(value)

        raise NameError("The field name is '"+str(field_name)+"', but "+
        "the field required to evaluate the 'SaveMeshVolumeRatioToRefe"+
        "renceVolume' post-process must be 'Displacement'. The diction"+
        "ary of fields names has the following keys:\n"+message+"\n\nT"+
        "he asked number was "+str(field_number))
    
    mpi_print(output_object.comm_object, "Updates the saving of the ra"+
    "tio of the meshe's volume to the initial volume of the mesh\n")

    # Gets the jacobian

    I = Identity(3)

    F = None

    # If there is a single field, field number will be -1

    if field_number==-1:

        F = grad(field)+I 

    else:

        F = grad(field[field_number])+I 

    J = det(F)
    
    # Calculates the new mesh volume

    new_volume = assemble(J*output_object.dx) 

    # Gets the ratio of the new volume to the reference

    ratio = new_volume/output_object.initial_volume

    # Appends this result to the output class

    output_object.result.append([time, ratio])

    # Saves the list of volume ratios in a txt file

    mpi_execute_function(output_object.comm_object, 
    file_tools.list_toTxt, output_object.result, output_object.file_name, 
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

    comm_object = direct_codeData[3]

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
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, homogenized_fieldList, inverse_volume, dx, 
        subdomain, file_name):

            # Saves the comm object

            self.comm_object = comm_object
            
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
    
    mpi_print(output_object.comm_object, "Updates the homogenization o"+
    "f the "+str(field_number)+" field\n")

    # If the problem has a single field

    if field_number==-1:

        # Homogenizes the field and updates the list of homogenized field
        # along time

        output_object.result = homogenization_tools.homogenize_genericField(
        field, output_object.result, time, output_object.inverse_volume, 
        output_object.dx, output_object.subdomain, 
        output_object.file_name, output_object.comm_object)

        return output_object

    # If the problem has multiple fields

    else:

        # Homogenizes the field and updates the list of homogenized field
        # along time

        output_object.result = homogenization_tools.homogenize_genericField(
        field[field_number], output_object.result, time, 
        output_object.inverse_volume, output_object.dx, 
        output_object.subdomain, output_object.file_name, 
        output_object.comm_object)

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
    
    mpi_print(output_object.comm_object, "Updates the homogenization o"+
    "f the gradient of the "+str(field_number)+" field\n")

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

    comm_object = direct_codeData[4]

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
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, homogenized_firstPiolaList, inverse_volume, dx, 
        subdomain, file_name, constitutive_model, physical_groupsList,
        physical_groupsNamesToTags):

            # Saves the comm object

            self.comm_object = comm_object
            
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
    
    mpi_print(output_object.comm_object, "Updates the homogenization o"+
    "f the first Piola-Kirchhoff stress field\n")

    output_object.result = homogenization_tools.homogenize_stressTensor(
    field, output_object.constitutive_model, "first_piola_kirchhoff", 
    "first_piolaStress", output_object.result, time, 
    output_object.inverse_volume, output_object.dx, 
    output_object.subdomain,output_object.file_name, 
    output_object.physical_groupsList, 
    output_object.physical_groupsNamesToTags, fields_namesDict, 
    output_object.required_fieldsNames, output_object.comm_object)

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

    comm_object = direct_codeData[5]

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
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, homogenized_firstPiolaList, inverse_volume, dx, 
        subdomain, file_name, constitutive_model, physical_groupsList,
        physical_groupsNamesToTags, position_vector):

            # Saves the comm object

            self.comm_object = comm_object
            
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
    
    mpi_print(output_object.comm_object, "Updates the homogenization o"+
    "f the couple first Piola-Kirchhoff stress field\n")

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
    output_object.required_fieldsNames, output_object.comm_object)

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

    comm_object = direct_codeData[4]

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
        
    # Takes out the termination of the file name

    file_name = path_tools.take_outFileNameTermination(
    file_name)

    # Gets the name of the file with the path to it

    file_name = path_tools.verify_path(parent_path, file_name)

    # Assembles the output. This post-process does not have a variable
    # that can be shared with a submesh

    class OutputObject:

        def __init__(self, homogenized_cauchyList, inverse_volume, dx, 
        subdomain, file_name, constitutive_model, physical_groupsList,
        physical_groupsNamesToTags):

            # Saves the comm object

            self.comm_object = comm_object
            
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
    
    mpi_print(output_object.comm_object, "Updates the homogenization o"+
    "f the Cauchy stress field\n")

    output_object.result = homogenization_tools.homogenize_stressTensor(
    field, output_object.constitutive_model, "cauchy", "cauchy_stress", 
    output_object.result, time, output_object.inverse_volume, 
    output_object.dx, output_object.subdomain, output_object.file_name, 
    output_object.physical_groupsList, 
    output_object.physical_groupsNamesToTags, fields_namesDict, 
    output_object.required_fieldsNames, output_object.comm_object)

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
    
    mpi_print(output_object.comm_object, "Updates the homogenization o"+
    "f the couple Cauchy stress field\n")

    output_object.result = homogenization_tools.homogenize_stressTensor(
    field, output_object.constitutive_model, "couple_cauchy", "cauchy_"+
    "stress", output_object.result, time, output_object.inverse_volume, 
    output_object.dx, output_object.subdomain, output_object.file_name, 
    output_object.physical_groupsList, 
    output_object.physical_groupsNamesToTags, fields_namesDict, 
    output_object.required_fieldsNames, output_object.comm_object)

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

    comm_object = direct_codeData[5]

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

            # Saves the comm object

            self.comm_object = comm_object

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
    
    mpi_print(output_object.comm_object, "Updates the saving of the fi"+
    "rst elasticity tensor\n")

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
    
    mpi_print(output_object.comm_object, "Updates the saving of the se"+
    "cond elasticity tensor\n")

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
    
    mpi_print(output_object.comm_object, "Updates the saving of the th"+
    "ird elasticity tensor\n")

    return constitutive_tools.save_elasticityTensor(output_object, 
    field, time, "third_elasticityTensor", "third_elasticity_tensor",
    fields_namesDict)