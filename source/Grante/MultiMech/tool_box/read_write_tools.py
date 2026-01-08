# Routine to read and write files for MultiMech

from dolfin import *

from copy import deepcopy

from ...MultiMech.tool_box.functional_tools import FunctionalData, construct_monolithicFunctionSpace

from ...MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from ...PythonicUtilities.path_tools import get_parent_path_of_file, decapitalize_and_insert_underline, verify_file_existence, take_outFileNameTermination, verify_path

########################################################################
########################################################################
##                                Write                               ##
########################################################################
########################################################################

# Defines a function to write FEniCS fields (functions) into xdmf files

def write_field_to_xdmf(functional_data_class, time=0.0, field_name=None,
directory_path=None, visualization_copy=False, close_file=True, file=
None, visualization_copy_file=None, time_step=0, explicit_file_name=None,
code_given_mesh_data_class=None, field_type=None, interpolation_function=
None, polynomial_degree=None):
    
    """
    Function for writing a FEniCS function to xdmf files.
    
    functional_data_class: Instance of the FunctionalData class, it 
    constains function spaces, finite elements, and so forth. It can also 
    be a dictionary with keys 'dictionary of field names', 
    'monolithic solution', and 'mesh file'
    
    time: time value, when the function was evaluated
    
    field_name: name of the field
    
    directory_path: path to the directory where the file must be saved

    visualization_copy: flag to write a conventional xdmf copy for
    visualization, since write_checkpoint method may fail to be 
    visualized on ParaView

    close_file: flag to close a xdmf file after modifying it. It must be
    false if this file is meant to store a time series

    file: receives a xdmf file object from past iterations if a time 
    series is to be saved

    visualization_copy_file: receives a xdmf file object from past 
    iterations if a time series of the visualization copy is to be saved

    time_step: integer number with the index of this time step

    explicit_file_name: explicit file name to create the xdmf file 
    without using the automatic creator of this function
    """
    
    # If the directory path was not provided

    if directory_path is None:

        # Gets the path where the function which called this function is
        # located

        directory_path = get_parent_path_of_file(
        function_calls_to_retrocede=2)

    # If an explicit file name was provided

    if explicit_file_name is not None:

        # Takes out the termination of the file name

        explicit_file_name = (take_outFileNameTermination(
        explicit_file_name)+".xdmf")
    
    # Verifies if functional_data_class is indeed an instance of the 
    # functional data class

    fields_names_dict = None

    monolithic_solution = None

    mesh_file = None

    if isinstance(functional_data_class, FunctionalData):

        # Gets the dictionary of fields' names off of the functional da-
        # ta class

        fields_names_dict = functional_data_class.fields_names_dict

        monolithic_solution = functional_data_class.monolithic_solution

        mesh_file = functional_data_class.mesh_file

    elif isinstance(functional_data_class, dict):

        # if the functional data class is a dictionary, creates the 
        # functional data class automatically

        # Verifies if all keys are available

        if not ("monolithic solution" in functional_data_class):

            raise ValueError("'functional_data_class' in 'write_field_"+
            "to_xdmf' is a dictionary but it does not have the key 'mo"+
            "nolithic solution'")

        if not ("dictionary of field names" in functional_data_class):

            # Automatically creates a dictionary using the own name of
            # the function

            functional_data_class["dictionary of field names"] = {
            functional_data_class["monolithic solution"].name(): 0}

            """raise ValueError("'functional_data_class' in 'write_field_"+
            "to_xdmf' is a dictionary but it does not have the key 'di"+
            "ctionary of field_names'")"""

        if not ("mesh file" in functional_data_class):

            raise ValueError("'functional_data_class' in 'write_field_"+
            "to_xdmf' is a dictionary but it does not have the key 'me"+
            "sh file'")
        
        # Retrieves the data from the given dictionary

        if isinstance(functional_data_class["dictionary of field names"], 
        str):

            fields_names_dict = {functional_data_class["dictionary of "+
            "field names"]: 0}

        else:

            fields_names_dict = functional_data_class["dictionary of f"+
            "ield names"]

        monolithic_solution = functional_data_class["monolithic soluti"+
        "on"]

        mesh_file = functional_data_class["mesh file"]

        # Converts the functional data class to a true instance of the
        # FunctionalData class

        functional_data_class = FunctionalData(None, monolithic_solution,
        None, None, None, None, fields_names_dict, None, mesh_file=
        mesh_file)

    else:

        raise TypeError("'functional_data_class' is not an instance of"+
        " the FunctionalData class nor a dictionary with keys 'diction"+
        "ary of field names', 'monolithic solution', and 'mesh file'. "+
        "Thus, the fields cannot be written into a xdmf file using the"+
        " function 'write_field_to_xdmf'")

    # Verifies if there are multiple fields

    if len(fields_names_dict.keys())>1:

        # Splits the fields

        split_fields = list(monolithic_solution.split(
        deepcopy=True))

        # Verifies if a particular field has been asked for

        if field_name is not None:

            # Verifies if this field is in the available fields

            if field_name in fields_names_dict:

                # Gets the automatic file name

                if explicit_file_name is None:

                    explicit_file_name = (directory_path+"//"+
                    decapitalize_and_insert_underline(str(field_name))+
                    ".xdmf")

                print("Saves the field '"+str(field_name)+"' at "+
                explicit_file_name)

                print("")

                # Gets the individual field and renames it

                individual_field = split_fields[fields_names_dict[
                field_name]]

                individual_field.rename(field_name, "DNS")

                # Verifies if this file has already been created. If 
                # not, creates it

                append_flag = True

                if not isinstance(file, XDMFFile):

                    print("Creates a new XDMFFile instance\n")

                    file = XDMFFile(individual_field.function_space(
                    ).mesh().mpi_comm(), explicit_file_name)

                    # If the file was not provided, the append flag must
                    # must be false to not append a new checkpoint if no
                    # previous structure had been saved

                    append_flag = False

                # Writes the function

                file.write_checkpoint(individual_field, 
                individual_field.name(), time, append=append_flag)

                # Closes the file

                if close_file:

                    file.close()

            else:

                raise NameError("'field_name' is '"+str(field_name)+"'"+
                ", but it is not a name of proper field. See the avail"+
                "able fields' names: "+str(list(fields_names_dict.keys()
                )))
            
        # Otherwise, writes all fields

        else:

            for field_name in fields_names_dict.keys():

                # Gets the automatic file name

                if explicit_file_name is None:

                    explicit_file_name = (directory_path+"//"+
                    decapitalize_and_insert_underline(str(field_name
                    ))+".xdmf")

                print("Saves the field '"+str(field_name)+"' at "+
                explicit_file_name)

                print("")

                # Gets the individual field and renames it

                individual_field = split_fields[fields_names_dict[
                field_name]]

                individual_field.rename(field_name, "DNS")
                
                # Verifies if this file has already been created. If 
                # not, creates it

                append_flag = True

                if not isinstance(file, XDMFFile):

                    print("Creates a new XDMFFile instance\n")

                    file = XDMFFile(individual_field.function_space(
                    ).mesh().mpi_comm(), explicit_file_name)

                    # If the file was not provided, the append flag
                    # must be false to not append a new checkpoint
                    # if no previous structure had been saved

                    append_flag = False
                
                # Writes the field

                file.write_checkpoint(individual_field, 
                individual_field.name(), time, append=append_flag)

                # Closes the file

                if close_file:

                    file.close()

    # For single field problems

    else:

        if field_name is not None:

            # Verifies if this field is in the available fields

            if field_name in fields_names_dict:

                # Gets the automatic file name

                if explicit_file_name is None:

                    explicit_file_name = (directory_path+"//"+
                    decapitalize_and_insert_underline(str(field_name
                    ))+".xdmf")

                print("Saves the field '"+str(field_name)+"' at "+
                explicit_file_name)

                print("")

                # Gets the individual field and renames it

                individual_field = monolithic_solution

                individual_field.rename(field_name, "DNS")

                # Verifies if this file has already been created. If 
                # not, creates it

                append_flag = True

                if not isinstance(file, XDMFFile):

                    print("Creates a new XDMFFile instance\n")

                    file = XDMFFile(individual_field.function_space(
                    ).mesh().mpi_comm(), explicit_file_name)

                    # If the file was not provided, the append flag
                    # must be false to not append a new checkpoint
                    # if no previous structure had been saved

                    append_flag = False

                # Writes the function

                file.write_checkpoint(individual_field, 
                individual_field.name(), time, append=append_flag)

                # Closes the file

                if close_file:

                    file.close()

            else:

                raise NameError("'field_name' is '"+str(field_name)+
                "', but it is not a name of proper field. See the "+
                "available fields' names: "+str(
                list(fields_names_dict.keys())))
            
        # Otherwise, writes as a generic solution

        else:

            # Gets the name of the field

            field_name = list(fields_names_dict.keys())[0]

            # Gets the automatic file name

            if explicit_file_name is None:

                explicit_file_name = (directory_path+"//"+
                decapitalize_and_insert_underline(str(field_name))+
                ".xdmf")

            print("Saves the field '"+str(field_name)+"' at "+
            explicit_file_name)

            print("")

            # Gets the individual field and renames it

            individual_field = monolithic_solution

            individual_field.rename(field_name, "DNS")

            # Verifies if this file has already been created. If not,
            # creates it

            append_flag = True

            if not isinstance(file, XDMFFile):

                print("Creates a new XDMFFile instance\n")

                file = XDMFFile(individual_field.function_space(
                ).mesh().mpi_comm(), explicit_file_name)

                # If the file was not provided, the append flag must
                # be false to not append a new checkpoint if no pre-
                # vious structure had been saved

                append_flag = False
            
            # Writes the field

            file.write_checkpoint(individual_field, 
            individual_field.name(), time, append=append_flag)

            # Closes the file

            if close_file:

                file.close()
    
    # Verifies if a visualization copy must be made

    if visualization_copy:

        # Creates a dictionary with instructions to build the functional
        # data class for the read of the visualization copy. This new 
        # instance is created to not affect the old instance

        functional_data_dictionary = dict()

        if field_type is None:

            raise ValueError("'visualization_copy' in 'write_field_to_"+
            "xdmf' is True, but 'field_type' was not given. It must be"+
            " 'scalar', or 'vector', or 'tensor'")
        
        else:

            functional_data_dictionary["field type"] = field_type

        if interpolation_function is None:

            raise ValueError("'visualization_copy' in 'write_field_to_"+
            "xdmf' is True, but 'interpolation_function' was not given"+
            ". It must be 'CG', or 'Lagrange', or 'DG'")
        
        else:

            functional_data_dictionary["interpolation function"] = (
            interpolation_function)

        if polynomial_degree is None:

            raise ValueError("'visualization_copy' in 'write_field_to_"+
            "xdmf' is True, but 'polynomial_degree' was not given. It "+
            "must be an 1, 2, 3...'")
        
        else:

            functional_data_dictionary["polynomial degree"] = (
            polynomial_degree)

        # Calls the function to create the visualization copy
        
        visualization_copy_file = write_visualization_copy(
        functional_data_dictionary, explicit_file_name, mesh_file, 
        field_name, time=time, time_step=time_step, 
        visualization_copy_file=visualization_copy_file, close_file=
        close_file, code_given_mesh_data_class=code_given_mesh_data_class)

        return file, visualization_copy_file

    # Returns the file

    return file

# Defines a function to write a visualization copy using the conventi-
# onal write method

def write_visualization_copy(functional_data_dictionary, file_name, 
mesh_file, code_given_field_name, time=0.0, time_step=0, 
visualization_copy_file=None, close_file=True, 
code_given_mesh_data_class=None):

    # Reads the file back

    read_function = read_field_from_xdmf(file_name, mesh_file,
    functional_data_dictionary, time_step=time_step, rename_function=
    True, code_given_mesh_data_class=code_given_mesh_data_class,
    code_given_field_name=code_given_field_name)

    # Writes it using simple write

    copy_file_name = (take_outFileNameTermination(file_name)+"_visuali"+
    "zation_copy.xdmf")

    print("Saves the visualization copy at file '"+str(copy_file_name)+
    "'\n")

    # Verifies if this file has already been created. If not, creates it

    if not isinstance(visualization_copy_file, XDMFFile):

        print("Creates a new XDMFFile instance for the visualization c"+
        "opy file\n")

        visualization_copy_file = XDMFFile(read_function.function_space(
        ).mesh().mpi_comm(), copy_file_name)

    visualization_copy_file.write(read_function, time)

    # Closes the file

    if close_file:

        visualization_copy_file.close()

    return visualization_copy_file

########################################################################
########################################################################
##                                Read                                ##
########################################################################
########################################################################

# Defines a function to read FEniCS fields (functions) from xdmf files
# back into FEniCS functions

def read_field_from_xdmf(field_file, mesh_file, function_space_info,
directory_path=None, code_given_field_name=None, 
code_given_mesh_data_class=None, time_step=0, rename_function=True):
    
    # If the directory path is given, joins them

    if directory_path is not None:

        field_file = directory_path+"//"+field_file

        mesh_file = directory_path+"//"+mesh_file
    
    # Verifies if the field file exists and if it is a xdmf file

    field_file = take_outFileNameTermination(field_file)+".xdmf"

    verify_file_existence(field_file, termination=".xdmf")
    
    # Verifies if the mesh file exists and if it is a msh file

    mesh_file = take_outFileNameTermination(mesh_file)

    verify_file_existence(mesh_file+".msh")

    # Reads the mesh

    mesh_data_class = None

    # Verifies if there is a mesh data class that was given by the code

    if code_given_mesh_data_class is not None:

        # Verifies if its is a dictionary

        if isinstance(code_given_mesh_data_class, dict):

            if "mesh_data_class" in code_given_mesh_data_class:

                code_given_mesh_data_class = code_given_mesh_data_class[
                "mesh_data_class"]

            else:

                raise ValueError("The code given information to reuse "+
                "the mesh does not have the key 'mesh_data_class'. Che"+
                "ck the source.")

        # Verifies if the file of the mesh is the same as the asked now

        if mesh_file==code_given_mesh_data_class.mesh_file:

            mesh_data_class = code_given_mesh_data_class

    # If still no mesh was given, tries to read one

    if mesh_data_class is None:

        try:

            mesh_data_class = read_mshMesh(mesh_file)

        except Exception as e:

            raise ValueError("An error occurred while reading the mesh"+
            " file '"+str(mesh_file)+"' that is used to read the field"+
            " in file '"+str(field_file)+"'")
    
    # Verifies if function_space_info is an instance of the Functional-
    # Data class

    if not isinstance(function_space_info, FunctionalData):

        # Verifies if function space info is a dictionary

        if not isinstance(function_space_info, dict):

            raise TypeError("'function_space_info' in function 'read_f"+
            "ield_from_xdmf' is not a dictionary, but it must have the"+
            " keys: 'field type'; 'interpolation function'; 'polynomia"+
            "l degree'. Optionally, it may have the key 'field name' a"+
            "s well. Otherwise, it can have the format field_name: dic"+
            "tionary_with_necessary_keys. Currently, 'function_space_i"+
            "nfo' is: "+str(function_space_info))
        
        # Verifies if any of the necessary keys are in the dictionary

        necessary_keys = ['field type', 'interpolation function', ('po'+
        'lynomial degree')]

        for necessary_key in necessary_keys:

            if necessary_key in function_space_info:

                # As the dictionary has a necessary key, it means the 
                # dictionary is not discriminated by field. Thus, it 
                # must have the field name as key

                if 'field name' in function_space_info:

                    # Turns the field name as key to a new dictionary 
                    # compatible to the syntax used for creating finite 
                    # element spaces

                    function_space_info = {function_space_info["field "+
                    "name"]: function_space_info}

                # If no field name was provided by the user, but the co-
                # de did

                elif code_given_field_name is not None:

                    # Turns the field name as key to a new dictionary 
                    # compatible to the syntax used for creating finite 
                    # element spaces

                    function_space_info = {code_given_field_name: 
                    function_space_info}

                # Otherwise, throws an error

                else: 

                    raise KeyError("'function_space_info' has the obli"+
                    "gatory keys, such as '"+str(necessary_key)+"', bu"+
                    "t it does not have the key 'field name'")
                
                break 

        # Verifies if the dictionary has a single key-value pair

        if len(function_space_info.keys())!=1:

            raise KeyError("'function_space_info' has "+str(len(
            function_space_info.keys()))+" key-value pairs, whereas it"+
            " must have only one: field name <-> dictionary_with_neces"+
            "sary_keys. The necessary keys are: "+str(necessary_keys))
        
        # Verifies if the dictionary inside the single value has the o-
        # bligatory keys

        function_space_info_value = list(function_space_info.values())[0]

        for necessary_key in necessary_keys:

            if not (necessary_key in function_space_info_value):

                error_string = ("The dictionary 'function_space_info' "+
                "currently is: "+str(function_space_info)+". But it sh"+
                "ould have the following keys: ")

                for necessary_key_name in necessary_keys:

                    error_string += "\\n"+str(necessary_key_name)

                raise ValueError(error_string)
            
        # Creates the function space for this field. Creates function 
        # only, no variation or trial functions are created

        function_space_info = construct_monolithicFunctionSpace(
        function_space_info, mesh_data_class, function_only=True)

    # Renames the function

    field_name = None

    if code_given_field_name is None:

        field_name = list(function_space_info.fields_names_dict.keys()
        )[0]

    else:

        field_name = str(code_given_field_name)

    if rename_function:

        function_space_info.monolithic_solution.rename(field_name, "DNS")

    # Finally reads the xdmf file with the field

    try:

        with XDMFFile(mesh_data_class.mesh.mpi_comm(), field_file) as xdmf_file:

            xdmf_file.read_checkpoint(
            function_space_info.monolithic_solution, field_name, 
            time_step)

    except Exception as e:

        raise ValueError("An error ocurred while reading file '"+str(
        field_file)+"'.\n\nTwo main causes for this problem:\n1. The o"+
        "riginal field was not saved using function 'write_field_to_xd"+
        "mf';\n2. If 'write_field_to_xdmf' was indeed used, you might "+
        "be trying to read the visualization copy file, which is not m"+
        "ade for this purpose, rather for visualization only;\n3. The "+
        "name to the function (FEniCS function) you are trying to impo"+
        "se now is, '"+str(field_name)+"', and it may not be the same "+
        "as the one used when then function was saved.\n\nThe original"+
        " error message is: "+str(e))

    # Returns the function

    return function_space_info.monolithic_solution