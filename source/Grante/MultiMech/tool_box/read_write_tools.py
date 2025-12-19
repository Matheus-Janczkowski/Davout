# Routine to read and write files for MultiMech

from dolfin import *

from ...MultiMech.tool_box.functional_tools import FunctionalData, construct_monolithicFunctionSpace

from ...MultiMech.tool_box.mesh_handling_tools import read_mshMesh

from ...PythonicUtilities.path_tools import get_parent_path_of_file, decapitalize_and_insert_underline, verify_file_existence, take_outFileNameTermination

########################################################################
########################################################################
##                                Write                               ##
########################################################################
########################################################################

# Defines a function to write FEniCS fields (functions) into xdmf files

def write_field_to_xdmf(functional_data_class, time=0.0, field_name=None,
directory_path=None, visualization_copy=False):
    
    """
    Function for writing a FEniCS function to xdmf files.
    
    functional_data_class: Instance of the FunctionalData class, it 
    constains function spaces, finite elements, and so forth
    
    time: time value, when the function was evaluated
    
    field_name: name of the field
    
    directory_path: path to the directory where the file must be saved
    """
    
    # If the directory path was not provided

    if directory_path is None:

        # Gets the path where the function which called this function is
        # located

        directory_path = get_parent_path_of_file(
        function_calls_to_retrocede=2)
    
    # Verifies if functional_data_class is indeed an instance of the 
    # functional data class

    if isinstance(functional_data_class, FunctionalData):

        # Gets the dictionary of fields' names off of the functional da-
        # ta class

        fields_names_dict = functional_data_class.fields_names_dict

        # Verifies if there are multiple fields

        if len(fields_names_dict.keys())>1:

            # Splits the fields

            split_fields = list(functional_data_class.monolithic_solution.split(
            deepcopy=True))

            # Verifies if a particular field has been asked for

            if field_name is not None:

                # Verifies if this field is in the available fields

                if field_name in fields_names_dict:

                    # Gets the automatic file name

                    file_name = (directory_path+"//"+
                    decapitalize_and_insert_underline(str(field_name))+
                    ".xdmf")

                    print("Saves the field '"+str(field_name)+"' at "+
                    file_name)

                    print("")

                    # Gets the individual field and renames it

                    individual_field = split_fields[fields_names_dict[
                    field_name]]

                    individual_field.rename(field_name, "DNS")

                    # Writes the function

                    file = XDMFFile(individual_field.function_space(
                    ).mesh().mpi_comm(), file_name)

                    file.write_checkpoint(individual_field, 
                    individual_field.name(), time)

                    # Closes the file

                    file.close()

                else:

                    raise NameError("'field_name' is '"+str(field_name)+
                    "', but it is not a name of proper field. See the "+
                    "available fields' names: "+str(
                    list(fields_names_dict.keys())))
                
            # Otherwise, writes all fields

            else:

                for field_name, field_number in fields_names_dict.items(
                ):

                    # Gets the automatic file name

                    file_name = (directory_path+"//"+
                    decapitalize_and_insert_underline(str(field_name))+
                    ".xdmf")

                    print("Saves the field '"+str(field_name)+"' at "+
                    file_name)

                    print("")

                    # Gets the individual field and renames it

                    individual_field = split_fields[fields_names_dict[
                    field_name]]

                    individual_field.rename(field_name, "DNS")
                    
                    # Writes the field

                    file = XDMFFile(individual_field.function_space(
                    ).mesh().mpi_comm(), file_name)

                    file.write_checkpoint(individual_field, 
                    individual_field.name(), time)

                    # Closes the file

                    file.close()

        # For single field problems

        else:

            if field_name is not None:

                # Verifies if this field is in the available fields

                if field_name in fields_names_dict:

                    # Gets the automatic file name

                    file_name = (directory_path+"//"+
                    decapitalize_and_insert_underline(str(field_name))+
                    ".xdmf")

                    print("Saves the field '"+str(field_name)+"' at "+
                    file_name)

                    print("")

                    # Gets the individual field and renames it

                    individual_field = functional_data_class.monolithic_solution

                    individual_field.rename(field_name, "DNS")

                    # Writes the function

                    file = XDMFFile(individual_field.function_space(
                    ).mesh().mpi_comm(), file_name)

                    file.write_checkpoint(individual_field, 
                    individual_field.name(), time)

                    # Closes the file

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

                file_name = (directory_path+"//"+
                decapitalize_and_insert_underline(str(field_name))+".x"+
                "dmf")

                print("Saves the field '"+str(field_name)+"' at "+
                file_name)

                print("")

                # Gets the individual field and renames it

                individual_field = functional_data_class.monolithic_solution

                individual_field.rename(field_name, "DNS")
                
                # Writes the field

                file = XDMFFile(individual_field.function_space().mesh(
                ).mpi_comm(), file_name)

                file.write_checkpoint(individual_field, 
                individual_field.name(), time)

                # Closes the file

                file.close()

    else:

        raise TypeError("'functional_data_class' is not an instance of"+
        " the FunctionalData class, thus, the fields cannot be written"+
        " into a xdmf file using the function 'write_field_to_xdmf'")
    
    # Verifies if a visualization copy must be made

    if visualization_copy:

        # Reads the file back

        read_function = read_field_from_xdmf(file_name, 
        functional_data_class.mesh_file, functional_data_class)

        # Writes it using simple write

        copy_file_name = (take_outFileNameTermination(file_name)+"_vis"+
        "ualization_copy.xdmf")

        print("Saves the visualization copy at file '"+str(
        copy_file_name)+"'\n")

        file = XDMFFile(individual_field.function_space().mesh(
        ).mpi_comm(), copy_file_name)

        file.write(read_function, time)

        # Closes the file

        file.close()

########################################################################
########################################################################
##                                Read                                ##
########################################################################
########################################################################

# Defines a function to read FEniCS fields (functions) from xdmf files
# back into FEniCS functions

def read_field_from_xdmf(field_file, mesh_file, function_space_info,
directory_path=None, code_given_field_name=None):
    
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

    try:

        mesh_data_class = read_mshMesh(mesh_file)

    except Exception as e:

        raise ValueError("An error occurred while reading the mesh fil"+
        "e '"+str(mesh_file)+"' that is used to read the field in file"+
        " '"+str(field_file)+"'")
    
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

    field_name = list(function_space_info.fields_names_dict.keys()
    )[0]

    function_space_info.monolithic_solution.rename(field_name, "DNS")

    # Finally reads the xdmf file with the field

    try:

        with XDMFFile(mesh_data_class.mesh.mpi_comm(), field_file) as xdmf_file:

            xdmf_file.read_checkpoint(
            function_space_info.monolithic_solution, field_name, 0)

    except Exception as e:

        raise ValueError("An error ocurred while reading file '"+str(
        field_file)+"'. The original error message is: "+str(e))

    # Returns the function

    return function_space_info.monolithic_solution