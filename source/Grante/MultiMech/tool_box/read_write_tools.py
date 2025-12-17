# Routine to read and write files for MultiMech

from dolfin import *

from ...MultiMech.tool_box.functional_tools import FunctionalData

from ...PythonicUtilities.path_tools import get_parent_path_of_file, decapitalize_and_insert_underline

########################################################################
########################################################################
##                                Write                               ##
########################################################################
########################################################################

# Defines a function to write FEniCS fields (functions) into xdmf files

def write_field_to_xdmf(functional_data_class, time=0.0, field_name=None,
directory_path=None):
    
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

                    # Gets the individual field and renames it

                    individual_field = split_fields[fields_names_dict[
                    field_name]]

                    individual_field.rename(field_name, "DNS")

                    # Writes the function

                    file = XDMFFile(file_name)

                    file.write(individual_field, time)

                    # Closes the file

                    file.close()

                else:

                    raise NameError("'field_name' is '"+str(field_name)+
                    "', but it is not a name of proper field. See the "+
                    "available fields' names: "+str(
                    fields_names_dict.keys()))
                
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

                    # Gets the individual field and renames it

                    individual_field = split_fields[fields_names_dict[
                    field_name]]

                    individual_field.rename(field_name, "DNS")
                    
                    # Writes the field

                    file = XDMFFile(file_name)

                    file.write(individual_field, time)

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

                    # Gets the individual field and renames it

                    individual_field = functional_data_class.monolithic_solution

                    individual_field.rename(field_name, "DNS")

                    # Writes the function

                    file = XDMFFile(file_name)

                    file.write(individual_field, time)

                    # Closes the file

                    file.close()

                else:

                    raise NameError("'field_name' is '"+str(field_name)+
                    "', but it is not a name of proper field. See the "+
                    "available fields' names: "+str(
                    fields_names_dict.keys()))
                
            # Otherwise, writes as a generic solution

            else:

                # Gets the name of the field

                field_name = fields_names_dict.keys()[0]

                # Gets the automatic file name

                file_name = (directory_path+"//"+
                decapitalize_and_insert_underline(str(field_name))+".x"+
                "dmf")

                print("Saves the field '"+str(field_name)+"' at "+
                file_name)

                # Gets the individual field and renames it

                individual_field = functional_data_class.monolithic_solution

                individual_field.rename(field_name, "DNS")
                
                # Writes the field

                file = XDMFFile(file_name)

                file.write(individual_field, time)

                # Closes the file

                file.close()

    else:

        raise TypeError("'functional_data_class' is not an instance of"+
        " the FunctionalData class, thus, the fields cannot be written"+
        " into a xdmf file using the function 'write_field_to_xdmf'")