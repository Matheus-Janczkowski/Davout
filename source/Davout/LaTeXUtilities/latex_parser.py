# Routine to open a zip file of a LaTeX project and substitute all fea-
# tures of LaTeXUtilities by their raw LaTeX syntax

from zipfile import ZipFile, ZIP_DEFLATED

from pathlib import Path

import re

from ..PythonicUtilities.path_tools import verify_file_existence, take_outFileNameTermination

from ..LaTeXUtilities.tool_box.sty_file_tools import get_commands_names_and_codes

from .tool_box.parsing_tools import find_commands_and_substitute_raw_code, get_indices_of_first_character_of_each_line

# Defines a function to read the zip file and to substitute all commands
# of LaTeXUtilities

def substitute_utilities_by_raw_latex(zip_file_name, 
files_to_be_compiled="main.tex", parent_path=None, 
latex_utilities_full_path=None, verbose=False, 
character_encoding_standard="utf-8", 
maximum_iterations_to_look_for_commands=10000):

    # If the parent path was given

    if (parent_path is not None) and isinstance(parent_path, str) and (
    len(parent_path)>0):

        # Completes the path of the zip file

        zip_file_name = parent_path+"//"+zip_file_name

    # Verifies the existence of the file

    if not verify_file_existence(zip_file_name):

        raise FileNotFoundError("The given file name for the zip file:"
        "\n'"+str(zip_file_name)+"'\n is not a valid file name. This p"+
        "ath does not exist")

    # Creates a list of files within the zip

    files_inside_zip = []

    with ZipFile(zip_file_name) as zip_file:

        files_inside_zip.extend(zip_file.namelist())

    if verbose:

        printed_files_list = ""

        for file_name in files_inside_zip:

            printed_files_list += "\n'"+str(file_name)+"'"

        print("The list of files within the zip '"+str(zip_file_name)+
        "' is:"+str(printed_files_list))

    # If the path for the LaTeXUtilities.sty file was not given, checks
    # the zip file

    sty_file = None

    sty_path = None

    if latex_utilities_full_path is None:

        # Verifies if there is a .sty file in the zip

        if not ("LaTeXUtilities.sty" in files_inside_zip):

            printed_files_list = ""
            
            for file_name in files_inside_zip:
    
                printed_files_list += "\n'"+str(file_name)+"'"

            raise FileNotFoundError("The given 'latex_utilities_full_p"+
            "ath' was not given, but the zip file '"+str(zip_file_name)+
            "' contains file 'LaTeXUtilities.sty'. Check the available"+
            " files and their paths:"+str(printed_files_list))

        # Otherwise, captures the path of the sty file

        sty_path = zip_file_name+"//LaTeXUtilities.sty"

        # Reads the .sty file from the zip

        with ZipFile(zip_file_name, "r") as zip_file:

            sty_file = zip_file.read("LaTeXUtilities.sty").decode(
            character_encoding_standard)

    # Otherwise, reads the file

    else:

        if not verify_file_existence(latex_utilities_full_path):

            raise FileNotFoundError("'latex_utilities_full_path' was e"+
            "xplicitely given as:\n'"+str(latex_utilities_full_path)+
            "'\n But this path was not found")

        sty_path = Path(latex_utilities_full_path)

        sty_file = sty_path.read_text(encoding=
        character_encoding_standard)

    # From the sty file, gets a list of commands names and another list
    # of commands data classes. The names are the command calls (example: 
    # "\Displacement") and the data classes store, beyond other things,
    # the Latex code (example: "\boldsymbol{u}")

    (commands_names, commands_data_classes, parseable_commands_names,
    parseable_commands_data_classes) = get_commands_names_and_codes(
    sty_file, sty_path, verbose=verbose, 
    maximum_iterations_to_look_for_commands=
    maximum_iterations_to_look_for_commands)

    # Verifies if the given file to be compiled is not a list

    if not isinstance(files_to_be_compiled, list):

        # Converts to a list, but ensures it is a string

        files_to_be_compiled = [str(files_to_be_compiled)]

    # Initializes a dictionary of recompiled files. The keys are the fi-
    # les names and the values are the recompiled files' strings

    recompiled_files = {}

    # Iterates over the files to be compiled

    for compiled_file_path in files_to_be_compiled:

        print("\n")

        # Verifies if it exists inside the zip

        if not (compiled_file_path in files_inside_zip):
       
            printed_files_list = ""
            
            for file_name in files_inside_zip:
    
                printed_files_list += "\n'"+str(file_name)+"'"

            raise FileNotFoundError("The given file to be compiled '"+
            str(compiled_file_path)+" was not given. Check the availab"+
            "le files and their paths:"+str(printed_files_list))

        # Otherwise, captures the path of the file to be compiled
        
        compiled_file_full_path = zip_file_name+"//"+compiled_file_path

        # Reads the file from the zip

        compiled_live_file = None

        with ZipFile(zip_file_name, "r") as zip_file:

            compiled_live_file = zip_file.read(compiled_file_path
            ).decode(character_encoding_standard)

        # Gets a list of the indices of the first character of each line

        list_of_first_character_indices = get_indices_of_first_character_of_each_line(
        compiled_live_file)

        # Iterates over the raw code to substitute any cross dependencies
        # between macros
    
        flag_cross_dependencies = True 
    
        number_of_iterations = 0
    
        while flag_cross_dependencies:
    
            # Updates the number of iterations and verifies early termi-
            # nation
    
            number_of_iterations += 1
    
            if number_of_iterations>maximum_iterations_to_look_for_commands:
    
                raise NameError("The maximum number of iterations to l"+
                "ook for commands inside other commands, "+str(
                maximum_iterations_to_look_for_commands)+", in sty fil"+
                "e was reached. This means the sty file has circular d"+
                "efinition. Check the file at:\n\n"+str(
                compiled_file_full_path))

            print("Initializes the "+str(number_of_iterations)+"-th fi"+
            "xed point eration to recompile commands in file '"+str(
            compiled_file_path))
    
            # Modifies the flag of cross dependencies to False to test 
            # it again
    
            flag_cross_dependencies = False
    
            # Finds other commands in the file and returns the raw code 
            # with the raw code of the found commands inserted in their 
            # former place

            (raw_code, flag_found_commands, 
            list_of_first_character_indices) = find_commands_and_substitute_raw_code(
            compiled_live_file, parseable_commands_names, 
            parseable_commands_data_classes, 
            list_of_first_character_indices=
            list_of_first_character_indices, file_path=
            compiled_file_full_path, 
            return_list_of_first_character_indices=True)
    
            # If commands were found, updates the raw code of the file
            # and updates the flag of cross dependencies

            if flag_found_commands:

                compiled_live_file = str(raw_code)

                flag_cross_dependencies = True

        # Saves the file string into the dictionary of recompiled files

        recompiled_files[compiled_file_path] = compiled_live_file

    # Creates the name of the zip file that will receive the compiled 
    # files

    new_zip_file_name = (take_outFileNameTermination(zip_file_name)+"_"+
    "recompiled.zip")

    # Opens the original zip file and a copy that will receive the com-
    # piled files

    with (ZipFile(zip_file_name, "r") as source_zip, ZipFile(
    new_zip_file_name, "w", ZIP_DEFLATED) as destination_zip):

        # Iterates over the files of the original zip

        for item in source_zip.infolist():

            # Checks is this item is not one of the files to be compiled

            if not (item.filename in files_to_be_compiled):

                # Simply copies the old file

                destination_zip.writestr(item, source_zip.read(
                item.filename))

            # Otherwise, writes the recompiled file from the dictionary
            # of recompiled files

            else:

                # Gets the original name

                original_name = str(item.filename)

                # Changes the name

                item.filename = take_outFileNameTermination(original_name
                )+"_recompiled.tex"

                destination_zip.writestr(item, recompiled_files[
                original_name].encode(character_encoding_standard))