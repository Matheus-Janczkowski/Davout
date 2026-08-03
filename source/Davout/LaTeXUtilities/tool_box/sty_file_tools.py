# Routine to parse the .sty file

import re

from ..tool_box.command_tools import CommandData

from ..tool_box.parsing_tools import find_commands_and_substitute_raw_code

########################################################################
#                        Command reconnaissance                        #
########################################################################

# Defines a function to get the LaTeXUtilities.sty file and return a 
# list of commands names (calls, such as \Displacement) and another list
# of commands code (such as \boldsymbol{u})

def get_commands_names_and_codes(sty_file, sty_file_path, verbose=False,
maximum_iterations_to_look_for_commands=100):

    # Instantiates the class of command data

    command_data_class = CommandData()

    # Gets the pattern to looking for commands

    macros_pattern = command_data_class()

    # Remove comment lines (i.e. those initiated by %)

    sty_file = re.sub(r"(?m)^[ \t]*%.*(?:\n|$)", "", sty_file)

    # Splits the sty file using the pattern for macro definition

    bits = macros_pattern.split(str(sty_file))

    # Gets a list of macro headers and another list of what was written
    # from a macro header to the next. Note that the bits are ["header",
    # "code", "header", "code", ...]

    macro_headers = bits[1::2]

    # Initializes a list that stores the names of the commands

    commands_names = []

    # Initializes two lists of commands names and classes that can be 
    # parsed

    parseable_commands_names = []

    parseable_commands_data_classes = []

    # Strips the bodies of code from any line-breaking character and 
    # white spaces

    macro_data_classes = []

    for header, bit in zip(macro_headers, bits[2::2]):

        # Removes LaTeX line continuations "%\n"

        bit = re.sub(r"%\r?\n", "", bit)

        # Replaces remaining line breaks with spaces

        bit = re.sub(r"\r?\n", " ", bit)

        # Removes whitespace at the beginning and end

        bit = bit.strip()

        # Gets the name and the raw data of the command using the method
        # of the class of command data. All data is stored in a data 
        # class

        macro_definition_data = command_data_class.command_headers[
        header](bit)

        # Appends the command raw code to the bits of macro codes

        macro_data_classes.append(macro_definition_data)

        # Appends the name of this command

        commands_names.append(macro_definition_data.name)

        # Verifies if this command is parseable

        if macro_definition_data.parseable_command:

            parseable_commands_names.append(macro_definition_data.name)

            parseable_commands_data_classes.append(macro_definition_data)

    # Iterates over the raw code to substitute any cross dependencies
    # between macros

    flag_cross_dependencies = True 

    number_of_iterations = 0

    while flag_cross_dependencies:

        # Modifies the flag of cross dependencies to False to test it a-
        # gain

        flag_cross_dependencies = False

        # Iterates over the list of macros and their data classes. But
        # verifies only the parseable commands

        for i in range(len(parseable_commands_data_classes)):

            # Gets the raw code

            raw_code = parseable_commands_data_classes[i].raw_code

            # Finds other commands in the raw code of this command and 
            # returns the raw code with the raw code of the found com-
            # mands inserted in their former place

            raw_code, flag_found_commands = find_commands_and_substitute_raw_code(
            raw_code, parseable_commands_names, 
            parseable_commands_data_classes)

            # If commands were found, updates the raw code of this com-
            # mand and updates the flag of cross dependencies

            if flag_found_commands:

                parseable_commands_data_classes[i].raw_code = raw_code

                flag_cross_dependencies = True

        # Updates the number of iterations and verifies early termination

        number_of_iterations += 1

        if number_of_iterations>maximum_iterations_to_look_for_commands:

            raise NameError("The maximum number of iterations to look "+
            "for commands inside other commands in sty file was reache"+
            "d. This means the sty file has circular definition. Check"+
            " the file at:\n\n"+str(sty_file_path))

    # Adds the command to build the header in place of \usepackage{
    # LaTeXUtilities}

    header_builder = command_data_class.build_header(macro_data_classes)

    macro_data_classes.append(header_builder)

    commands_names.append(header_builder.name)

    parseable_commands_data_classes.append(header_builder)

    parseable_commands_names.append(header_builder.name)

    # Shows the dictionary of commands if required

    if verbose:

        string_dictionary = ""

        for name, data in zip(commands_names, macro_data_classes):

            string_dictionary += "\n'"+str(name)+"': '"+str(
            data.raw_code)+"'"

        print("The dictionary of commands is:\n"+str(string_dictionary)+
        "\n")

    return (commands_names, macro_data_classes, parseable_commands_names,
    parseable_commands_data_classes)