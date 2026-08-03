# Routine to parse the .sty file

import re

from dataclasses import dataclass

from ..tool_box.parsing_tools import extract_raw_code, parse_string_until_stopping_character

# Defines a class for meta data

class MacroDefinition:

    def __init__(self, name=None, number_of_arguments=None, raw_code=
    None, header=None, color_system=None, to_be_put_in_document_header=
    False, parseable_command=True, default_arguments=None):

        self.name = name

        self.number_of_arguments = number_of_arguments

        self.raw_code = raw_code

        self.header = header

        self.color_system = color_system

        self.to_be_put_in_document_header = to_be_put_in_document_header

        self.parseable_command = parseable_command

        self.default_arguments = default_arguments

# Defines a class that stores command data

class CommandData:

    def __init__(self):

        # Defines a dictionary of command header that will be used to 
        # find macros in the sty file. The values are functions that
        # get the body of code and retrieve the necessary information

        self.command_headers = {r"\newcommand": self.newcommand_info,
        r"\newcommand*": self.newcommand_star_info,
        r"\DeclareRobustCommand": self.declare_robust_command_info, 
        r"\definecolor": self.define_color_info,
        r"\RequirePackage": self.require_package_info,
        r"\usepackage{LaTeXUtilities}": self.build_header}

    def __call__(self):

        # Sorts the headers by length first. This avoids that * is lost
        # in \newcommand*

        headers = sorted(self.command_headers.keys(), key=len, reverse=
        True)

        # Assembles the pattern to find macros using re library

        pattern_string = "("

        for header in headers:

            pattern_string += re.escape(header)+"|"

        # Takes out the last | character and adds the closing parenthesis

        pattern_string = pattern_string[0:-1]+")"

        self.macro_pattern = re.compile(pattern_string)

        # Returns all relevant information

        return self.macro_pattern

    # Defines a function to retrieve information from \newcommand

    def newcommand_info(self, code_body, character_counter=1, header=
    r"\newcommand"):

        # Initializes the infos

        number_of_arguments = ""

        # Checks if after \newcommand comes { or something else

        closing_character = "}"

        if code_body[character_counter-1]!="{":

            closing_character = "{"

        # Gets the command name

        (command_name, character_counter
        ) = parse_string_until_stopping_character(code_body, 
        character_counter, closing_character)

        # If the current character is a bracket, it indicates the number
        # of arguments of this command

        if code_body[character_counter]=="[":

            character_counter += 1

            # Gets the number of arguments

            (number_of_arguments, character_counter
            ) = parse_string_until_stopping_character(code_body, 
            character_counter, "]")

        if len(number_of_arguments)>0:

            # Converts the number of arguments to an integer

            number_of_arguments = int(number_of_arguments)

        else:

            number_of_arguments = 0

        # Verifies the existence of default arguments

        default_arguments = []

        for i in range(number_of_arguments):

            # If the current character is a bracket, it indicates the e-
            # xistence of a default argument of this command
    
            if code_body[character_counter]=="[":
    
                character_counter += 1
    
                # Gets the default_argument
    
                (default_argument, character_counter
                ) = parse_string_until_stopping_character(code_body, 
                character_counter, "]")

                default_arguments.append(default_argument)

            # Otherwise, breaks, because it will not have more default
            # arguments

            else:

                break

        # Extracts the raw code that is left, but stops parsing when all
        # open brackets were closed

        raw_code = extract_raw_code(code_body, character_counter)

        return MacroDefinition(name=command_name, number_of_arguments=
        number_of_arguments, raw_code=raw_code, header=header, 
        default_arguments=default_arguments)

    # Defines a function to retrieve information from \newcommand*

    def newcommand_star_info(self, code_body):

        # Uses the same function as the newcommand, but alters the header

        return self.newcommand_info(code_body, character_counter=1, 
        header=r"\newcommand*")

    # Defines a function to retrieve information from \DeclareRobustCom-
    # mand

    def declare_robust_command_info(self, code_body):

        # Initializes a character counter
        
        character_counter = 1

        # Initializes the infos

        number_of_arguments = ""

        # Gets the command name

        (command_name, character_counter
        ) = parse_string_until_stopping_character(code_body, 
        character_counter, "}")

        # If the current character is a bracket, it indicates the number
        # of arguments of this command

        if code_body[character_counter]=="[":

            character_counter += 1

            # Gets the number of arguments

            (number_of_arguments, character_counter
            ) = parse_string_until_stopping_character(code_body, 
            character_counter, "]")

        if len(number_of_arguments)>0:

            # Converts the number of arguments to an integer

            number_of_arguments = int(number_of_arguments)

        else:

            number_of_arguments = 0

        # Extracts the raw code that is left, but stops parsing when all
        # open brackets were closed

        raw_code = extract_raw_code(code_body, character_counter)

        return MacroDefinition(name=command_name, number_of_arguments=
        number_of_arguments, raw_code=raw_code, header=
        r"\DeclareRobustCommand")

    # Defines a function to retrieve information from \definecolor

    def define_color_info(self, code_body):

        # Initializes a character counter
        
        character_counter = 1

        # Gets the command name

        (color_name, character_counter
        ) = parse_string_until_stopping_character(code_body, 
        character_counter, "}")

        # Gets the color system (example: rgb)
        
        (color_system, character_counter
        ) = parse_string_until_stopping_character(code_body, 
        character_counter+1, "}")

        # Extracts the raw code that is left, but stops parsing when all
        # open brackets were closed

        raw_code = extract_raw_code(code_body, character_counter)

        # Assembles the raw code using the LaTeX color format

        raw_code = "["+str(color_system)+"]{"+str(raw_code)+"}"

        return MacroDefinition(name=color_name, number_of_arguments=0,
        color_system=color_system, raw_code=raw_code, header=
        r"\definecolor")

    # Defines a function to retrieve information from the \RequirePackage
    # command

    def require_package_info(self, code_body):

        # Initializes a character counter
        
        character_counter = 0

        # Gets options for the package

        package_options = ""

        if code_body[character_counter]=="[":

            (package_options, character_counter
            ) = parse_string_until_stopping_character(code_body, 
            character_counter+1, "]")

        # Skips the first curly brackets

        character_counter += 1

        # Gets the package name
        
        (package_name, character_counter
        ) = parse_string_until_stopping_character(code_body, 
        character_counter, "}")

        # Assembles the raw code

        raw_code = r"\usepackage"

        # If there are options
        
        if len(package_options)>0:

            raw_code += "["+str(package_options)+"]"

        raw_code += "{"+str(package_name)+"}"

        # Flags not to parse this command since its name will also be in
        # its raw code

        return MacroDefinition(name=package_name, number_of_arguments=0,
        to_be_put_in_document_header=True, raw_code=raw_code, header=
        r"\RequirePackage", parseable_command=False)

    # Defines a function to build the header with package imports in the
    # final parsed file

    def build_header(self, list_of_data_classes):

        # Initializes the header

        document_header = ""

        # Iterates over the data classes of the commands

        for data_class in list_of_data_classes:

            # Verifies if the flag to put this command in the header is
            # True

            if data_class.to_be_put_in_document_header:

                # Gets the raw code and adds it

                document_header += "\n"+data_class.raw_code

        # Returns this as a conventional command using the data class

        return MacroDefinition(name=r"\usepackage{LaTeXUtilities}", 
        raw_code=document_header, parseable_command=False,
        number_of_arguments=0)