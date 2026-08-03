# Routine to store a set of methods to read a string and separate bits 
# of code from it

import bisect

########################################################################
#                            Command parser                            #
########################################################################

# Defines a function to verify if the given commands are contained in a
# string. If so, substitutes the method's raw code

def find_commands_and_substitute_raw_code(original_string, 
commands_names, commands_data_class, list_of_first_character_indices=
None, file_path=None, return_list_of_first_character_indices=False):

    # Initializes a flag to tell if commands were indeed found

    flag_found_commands = False

    # Makes a copy of the original string

    modified_string = str(original_string)

    # Initializes the first character to start the search for commands

    start_character_index = 0

    # Iterates over the commands

    for name, command_data_class in zip(commands_names, 
    commands_data_class):

        # Finds the command name in the string and the index of its 
        # first character

        first_character_index = modified_string.find(name,
        start_character_index)

        # Gets the line of the text where this bit was found

        line_number = "'no line numbering is available'"

        line_text = "no line text is available"

        if list_of_first_character_indices is not None:

            # Uses the bissection method to find the index of the first
            # character of the next line that comes right after the 
            # first character of this name

            line_number = bisect.bisect_right(
            list_of_first_character_indices, first_character_index)

            line_text = modified_string[list_of_first_character_indices[
            line_number-1]:list_of_first_character_indices[line_number]]

        # If the index of the first character is not -1, it was indeed
        # found

        if first_character_index!=-1:

            # If there are more characters after this command name, ve-
            # rify if this character is a letter

            if ((first_character_index+len(name))<len(modified_string)
            ) and modified_string[first_character_index+len(name)
            ].isalpha():

                # If it is a letter, the name that was found is in fact
                # part of a larger command name, thus ignore it

                continue

            # Updates the flag of found commands to True

            flag_found_commands = True

            # Gets possible default arguments

            default_arguments = []

            if command_data_class.default_arguments is not None:

                default_arguments = command_data_class.default_arguments

            # Recovers the list of the values of the arguments

            arguments_list, final_character_index = get_arguments_from_code(
            modified_string, first_character_index+len(name), 
            command_data_class.number_of_arguments, name, 
            default_arguments, line_number=line_number, file_path=
            file_path, line_text="")

            before_change = modified_string[first_character_index:min(
            final_character_index+20, len(modified_string))]

            # Gets a copy of the raw code of the command and inserts the
            # values of the arguments in the code

            raw_code_with_arguments = substitute_arguments_into_copy(
            command_data_class.raw_code, arguments_list)

            # Substitutes the code with the arguments into the original
            # string that has been iteratively modified

            modified_string = (modified_string[:first_character_index]
            +raw_code_with_arguments+modified_string[(
            final_character_index+1):])

            """if return_list_of_first_character_indices:

                print("\n\n\nbefore_change: "+str(before_change))

                print("\narguments_list: "+str(arguments_list))

                print("\nraw code without arguments: "+str(command_data_class.raw_code))

                print("\nraw_code_with_arguments: "+str(raw_code_with_arguments))

                print("\nnew string: "+str(modified_string[max(
                first_character_index-20,0):min(len(modified_string),
                first_character_index+len(raw_code_with_arguments)+20)]))

                answer = "a"#input()

                if answer=="b":

                    float(a)"""

            # Since the string has been modified, the list of indices of
            # the first character of each line must also be updated

            difference_of_characters = len(raw_code_with_arguments)-(
            final_character_index-first_character_index)

            if list_of_first_character_indices is not None:

                for line_index in range(line_number, len(
                list_of_first_character_indices)):

                    list_of_first_character_indices[line_index] += (
                    difference_of_characters)

            # Updates the start index for the next search

            start_character_index = first_character_index+len(
            raw_code_with_arguments)

    # If the list of first character indices must also be return

    if return_list_of_first_character_indices:

        return (modified_string, flag_found_commands, 
        list_of_first_character_indices)

    # Returns the modified string and the flag of found commands

    return modified_string, flag_found_commands

########################################################################
#                           Argument parser                            #
########################################################################

# Defines a function to retrieve the values of the arguments of a com-
# mand from a bit of raw code

def get_arguments_from_code(raw_code, initial_character_counter, 
number_of_arguments, command_name, default_arguments, line_number="", 
line_text="", file_path=None):

    start_character_index = initial_character_counter+0

    # If default arguments were given, the reading of arguments must
    # take shape till the next bracket

    stopping_character = None

    character_to_be_ignored = None

    if len(default_arguments)>0:

        stopping_character = "]"

        character_to_be_ignored = "["

    # Initializes a list of arguments

    arguments_list = []

    # Iterates over the necessary arguments

    if number_of_arguments is None:

        raise ValueError("Command '"+str(command_name)+"' currently ha"
        "s 'number_of_arguments' as None, which is not allowable")

    for i in range(number_of_arguments):

        initial_character = raw_code[initial_character_counter]

        # Verifies if there are default arguments, but the current cha-
        # racter is not { nor [, ends the loop

        if len(default_arguments)>0 and (initial_character!="{" and (
        initial_character!="[")):

            break

        # Verifies if the current character is an opening bracket

        elif (initial_character_counter>=len(raw_code) or (
        initial_character!="{")) and len(default_arguments)==0:

            # Otherwise, describes the problem

            problem = ""

            if initial_character_counter>=len(raw_code):

                problem = ("There are no more characters after the giv"+
                "en command")

            elif raw_code[initial_character_counter]!="{":

                problem = ("After the command, there is no { to indica"+
                "te arguments")

            given_arguments = "\n"

            for argument_value in arguments_list:

                given_arguments += "\n"+str(argument_value)

            raise ValueError("At line:\n"+str(line_text)+"\n\nCommand "+
            "'"+str(command_name)+"' requires "+str(number_of_arguments
            )+" arguments, but only "+str(i)+" were given: "+
            given_arguments+"\n\nThe problem is:\n\n'"+problem+"'\n\nI"+
            "t happended in the followind snippet of code:\n"+
            str(raw_code[start_character_index:min(len(raw_code),20+
            initial_character_counter)])+"\n\nwhich lies in line numbe"+
            "rred "+str(line_number)+"\n\nof file:\n"+str(file_path))

        # Parses the string until the next curly bracket

        argument_string, initial_character_counter = extract_raw_code(
        raw_code, initial_character_counter, return_final_counter=True,
        stopping_character=stopping_character, character_to_be_ignored=
        character_to_be_ignored)

        # Appends the argument according to its classification

        if initial_character=="[":

            default_arguments[i] = argument_string

        else:

            arguments_list.append(argument_string)

    # If default arguments were given

    if len(default_arguments)>0:

        # Adds the two cases of arguments

        default_arguments.extend(arguments_list)

        # Subtracts -1 of the character counter to yield the index of 
        # the last curly bracket
    
        return default_arguments, initial_character_counter-1

    # Subtracts -1 of the character counter to yield the index of the 
    # last curly bracket

    return arguments_list, initial_character_counter-1

# Defines a function to substitute the arguments into a copy of the raw
# code of the command

def substitute_arguments_into_copy(raw_code, arguments_list):

    raw_code_copy = str(raw_code)

    # Iterates over the list of arguments

    for i, argument in enumerate(arguments_list):

        # Finds the argument indicator in the raw code and replaces by
        # the argument

        argument_indicator = "#"+str(i+1)

        raw_code_copy = raw_code_copy.replace(argument_indicator, 
        argument)

    # Returns the replaced raw code

    return raw_code_copy

########################################################################
#                          Parsing utilities                           #
########################################################################

# Defines a function to extract raw code such that all open brackets are
# closed within the extracted raw code

def extract_raw_code(code_string, initial_character_counter, 
return_final_counter=False, stopping_character=None, 
character_to_be_ignored=None):

    # Initializes the raw code

    raw_code = ""

    # Initializes a counter of open brackets

    open_brackets_counter = 0

    for i in range(initial_character_counter, len(code_string)):

        # Gets the character

        character = code_string[i]

        initial_character_counter = i+1

        # If the character is {, adds an open bracket to that counter

        if character=="{":

            open_brackets_counter += 1

            # If it is the second open brackets, saves it to the raw
            # code

            if open_brackets_counter>1:

                raw_code += character

        # If it is }, removes one open brackets

        elif character=="}":

            open_brackets_counter -= 1

            # If it is the last closing brackets, breaks the loop

            if open_brackets_counter==0:

                break 

            # Otherwise, saves to the raw code

            else:

                raw_code += character

        # Verifies if there is a stopping character to terminate

        elif (stopping_character is not None) and character==(
        stopping_character):

            break

        # Otherwise, if it is not the character to be ignored, saves to 
        # the raw code

        elif (character_to_be_ignored is None) or (character!=
        character_to_be_ignored):

            raw_code += character

    if return_final_counter:

        return raw_code, initial_character_counter

    return raw_code

# Defines a function to parse a string until a character is found

def parse_string_until_stopping_character(original_string, 
character_counter, stopping_character):

    # Initializes the string to be parsed

    parsed_information = ""

    # Iterates over the characters to parse the command name
    
    for i in range(character_counter, len(original_string)):

        # Gets the character

        character = original_string[i]

        # Updates the character counter

        character_counter += 1

        # If the character is the stopping character, it is the end of 
        # the string to be parsed

        if character==stopping_character:

            break

        # Otherwise, updates the command name

        parsed_information += character

    return parsed_information, character_counter

# Defines a function to get a list of the index of the first character
# of each line of a file

def get_indices_of_first_character_of_each_line(full_string):

    # Initializes the list of indices 

    lines_first_character_indices = [0]

    for index, character in enumerate(full_string):

        # If the character is \n

        if character=="\n":

            lines_first_character_indices.append(index)

    return lines_first_character_indices