# Routine to interact with the user

import numpy as np

from ..PythonicUtilities.string_tools import string_toList

########################################################################
#                            Terminal input                            #
########################################################################

# Defines a function to convert strings to useful formats

def convert_string(string, default_value, necessary_type=None, 
throw_error=True):

    # If the string has no length, returns the default value

    if len(string)==0:

        return default_value

    # Tries to convert to an integer

    try:

        string = int(string)

    except:

        # Tries to convert to a float

        try:

            string = float(string)

        except:

            # Tries to convert to a list

            if (string[0]=="[" and string[-1]=="]"):

                string = string_toList(string)

    # Verifies the type of the string if needed

    if (necessary_type is not None) and (not isinstance(string, 
    necessary_type)):
        
        if throw_error:
        
            raise TypeError("'"+str(string)+"' has type '"+str(type(
            string))+"', but the necessary type is "+str(necessary_type))
        
        # Otherwise, just prints the error

        else:

            print("'"+str(string)+"' has type '"+str(type(string))+"',"+
            " but the necessary type is "+str(necessary_type))

            return None

    # If the string is still a string and the default value is not None,
    # returns the default value

    if isinstance(string, str) and (default_value is not None) and (
    not isinstance(default_value, str)):

        return default_value
    
    # Otherwise, returns the original or converted value

    return string

# Defines a function to repeat a call until the right answer is given

def input_repeater(input_text, reviewer_function=None, default_value=
None, necessary_type=None):

    # Initializes a flag to keep repeating the question

    flag_repeat = True 

    # Repeats the question until a right answer is given

    while flag_repeat:

        # Asks the user for the input

        input_string =  input(input_text)
        
        # Converts the input string to another format if possible

        input_string = convert_string(input_string, default_value,
        necessary_type=necessary_type, throw_error=False)

        # If input string has not been falsified during conversion

        if input_string is not None:

            # Verifies if a right answer has been given

            if reviewer_function is not None:

                processed_answer = reviewer_function(input_string)

                # If it is not false, returns it

                if processed_answer:

                    flag_repeat = False 

                    return processed_answer
                
            # If the reviewer function is None, returns the answer any-
            # ways

            else:

                flag_repeat = False 

                return input_string

########################################################################
#                             Loop printing                            #
########################################################################

# Defines a class to print informations in loops inside a rectangular]
# frame of hashtags

class PrintForLoopInfo:

    def __init__(self, list_of_message_lines, 
    number_of_characters_per_line=72):

        self.list_of_message_lines = list_of_message_lines

        self.number_of_characters_per_line = (
        number_of_characters_per_line)

        # Verifies if the list of messages is a list

        if not isinstance(list_of_message_lines, list):

            raise TypeError("'list_of_message_lines' in 'PrintForLoopI"+
            "nfo' must be a list of messages to be printed. Currently,"+
            " it is not a list, but:\n"+str(list_of_message_lines))

        # Checks if each line of the message fits in the number of cha-
        # racters

        for line_index, line in enumerate(list_of_message_lines):

            # Disconsiders two characters for the left and right bounding
            # hashtags

            if len(line)>(number_of_characters_per_line-2):

                raise ValueError("The "+str(line_index+1)+"-th line of"+
                " 'list_of_message_lines' in 'PrintForLoopInfo' has "+
                str(len(line))+" characters, which is larger than "+str(
                number_of_characters_per_line-2)+".\nThe provided maxi"+
                "mum number of characters per line is "+str(
                number_of_characters_per_line)+", but 2 are restricted"+
                " to the left and right bounding hashtags")

        # Saves the number of lines to be plotted

        self.number_of_lines = len(list_of_message_lines)

    # Defines a method to actually print the information

    def __call__(self, list_of_results_to_be_printed):

        # Verifies if the number of given information is the same as the
        # number of message lines

        if (not isinstance(list_of_results_to_be_printed, list)) or len(
        list_of_results_to_be_printed)!=self.number_of_lines:

            raise IndexError("'list_of_results_to_be_printed' in 'Prin"+
            "tForLoopInfo' is not a list or it has not the same length"+
            " as the number of lines of the message list, "+str(
            self.number_of_lines))

        # Prints the first horizontal bar

        print("\n\n###################################################"+
        "#####################")

        # Iterates over the message lines and the corresponding results

        for message_line, result in zip(self.list_of_message_lines,
        list_of_results_to_be_printed):

            # Transforms the result to a string

            result_string = str(result)

            # Evaluates the number of characters of the result

            n_characters = len(result_string)+len(message_line)+2

            # Verifies if there is enough space for the whole step information

            if n_characters<=self.number_of_characters_per_line:

                # Calculates the clearance to each side

                clearance_left = int(np.ceil(0.5*(
                self.number_of_characters_per_line-n_characters)))

                clearance_right = (self.number_of_characters_per_line-
                clearance_left-n_characters)

                # Makes the clearance space

                clearance_space_left = ""

                clearance_space_right = ""

                for i in range(clearance_left):

                    clearance_space_left += " "

                for i in range(clearance_right):

                    clearance_space_right += " "

                print("#"+clearance_space_left+message_line+
                result_string+clearance_space_right+"#")

            # Otherwise, if there is no enough space for message in ad-
            # dition to the result

            else:

                # Calculates the clearance to each side of the message

                clearance_left_message = int(np.ceil(0.5*(
                self.number_of_characters_per_line-len(message_line)))-1)

                clearance_right_message = (
                self.number_of_characters_per_line-clearance_left_message
                -len(message_line)-2)

                # Calculates the clearance to each side of the result

                clearance_left_result = int(np.ceil(0.5*(
                self.number_of_characters_per_line-len(result_string)))
                -1)

                clearance_right_result = (
                self.number_of_characters_per_line-clearance_left_result
                -len(result_string)-2)

                # Makes the clearance space for the message line

                clearance_space_left_message = ""

                clearance_space_right_message = ""

                for i in range(clearance_left_message):

                    clearance_space_left_message += " "

                for i in range(clearance_right_message):

                    clearance_space_right_message += " "

                # Makes the clearance space for the result line

                clearance_space_left_result = ""

                clearance_space_right_result = ""

                for i in range(clearance_left_result):

                    clearance_space_left_result += " "

                for i in range(clearance_right_result):

                    clearance_space_right_result += " "

                # Prints both message and result and two lines

                print("#"+clearance_space_left_message+message_line+
                clearance_space_right_message+"#\n#"+
                clearance_space_left_result+result_string+
                clearance_space_right_result+"#")

        # Prints the final horizontal bar

        print("#######################################################"+
        "#################\n")