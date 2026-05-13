# Routine to interact with the user

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