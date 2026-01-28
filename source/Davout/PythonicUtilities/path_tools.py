# Routine to store tools to work with path

import os

from pathlib import Path

import inspect

import unicodedata

########################################################################
#                              Path tools                              #
########################################################################

# Defines a function to get a list of directories within a path

def get_list_of_directories(whole_path):

    # Initializes a list of directories to create

    directories = [""]

    # Iterates through the path

    for i in range(len(whole_path)):

        # If the character is a bar, this means the last directory name
        # has been finished

        if whole_path[-(i+1)]=="/":

            # Verifies if the last directory is not empty

            if directories[-1]!="":

                directories.append("")

        # Otherwise, saves the characters

        else:

            directories[-1] = whole_path[-(i+1)]+directories[-1]

    # Checks if the last saved directory is empty

    if directories[-1]=="":

        directories = directories[0:-1]

    return directories

# Defines a function to verify if a file exists

def verify_file_existence(file_path, saving_function=None, termination=
None):

    # Verifies is the file path is a string

    if not isinstance(file_path, str):

        raise TypeError("The 'file_path'='"+str(file_path)+"' is not a"+
        " string, so the existence of this file cannot be asserted")
    
    # Verifies if the file has a particular termination

    if termination is not None:

        # Makes sure termination is a string

        if not isinstance(termination, str):

            raise TypeError("The 'termination' must be a string to ver"+
            "ify a file existence. However, it is: "+str(termination))
        
        # Verify if the length of the file path is greater than the ter-
        # mination

        if len(termination)>len(file_path):

            raise IndexError("The 'termination'="+str(termination)+" h"+
            "as more characters than the 'file_path' itself. Thus, the"+
            " file existence cannot be asserted")
        
        # Verifies if the termination is equal to that of the file

        if file_path[-len(termination):len(file_path)]!=termination:

            raise NameError("The termination="+str(termination)+" is n"+
            "ot the same as the 'file_path' "+str(file_path)+". Thus, "+
            "the file existence cannot be asserted")

    if not os.path.exists(file_path):

        # If a saving function has been provided uses it to save it

        if saving_function is None:

            raise FileNotFoundError("The file at '"+str(file_path)+"' "+
            "was not found")
        
        else:

            try:

                saving_function(file_path)

            except Exception as error_message:

                raise TypeError("It is not possible to save the file '"+
                str(file_path)+"' using function '"+str(
                saving_function.__name__)+"' due to the following exce"+
                "ption:\n"+str(error_message))

# Defines a function to verify if a path exists or not. If not, create
# it

def verify_path(parent_path, file_name):

    if (parent_path is None) or parent_path=="":

        return file_name

    # Checks if the parent path exists

    if not os.path.exists(parent_path):

        # Gets a list of the individual bits of path

        directories = get_list_of_directories(parent_path)

        # Initializes the path

        path = ""

        # Iterates through the directories

        for i in range(len(directories)):

            # Appends this bit of directory to the path

            path += "//"+directories[-(i+1)]

            # Checks if this path exists

            if not os.path.exists(path):

                # Creates the directory

                os.mkdir(path)

    # Joins everything together

    return parent_path+"//"+file_name
    
# Defines a function to get the path to a file

def get_parent_path_of_file(file="__file__", path_bits_to_be_excluded=1,
function_calls_to_retrocede=1):

    if file=="__file__":

        # Tries to automatically retrieve where this function has been
        # called

        # Gets the previous frame, where the function has been called

        last_frame = inspect.stack()[function_calls_to_retrocede]

        # Gets the module where it's been called

        module = inspect.getmodule(last_frame[0])

        # Verifies if this module is valid and gets the __file__ attri-
        # bute

        if (module is not None) and hasattr(module, "__file__"):

            file = module.__file__ 

        else:

            raise NameError("To get the parent path of a file, you sho"+
            "uld type file=file, where file is an imported file; or yo"+
            "u should type file=__file__ to get the parent path to the"+
            " current file. Otherwise, it will try to get the path of "+
            "the file where this function, 'get_parent_path_of_file', "+
            "has been called last. Unfortunately, to no avail this tim"+
            "e")
    
    # Gets a list of the bits of the whole path
    
    current_path = Path(os.path.abspath(file)).parts

    # Joins them again without the last bit, which is the name of the 
    # file itself

    return str(Path(*current_path[0:-path_bits_to_be_excluded]))

########################################################################
#                         File name alteration                         #
########################################################################

# Defines a function to delete a file

def delete_file(file_name, parent_path=None, ignore_non_existing_file=
False):

    # If the parent path is not None, joins it

    if parent_path is not None:

        file_name = parent_path+"//"+file_name

    # Verifies if the file exists

    if not ignore_non_existing_file:

        verify_file_existence(file_name)

        # Removes it

        os.remove(file_name)

# Defines a function to rename a file

def rename_file(old_file_name, new_file_name, parent_path=None,
saving_function=None):

    # If the parent path is not None, joins it

    if parent_path is not None:

        old_file_name = parent_path+"//"+old_file_name

    # Gets the termination of both files

    old_path, termination_old = take_outFileNameTermination(
    old_file_name, get_termination=True)

    new_file_name, termination_new = take_outFileNameTermination(
    new_file_name, get_termination=True)

    # Verifies if this file exists, and saves it if it does not exist
    # yet

    verify_file_existence(old_path+"."+termination_old, saving_function=
    saving_function)

    # Verifies if both terminations are the same

    if termination_old!=termination_new:

        raise NameError("It's not possible to change the name of the f"+
        "ile at '"+str(old_path)+"' to '"+str(new_file_name)+"' for th"+
        "e termination of the file is '"+str(termination_old)+"', and "+
        "the termination of the new name is '"+str(termination_new)+"'")

    # Transforms into Path and adds the new name

    old_path = Path(old_path+"."+termination_old)

    new_path = old_path.with_name(new_file_name+"."+termination_new)

    # Renames it and returns as string

    return str(old_path.rename(new_path))

# Defines a function to take out the termination file name of a string

def take_outFileNameTermination(file_name, get_termination=False):

    # Initializes the new name

    clean_fileName = ""

    # Initializes the termination

    termination = ""

    termination_reading = False

    # Iterates through the file name

    for character in file_name:

        if character=="." or character=='.':

            if not get_termination:

                break 

            else:

                termination_reading = True

        else:

            if termination_reading:

                termination += character

            else:

                clean_fileName += character

    # Returns the file name without the termination

    if get_termination:

        return clean_fileName, termination
    
    else:

        return clean_fileName
    
# Defines a function to decapitalize letters and change blank spaces for
# underline characters

def decapitalize_and_insert_underline(file_name):

    # Verifies if it is a string

    if not isinstance(file_name, str):

        raise TypeError("'file_name' is not a string in 'decapitalize_"+
        "and_insert_underline'. Thus, it cannot be used for decapitali"+
        "zing letters and changing blank spaces for underline characte"+
        "rs")
    
    # Initializes a new file name

    new_file_name = ""

    # Iterates through the characters

    for character in file_name:

        # Verifies if the character is not ASCII

        if ord(character)>127:
        
            character = unicodedata.normalize("NFKD", character
            ).encode("ascii", "ignore").decode("ascii")

        # Verifies if it is a blank space

        if character==" ":

            # Adds an underline to the new file name

            new_file_name += "_"

        # Verifies if the character is upper case

        elif character.isupper():

            # Converts to lower case

            new_file_name += character.lower()

        # Otherwise, just adds the character

        else:

            new_file_name += character 

    # Returns the new file name

    return new_file_name