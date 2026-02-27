# Routine to store decorators and programming utilities

import functools

import inspect

import os

from copy import deepcopy

import subprocess

from collections import OrderedDict

#from ..PythonicUtilities import dictionary_tools

########################################################################
#                           Imports preamble                           #
########################################################################

from pathlib import Path

from importlib import util

import sys

from tqdm import tqdm

# Gets the parent paths of the current 

broken_path = Path(__file__).parents

# Imports recursion tools

specifications = util.spec_from_file_location("dictionary_tools", 
broken_path[1]/"PythonicUtilities"/"dictionary_tools.py")

dictionary_tools = util.module_from_spec(specifications)

sys.modules["dictionary_tools"] = dictionary_tools

specifications.loader.exec_module(dictionary_tools)

########################################################################
#                             Progress bar                             #
########################################################################

# Defines a class of colors to colorize text on terminal

class TerminalColor:

    def __init__(self):

        # Defines a dictionary of colors and their terminal keys

        self.color_dictionary = {"green": "\033[92m", "red": "\033[91m", 
        "yellow": "\033[93m", "cyan": "\033[96m", "bold light blue": 
        "\033[1m\033[94m", "bold red": "\033[1m\033[91m", "reset": 
        "\033[0m", "purple": "\033[95m"}

    # Defines a method to get color and text and combine them into a sin-
    # gle setring

    def __call__(self, text, color_name):
        
        # Verifies if color name is in the dictionary of colors

        if not (color_name in self.color_dictionary):

            available_colors = ""

            for name in self.color_dictionary.keys():

                available_colors += "\n'"+str(name)+"'"

            raise ValueError("'"+str(color_name)+"' is not a valid col"+
            "or name. Check the available names: "+available_colors)
        
        # Concatenates the command with the text

        return (self.color_dictionary[color_name]+str(text)+
        self.color_dictionary["reset"])

# Defines a function to create a progress bar

def progress_bar(iteration_set, message=None, colorful_bar=True):

    # If message is None, gets an empty string

    if message is None:

        message = ""

    # Customizes the progress bar, ANSI color codes for the terminal 

    bold_light_blue = ""

    bold_red = ""

    reset = ""

    # Verifies if color is desired and if the terminal allows it

    if colorful_bar and hasattr(sys.stdout, "isatty") and (
    sys.stdout.isatty()):

        bold_light_blue = "\033[1m\033[94m"
    
        bold_red = "\033[1m\033[91m" 
        
        reset = "\033[0m"

    # Creates the string for the custom bar

    custom_bar = ("{l_bar}{bar}| {n_fmt}/{total_fmt} ["+bold_light_blue+
    "Elapsed: {elapsed}"+reset+" | "+bold_red+"Estimated time"+
    ": {remaining}"+reset+"]")
    
    return tqdm(iteration_set, desc=message, bar_format=custom_bar) 

########################################################################
#               Handler and executioner of other scripts               #
########################################################################

# Defines a function to run a script as if you would run it yourself in
# a terminal. This helper function is useful to run files that require
# special python interpreters

def script_executioner(file_name, python_interpreter="python3",
function_name=None, arguments_list=None, keyword_argumentsDict=None,
parent_path=None, execution_rootPath=None, verbose=True, run_as_module=
False, LIBGL_ALWAYS_SOFTWARE=False, extra_execution_arguments=None):
    
    # Verifies the existence of the parent path

    if not (parent_path is None):

        if not os.path.exists(parent_path):

            raise FileNotFoundError("The parent path '"+str(parent_path
            )+"' was not found, and, thus, cannot be used to run the s"+
            "cript '"+str(file_name)+"'")
        
        elif not os.path.exists(parent_path+"//"+file_name):

            raise FileNotFoundError("The file at '"+str(parent_path)+
            "' by the name '"+str(file_name)+"' was not found, and, th"+
            "us, cannot be run externally")
    
        # Gets the path together

        file_name = os.path.join(parent_path, file_name)
        
    elif not run_as_module:

        # Verifies the existence of the file
        
        if not os.path.exists(file_name):

            raise FileNotFoundError("The file at '"+str(file_name)+
            "' was not found, and, thus, cannot be run externally")
        
    # Verifies if the file name has the termination .py

    if not run_as_module:

        if len(file_name)>2:

            if file_name[-3:len(file_name)]!=".py":

                raise NameError("The file name '"+str(file_name)+"' do"+
                "es not end with '.py', thus, cannot be run externally")
            
        else:

            raise NameError("The file name '"+str(file_name)+"' does n"+
            "ot have at least 3 characters. Hence, it is not possibly "+
            "a runnable file")
        
    # Gets the arguments of the execution

    exe_arguments = None 

    # If the python file is to be run as a module

    if run_as_module:

        # Looks for // in the file name and substitute for .

        file_name_copy = deepcopy(file_name)

        file_name = ""

        # Iterates through the characters of the copy

        for index, character in enumerate(file_name_copy):

            # Adds the character if it is not a bar

            if character!="/":

                file_name += character

            # Otherwise

            else:

                file_name += "."

        # Adds the file name

        exe_arguments = [python_interpreter, "-m", file_name]

    # Otherwise, just as a script

    else:

        exe_arguments = [python_interpreter, file_name]

    # Verifies if extra arguments are given

    if extra_execution_arguments is not None:

        exe_arguments = [exe_arguments[0], *extra_execution_arguments, 
        *exe_arguments[1:len(exe_arguments)]]

    # Initializes a copy of the arguments to print on screen

    arguments_copy = {"python interpreter": python_interpreter, "file "+
    "name": file_name}

    # Verifies if a specific function must be run

    if not (function_name is None):

        # Checks if the function name is a string

        if not isinstance(function_name, str):

            raise TypeError("The function "+str(function_name)+" is no"+
            "t a string. To execute functions externally in script_exe"+
            "cutioner, the function_name argument must be a string")

        exe_arguments.append(function_name)

        # Saves the arguments for printing

        if verbose:

            arguments_copy["function name"] = function_name

    # Verifies if positional arguments were given

    if not (arguments_list is None):

        if not isinstance(arguments_list, list):

            raise TypeError("'arguments_list' must be a list, to pass "+
            "it to the function '"+str(function_name)+"' and run it ex"+
            "ternally")

        exe_arguments.extend(arguments_list)

        # Saves the arguments for printing

        if verbose:

            arguments_copy["arguments list"] = arguments_list

    # Verifies if keyword arguments were given

    if not (keyword_argumentsDict is None):

        if not isinstance(keyword_argumentsDict, dict):

            raise TypeError("'keyword_argumentsDict' must be a diction"+
            "ary, the keys are the names of the keyword arguments and "+
            "the values are the keyword arguments' values")
        
        for key, value in keyword_argumentsDict.items():

            # Disregards None values, because they might be non-desired
            # keyword arguments in the first place that were automati-
            # cally put in

            if not (value is None):

                # Transforms the key-value pair into a string

                exe_arguments.append(f"--{key}")

                exe_arguments.append(str(value))

                # Saves the arguments for printing

                if verbose:

                    arguments_copy[f"--{key}"] = value

    if verbose:

        print("Script_executioner arguments:\n")

        for argument_name, argument_value in arguments_copy.items():

            print(str(argument_name)+": "+str(argument_value)+"\n")

    # Retrieves the environment

    env = os.environ.copy()

    # Verifies if the flag LIBGL_ALWAYS_SOFTWARE is to be set

    if LIBGL_ALWAYS_SOFTWARE:

        env["LIBGL_ALWAYS_SOFTWARE"] = "1"

    # If the root path is passed 

    if not (execution_rootPath is None):

        env["PYTHONPATH"] = execution_rootPath+(":"+env.get("PYTHONPAT"+
        "H", ""))
    
    # Executes the file

    execution = subprocess.run(exe_arguments, capture_output=True, text=
    True, env=env)

    # Verifies if there was an error 

    if verbose:

        if execution.returncode!=0:

            print("An error ocurred running "+str(python_interpreter)+
            " scrip at "+str(file_name)+":\n")

            print(execution.stderr)

        else:

            print("The script at "+str(file_name)+" was successfully r"+
            "un externally\n")

            print(execution.stdout)

########################################################################
#                         Memory-handling tools                        #
########################################################################

# Defines a function to handle optional mutable arguments, to avoid re-
# writing in memory during multiple method calls. Use it as a decorator:
#
# @optional_argumentsInitializer({argument1: value1, argument2: value2})
# def function_example(arguments, argument1=None, argument2=None):
#     ...
# The value must be function so that it gives fresh values at every time

def optional_argumentsInitializer(default_argumentsDictionary):

    # Checks if all the values of the dictionary are callable, i.e. if
    # they are all functions

    for key, value in default_argumentsDictionary.items():

        if not callable(value):

            raise TypeError("All values in the dictionary of the decor"+
            "ator to get the default values for optional arguments mus"+
            "t be callable. Use lambda functions or functions.")

    # Defines the decorator function. It receives the decorated function
    # as argument

    def decorator(decorated_function):

        @functools.wraps(decorated_function)

        # Defines the wrapper to get the arguments and check if any of 
        # them is in the list of arguments to be initialized as prescri-
        # bed

        def wrapper(*positional_arguments, **keyword_argument):

            # Gets the names of the arguments using internal Python syn-
            # tax. Then, turns the arguments as a list

            arguments_names = decorated_function.__code__.co_varnames

            positional_arguments = list(positional_arguments)

            # Iterates through the arguments names and gets the index of
            # each name

            for index, name in enumerate(arguments_names):

                # Checks if this argument is in the names of arguments
                # that are prescribed by default

                if name in default_argumentsDictionary:

                    # If the index is still less than the number of po-
                    # sitionally given arguments

                    if index<len(positional_arguments):

                        # Checks if the given value is None; if so, 
                        # substitutes it by the default value

                        if positional_arguments[index] is None:

                            positional_arguments[index] = (
                            default_argumentsDictionary[name]())

                    # If it is a keyword argument

                    else:

                        # Checks if the given value is None; if so, 
                        # substitutes it by the default value

                        if keyword_argument.get(name, None) is None:

                            keyword_argument[name] = (
                            default_argumentsDictionary[name]())

            return decorated_function(*positional_arguments, 
            **keyword_argument)
        
        return wrapper
    
    return decorator

########################################################################
#                           Dictionary tools                           #
########################################################################

# Defines a function to swap the keys of a dictionary. Uses a list of 
# lists to change some or all keys; each sublist is a pair of old and 
# new keys

def change_dictionaryKeys(original_dictionary, old_keysList):

    for old_key, new_key in old_keysList:

        # Tests if the key is in the dictionary

        if old_key in original_dictionary:

            # Adds the new key

            original_dictionary[new_key] = original_dictionary[old_key]

            # Deletes the old key

            original_dictionary.pop(old_key)

        # If this key is not found, raises an error message

        else:

            raise KeyError("The key '"+str(old_key)+"' cannot be delet"+
            "ed and swapped by the new key '"+str(new_key)+"' because "+
            "the old keys does not exist in the dictionary.")

    return original_dictionary

########################################################################
#                         Return-handling tools                        #
########################################################################

# Defines a function to get a particular variable from the result of a 
# function

def get_result(whole_result, variable_name):

    # Verifies whether the whole result is a dictionary

    if isinstance(whole_result, dict):

        # Tries to read the variable name as a key

        try:

            return whole_result[variable_name]
        
        except:

            raise KeyError("The name '"+str(variable_name)+"' is not a"+
            " key of the result dictionary of this function. Do you me"+
            "an one of the following possible keys? "+str(set(
            whole_result.keys()))[1:-1])
        
    # Verifies if it is a tuple

    if isinstance(whole_result, tuple):

        # Verifies if it is a named tuple
        
        if hasattr(whole_result, '_fields'):
        
            try:

                return getattr(whole_result, variable_name)
            
            except:

                raise AttributeError("The named tuple returned by the "+
                "function does not have a variable named "+str(
                variable_name)+". This named tuple has the following a"+
                "ttributes:\n"+str(whole_result._fields)[1:-1])
            
        # If it is not a named tuple, raise an exception

        else:

            raise TypeError("The result of the function is a tuple but"+
            " it is not a named tuple. Thus, the variable name '"+str(
            variable_name)+"' cannot be used")
        
    # Simply gives the result
        
    else:

        return whole_result
    
# Defines a function to verify whether a class has an attribute. If it
# does, returns the attribute, otherwise gives an error message

def get_attribute(class_object, attribute_name, error_message, 
dictionary_of_methods=False, delete_init_key=False, reserved_methods=
None, sort_methods_alphabetically=True):

    """
    Function to verify if a class instance has an attribute and to build 
    a dictionary of methods defined in a class. The keys are the methods 
    names, and the values are the executable methods themselves

    class_object: a class instance
    
    attribute_name: a string with the attribute name
    
    error_message: a string with the error message in case the class 
    does not have this attribute. This function is meant to give 
    meaningful error messages
    
    If 'dictionary_of_methods' is True, a dictionary of methods (method
    name - callable method function) of the instance of a class will be 
    returned. If you want just a dictionary of methods. Put simply 'att
    ribute_name' and 'error_message' as None"""

    if dictionary_of_methods:

        # Transforms the reserved methods to the default empty list
        
        if reserved_methods is None:
    
            reserved_methods = []

        # Initializes a dictionary of methods

        dictionary_of_methods = OrderedDict()

        # If methods are to be alphabetically ordered

        if sort_methods_alphabetically:

            # Iterates over the methods of this class instance

            for name, member in inspect.getmembers(class_object, 
            predicate=inspect.ismethod):
                
                # Verifies if the name is not in one of the reserved
                # functions

                if not (name in reserved_methods):

                    dictionary_of_methods[name] = member

        # Otherwise, just use the dictionary of the class to preserve 
        # the implementation order of the methods

        else:

            # Iterates over the objects of the class
        
            for name, member in class_object.__class__.__dict__.items():

                # Verifies if the object is callable, i.e. a function

                if inspect.isfunction(member):

                    # If it is not in the list of reserved methods, sa-
                    # ves it

                    if not (name in reserved_methods):

                        dictionary_of_methods[name] = getattr(
                        class_object, name)

        # If the __init__ method is to be taken away

        if delete_init_key:

            return dictionary_tools.delete_dictionary_keys(
            dictionary_of_methods, "__init__")

        return dictionary_of_methods

    # Verifies whether this class has this attribute

    if hasattr(class_object, attribute_name):

        return getattr(class_object, attribute_name)
    
    else:

        raise AttributeError(error_message)

########################################################################
#                Classes from file automatic retriever                 #
########################################################################

# Defines a function to get the entities of a python module (
# searched_file) which are subclasses of a parent class. The 
# reserved_entities is a list of entities that shall not be given as a 
# process for actual evaluation, they can be parent classes for example

@optional_argumentsInitializer({'reserved_entities': lambda: []})

def get_allProcesses(searched_file, parent_class=None, reserved_entities=
None, searched_entity="classes"):
    
    # Sets the stuff you want to search for

    if searched_entity=="classes":

        searched_entity = inspect.isclass

    elif searched_entity=="functions":

        searched_entity = inspect.isfunction

    else:

        raise NameError("The entitities '"+str(searched_entity)+"' can"+
        "not be searched. You can select from either 'classes' or 'fun"+
        "ctions'")

    # Initializes a dictionary of entities for the entities found in a 
    # file

    entities_dict = dict()

    # If the searched entity is a function or the searched classes are 
    # not children of some parent class

    if searched_entity==inspect.isfunction or (parent_class is None):

        # Checks for the entities in the module using inspect library

        for entity_name, entity_object in inspect.getmembers(
        searched_file, searched_entity):
            
            # Checks if the class is a subclass of the parent class and
            # if the entity is local to the searched file

            if (not (entity_object in reserved_entities)) and (
            entity_object.__module__==searched_file.__name__):
                
                # Appends this subclass

                entities_dict[entity_name] = entity_object

    # If the searched entity is a class and the parent class is not None

    elif searched_entity==inspect.isclass:

        # Checks for the entities in the module using inspect library

        for entity_name, entity_object in inspect.getmembers(
        searched_file, searched_entity):
            
            # Checks if the class is a subclass of the parent class

            if (issubclass(entity_object, parent_class) and (not (
            entity_object in reserved_entities)) and (
            entity_object.__module__==searched_file.__name__)):
                
                # Appends this subclass

                entities_dict[entity_name] = entity_object

    # Returns the lis of classes

    return entities_dict

# Defines a function to initializes the needed methods' classes

@optional_argumentsInitializer({'reserved_entities': lambda: []})

def dispatch_classes(methods_names, searched_file, parent_class=None, 
class_input=None, reserved_entities=None):

    # Assures methods_names is a list, or a tuple or a dictionary

    if not (isinstance(methods_names, list) or isinstance(methods_names, 
    tuple) or isinstance(methods_names, dict)):

        methods_names = [methods_names]

    # Initializes a dictionary of classes that will be used later

    methods_classes = dict()

    # Gets the methods classes of this file as a dictionary, the keys a-
    # re the names of the classes whereas the values are the classes ob-
    # jects

    methods_classesDict = get_allProcesses(searched_file, parent_class=
    parent_class, reserved_entities=reserved_entities)

    available_methodsNames = list(methods_classesDict.keys())

    # Iterates through the methods names

    if class_input is None:

        for method_name in methods_names:

            # If the name is in the list of available methods, updates 
            # the dictionary of classes

            if method_name in available_methodsNames:

                methods_classes[method_name] = methods_classesDict[
                method_name]()

            else:

                available_list = "\n"

                for name in available_methodsNames:

                    available_list += "'"+name+"'\n"

                raise NameError("'"+str(method_name)+"' is not an avai"+
                "lable class method. Find one in the list:"+
                available_list)

    elif isinstance(class_input, tuple):

        for method_name in methods_names:

            # If the name is in the list of available methods, updates 
            # the dictionary of classes

            if method_name in available_methodsNames:

                methods_classes[method_name] = methods_classesDict[
                method_name](*class_input)

            else:

                available_list = "\n"

                for name in available_methodsNames:

                    available_list += "'"+name+"'\n"

                raise NameError("'"+str(method_name)+"' is not an avai"+
                "lable class method. Find one in the list:"+
                available_list)

    else:

        for method_name in methods_names:

            # If the name is in the list of available methods, updates 
            # the dictionary of classes

            if method_name in available_methodsNames:

                methods_classes[method_name] = methods_classesDict[
                method_name](class_input)

            else:

                available_list = "\n"

                for name in available_methodsNames:

                    available_list += "'"+name+"'\n"

                raise NameError("'"+str(method_name)+"' is not an avai"+
                "lable class method. Find one in the list:"+
                available_list)
        
    # Returns the dictionary of classes that will be used given the na-
    # mes and the available classes

    return methods_classes

# Defines a function to initializes the needed methods' functions

@optional_argumentsInitializer({'reserved_entities': lambda: []})

def dispatch_functions(methods_names, searched_file, reserved_entities=
None, fixed_inputVariablesDict=None, methods_functionsDict=None, 
return_list=False, return_singleFunction=False, all_argumentsFixed=False,
second_sourceFixedArguments=None):

    # Assures methods_names is a list, or a tuple or a dictionary

    if not (isinstance(methods_names, list) or isinstance(methods_names, 
    tuple) or isinstance(methods_names, dict)):

        methods_names = [methods_names]

    # Initializes a dictionary of functions that will be used later

    methods_functions = OrderedDict()

    # Gets the functions of this file as a dictionary, the keys are the
    # names of the functions whereas the values are the functions objects

    if methods_functionsDict is None:

        methods_functionsDict = get_allProcesses(searched_file, 
        reserved_entities=reserved_entities, searched_entity="functions")

    available_methodsNames = list(methods_functionsDict.keys())

    # Iterates through the methods names

    if (fixed_inputVariablesDict is None) and (
    second_sourceFixedArguments is None):

        # Iterates through the methods

        for method_name in methods_names:

            # If the name is in the list of available methods, updates 
            # the dictionary of classes

            if method_name in available_methodsNames:

                methods_functions[method_name] = methods_functionsDict[
                method_name]

            else:

                available_list = "\n"

                for name in available_methodsNames:

                    available_list += "'"+name+"'\n"

                raise NameError("'"+str(method_name)+"' is not an avai"+
                "lable function method. Find one in the list:"+
                available_list)

    else:

        # Verifies if the dictionary of input variables to be fixed is 
        # indeed a dictionary

        if not isinstance(fixed_inputVariablesDict, dict):

            raise TypeError("The dictionary of fixed input variables m"+
            "ust be a dictionary")

        # Iterates through the methods

        for method_name in methods_names:

            # Verifies if the dictionary of input variables to be fixed
            # has this method as a key

            fixed_arguments = []

            if not (method_name in available_methodsNames):

                error_string = ("The '"+str(method_name)+"' is not an "+
                "available method. Check out the available methods: ")

                for available_name in available_methodsNames:

                    error_string += "\n"+str(available_name)

                raise ValueError(error_string)

            if isinstance(fixed_inputVariablesDict, dict):

                # Checks if the dictionary of fixed input variables is a
                # dictionary of dictionaries

                if isinstance(fixed_inputVariablesDict[list(
                fixed_inputVariablesDict.keys())[0]], dict):

                    if not (method_name in fixed_inputVariablesDict):

                        raise KeyError("The dictionary of fixed input vari"+
                        "ables does not have the method '"+str(method_name)+
                        "' as a key. The available methods are: "+str(
                        fixed_inputVariablesDict.keys()))

                    fixed_arguments = fixed_inputVariablesDict[method_name]
                
                else:

                    fixed_arguments = fixed_inputVariablesDict
            
            else:

                raise TypeError("The corresponding value for the metho"+
                "d '"+str(method_name)+"' in the dictionary of fixed i"+
                "nput variables is not a dictionary")

            # If the name is in the list of available methods, updates 
            # the dictionary of classes

            if method_name in available_methodsNames:

                # If there is no fixed arguments, just gets the function

                if (len(fixed_arguments.keys())==0) and (
                second_sourceFixedArguments is None):

                    methods_functions[method_name] = methods_functionsDict[
                    method_name]

                # Otherwise, separates the fixed arguments from the free
                # ones

                else:

                    # Sets the new function as a driver, i.e. fixing so-
                    # me arguments given by the fixed arguments dictio-
                    # nary
                    
                    methods_functions[method_name] = driver_maker(
                    methods_functionsDict[method_name], method_name, 
                    fixed_arguments, all_argumentsFixed=
                    all_argumentsFixed, second_sourceFixedArguments=
                    second_sourceFixedArguments)

            else:

                available_list = "\n"

                for name in available_methodsNames:

                    available_list += "'"+name+"'\n"

                raise NameError("'"+str(method_name)+"' is not an avai"+
                "lable function method. Find one in the list:"+
                available_list)
            
    # Verifies if the output must be a list instead of a dictionary. When
    # usage is direct

    if return_list:

        methods_functions = list(methods_functions.values())

        if return_singleFunction and len(methods_functions)==1:

            methods_functions = methods_functions[0]

    return methods_functions, methods_functionsDict

# Defines a maker of driver functions by fixing the arguments of some 
# other function

def driver_maker(method_function, method_name, fixed_arguments, 
all_argumentsFixed=False, second_sourceFixedArguments=None):

    # Gets the names of the arguments of the function that is being as-
    # signed

    arguments_names = inspect.signature(method_function).parameters

    # Separates the positional arguments from the keyword ones

    positional_arguments = [name for name, param in (
    arguments_names.items()) if param.default is inspect.Parameter.empty]

    keyword_arguments = [name for name, param in arguments_names.items(
    ) if param.default is not inspect.Parameter.empty]

    # Initializes a dictionary of fixed arguments that are really argu-
    # ments of the function

    functions_fixedArguments = dict()

    # If more fixed arguments were provided, thus, enabling all the ar-
    # guments to be picked from a larger pool of arguments

    if all_argumentsFixed:

        # Verifies if the additional dictionary of arguments is None. 
        # This additional dictionary is used to set up input that comes
        # from other source

        if second_sourceFixedArguments is None:

            # Verifies if any of the positional arguments was not provi-
            # ded in the fixed arguments

            for fixed_argument in positional_arguments:

                if not (fixed_argument in fixed_arguments):

                    raise NameError("The argument '"+str(fixed_argument
                    )+"' was not provided in the list of fixed argumen"+
                    "ts. Check the list of fixed arguments: "+str(
                    fixed_arguments.keys())+"\n\nCheck the list of req"+
                    "uired positional arguments: "+str(
                    positional_arguments))
        
                # If this argument's name is in the general dictionary 
                # of fixed arguments, updates the functions own dictio-
                # nary of this kind
                
                else:
                    
                    functions_fixedArguments[fixed_argument] = (
                    fixed_arguments[fixed_argument])

            # Verifies if any of the keyword arguments was not provided 
            # in the fixed arguments

            for fixed_argument in keyword_arguments:

                if fixed_argument in fixed_arguments:
                    
                    functions_fixedArguments[fixed_argument] = (
                    fixed_arguments[fixed_argument])

        elif isinstance(second_sourceFixedArguments, dict):

            # Verifies if any of the positional arguments was not provi-
            # ded in the fixed arguments

            for fixed_argument in positional_arguments:
        
                # If this argument's name is in the general dictionary 
                # of fixed arguments, updates the functions own dictio-
                # nary of this kind
                
                if fixed_argument in fixed_arguments:
                    
                    functions_fixedArguments[fixed_argument] = (
                    fixed_arguments[fixed_argument])

                # Or in the second source of arguments

                elif fixed_argument in second_sourceFixedArguments:
                    
                    functions_fixedArguments[fixed_argument] = (
                    second_sourceFixedArguments[fixed_argument])

                # Otherwise, throws an error

                else:

                    raise NameError("The argument '"+str(fixed_argument
                    )+"' was not provided in the dictionary of fixed a"+
                    "rguments nor in. Check the list of fixed argument"+
                    "s: "+str(fixed_arguments.keys())+"\nCheck the lis"+
                    "t of second source arguments"+str(
                    second_sourceFixedArguments.keys())+"\n\nCheck the"+
                    " list of required positional arguments: "+str(
                    positional_arguments))

            # Verifies if any of the keyword arguments was not provided 
            # in the fixed arguments

            for fixed_argument in keyword_arguments:

                if fixed_argument in fixed_arguments:
                    
                    functions_fixedArguments[fixed_argument] = (
                    fixed_arguments[fixed_argument])

                elif fixed_argument in second_sourceFixedArguments:
                    
                    functions_fixedArguments[fixed_argument] = (
                    second_sourceFixedArguments[fixed_argument])

            # If any information of the second source is not in the list
            # of keyword arguments, throw an error

            for fixed_argument in second_sourceFixedArguments:

                if ((not (fixed_argument in keyword_arguments)) and (
                not (fixed_argument in positional_arguments))):

                    raise KeyError("The keyword argument '"+str(
                    fixed_argument)+"' was given by the second source "+
                    "of arguments, but was not found in the function's"+
                    " keyword arguments. Check the list of available k"+
                    "eyword arguments of the function '"+str(method_name
                    )+"': "+str(keyword_arguments))

        else:

            raise TypeError("The second source fixed arguments must be"+
            " a dictionary or None, whereas the provided one is "+str(
            second_sourceFixedArguments))

    else:

        # Verifies if any of the fixed arguments is not really an argu-
        # ment of the required function

        for fixed_argument in fixed_arguments:

            if not (fixed_argument in arguments_names):

                raise NameError("The fixed argument '"+str(
                fixed_argument)+"' is not an argument of the function "+
                "'"+str(method_name)+"'. Did you mean any of the follo"+
                "wing arguments? "+str(arguments_names))
    
            # If this argument's name is in the general dictionary of 
            # fixed arguments, updates the functions own dictionary of
            # this kind
            
            else:
                
                functions_fixedArguments[fixed_argument] = (
                fixed_arguments[fixed_argument])

    # Gets the free arguments from the difference with the list of fixed 
    # arguments. Again differentiates the positional from the keyword 
    # arguments

    free_positionalNames = [free_name for free_name in (
    positional_arguments) if free_name not in functions_fixedArguments]

    free_keywordNames = [free_name for free_name in (keyword_arguments
    ) if free_name not in functions_fixedArguments]

    # Creates the driver of the function using a wrapper to keep the o-
    # riginal function's information

    @functools.wraps(method_function)

    def driver_function(*arguments, **key_arguments):

        # Checks if the arguments given have the same length of the re-
        # quired arguments

        if len(arguments)!=len(free_positionalNames):

            raise ValueError("You've provided "+str(len(arguments))+" "+
            "positional arguments to the function '"+str(method_name)+
            "' whereas only "+str(len(free_positionalNames))+" are req"+
            "uired. To check, read the list of positional arguments ah"+
            "ead: "+str(free_positionalNames))

        # Checks if the key arguments are really keys

        for key in key_arguments.keys():

            if not (key in free_keywordNames):

                raise KeyError("The key '"+str(key)+"' is not keyword "+
                "argument of the function '"+str(method_name)+"'. The "+
                "following is a list of the keyword arguments of this "+
                "method: "+str(free_keywordNames))

        return method_function(**{free_positionalNames[i]: arguments[i
        ] for i in range(len(free_positionalNames))}, **key_arguments, 
        **functions_fixedArguments)
    
    return driver_function

########################################################################
#                                 Tests                                #
########################################################################

"""if __name__=="__main__":

    import time

    for i in progress_bar(range(10), message="Test"):

        time.sleep(0.5)"""