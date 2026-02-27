# Routine to store tools for testing other routines and functionalities

from time import time

import traceback

from ..PythonicUtilities.programming_tools import get_attribute

from ..PythonicUtilities.programming_tools import TerminalColor

# Defines a function to perform a series of tests defined as methods in-
# side a class

def run_class_of_tests(class_of_tests):

    # Instantiates the class of colors in the terminal

    terminal_color = TerminalColor()

    # Gets a dictionary of the methods inside this instance except for
    # the __init__ method

    methods_dictionary = get_attribute(class_of_tests, None, None, 
    dictionary_of_methods=True, delete_init_key=True)

    # Initializes the success and failure counters

    success_counter = 0

    failure_counter = 0

    # Initializes a list to plot information per method

    log_per_method = []

    # Iterates through the methods 

    initial_overall_time = time()

    for name, method in methods_dictionary.items():

        print("\nRuns method '"+str(name)+"'\n")

        flag_sucess = False

        initial_method_time = time()

        try:

            method()

            success_counter += 1

            flag_sucess = True

        except Exception as e:

            print("Method '"+str(name)+"' failed:\n"+str(e)+"\n")

            traceback.print_exc()

            failure_counter += 1

        # Gets the time

        method_time = time()-initial_method_time

        # Appends the necessary information

        log_per_method.append([name, flag_sucess, method_time])

        print("\n#####################################################"+
        "###################\nMethod '"+str(name)+" took "+str(time()-
        initial_method_time)+" seconds\n##############################"+
        "##########################################\n")

    print("\n#########################################################"+
    "###############\n#                             Execution log     "+
    "                       #\n#######################################"+
    "#################################\n")

    print(str(success_counter)+" methods were successfully executed\n")

    print(str(failure_counter)+" methods failed to be executed")

    print(terminal_color("\nThe full log follows below:", "purple"))

    # Print the complete log

    for method_log in log_per_method:

        if method_log[1]:

            print("\n'"+terminal_color(method_log[0], "bold light blue"
            )+"' was "+terminal_color("SUCCESSFUL", "green")+" and too"+
            "k "+terminal_color(method_log[2], "yellow")+" seconds")

        else:

            print("\n'"+terminal_color(method_log[0], "bold light blue"
            )+"' was "+terminal_color("NOT SUCCESSFUL", "bold red")+" "+
            "and took "+terminal_color(method_log[2], "bold red")+" se"+
            "conds", "yellow")

    print("\nThe whole testing operation took "+str(time()-
    initial_overall_time)+" seconds.")