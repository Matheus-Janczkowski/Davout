# Routine to store tools for testing other routines and functionalities

from time import time

import traceback

from ..PythonicUtilities.programming_tools import get_attribute

# Defines a function to perform a series of tests defined as methods in-
# side a class

def run_class_of_tests(class_of_tests):

    # Gets a dictionary of the methods inside this instance except for
    # the __init__ method

    methods_dictionary = get_attribute(class_of_tests, None, None, 
    dictionary_of_methods=True, delete_init_key=True)

    # Initializes the success and failure counters

    success_counter = 0

    failure_counter = 0

    # Iterates through the methods 

    initial_overall_time = time()

    for name, method in methods_dictionary.items():

        print("\nRuns method '"+str(name)+"'\n")

        initial_method_time = time()

        try:

            method()

            success_counter += 1

        except Exception as e:

            print("Method '"+str(name)+"' failed:\n"+str(e)+"\n")

            traceback.print_exc()

            failure_counter += 1

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

    print("\nThe whole testing operation took "+str(time()-
    initial_overall_time)+" seconds.")