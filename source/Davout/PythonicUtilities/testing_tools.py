# Routine to store tools for testing other routines and functionalities

import time

import tensorflow as tf

import datetime

import os

import psutil

import resource

import numpy as np

import gc

import traceback

from ..PythonicUtilities.programming_tools import get_attribute

from ..PythonicUtilities.programming_tools import TerminalColor

from ..PythonicUtilities.string_tools import float_to_scientific_notation

from ..StochasticUtilities.tool_box.statistics_tools import get_confidence_radius_student

########################################################################
#                            Testing tools                             #
########################################################################

# Defines a function to perform a series of tests defined as methods in-
# side a class

def run_class_of_tests(class_of_tests, reserved_methods=None,
sort_methods_alphabetically=True):

    """
    Function to test all methods defined in a class except the __init__
    method and those especially reserved.
    
    class_of_tests: python class with functions to be tested defined as
    methods in the class
    
    reserved_methods: list of strings, where each string is the name of
    methods that are not to be tested
    
    sort_methods_alphabetically: flag to tell if the methods of the class
    must be run using alphabetical order"""

    # Instantiates the class of colors in the terminal

    terminal_color = TerminalColor()

    # Gets a dictionary of the methods inside this instance except for
    # the __init__ method

    methods_dictionary = get_attribute(class_of_tests, None, None, 
    dictionary_of_methods=True, delete_init_key=True, reserved_methods=
    reserved_methods, sort_methods_alphabetically=
    sort_methods_alphabetically)

    # Initializes the success and failure counters

    success_counter = 0

    failure_counter = 0

    # Initializes a list to plot information per method

    log_per_method = []

    # Iterates through the methods 

    initial_overall_time = time.time()

    for name, method in methods_dictionary.items():

        print("\nRuns method '"+str(name)+"'\n")

        flag_sucess = False

        initial_method_time = time.time()

        try:

            method()

            success_counter += 1

            flag_sucess = True

        except Exception as e:

            print("Method '"+str(name)+"' failed:\n"+str(e)+"\n")

            traceback.print_exc()

            failure_counter += 1

        # Gets the time

        method_time = time.time()-initial_method_time

        # Appends the necessary information

        log_per_method.append([name, flag_sucess, method_time])

        print("\n#####################################################"+
        "###################\nMethod '"+str(name)+" took "+str(time.time(
        )-initial_method_time)+" seconds\n############################"+
        "############################################\n")

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
            "and took "+terminal_color(method_log[2], "yellow")+" se"+
            "conds")

    print("\nThe whole testing operation took "+str(time.time()-
    initial_overall_time)+" seconds.")

########################################################################
#                           Performance tools                          #
########################################################################

# Defines a function to take an argument-less function as an object and
# to run it to evaluate performance, both in computational time and me-
# mory footprint

def evaluate_function_performance(function_object, n_warm_up_runs=1,
n_evaluation_runs=10, confidence_level=0.95, 
n_evaluations_to_show_memory_data=None, ten_notation=" E"):

    # Iterates over the warm-up runs to get all possible graphs compiled

    print("Starts the "+str(n_warm_up_runs)+" warm-up runs\n")

    start_warm_up_time = time.perf_counter()

    for _ in range(n_warm_up_runs):

        function_object()

    warm_up_time = time.perf_counter()-start_warm_up_time

    current_computer_time = datetime.datetime.now()

    # Estimates the time the evaluations will be completed

    estimated_time = (n_evaluation_runs/n_warm_up_runs)*warm_up_time

    completion_time = current_computer_time+datetime.timedelta(seconds=
    estimated_time)

    print("The "+str(n_warm_up_runs)+" warm-up runs took "+str(
    warm_up_time)+" seconds. Thus, it is estimated\nthat the "+str(
    n_evaluation_runs)+" will took "+str(estimated_time)+" seconds\nNo"+
    "w, the computer strikes its clock at "+str(
    current_computer_time.strftime('%H:%M:%S'))+";\nthe evaluations ar"+
    "e estimated to be completed at "+str(completion_time.strftime('%H'+
    ':%M:%S'))+"\n")

    # Collects all memory garbage to avoid any information leakage that
    # might affect future computations

    gc.collect()

    # Initializes the object that tracks information of the hardware du-
    # ring processes

    process_data = psutil.Process(os.getpid())

    # Initializes a list of time intervals for each run

    time_intervals_list = []

    # Initializes lists to measure memory usage before and after each e-
    # valuation of the function, and also the peak memory usage

    memory_usage_before_evaluation = []

    memory_usage_after_evaluation = []

    memory_peak_during_evaluation = []

    # Evaluates the running time

    start_general_time = time.perf_counter()

    print("Starts evaluating the time interval for each run of a total"+
    " of "+str(n_evaluation_runs)+" runs\n")

    for _ in range(n_evaluation_runs):

        # Cleans the garbage in memory and records the memory usage be-
        # fore evaluation. Converts to MB

        gc.collect()

        memory_usage_before_evaluation.append(process_data.memory_info(
        ).rss/(1024**2))

        # Starts to count this step's time interval

        start_interval = time.perf_counter()

        # Runs the function

        return_object = function_object()

        # Verifies if it is a tensorflow tensor and materializes is to
        # force synchronization

        if tf.is_tensor(return_object):

            return_object.numpy()

        # Records the memory usage after the function evaluation

        memory_usage_after_evaluation.append(process_data.memory_info(
        ).rss/(1024**2))

        # Records the peak memory. The conversion to MB is different
        # since Linux records peak memory in KiB

        memory_peak_during_evaluation.append(resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss/(1000**2)*1024)

        # Gets this time interval

        time_intervals_list.append(time.perf_counter()-start_interval)

    # Gets the final general time

    general_time_interval = time.perf_counter()-start_general_time

    # Converts the list of time intervals to numpy array and computes 
    # its average and standard deviation

    time_intervals_list = np.asarray(time_intervals_list)

    average_time_interval = np.mean(time_intervals_list)

    standard_deviation_time_interval = np.std(time_intervals_list)

    # Gets the confidence radius using Student's t distribution

    confidence_radius = get_confidence_radius_student(n_evaluation_runs,
    standard_deviation_time_interval, confidence_level=confidence_level)

    function_name = "without name"

    if hasattr(function_object, "__name__"):

        function_name = str(function_object.__name__)

    print("\n############################# Running time ##############"+
    "###############\n")   

    print("Function '"+function_name+"' was evaluated for "+str(
    n_warm_up_runs)+" warm-up runs and\nfor "+str(n_evaluation_runs)+
    " evaluation runs.\n\nThe average time interval for each evaluatio"+
    "n is...........................: "+str(average_time_interval)+"\n"+
    "The standard deviation is........................................"+
    "..........: "+str(standard_deviation_time_interval)+"\nThe averag"+
    "e time considering an interrupted time measurement across runs is"+
    ": "+str(general_time_interval/n_evaluation_runs)+"\n\nUsing the S"+
    "tudent's t distribution and a confidence level of "+str(
    confidence_level)+", the time\ninterval per iteration, T, is......"+
    "..........................................: "+str(
    average_time_interval-confidence_radius)+" <= t <= "+str(
    average_time_interval+confidence_radius)+"\n")

    # Presents the result of memory usage

    string_result = ""

    # Verifies if a maximum number of evaluations was given to show me-
    # mory data

    if n_evaluations_to_show_memory_data is None:

        # If no maximum number of evaluations was given, show all of them

        n_evaluations_to_show_memory_data = len(
        memory_usage_before_evaluation)

    else:

        n_evaluations_to_show_memory_data = min(n_evaluation_runs,
        n_evaluations_to_show_memory_data)

    # Iterates over the evaluation runs

    for evaluation_number in range(n_evaluations_to_show_memory_data):

        memory_before = memory_usage_before_evaluation[evaluation_number]

        memory_after = memory_usage_after_evaluation[evaluation_number]

        memory_peak = memory_peak_during_evaluation[evaluation_number]

        string_result += ("\n         "+float_to_scientific_notation(
        memory_before, decimal_places=5, ten_notation=ten_notation, 
        n_digits_for_power_of_ten=2)+" MB        |          "+
        float_to_scientific_notation(memory_after, decimal_places=5, 
        ten_notation=ten_notation, n_digits_for_power_of_ten=2)+" MB  "+
        "      |         "+float_to_scientific_notation(memory_peak, 
        decimal_places=5, ten_notation=ten_notation, 
        n_digits_for_power_of_ten=2)+" MB")

    print("\n############################# Memory usage ##############"+
    "###############\n")   

    print("Function '"+function_name+"' was evaluated for "+str(
    n_warm_up_runs)+" warm-up runs and\nfor "+str(n_evaluation_runs)+
    " evaluation runs.\n\n  Memory before evaluation    |    Memory af"+
    "ter evaluation    | Peak memory during evaluation"+string_result+
    "\n\n    Average memory before     |      Average memory after    "+
    " |    Average peak memory during\n         "+
    float_to_scientific_notation(np.mean(np.asarray(
    memory_usage_before_evaluation)), decimal_places=5, ten_notation=
    ten_notation, n_digits_for_power_of_ten=2)+" MB        |          "+
    float_to_scientific_notation(np.mean(np.asarray(
    memory_usage_after_evaluation)), decimal_places=5, ten_notation=
    ten_notation, n_digits_for_power_of_ten=2)+" MB        |         "+
    float_to_scientific_notation(np.mean(np.asarray(
    memory_peak_during_evaluation)), decimal_places=5, ten_notation=
    ten_notation, n_digits_for_power_of_ten=2)+" MB")    

    # Collects all memory garbage to avoid any information leakage that
    # might affect future computations

    gc.collect()