# Routine to store methods to work with importing other modules

import pkgutil

import importlib

import inspect

# Defines a function to import classes from all modules in a package.
# Modules are the .py file, whereas package is a directory with a 
# __init__ file in it

def load_classes_from_package(package, necessary_attributes=None,
classes_list=None):

    # If no necessary attributes are asked, creates an empty list

    if necessary_attributes is None:

        necessary_attributes = []
    
    # Initializes a list of classes

    if classes_list is None:

        classes_list = []

    # Iterates through the modules inside the package

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):

        # Imports the module from the package using its name

        module = importlib.import_module(str(package.__name__)+"."+str(
        module_name))

        # Iterates through the class objects inside this module

        for _, class_object in inspect.getmembers(module, inspect.isclass):

            # Only keeps classes defined in this module (not imports)

            if class_object.__module__ == module.__name__:

                # Verifies if the class has the necessary attributes (be-
                # fore instantiation)

                flag_has_attributes = True 

                for necessary_attribute in necessary_attributes:

                    if not hasattr(class_object, necessary_attribute):

                        flag_has_attributes = False 

                        break 

                if flag_has_attributes:

                    classes_list.append(class_object)

    return classes_list