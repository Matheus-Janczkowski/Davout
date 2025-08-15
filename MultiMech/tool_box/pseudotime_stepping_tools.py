# Routine to store methods for pseudotime stepping

from dolfin import *

import numpy as np

import time

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import post_processing_tools

from ..tool_box import programming_tools

from ..post_processes import post_processes_classes as post_classes

from ..tool_box import functional_tools

########################################################################
#                        Newton-Raphson schemes                        #
########################################################################

# Defines a function to iterate through a Newton-Raphson loop of a vari-
# ational problem of a single field

@programming_tools.optional_argumentsInitializer({'post_processesDict': 
lambda: dict(), 'post_processesSubmeshDict': lambda: dict(), 
'dirichlet_loads': lambda: [], 'neumann_loads': lambda: [], ('solution'+
'_name'): lambda: ["solution", "DNS"], 'volume_physGroupsSubmesh': 
lambda: [], 'macro_quantitiesClasses': lambda: []})

def newton_raphsonSingleField(solver, solution_field, fields_namesDict,
mesh_dataClass, constitutive_model, post_processesDict=None, 
post_processesSubmeshDict=None, dirichlet_loads=None, neumann_loads=None, 
solution_name=None, volume_physGroupsSubmesh=None, 
macro_quantitiesClasses=None, t=None, t_final=None, maximum_loadingSteps=
None, field_correction=None):
    
    print("\n#########################################################"+
    "###############\n#              The Newton-Raphson scheme will be"+
    " initiated             #\n#######################################"+
    "#################################\n")

    print("There are "+str(len(solution_field.vector()))+" degrees of "+
    "freedom in the mesh\n")

    # Verifies if the classes of macroscale quantities are indeed clas-
    # ses

    time_keys = []

    for MacroScaleClass in macro_quantitiesClasses:

        if not hasattr(MacroScaleClass, '__class__'):

            raise TypeError("The objects in 'macro_quantitiesClasses' "+
            "must be classes")
        
        # Gets the time keys for these macro quantities

        if len(time_keys)==0:

            time_keys = MacroScaleClass.time_keys

        else:

            # Verifies if all classes have the same time keys

            functional_tools.test_timeKeysConsistency(time_keys, 
            MacroScaleClass.time_keys, variable_name=type(
            MacroScaleClass).__name__)

    # Constructs the class of code-provided information for the post-
    # processes

    context_class = post_classes.PostProcessContext(
    solution_field.function_space().mesh(), 
    constitutive_model, mesh_dataClass.dx, mesh_dataClass.x,
    mesh_dataClass.domain_physicalGroupsNameToTag, mesh_dataClass.ds,
    mesh_dataClass.boundary_physicalGroupsNameToTag, mesh_dataClass.n)

    # Transforms the dictionary of post-processing methods instructions
    # into a live-wire dictionary with the proper methods and needed in-
    # formation
    
    post_processes = post_processing_tools.post_processingSelectionSingleField(
    post_processesDict, context_class) 
    
    # Initializes the dictionary of submesh post processes

    post_processesSubmesh = dict()
    
    # Verifies if the physical groups for the submesh is an integer

    if (isinstance(volume_physGroupsSubmesh, int) or isinstance(
    volume_physGroupsSubmesh, str)):

        # Transforms into a list

        volume_physGroupsSubmesh = [volume_physGroupsSubmesh]
    
    # If there are volumetric physical groups to build a submesh

    solution_submesh = None

    if len(volume_physGroupsSubmesh)>0:

        # Gets the function space of the solution field

        function_space = solution_field.function_space()

        (RVE_submesh, domain_meshFunction, function_spaceSubmesh,
        RVE_meshMapping, parent_meshMapping, solution_submesh, 
        RVE_toParentCellMap, dx_submesh, x_submesh) = mesh_tools.create_submesh(
        mesh_dataClass.domain_meshCollection, 
        mesh_dataClass.domain_meshFunction, volume_physGroupsSubmesh, 
        function_space, domain_physicalGroupsNameToTag=
        mesh_dataClass.domain_physicalGroupsNameToTag)

        # Constructs the class of code-provided information for the post-
        # processes

        context_classRVE = post_classes.PostProcessContext(RVE_submesh, 
        constitutive_model, dx_submesh, x_submesh, 
        mesh_dataClass.domain_physicalGroupsNameToTag, mesh_dataClass.ds,
        mesh_dataClass.boundary_physicalGroupsNameToTag, 
        mesh_dataClass.n)

        # Initializes the post process for the submesh if there's any

        post_processesSubmesh = post_processing_tools.post_processingSelectionSingleField(
        post_processesSubmeshDict, context_classRVE) 
    
    # Initializes a dictionary of post processes objects, files for e-
    # xample

    post_processingObjects = dict()

    for post_processName, post_process in post_processes.items():

        post_processingObjects[post_processName] = (
        post_process.initialization_function(
        post_process.additional_information, 
        post_process.code_providedInfo, False))

    # Initializes a dictionary of post processes objects for the submesh, 
    # files for example, for each field

    post_processingObjectsSubmesh = dict()

    for post_processName, post_process in post_processesSubmesh.items():

        post_processingObjectsSubmesh[post_processName] = (
        post_process.initialization_function(
        post_process.additional_information, 
        post_process.code_providedInfo, True))
    
    # Verifies if there are no loads

    if len(dirichlet_loads)==0 and len(neumann_loads)==0:

        print("\nWARNING: there are no Dirichlet boundary conditions n"+
        "or Neumann boundary conditions\n")

    # Initializes the pseudotime counter

    time_counter = 0

    correction_projection = None

    intermediate_field = None

    if not (field_correction is None):

        correction_projection = Function(field_correction[2])

    # Checks if the time keys of the macro quantities for multiscale a-
    # nalysis are empty. If so, creates a range of time points

    if len(time_keys)==0:

        # Verifies if initial time and final time have been supplied

        if t is None:

            raise ValueError("The initial time value for the pseudotim"+
            "e stepping algorithm was not given, even though no macro "+
            "quantities with their respective time points were supplie"+
            "d")

        if t_final is None:

            raise ValueError("The final time value for the pseudotime "+
            "stepping algorithm was not given, even though no macro qu"+
            "antities with their respective time points were supplied")

        if maximum_loadingSteps is None:

            raise ValueError("The maximum number of loading steps for "
            "the pseudotime stepping algorithm was not given, even tho"+
            "ugh no macro quantities with their respective time points"+
            " were supplied")

        time_keys = np.linspace(t, t_final, maximum_loadingSteps)

    # Iterates through the pseudotime stepping

    for t in time_keys:

        # Updates the pseudo time variables and the counter
        
        time_counter += 1

        # Prints step information

        print_stepInfo(time_counter, t)

        # Updates the Dirichlet boundary conditions 

        for dirichlet_load in dirichlet_loads:

            # Tests whether this load is a dolfin Constant

            if isinstance(dirichlet_load, Constant):

                dirichlet_load.assign(t)

            # If the load is a class and has an attribute "update"

            elif hasattr(dirichlet_load, "update_load"):

                dirichlet_load.update_load(t)

            # Otherwise, updates it as a class (Expressions are classes)

            elif hasattr(dirichlet_load, "t"):

                dirichlet_load.t = t

            else:

                raise AttributeError("Cannot update the dirichlet load"+
                " because the class '"+str(dirichlet_load)+"' does not"+
                " have the attribute 't'")

        # Updates the Neumann boundary conditions 

        for neumann_load in neumann_loads:

            # Tests whether this load is a dolfin Constant

            if isinstance(neumann_load, Constant):

                neumann_load.assign(t)

            # If the load is a class and has an attribute "update"

            elif hasattr(neumann_load, "update_load"):

                neumann_load.update_load(t)

            # Otherwise, updates it as a class (Expressions are classes)

            elif hasattr(neumann_load, "t"):

                neumann_load.t = t

            else:

                raise AttributeError("Cannot update the neumann load b"+
                "ecause the class '"+str(neumann_load)+"' does not hav"+
                "e the attribute 't'")

        # Updates the classes of macroscale quantities

        for MacroScaleClass in macro_quantitiesClasses:

            MacroScaleClass.update(t)

        # Solves the nonlinear variational problem 

        start_time = time.time()

        solver.solve()

        end_time = time.time()

        print("The solution of this pseudotime took "+str(end_time-
        start_time)+" seconds\n\n")

        # If the field has to be corrected, like in multiscale analysis, 
        # using a correction field

        if not (field_correction is None):

            # Interpolates the correction of the field by the given
            # function space, and, then, adds the resulting vector of
            # parameters to the solution's one

            correction_projection.interpolate(field_correction[1])

            # Takes care if the solution field has the same number of 
            # degrees of freedom as the correction does

            if len(solution_field.vector())!=len(
            correction_projection.vector()):
                
                # If there's a difference, an intermediate field has to
                # be created in the function space of the correction 
                
                if intermediate_field is None:

                    intermediate_field = Function(field_correction[2])

                # In this space, the solution will be projected and the
                # interpolated correction will be added
                
                intermediate_field.vector()[:] = (project(solution_field, 
                field_correction[2]).vector()[:]+
                correction_projection.vector()[:])

            else:

                solution_field.vector()[:] += (
                correction_projection.vector()[:])

        # Renames the solution

        if len(solution_name)>0:

            if intermediate_field is None:

                solution_field.rename(*solution_name)

            else:

                intermediate_field.rename(*solution_name)

        # Updates the post processes objects

        for post_processName, post_process in post_processes.items():

            # Sets the field number as -1, for this problem has a single
            # field only. It must be -1 in this case

            if intermediate_field is None:

                post_processingObjects[post_processName] = (
                post_process.update_function(post_processingObjects[
                post_processName], solution_field, -1, t, 
                fields_namesDict))

            else:

                post_processingObjects[post_processName] = (
                post_process.update_function(post_processingObjects[
                post_processName], intermediate_field, -1, t, 
                fields_namesDict))

        # If a submesh is to be populated with part of the solution

        if len(volume_physGroupsSubmesh)>0 and (len(list(
        post_processesSubmesh.keys()))>0):

            if intermediate_field is None:

                solution_submesh = mesh_tools.field_parentToSubmesh(
                RVE_submesh, solution_field, RVE_toParentCellMap, 
                sub_meshMapping=RVE_meshMapping, parent_meshMapping=
                parent_meshMapping, field_submesh=solution_submesh)

            else:

                solution_submesh = mesh_tools.field_parentToSubmesh(
                RVE_submesh, intermediate_field, RVE_toParentCellMap, 
                sub_meshMapping=RVE_meshMapping, parent_meshMapping=
                parent_meshMapping, field_submesh=solution_submesh)

            if len(solution_name)>0:

                solution_submesh.rename(*solution_name)

            # Updates the post processes objects

            for post_processName, post_process in post_processesSubmesh.items():

                # Verifies if this post-process has been evaluated in 
                # parent mesh

                shared_result = False

                if post_processName in post_processes.keys():

                    # A process to be transitioned from a parent mesh to
                    # a submesh must have the variable 
                    # 'parent_toChildMeshResult' in its output class

                    if hasattr(post_processingObjects[post_processName], 
                    'parent_toChildMeshResult'):

                        # Access the result variable of the post-process 
                        # class to directly allocate the result from the 
                        # parent mesh

                        post_processingObjectsSubmesh[post_processName
                        ].parent_toChildMeshResult = mesh_tools.field_parentToSubmesh(
                        RVE_submesh, post_processingObjects[
                        post_processName].parent_toChildMeshResult, 
                        RVE_toParentCellMap)

                        # Updates the process to save whatever informa-
                        # tion was transfered. Sets the field number as 
                        # -1, for this problem has a single field only. 
                        # It must be -1 in this case

                        post_processingObjectsSubmesh[post_processName
                        ] = post_process.update_function(
                        post_processingObjectsSubmesh[post_processName], 
                        solution_submesh, -1, t, fields_namesDict,
                        flag_parentMeshReuse=True)

                        # Updates the flag to inform this process has
                        # been taken from the parent mesh

                        shared_result = True

                # If this post process has not been found on the parent
                # mesh or it is not a sharable process, evaluates it in
                # the submesh

                if not shared_result:

                    # Sets the field number as -1, for this problem has a 
                    # single field only. It must be -1 in this case

                    post_processingObjectsSubmesh[post_processName] = (
                    post_process.update_function(
                    post_processingObjectsSubmesh[post_processName], 
                    solution_submesh, -1, t, fields_namesDict))

        end_postProcessingTime = time.time()

        print("\n\nThe post-processing phase took "+str(
        end_postProcessingTime-end_time)+" seconds\n\n")

# Defines a function to iterate through a Newton-Raphson loop of a vari-
# ational problem of multiple fields

@programming_tools.optional_argumentsInitializer({'post_processesList': 
lambda: [], 'post_processesSubmeshList': lambda: [], 'dirichlet_loads': 
lambda: [], 'neumann_loads': lambda: [], 'solution_name': lambda: ["so"+
"lution", "DNS"], 'volume_physGroupsSubmesh': lambda: [], ('macro_quan'+
'titiesClasses'): lambda: [], 'fields_corrections': lambda: dict()})

def newton_raphsonMultipleFields(solver, solution_field, 
fields_namesDict, mesh_dataClass, constitutive_model, post_processesList=
None, post_processesSubmeshList=None, dirichlet_loads=None, 
neumann_loads=None, solution_name=None, volume_physGroupsSubmesh=None, 
macro_quantitiesClasses=None, t=None, t_final=None, maximum_loadingSteps=
None, fields_corrections=None):

    # Verifies if the classes of macroscale quantities are indeed ins-
    # tances of some class

    time_keys = []

    for MacroScaleClass in macro_quantitiesClasses:

        if not hasattr(MacroScaleClass, '__class__'):

            raise TypeError("The objects in 'macro_quantitiesClasses' "+
            "must be classes")
        
        # Gets the time keys for these macro quantities

        if len(time_keys)==0:

            time_keys = MacroScaleClass.time_keys

        else:

            # Verifies if all classes have the same time keys

            functional_tools.test_timeKeysConsistency(time_keys, 
            MacroScaleClass.time_keys, variable_name=type(
            MacroScaleClass).__name__)
    
    # Gets the number of fields in the mixed element. But, verifies if
    # the solution is really a mixed solution space

    n_fields = 1

    mixed_element = solution_field.function_space().ufl_element()

    if mixed_element.family()=="Mixed":

        n_fields = mixed_element.num_sub_elements()

    # Otherwise, the solution is indeed composed by a single field and
    # the single field pseudotime stepping function must be called

    else:

        # Treats the lists of post processes to convert them to dictio-
        # naries (the format used in the single field pseudotime step-
        # ping function)

        if isinstance(post_processesList, list):

            # Takes the first component

            if len(post_processesList)>0:

                if len(post_processesList[0])>0:

                    post_processesList = post_processesList[0][1]

                else:

                    post_processesList = dict()

            else:

                post_processesList = dict()

        if isinstance(post_processesSubmeshList, list):

            # Takes the first component

            if len(post_processesSubmeshList)>0:

                if len(post_processesSubmeshList[0])>0:

                    post_processesSubmeshList = post_processesSubmeshList[
                    0][1]

                else:

                    post_processesSubmeshList = dict()

            else:

                post_processesSubmeshList = dict()

        # Updates the solution name for one field

        if len(solution_name)>0:

            if isinstance(solution_name[0], list):

                solution_name = solution_name[0]

            else:

                solution_name = ["field", "Microscale"]

        else:

            solution_name = ["field", "Microscale"]

        # Updates the field corrections for the case of a single field

        fields_corrections = list(fields_corrections.values())

        if len(fields_corrections)>0:

            fields_corrections = fields_corrections[0]

        # Calls the appropriate function to iterate in a single-field 
        # problem

        return newton_raphsonSingleField(solver, solution_field, 
        fields_namesDict, mesh_dataClass, constitutive_model, 
        post_processesDict=post_processesList, post_processesSubmeshDict=
        post_processesSubmeshList, dirichlet_loads=dirichlet_loads, 
        neumann_loads=neumann_loads, solution_name=solution_name, 
        volume_physGroupsSubmesh=volume_physGroupsSubmesh, 
        macro_quantitiesClasses=macro_quantitiesClasses, t=t, t_final=
        t_final, maximum_loadingSteps=maximum_loadingSteps,
        field_correction=fields_corrections)
    
    print("\n#########################################################"+
    "###############\n#              The Newton-Raphson scheme will be"+
    " initiated             #\n#######################################"+
    "#################################\n")

    # Splits the solution to show how many DOFs are at each field

    split_solution = list(solution_field.split(deepcopy=True))

    for i in range(n_fields):

        print("There are "+str(len(split_solution[i].vector()))+" degrees "+
        "of freedom in the "+str(i+1)+"-th field of the mesh\n")

    # Constructs the class of code-provided information for the post-
    # processes

    context_class = post_classes.PostProcessContext(
    solution_field.function_space().mesh(), constitutive_model, 
    mesh_dataClass.dx, mesh_dataClass.x, 
    mesh_dataClass.domain_physicalGroupsNameToTag, mesh_dataClass.ds, 
    mesh_dataClass.boundary_physicalGroupsNameToTag, mesh_dataClass.n)

    # Verifies if the post processes is a list

    if not isinstance(post_processesList, list):

        raise ValueError("post_processesList must be a list of diction"+
        "aries in newton_raphsonMultipleFields, because there must be "+
        "post-processing steps for each field.")

    # Verifies if the post processes for the submesh is a list

    if not isinstance(post_processesSubmeshList, list):

        raise ValueError("post_processesSubmeshList must be a list of "+
        "dictionaries in newton_raphsonMultipleFields, because there m"+
        "ust be post-processing steps for each field.")

    # Transforms the list of dictionaries of post-processing methods 
    # instructions into a list of live-wire dictionaries with the proper 
    # methods and needed information. A dictionary is created for the 
    # post-processing methods that work on a single field
    
    (post_processes, post_processesNamesList
    ) = post_processing_tools.post_processingSelectionMultipleFields(
    post_processesList, context_class, fields_namesDict) 
    
    # Initializes the list of submesh post processes. It is a list, be-
    # cause each field will ocupy a component

    post_processesSubmesh = []
    
    # Verifies if the physical groups for the submesh is an integer

    if (isinstance(volume_physGroupsSubmesh, int) or isinstance(
    volume_physGroupsSubmesh, str)):

        # Transforms into a list

        volume_physGroupsSubmesh = [volume_physGroupsSubmesh]
    
    # If there are volumetric physical groups to build a submesh

    if len(volume_physGroupsSubmesh)>0:

        function_space = solution_field.function_space()

        (RVE_submesh, domain_meshFunction, function_spaceSubmesh, 
        RVE_meshMapping, parent_meshMapping, solution_submesh, 
        RVE_toParentCellMap, dx_submesh, x_submesh) = mesh_tools.create_submesh(
        mesh_dataClass.domain_meshCollection, 
        mesh_dataClass.domain_meshFunction, volume_physGroupsSubmesh, 
        function_space, domain_physicalGroupsNameToTag=
        mesh_dataClass.domain_physicalGroupsNameToTag)

        # Constructs the class of code-provided information for the post-
        # processes

        context_classRVE = post_classes.PostProcessContext(RVE_submesh, 
        constitutive_model, dx_submesh, x_submesh,
        mesh_dataClass.domain_physicalGroupsNameToTag, mesh_dataClass.ds,
        mesh_dataClass.boundary_physicalGroupsNameToTag, 
        mesh_dataClass.n)

        # Initializes the post process for the submesh if there's any

        post_processesSubmesh, *_ = post_processing_tools.post_processingSelectionMultipleFields(
        post_processesSubmeshList, context_classRVE, fields_namesDict) 
    
    # Initializes a dictionary of post processes objects, files for e-
    # xample, for each field

    post_processingObjects = []

    for i in range(len(post_processes)):

        post_processingObjects.append(dict())

        for post_processName, post_process in post_processes[i].items():

            if post_processName!="field number":

                post_processingObjects[-1][post_processName] = (
                post_process.initialization_function(
                post_process.additional_information, 
                post_process.code_providedInfo, False))

    # Makes the same thing for the submesh post-processes

    if not isinstance(post_processesSubmesh, list):

        raise ValueError("post_processesSubmesh must be a list of dict"+
        "ionaries in newton_raphsonMultipleFields, because there must "+
        "be post-processing steps for each field.")
    
    # Initializes a dictionary of post processes objects, files for e-
    # xample, for each field

    post_processingObjectsSubmesh = []

    for i in range(len(post_processesSubmesh)):

        post_processingObjectsSubmesh.append(dict())

        for post_processName, post_process in post_processesSubmesh[i
        ].items():

            if post_processName!="field number":

                post_processingObjectsSubmesh[-1][post_processName] = (
                post_process.initialization_function(
                post_process.additional_information, 
                post_process.code_providedInfo, True))
    
    # Verifies if there are no loads

    if len(dirichlet_loads)==0 and len(neumann_loads)==0:

        print("\nWARNING: there are no Dirichlet boundary conditions n"+
        "or Neumann boundary conditions\n")

    # Initializes the pseudotime counter

    time_counter = 0

    # Checks if the time keys of the macro quantities for multiscale a-
    # nalysis are empty. If so, creates a range of time points

    if len(time_keys)==0:

        # Verifies if initial time and final time have been supplied

        if t is None:

            raise ValueError("The initial time value for the pseudotim"+
            "e stepping algorithm was not given, even though no macro "+
            "quantities with their respective time points were supplie"+
            "d")

        if t_final is None:

            raise ValueError("The final time value for the pseudotime "+
            "stepping algorithm was not given, even though no macro qu"+
            "antities with their respective time points were supplied")

        if maximum_loadingSteps is None:

            raise ValueError("The maximum number of loading steps for "
            "the pseudotime stepping algorithm was not given, even tho"+
            "ugh no macro quantities with their respective time points"+
            " were supplied")

        time_keys = np.linspace(t, t_final, maximum_loadingSteps)

    # Iterates through the pseudotime stepping

    for t in time_keys:

        # Updates the pseudo time variables and the counter
        
        time_counter += 1

        # Prints step information

        print_stepInfo(time_counter, t)

        # Updates the Dirichlet boundary conditions 

        for dirichlet_load in dirichlet_loads:

            # Tests whether this load is a dolfin Constant

            if isinstance(dirichlet_load, Constant):

                dirichlet_load.assign(t)

            # If the load is a class and has an attribute "update"

            elif hasattr(dirichlet_load, "update_load"):

                dirichlet_load.update_load(t)

            # Otherwise, updates it as a class (Expressions are classes)

            elif hasattr(dirichlet_load, "t"):

                dirichlet_load.t = t

            else:

                raise AttributeError("Cannot update the dirichlet load"+
                " because the class '"+str(dirichlet_load)+"' does not"+
                " have the attribute 't'")

        # Updates the Neumann boundary conditions 

        for neumann_load in neumann_loads:

            # Tests whether this load is a dolfin Constant

            if isinstance(neumann_load, Constant):

                neumann_load.assign(t)

            # If the load is a class and has an attribute "update"

            elif hasattr(neumann_load, "update_load"):

                neumann_load.update_load(t)

            # Otherwise, updates it as a class (Expressions are classes)

            elif hasattr(neumann_load, "t"):

                neumann_load.t = t

            else:

                raise AttributeError("Cannot update the neumann load b"+
                "ecause the class '"+str(neumann_load)+"' does not hav"+
                "e the attribute 't'")

        # Updates the classes of macroscale quantities

        for MacroScaleClass in macro_quantitiesClasses:

            MacroScaleClass.update(t)

        # Solves the nonlinear variational problem 

        start_time = time.time()

        solver.solve()

        end_time = time.time()

        print("The solution of this pseudotime took "+str(end_time-
        start_time)+" seconds\n\n")

        # Splits the solution and appends each field to a list. Adds the
        # correction if needed (the correction is another field; linear
        # or quadratic, as examples)

        split_solution = list(solution_field.split(deepcopy=True))

        for field_name, field_correction in fields_corrections.items():

            field_index = 0

            try:

                field_index = fields_namesDict[field_name]

            except:

                raise KeyError("The field correction of the '"+str(
                field_name)+"' cannot be added to the solution for thi"+
                "s name was not found in the dictionary of fields' nam"+
                "es':\n"+str(list(fields_namesDict.keys())))

            # Interpolates the correction of the field by the given 
            # function space, and, then, adds the resulting vector of
            # parameters to the solution's one

            correction_projection = interpolate(field_correction[1], 
            field_correction[2])

            split_solution[field_index].vector()[:] += (
            correction_projection.vector()[:])

        # Renames the solution fields

        if len(solution_name)==n_fields:
        
            for i in range(n_fields):

                split_solution[i].rename(*solution_name[i])

        # Evaluates the post-processes

        for i in range(len(post_processes)):

            # Get the field number for this process

            field_number = post_processes[i]["field number"]

            # Updates the post processes objects

            for post_processName, post_process in post_processes[i
            ].items():
                
                if post_processName!="field number":

                    post_processingObjects[i][post_processName] = (
                    post_process.update_function(post_processingObjects[
                    i][post_processName], split_solution, field_number, 
                    t, fields_namesDict))

        # If a submesh is to be populated with part of the solution

        if len(volume_physGroupsSubmesh)>0 and (len(post_processesSubmesh
        )>0):

            solution_submesh = mesh_tools.field_parentToSubmesh(
            RVE_submesh, solution_field, RVE_toParentCellMap, 
            sub_meshMapping=RVE_meshMapping, parent_meshMapping=
            parent_meshMapping, field_submesh=solution_submesh)

            # Splits the solution

            split_solutionSubmesh = list(solution_submesh.split(deepcopy
            =True))

            # Post-process the submesh solution

            # Updates the name of the fields

            if len(solution_name)==n_fields:
                    
                for i in range(n_fields):

                    split_solutionSubmesh[i].rename(*solution_name[i])

            # Updates the post processes objects

            for i in range(len(post_processesSubmesh)):

                # Get the field number for this process

                field_number = post_processesSubmesh[i]["field number"]

                for post_processName, post_process in post_processesSubmesh[i
                ].items():
                    
                    if post_processName!="field number":

                        # Verifies if the pair of this post-process of
                        # this particular field has already been evalua-
                        # ted in the parent mesh

                        shared_result = False

                        pair_fieldName = [field_number, post_processName]

                        # Verifies if and where this pair is in the list
                        # of pairs field number and names of the parent 
                        # mesh

                        process_index = None 

                        for j in range(len(post_processesNamesList)):

                            if pair_fieldName in post_processesNamesList[
                            j]:

                                # Gets the index

                                process_index = j+0

                        if not (process_index is None):

                            # A process to be transitioned from a parent
                            # mesh to a submesh must have the variable
                            # 'parent_toChildMeshResult' in its output
                            # class

                            if hasattr(post_processingObjects[
                            process_index][post_processName], 'parent_'+
                            'toChildMeshResult'):

                                # Access the result variable of the post-
                                # process class to directly allocate the 
                                # result from the parent mesh

                                post_processingObjectsSubmesh[i][
                                post_processName
                                ].parent_toChildMeshResult = mesh_tools.field_parentToSubmesh(
                                RVE_submesh, post_processingObjects[
                                process_index][post_processName
                                ].parent_toChildMeshResult, 
                                RVE_toParentCellMap)

                                # Updates the process to save whatever 
                                # information was transfered

                                post_processingObjectsSubmesh[i][
                                post_processName] = post_process.update_function(
                                post_processingObjectsSubmesh[i][
                                post_processName], split_solutionSubmesh, 
                                field_number, t, fields_namesDict, 
                                flag_parentMeshReuse=True)

                                # Updates the flag to inform this process
                                # has been taken from the parent mesh

                                shared_result = True

                        # If it hasn't been evaluated in the parent mesh,
                        # evaluates it in the submesh

                        if not shared_result:

                            post_processingObjectsSubmesh[i][
                            post_processName] = post_process.update_function(
                            post_processingObjectsSubmesh[i][
                            post_processName], split_solutionSubmesh, 
                            field_number, t, fields_namesDict)

        end_postProcessingTime = time.time()

        print("\n\nThe post-processing phase took "+str(
        end_postProcessingTime-end_time)+" seconds\n\n")

########################################################################
#                              Utilities                               #
########################################################################

# Defines a function to print the stepping information

def print_stepInfo(step, time):

    # Transforms the time and the step into strings

    step = str(step)

    time = str(time)

    # Evaluates the number of characters 

    n_characters = len(step)+len(time)

    # Verifies if there is enough space for the whole step information

    if n_characters<=36:

        # Calculates the clearance to each side

        clearance_left = int(np.ceil(0.5*(36-n_characters)))

        clearance_right = 36-clearance_left-n_characters

        # Makes the clearance space

        clearance_spaceLeft = ""

        clearance_spaceRight = ""

        for i in range(clearance_left):

            clearance_spaceLeft += " "

        for i in range(clearance_right):

            clearance_spaceRight += " "

        print("\n#####################################################"+
        "###################\n#"+clearance_spaceLeft+"Incremental step"+
        ": "+step+"; current time: "+time+clearance_spaceRight+"#\n###"+
        "#############################################################"+
        "########\n")

    else:

        # Calculates the clearance to each side

        clearance_left = int(np.ceil(0.5*(52-len(step))))

        clearance_right = 52-clearance_left-len(step)

        # Makes the clearance space

        clearance_spaceLeft = ""

        clearance_spaceRight = ""

        for i in range(clearance_left):

            clearance_spaceLeft += " "

        for i in range(clearance_right):

            clearance_spaceRight += " "

        print("\n#####################################################"+
        "###################\n#"+clearance_spaceLeft+"Incremental step"+
        ": "+step+clearance_spaceRight+"#")

        # Calculates the clearance to each side

        clearance_left = int(np.ceil(0.5*(56-len(time))))

        clearance_right = 56-clearance_left-len(time)

        # Makes the clearance space

        clearance_spaceLeft = ""

        clearance_spaceRight = ""

        for i in range(clearance_left):

            clearance_spaceLeft += " "

        for i in range(clearance_right):

            clearance_spaceRight += " "

        print("#"+clearance_spaceLeft+"current time: "+time+
        clearance_spaceRight+"#\n#####################################"+
        "###################################\n")