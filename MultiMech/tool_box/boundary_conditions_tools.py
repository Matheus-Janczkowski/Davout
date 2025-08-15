# Routine to store methods to neatly apply boundary conditions

from dolfin import *

from ..tool_box import programming_tools

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import numerical_tools

########################################################################
#              Heterogeneous Dirichlet boundary conditions             #
########################################################################

# Defines a function to apply prescribed boundary conditions to a pro-
# blem using an Expression as load prescriber

@programming_tools.optional_argumentsInitializer({'boundary_conditions':
lambda: [], 'dirichlet_loads': lambda: []})

def PrescribedDirichletBC(bc_informationsDict, 
field_functionSpace, mesh_dataClass, fields_namesDict, 
complex_bcsFunctionsDict, boundary_conditions=None, dirichlet_loads=None, 
t_initial=0.0, t_final=1.0, boundary_physicalGroups=None):
    
    # Sets the method arguments as the dictionary for arguments to the
    # more complex BC generators

    method_arguments = {"mesh_dataClass": mesh_dataClass, "boundary_ph"+
    "ysicalGroups": boundary_physicalGroups}
    
    # Verifies if the boundary physical groups is a dictionary or a list
    # if the physical group is given

    if isinstance(bc_informationsDict, list):

        if boundary_physicalGroups is None:

            raise ValueError("To apply PrescribedDirichletBC, the bc_i"+
            "nformationsDict variable can be a list if and only if the"+
            " boundary_physicalGroups is provided, but it's None")
        
        # Transforms in a dictionary
        
        bc_informationsDict[boundary_physicalGroups
        ] = bc_informationsDict

    # Verifies if the dictionary of boundary conditions' information has
    # physical groups as keys

    elif isinstance(bc_informationsDict, dict):

        for key in bc_informationsDict.keys():

            if isinstance(key, str):
                
                if not (key in mesh_dataClass.boundary_physicalGroupsNameToTag):

                    # This dictionary does not have keys as physical 
                    # groups, thus, makes this a value with the physical
                    # group as key

                    if boundary_physicalGroups is None:

                        raise ValueError("The bc_informationsDict is n"+
                        "ot a dictionary with physical groups as keys."+
                        " Nevertheless, the boundary_physicalGroups va"+
                        "riable is None, so no physical group informat"+
                        "ion can be retrieved")
                    
                    else:

                        bc_informationsDict = {boundary_physicalGroups:
                        bc_informationsDict}

                    break

    elif not isinstance(bc_informationsDict, dict):

        raise ValueError("The bc_informationsDict variable in the Pres"+
        "cribedDirichletBC method must be a dictionary, where the keys"+
        " are the physical groups of the boundary regions and values a"+
        "re the information to prescribe the field.")

    # If the physical groups variable is null, returns the empty list of
    # boundary conditions

    if len(list(bc_informationsDict.keys()))==0:

        if mesh_dataClass.verbose:

            print("Creates no prescribed boundary condition.\n")

        return boundary_conditions, dirichlet_loads
    
    # Tests if the element is mixed and gets the number of fields

    n_fields = 1

    element_type = field_functionSpace.ufl_element().family()

    if element_type=='Mixed':

        # Gets the number of fields

        n_fields = field_functionSpace.ufl_element().num_sub_elements()

    # Gets the default list of degrees of freedom to apply the prescribed
    # value to all 

    prescribed_dofs = []

    if n_fields==1:

        for i in range(field_functionSpace.ufl_element(
        ).num_sub_elements()):

            prescribed_dofs.append(i)

    else:

        counter = 0

        for j in range(n_fields):

            prescribed_dofs.append([])

            for i in range(field_functionSpace.sub(j).ufl_element(
            ).num_sub_elements()):

                prescribed_dofs[-1].append(counter)

                counter += 1

    # Iterates through the keys and values of the dictionary of prescri-
    # bed Dirichlet boundary conditions. The key is a physical group or
    # a tuple of physical groups, whereas the values are lists of: num-
    # ber of the field if there is more than one; degrees of freedom to
    # be prescribed; and the load expression

    for physical_groups, load_info in bc_informationsDict.items():

        # Verifies if load info is a dictionary and converts load info 
        # to the list format

        if isinstance(load_info, dict):

            new_loadInfo = []

            # Searches for the subfields to apply the boundary condition

            if "sub_fieldsToApplyBC" in load_info:

                new_loadInfo.append(load_info["sub_fieldsToApplyBC"])

            elif n_fields>1:

                raise KeyError("This simulation has "+str(n_fields)+" "+
                "fields. Thus, you have to provide the field you want "+
                "to prescribe Dirichlet boundary conditions to using t"+
                "he key 'sub_fieldsToApplyBC'")

            # Searches for the degrees of freedom to apply the boundary
            # condition

            if "degrees_ofFreedomList" in load_info:

                new_loadInfo.append(load_info["degrees_ofFreedomList"])

                if isinstance(new_loadInfo[-1], int):

                    new_loadInfo[-1] = [new_loadInfo[-1]]

                elif not isinstance(new_loadInfo[-1], list):

                    raise TypeError("The degrees of freedom to apply p"+
                    "rescribed Dirichlet boundary conditions must be a"+
                    "n integer or a list. The given information is nei"+
                    "ther, check it out: "+str(new_loadInfo[-1]))
                
            else:

                # Checks whether the degrees of freedom need to be spe-
                # cified if a complex boundary condition generator was 
                # not required nor a list of degrees of freedom

                if "load_function" in load_info:

                    if (not (load_info["load_function"] in complex_bcsFunctionsDict
                    )):
                        
                        if n_fields==1:

                            if len(prescribed_dofs)>0:

                                raise KeyError("To prescribe a boundar"+
                                "y condition in a field which is not a"+
                                " scalar and not setting a complex gen"+
                                "erator of boundary conditions, you ha"+
                                "ve to set a degree of freedom or a se"+
                                "t of them. Use the key 'degrees_ofFre"+
                                "edomList'")

                        else:

                            for subfield in load_info["sub_fieldsToApp"+
                            "lyBC"]:
                                
                                converted_subfield = convert_fieldsNamesToIndices(
                                subfield, fields_namesDict)

                                if len(prescribed_dofs[converted_subfield
                                ])>0:
                                    
                                    raise KeyError("To prescribe a bou"+
                                    "dary condition in a field which i"+
                                    "s not a scalar and not setting a "+
                                    "complex generator of boundary con"+
                                    "ditions, you have to set a degree"+
                                    " of freedom or a set of them. Use"+
                                    " the key 'degrees_ofFreedomList'")

            # Searches for the load function

            if "load_function" in load_info:

                new_loadInfo.append(load_info["load_function"])

            else:

                raise KeyError("The dictionary to inform how the presc"+
                "ribed load must be applied as boundary conditions has"+
                " no key 'load_function', thus, this boundary conditio"+
                "n cannot be applied. Check the dictionary you supplie"+
                "d: "+str(load_info))
            
            # Sets a dictionary for the keys that are not reserved and
            # sets this dictionary as a dictionary of additional parame-
            # ters

            additional_parameters = dict()

            reserved_keys = ["sub_fieldsToApplyBC", ("degrees_ofFreedo"+
            "mList"), "load_function"]

            # Iterates through the keys of the load_info

            for key, value in load_info.items():

                # Checks if any of the keys is one of the reserved keys

                if not (key in reserved_keys):

                    additional_parameters[key] = value

            # Checks if the load function is one of the complex ones 

            if new_loadInfo[-1] in complex_bcsFunctionsDict:

                # Gets the complex function working

                result = programming_tools.dispatch_functions(
                new_loadInfo[-1], None, fixed_inputVariablesDict=
                method_arguments, second_sourceFixedArguments=
                additional_parameters, methods_functionsDict=
                complex_bcsFunctionsDict, return_list=True, 
                return_singleFunction=True, all_argumentsFixed=True)[0]()

                # Verifies if the result is a tuple

                if isinstance(result, tuple):

                    new_loadInfo[-1] = result[0]

                    dirichlet_loads.append(result[1])

                else:

                    new_loadInfo[-1] = result

            # Otherwise, uses the generic loading curves

            elif numerical_tools.generate_loadingParametricCurves(
            new_loadInfo[-1], verify_curveNameExistence=True):

                # Gets the loading curve

                load_curve = numerical_tools.generate_loadingParametricCurves(
                new_loadInfo[-1], additional_parameters=
                additional_parameters)

                prescribed_BC = Constant(0.0)

                class LocalLoad:

                    def __init__(self):

                        self.time = Constant(t_initial)

                        self.maximum_time = Constant(t_final)

                        self.prescribed_BC = prescribed_BC

                    def update_load(self, t):

                        self.time.assign(t)

                        self.prescribed_BC.assign(load_curve(self.time/self.maximum_time))

                local_load = LocalLoad()

                new_loadInfo[-1] = prescribed_BC

                # Adds the time constant to the dirichlet_loads

                dirichlet_loads.append(local_load)

            else:

                # Tests whether this load is a dolfin Constant

                if isinstance(new_loadInfo[-1], Constant):

                    dirichlet_loads.append(new_loadInfo[-1])

                # If the load is a class and has an attribute "update"

                elif hasattr(new_loadInfo[-1], "update_load"):

                    dirichlet_loads.append(new_loadInfo[-1])

                # Otherwise, updates it as a class (Expressions are clas-
                # ses)

                elif hasattr(new_loadInfo[-1], "t"):

                    dirichlet_loads.append(new_loadInfo[-1])

                else:

                    raise AttributeError("The load prescribed to gener"+
                    "ate Dirichlet boundary condition is not a program"+
                    "-built generator nor a simple function. Besides i"+
                    "t, it has no attribute 't', nor 'update_load' met"+
                    "hod, neither is it a Constant. Thus, it cannot be"+
                    " appended to the list of dirichlet_loads to be up"+
                    "dated during time stepping. Check the given value"+
                    ": "+str(new_loadInfo[-1])+"\nCheck the list of co"+
                    "mplex boundary conditions generator: "+str(
                    complex_bcsFunctionsDict.keys()))
                
            # Makes the new_loadInfo the old one

            load_info = new_loadInfo

        # Verifies if load info is a list

        elif not isinstance(load_info, list):

            raise TypeError("The load_info for the prescribed Dirichle"+
            "t boundary condition is not a list, but: "+str(load_info)+
            "\nIt must be a list")
        
        # Verifies if the load itself is a string, which denotes that it
        # comes from a loading curve

        elif isinstance(load_info[-1], str):

            # Tries to find this load in the more complex boundary con-
            # ditions generators

            if load_info[-1] in complex_bcsFunctionsDict:

                raise TypeError("The generator '"+str(load_info)+"' wa"+
                "s required to create boundary conditions. To do this,"+
                "you have to encapsulate the load information into a d"+
                "ictionary, not a list")

            # Otherwise, uses the generic loading curves

            else:

                # Gets the loading curve

                prescribed_BC = Constant(0.0)

                load_curve = numerical_tools.generate_loadingParametricCurves(
                load_info[-1], additional_parameters=additional_parameters)

                class LocalLoad:

                    def __init__(self):

                        self.time = Constant(t_initial)

                        self.maximum_time = Constant(t_final)

                        self.prescribed_BC = prescribed_BC

                    def update_load(self, t):

                        self.time.assign(t)

                        self.prescribed_BC.assign(load_curve(self.time/
                        self.maximum_time))

                local_load = LocalLoad()

                load_info[-1] = prescribed_BC

                # Adds the time constant to the dirichlet_loads

                dirichlet_loads.append(local_load)

        # Tests whether the load is in the list of loads

        else:

            if not (load_info[-1] in dirichlet_loads):

                # Tests whether this load is a dolfin Constant

                if isinstance(load_info[-1], Constant):

                    dirichlet_loads.append(load_info[-1])

                # If the load is a class and has an attribute "update"

                elif hasattr(load_info[-1], "update_load"):

                    dirichlet_loads.append(load_info[-1])

                # Otherwise, updates it as a class (Expressions are clas-
                # ses)

                elif hasattr(load_info[-1], "t"):

                    dirichlet_loads.append(load_info[-1])

        # Verifies whether the boundary physical groups is a list or not

        if isinstance(physical_groups, tuple):

            # Iterates through the regions

            for physical_group in physical_groups:

                # Verifies if the physical group is a string

                physical_group = verify_stringPhysicalGroup(physical_group, 
                mesh_dataClass.boundary_physicalGroupsNameToTag)

                # Verifies if there is only one field

                if n_fields==1:

                    # Verifies if no DOFs are especified

                    if len(load_info)==1:

                        # Adds this particular boundary condition to the 
                        # lot

                        boundary_conditions.append(DirichletBC(
                        field_functionSpace, load_info[0], 
                        mesh_dataClass.boundary_meshFunction, 
                        physical_group))

                    # Otherwise, checks if there is a list of DOFs to be
                    # prescribed

                    else:

                        if isinstance(load_info[0], list):

                            # Iterates through the prescribed DOFs

                            for dof_number in load_info[0]:

                                # Adds this particular boundary condition 
                                # to the lot

                                boundary_conditions.append(DirichletBC(
                                field_functionSpace.sub(dof_number), 
                                load_info[1], 
                                mesh_dataClass.boundary_meshFunction, 
                                physical_group))

                        else:

                            # Adds this particular boundary condition to 
                            # the lot

                            boundary_conditions.append(DirichletBC(
                            field_functionSpace.sub(load_info[0]), 
                            load_info[1], 
                            mesh_dataClass.boundary_meshFunction, 
                            physical_group))

                # Otherwise, iterates through the fields

                else:

                    # Verifies if the load info has string values

                    load_info[0] = convert_fieldsNamesToIndices(
                    load_info[0], fields_namesDict)

                    for field in range(n_fields):

                        # Verifies if this field is to be constrained or 
                        # not

                        if field in load_info[0]:

                            # Checks whether a set of degrees of freedom
                            # of the field is to be prescribed

                            if len(load_info)>2:

                                # Verifies if there is a list of DOFs

                                if isinstance(load_info[1], list):

                                    # Iterates through the prescribed
                                    # DOFs

                                    for dof_number in load_info[1]:

                                        # Adds this particular boundary
                                        # condition to the lot

                                        boundary_conditions.append(DirichletBC(
                                        field_functionSpace.sub(field).sub(
                                        dof_number), load_info[2], 
                                        mesh_dataClass.boundary_meshFunction, 
                                        physical_group))

                                else:

                                    # Adds this particular boundary con-
                                    # dition to the lot

                                    boundary_conditions.append(DirichletBC(
                                    field_functionSpace.sub(field).sub(
                                    load_info[1]), load_info[2], 
                                    mesh_dataClass.boundary_meshFunction, 
                                    physical_group))

                            else:

                                # Adds this particular boundary condi-
                                # tion to the lot

                                boundary_conditions.append(DirichletBC(
                                field_functionSpace.sub(field), 
                                load_info[1], 
                                mesh_dataClass.boundary_meshFunction, 
                                physical_group))

        # Otherwise, if there is only one physical group to apply boundary 
        # conditions

        else:

            # Verifies if the physical group is a string

            physical_group = verify_stringPhysicalGroup(physical_groups, 
            mesh_dataClass.boundary_physicalGroupsNameToTag)

            # Verifies if there is only one field

            if n_fields==1:

                # Verifies if no DOFs are especified

                if len(load_info)==1:

                    # Adds this particular boundary condition to the lot

                    boundary_conditions.append(DirichletBC(
                    field_functionSpace, load_info[0], 
                    mesh_dataClass.boundary_meshFunction, physical_group
                    ))

                # Otherwise, checks if there is a list of DOFs to be
                # prescribed

                else:

                    if isinstance(load_info[0], list):

                        # Iterates through the prescribed DOFs

                        for dof_number in load_info[0]:

                            # Adds this particular boundary condition 
                            # to the lot

                            boundary_conditions.append(DirichletBC(
                            field_functionSpace.sub(dof_number), 
                            load_info[1], 
                            mesh_dataClass.boundary_meshFunction, 
                            physical_group))

                    else:

                        # Adds this particular boundary condition to 
                        # the lot

                        boundary_conditions.append(DirichletBC(
                        field_functionSpace.sub(load_info[0]), 
                        load_info[1], 
                        mesh_dataClass.boundary_meshFunction, 
                        physical_group))

            # Otherwise, iterates through the fields

            else:

                # Verifies if the load info has string values

                load_info[0] = convert_fieldsNamesToIndices(load_info[0], 
                fields_namesDict)

                for field in range(n_fields):

                    # Verifies if this field is to be constrained or 
                    # not

                    if field in load_info[0]:

                        # Checks whether a set of degrees of freedom
                        # of the field is to be prescribed

                        if len(load_info)>2:

                            # Verifies if there is a list of DOFs

                            if isinstance(load_info[1], list):

                                # Iterates through the prescribed
                                # DOFs

                                for dof_number in load_info[1]:

                                    # Adds this particular boundary
                                    # condition to the lot

                                    boundary_conditions.append(DirichletBC(
                                    field_functionSpace.sub(field).sub(
                                    dof_number), load_info[2], 
                                    mesh_dataClass.boundary_meshFunction, 
                                    physical_group))

                            else:

                                # Adds this particular boundary con-
                                # dition to the lot

                                boundary_conditions.append(DirichletBC(
                                field_functionSpace.sub(field).sub(
                                load_info[1]), load_info[2], 
                                mesh_dataClass.boundary_meshFunction, 
                                physical_group))

                        else:

                            # Adds this particular boundary condi-
                            # tion to the lot

                            boundary_conditions.append(DirichletBC(
                            field_functionSpace.sub(field), 
                            load_info[1], 
                            mesh_dataClass.boundary_meshFunction, 
                            physical_group))

    # Returns the boundary conditions list

    if mesh_dataClass.verbose:

        print("Finishes creating prescribed Dirichlet boundary conditi"+
        "ons.\n")

    # Always retuns the dirichlet_loads because this list is, at least,
    # verified
    
    return boundary_conditions, dirichlet_loads

########################################################################
#               Homogeneous Dirichlet boundary conditions              #
########################################################################

# Defines a function to apply homogeneous boundary conditions to a vec-
# tor field over all the 3 directions. The field can be mixed, i.e. the-
# re can be multiple field within the same solution. The number of 
# fields must be provided because, if one programatically tries to count
# the number of subfields in a function space and the function space is
# a vector function space for instance, it will count the dimensionality
# of the vector as subfields, which are not physical fields alone per
# se. If sub_fields is an empty list, all DOFs are constrained

@programming_tools.optional_argumentsInitializer({'boundary_conditions':
lambda: [], 'sub_fieldsToApplyBC': lambda: []})

def FixedSupportDirichletBC(field_functionSpace, mesh_dataClass,
fields_namesDict, boundary_physicalGroups=0, sub_fieldsToApplyBC=None, 
boundary_conditions=None):

    # If the physical groups variable is null, returns the empty list of
    # boundary conditions

    if boundary_physicalGroups==0:

        if mesh_dataClass.verbose:

            print("Creates no fixed support boundary condition.\n")

        return boundary_conditions
    
    # If subfields are required, use the dictionary of fields to convert
    # the names to the numbers

    if not (sub_fieldsToApplyBC is None):

        sub_fieldsToApplyBC = convert_fieldsNamesToIndices(
        sub_fieldsToApplyBC, fields_namesDict)
    
    # Tests if the element is mixed and gets the number of fields

    n_fields = 1

    element_type = field_functionSpace.ufl_element().family()

    if element_type=='Mixed':

        # Gets the number of fields

        n_fields = field_functionSpace.ufl_element().num_sub_elements()

    # Verifies whether the boundary physical groups is a list or not

    if isinstance(boundary_physicalGroups, list):

        # Iterates through the regions

        for physical_group in boundary_physicalGroups:

            # Verifies if the physical group is a string

            physical_group = verify_stringPhysicalGroup(physical_group, 
            mesh_dataClass.boundary_physicalGroupsNameToTag)

            # Verifies if there is only one field

            if n_fields==1:

                # Adds this particular boundary condition to the lot

                boundary_conditions.append(DirichletBC(
                field_functionSpace, Constant((0.0, 0.0, 0.0)), 
                mesh_dataClass.boundary_meshFunction, physical_group))

            # Otherwise, iterates through the fields

            else:

                for field in range(n_fields):

                    # Verifies if this field is to be constrained or not

                    if (field in sub_fieldsToApplyBC or (
                    sub_fieldsToApplyBC==[])):

                        # Adds this particular boundary condition to the 
                        # lot

                        boundary_conditions.append(DirichletBC(
                        field_functionSpace.sub(field), Constant((0.0, 0.0, 
                        0.0)), mesh_dataClass.boundary_meshFunction, 
                        physical_group))

    # Otherwise, if there is only one physical group to apply boundary 
    # conditions

    else:

        # Verifies if the physical group is a string

        boundary_physicalGroups = verify_stringPhysicalGroup(
        boundary_physicalGroups, 
        mesh_dataClass.boundary_physicalGroupsNameToTag)

        # Verifies if there is only one field

        if n_fields==1:

            # Adds this particular boundary condition to the lot

            boundary_conditions.append(DirichletBC(field_functionSpace,
            Constant((0.0, 0.0, 0.0)), 
            mesh_dataClass.boundary_meshFunction, boundary_physicalGroups
            ))

        # Otherwise, iterates through the fields

        else:

            for field in range(n_fields):

                # Verifies if this field is to be constrained or not

                if (field in sub_fieldsToApplyBC or (
                sub_fieldsToApplyBC==[])):

                    # Adds this particular boundary condition to the lot

                    boundary_conditions.append(DirichletBC(
                    field_functionSpace.sub(field), Constant((0.0, 0.0, 
                    0.0)), mesh_dataClass.boundary_meshFunction, 
                    boundary_physicalGroups))

    # Returns the boundary conditions list

    if mesh_dataClass.verbose:

        print("Finishes creating fixed support boundary conditions.\n")

    return boundary_conditions

# Defines a function to apply a simple support. If sub_fields is an empty
# list, all DOFs are constrained

@programming_tools.optional_argumentsInitializer({'boundary_conditions':
lambda: [], 'sub_fieldsToApplyBC': lambda: []})

def SimpleSupportDirichletBC(field_functionSpace, mesh_dataClass, 
bc_informationsDict, fields_namesDict, sub_fieldsToApplyBC=None, 
boundary_conditions=None, boundary_physicalGroups=None):
    
    # Verifies if the boundary physical groups is a dictionary or a list
    # if the physical group is given

    if isinstance(bc_informationsDict, list):

        if boundary_physicalGroups is None:

            raise ValueError("To apply SimpleSupportDirichletBC, the b"+
            "oundary_physicalGroups variables can be a list if and onl"+
            "y if the boundary_physicalGroups is provided, but it's No"+
            "ne")
        
        # Transforms in a dictionary
        
        bc_informationsDict[boundary_physicalGroups
        ] = bc_informationsDict

    # Verifies if the dictionary of boundary conditions' information has
    # physical groups as keys

    elif isinstance(bc_informationsDict, dict):

        for key in bc_informationsDict.keys():

            if isinstance(key, str):
                
                if not (key in mesh_dataClass.boundary_physicalGroupsNameToTag):

                    # This dictionary does not have keys as physical 
                    # groups, thus, makes this a value with the physical
                    # group as key

                    if boundary_physicalGroups is None:

                        raise ValueError("The bc_informationsDict is n"+
                        "ot a dictionary with physical groups as keys."+
                        " Nevertheless, the boundary_physicalGroups va"+
                        "riable is None, so no physical group informat"+
                        "ion can be retrieved")
                    
                    else:

                        bc_informationsDict[boundary_physicalGroups] = (
                        bc_informationsDict)

                    break

    elif not isinstance(bc_informationsDict, dict):

        raise ValueError("The bc_informationsDict variable in the Simp"+
        "leSupportDirichletBC method must be a dictionary, where the k"+
        "eys are the physical groups of the boundary regions and value"+
        "s are the list of DOFs to be constrained.")

    # If the physical groups variable is null, returns the empty list of
    # boundary conditions

    if len(list(bc_informationsDict.keys()))==0:

        if mesh_dataClass.verbose:

            print("Creates no simple support boundary condition.\n")

        return boundary_conditions
    
    # If subfields are required, use the dictionary of fields to convert
    # the names to the numbers

    if not (sub_fieldsToApplyBC is None):

        sub_fieldsToApplyBC = convert_fieldsNamesToIndices(
        sub_fieldsToApplyBC, fields_namesDict)
    
    # Tests if the element is mixed and gets the number of fields

    n_fields = 1

    element_type = field_functionSpace.ufl_element().family()

    if element_type=='Mixed':

        # Gets the number of fields

        n_fields = field_functionSpace.ufl_element().num_sub_elements()

    # Iterates through the regions

    for physical_group, list_constrainedDOFs in bc_informationsDict.items():

        # Verifies if the physical group is a string

        physical_group = verify_stringPhysicalGroup(physical_group, 
        mesh_dataClass.boundary_physicalGroupsDictNameToTag)

        # Verifies if there is only one field

        if n_fields==1:

            # Iterates through the DOFs to be constrained

            for i in range(field_functionSpace.num_sub_spaces()):
                
                if i in list_constrainedDOFs:

                    # Adds this particular boundary condition to the 
                    # lot

                    boundary_conditions.append(DirichletBC(
                    field_functionSpace.sub(i), Constant(0.0), 
                    mesh_dataClass.boundary_meshFunction, physical_group
                    ))

        # Otherwise, iterates through the fields

        else:

            for field in range(n_fields):

                # Verifies if this field is to be constrained or not

                if (field in sub_fieldsToApplyBC or (
                sub_fieldsToApplyBC==[])):
                    
                    # Iterates through the DOFs to be constrained

                    for i in range(field_functionSpace.sub(field
                    ).num_sub_spaces()):
                        
                        if i in list_constrainedDOFs:

                            # Adds this particular boundary condition 
                            # to the lot

                            boundary_conditions.append(DirichletBC(
                            field_functionSpace.sub(field).sub(i), 
                            Constant(0.0), 
                            mesh_dataClass.boundary_meshFunction, 
                            physical_group))

    # Returns the boundary conditions list

    if mesh_dataClass.verbose:

        print("Finishes creating simply supported boundary conditions."+
        "\n")

    return boundary_conditions

########################################################################
#                          Subdomain classes                           #
########################################################################

# Defines a function to generate a class to apply boundary conditions to 
# a node

def generate_nodeSubdomain(point_coordinates, mesh_dataClass, tolerance=
1E-5):

    # Gets the coordinates of the node closest to the point required

    _, node_coordinates = mesh_tools.find_nodeClosestToPoint(
    mesh_dataClass, point_coordinates, None, None)

    # Defines the class

    class FixedNode(SubDomain):

        def __init__(self, node_coordinates, tolerance):

            super().__init__()

            self.node_coordinates = node_coordinates

            self.tolerance = tolerance

        def inside(self, x, on_boundary):

            # Ignores the on_boudnary flag because a node inside the do-
            # main can be fixed

            return (near(x[0], self.node_coordinates[0], self.tolerance
            ) and near(x[1], self.node_coordinates[1], self.tolerance)
            and near(x[2], self.node_coordinates[2], self.tolerance))

    # Instantiates the class

    fixed_node = FixedNode(node_coordinates, tolerance)

    return fixed_node

########################################################################
#                              Utilities                               #
########################################################################

# Defines a function to verify if the physical group is a string and, 
# then, to convert it into a number tag using the dictionary of physical
# groups' names to tags

def verify_stringPhysicalGroup(physical_group, 
boundary_physGroupsNamesToTags):
    
    if isinstance(physical_group, str):

        # Converts this physical group to the corresponding number tag

        try:

            physical_group = boundary_physGroupsNamesToTags[
            physical_group]

        except:

            raise KeyError("The physical group '"+physical_group+"' is"+
            " not in the dictionary of physical groups names' to tags."+
            " This dictionary has the following keys and values: "+str(
            boundary_physGroupsNamesToTags))
        
    return physical_group

# Defines a function to convert a list or a value of fields to a list of
# their corresponding indices

def convert_fieldsNamesToIndices(sub_fieldsToApplyBC, fields_namesDict,
dont_convertToList=False):

    if isinstance(sub_fieldsToApplyBC, list):

        for i in range(len(sub_fieldsToApplyBC)):

            if isinstance(sub_fieldsToApplyBC[i], str):

                if sub_fieldsToApplyBC[i] in fields_namesDict:

                    sub_fieldsToApplyBC[i] = fields_namesDict[
                    sub_fieldsToApplyBC[i]]

                else:

                    raise KeyError("The field '"+str(sub_fieldsToApplyBC[
                    i])+"' was not found in the dictionary of fields' "+
                    "names. Hence, it is not possible to create Dirich"+
                    "let boundary conditions for this field. Check out"+
                    " the list of available fields names: "+str(
                    fields_namesDict.keys()))
                
            elif not isinstance(sub_fieldsToApplyBC[i], int):

                # If the field is not an integer, the index of the field,
                # throws an error

                raise ValueError("Only integer or strings can be used "+
                "to recover fields to create Dirichlet boundary condit"+
                "ions. The value "+str(sub_fieldsToApplyBC[i])+" was g"+
                "iven")
            
    elif isinstance(sub_fieldsToApplyBC, str):
        
        if sub_fieldsToApplyBC in fields_namesDict:

            if dont_convertToList:

                # Does not transform to a list

                sub_fieldsToApplyBC = fields_namesDict[
                sub_fieldsToApplyBC]

            else:

                # Transforms to a list also

                sub_fieldsToApplyBC = [fields_namesDict[
                sub_fieldsToApplyBC]]

        else:

            raise KeyError("The field '"+str(sub_fieldsToApplyBC)+"' w"+
            "as not found in the dictionary of fields' names. Hence, i"+
            "t is not possible to create a Dirichlet boundary conditio"+
            " for this field. Check out the list of available fields n"+
            "ames: "+str(fields_namesDict.keys()))
        
    elif not isinstance(sub_fieldsToApplyBC, int):

        # If the field is not an integer, the index of the field, throws
        # an error

        raise ValueError("Only integer or strings can be used to recov"+
        "er fields to create Dirichlet boundary conditions. The value "+
        str(sub_fieldsToApplyBC[i])+" was given")
    
    return sub_fieldsToApplyBC