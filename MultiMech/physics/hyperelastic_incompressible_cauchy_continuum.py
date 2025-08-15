# Routine to store the variational form and other accessories for a hy-
# perelastic Cauchy continuum in solid mechanics

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import variational_tools

from ..tool_box import pseudotime_stepping_tools as newton_raphson_tools

from ..tool_box import programming_tools

# Defines a function to model a hyperelastic problem with displacement,
# pressure and jacobian fields

@programming_tools.optional_argumentsInitializer({'neumann_loads': 
lambda: [], 'dirichlet_loads': lambda: [], 'solution_name': lambda: [
"solution", "DNS"], 'volume_physGroupsSubmesh': lambda: [], ('post_pro'+
'cessesSubmesh'): lambda: dict(), 'prescribed_displacement': lambda: 
dict(), 'dirichlet_boundaryConditions': lambda: dict()})

def hyperelasticity_threeFields(constitutive_model, traction_dictionary, 
maximum_loadingSteps, t_final, post_processes, mesh_fileName, 
solver_parameters, neumann_loads=None, dirichlet_loads=None,  
polynomial_degreeDisplacement=2, polynomial_degreePressure=1, t=0.0, 
volume_physGroupsSubmesh=None, post_processesSubmesh=None, solution_name=
None, dirichlet_boundaryConditions=None):

    ####################################################################
    #                               Mesh                               #
    ####################################################################

    # Reads the mesh and constructs some fenics objects using the xdmf 
    # file
    
    mesh_dataClass = mesh_tools.read_mshMesh(mesh_fileName, verbose=
    verbose)

    ####################################################################
    #                          Function space                          #
    ####################################################################

    # Assembles a dictionary of finite elements for the two primal 
    # fields: displacement and microrotation. Each field has a key and
    # the corresponding value is another dictionary, which has keys for
    # necessary information to create finite elements

    elements_dictionary = {"displacement": {"field type": "vector", "i"+
    "nterpolation function": "CG", "polynomial degree": 
    polynomial_degreeDisplacement}, "pressure": {"field type": "scalar", 
    "interpolation function": "CG", "polynomial degree": 
    polynomial_degreePressure}}

    # From the dictionary of elements, the finite elements are created,
    # then, all the rest is created: the function spaces, trial and test 
    # functions, solution function. Everything is split and named by ac-
    # cording to the element's name

    (monolithic_functionSpace, solution_new, fields_names, 
    solution_fields, variation_fields, delta_solution, fields_namesDict
    ) = functional_tools.construct_monolithicFunctionSpace(
    elements_dictionary, mesh_dataClass, verbose=verbose)

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Defines the boundary conditions and the list of displacement loads
    # using the dictionary of boundary conditions

    bc, dirichlet_loads = functional_tools.construct_DirichletBCs(
    dirichlet_boundaryConditions, fields_namesDict, 
    monolithic_functionSpace, mesh_dataClass, dirichlet_loads=
    dirichlet_loads)

    ####################################################################
    #                         Variational forms                        #
    ####################################################################

    # Constructs the variational form for the inner work

    internal_VarForm = variational_tools.hyperelastic_internalWorkFirstPiola(
    "displacement", solution_fields, variation_fields, 
    constitutive_model, mesh_dataClass)

    # Constructs the variational forms for the traction work

    traction_VarForm, neumann_loads = variational_tools.traction_work(
    traction_dictionary, "displacement", solution_fields, 
    variation_fields, solution_new, fields_namesDict, mesh_dataClass, 
    neumann_loads)

    ####################################################################
    #              Problem and solver parameters setting               #
    ####################################################################

    # Assembles the residual and the nonlinear problem object. Sets the
    # solver parameters too

    residual_form = internal_VarForm-traction_VarForm-moment_VarForm

    solver = functional_tools.set_nonlinearProblem(residual_form, 
    solution_new, delta_solution, bc, solver_parameters=
    solver_parameters)

    ####################################################################
    #                 Solution and pseudotime stepping                 #
    ####################################################################

    # Iterates through the pseudotime stepping algorithm 

    if len(solution_name)==0:

        for field_name in fields_namesDict:

            solution_name.append([field_name, "DNS"])

    newton_raphson_tools.newton_raphsonMultipleFields(solver, 
    solution_new, fields_namesDict, mesh_dataClass, constitutive_model, 
    post_processesList=post_processes, post_processesSubmeshList=
    post_processesSubmesh, dirichlet_loads=dirichlet_loads, neumann_loads
    =neumann_loads, volume_physGroupsSubmesh=volume_physGroupsSubmesh, 
    solution_name=solution_name, t=t, t_final=t_final, 
    maximum_loadingSteps=maximum_loadingSteps)