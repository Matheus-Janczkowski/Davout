# Routine to store the variational form and other accessories for a hy-
# perelastic micropolar (Cosserat) continuum in solid mechanics

from dolfin import *

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import variational_tools

from ..tool_box import functional_tools

from ..tool_box import pseudotime_stepping_tools as newton_raphson_tools

from ..tool_box import programming_tools

# Defines a function to model a hyperelastic problem with a displacement
# and a microrotation fields only

@programming_tools.optional_argumentsInitializer({'neumann_loads': 
lambda: [], 'dirichlet_loads': lambda: [], 'solution_name': lambda: [], 
'volume_physGroupsSubmesh': lambda: [], 'post_processesSubmesh': lambda: 
[], 'dirichlet_boundaryConditions': lambda: dict(), 'body_forcesDict':
lambda: dict(), 'body_momentsDict': lambda: dict()})

def hyperelasticity_displacementMicrorotationBased(constitutive_model, 
traction_dictionary, moment_dictionary, maximum_loadingSteps, t_final, 
post_processes, mesh_fileName, solver_parameters, 
polynomial_degreeDisplacement=2, polynomial_degreeMicrorotation=2, 
t=0.0, neumann_loads=None, dirichlet_loads=None, solution_name=None,
volume_physGroupsSubmesh=None, post_processesSubmesh=None, 
dirichlet_boundaryConditions=None, verbose=False, body_forcesDict=None,
body_momentsDict=None):

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

    elements_dictionary = {"Displacement": {"field type": "vector", "i"+
    "nterpolation function": "CG", "polynomial degree": 
    polynomial_degreeDisplacement}, "Microrotation": {"field type": "v"+
    "ector", "interpolation function": "CG", "polynomial degree": 
    polynomial_degreeMicrorotation}}

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

    internal_VarForm = variational_tools.hyperelastic_micropolarInternalWorkFirstPiola(
    "Displacement", "Microrotation", solution_fields, variation_fields, 
    constitutive_model, mesh_dataClass)

    # Constructs the variational forms for the traction work

    traction_VarForm, neumann_loads = variational_tools.traction_work(
    traction_dictionary, "Displacement", solution_fields, 
    variation_fields, solution_new, fields_namesDict, mesh_dataClass, 
    neumann_loads)

    # Constructs the variational forms for the moment work on the boun-
    # dary. Note that the function traction_work was reused, because the
    # variational construction is the same for traction and for moment

    moment_VarForm, neumann_loads = variational_tools.traction_work(
    moment_dictionary, "Microrotation", solution_fields, 
    variation_fields, solution_new, fields_namesDict, mesh_dataClass, 
    neumann_loads)

    # Constructs the variational form for the work of the body forces

    body_forcesVarForm, neumann_loads = variational_tools.body_forcesWork(
    body_forcesDict, "Displacement", solution_fields, variation_fields, 
    solution_new, fields_namesDict, mesh_dataClass, neumann_loads)

    # Constructs the variational form for the work of the body moments

    body_momentsVarForm, neumann_loads = variational_tools.body_forcesWork(
    body_momentsDict, "Microrotation", solution_fields, variation_fields, 
    solution_new, fields_namesDict, mesh_dataClass, neumann_loads)

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