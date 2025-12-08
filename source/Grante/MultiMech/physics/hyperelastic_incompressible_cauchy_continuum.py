# Routine to store the variational form and other accessories for an in-
# compressible hyperelastic Cauchy continuum in solid mechanics 

from dolfin import *

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import variational_tools

from ..tool_box import functional_tools

from ..tool_box import pseudotime_stepping_tools as newton_raphson_tools

from ...PythonicUtilities import programming_tools

# Defines a function to model a hyperelastic problem with displacement,
# and pressure fields

@programming_tools.optional_argumentsInitializer({'neumann_loads': 
lambda: [], 'dirichlet_loads': lambda: [], 'solution_name': lambda: [
["Displacement", "DNS"], ["Pressure", "DNS"]], 'volume_physGroupsSubme'+
'sh': lambda: [], ('post_processesSubmesh'): lambda: dict(), 'prescrib'+
'ed_displacement': lambda: dict(), 'dirichlet_boundaryConditions': 
lambda: dict(), "body_forcesDict": lambda: dict()})

def hyperelasticity_two_fields(constitutive_model, traction_dictionary, 
maximum_loadingSteps, t_final, post_processes, mesh_fileName, 
solver_parameters, neumann_loads=None, dirichlet_loads=None,  
polynomial_degree_displacement=2, polynomial_degree_pressure=1, t=0.0, 
volume_physGroupsSubmesh=None, post_processesSubmesh=None, solution_name=
None, dirichlet_boundaryConditions=None, body_forcesDict=None, verbose=
False):

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
    polynomial_degree_displacement}, "Pressure": {"field type": "scalar", 
    "interpolation function": "CG", "polynomial degree": 
    polynomial_degree_pressure}}

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

    # Checks if there is any subfield information on the boundary condi-
    # tions. Incompressible problems usually have boundary conditions on 
    # the displacement field only

    for region, boundary_condition in dirichlet_boundaryConditions.items():

        if isinstance(boundary_condition, dict):

            # Checks if there is a key telling the subfield, otherwise
            # assumes this boundary condition is for displacement

            if not ("sub_fieldsToApplyBC" in boundary_condition):

                dirichlet_boundaryConditions[region]["sub_fieldsToAppl"+
                "yBC"] = "Displacement"

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
    "Displacement", solution_fields, variation_fields, 
    constitutive_model, mesh_dataClass)

    # Evaluates the entities necessary for the incompressibility cons-
    # traint

    I = Identity(3)

    F = grad(solution_fields["Displacement"])+I 

    C = (F.T)*F

    C = variable(C)

    I_1 = tr(C)

    psi = (constitutive_model.mu/2)*(I_1-3)

    S = 2*diff(psi,C)

    P = F*S

    F_invT = inv(F).T

    J = det(F)

    inv_V0 = Constant(1/assemble(1.0*mesh_dataClass.dx))

    factor = 0.0

    internal_VarForm = ((inner(constitutive_model.first_piolaStress(
    solution_fields["Displacement"])+(solution_fields["Pressure"]*
    inv_V0*J*F_invT), grad(variation_fields["Displacement"]))*mesh_dataClass.dx)+(
    inv_V0*(((J-1)*variation_fields["Pressure"])*mesh_dataClass.dx)))

    internal_VarForm = ((inner(constitutive_model.first_piolaStress(
    solution_fields["Displacement"])+(solution_fields["Pressure"]*
    inv_V0*J*F_invT), grad(variation_fields["Displacement"
    ]))*mesh_dataClass.dx)+(
    inv_V0*(((J-1)*variation_fields["Pressure"])*mesh_dataClass.dx)))

    internal_VarForm = ((inner(P+(factor*inv_V0*solution_fields["Pressure"]*J*
    F_invT), grad(variation_fields["Displacement"]))*mesh_dataClass.dx)+(
    inv_V0*(((J-1)*variation_fields["Pressure"])*mesh_dataClass.dx)))

    # Adds the contribution of the incompressibility constraint

    """internal_VarForm += ((inner(inv_V0*solution_fields["Pressure"]*J*
    F_invT, grad(variation_fields["Displacement"]))*mesh_dataClass.dx)+(
    inv_V0*((J-1.0)*variation_fields["Pressure"])*mesh_dataClass.dx))"""

    # Constructs the variational forms for the traction work

    traction_VarForm, neumann_loads = variational_tools.traction_work(
    traction_dictionary, "Displacement", solution_fields, 
    variation_fields, solution_new, fields_namesDict, mesh_dataClass, 
    neumann_loads)

    # Constructs the variational form for the work of the body forces

    body_forcesVarForm, neumann_loads = variational_tools.body_forcesWork(
    body_forcesDict, "Displacement", solution_fields, variation_fields, 
    solution_new, fields_namesDict, mesh_dataClass, neumann_loads)

    ####################################################################
    #              Problem and solver parameters setting               #
    ####################################################################

    # Assembles the residual and the nonlinear problem object. Sets the
    # solver parameters too

    residual_form = internal_VarForm-traction_VarForm-body_forcesVarForm

    solver = functional_tools.set_nonlinearProblem(residual_form, 
    solution_new, delta_solution, bc, solver_parameters=
    solver_parameters)

    ####################################################################
    #                   Post-processes verification                    #
    ####################################################################

    # Post-processes for single-field simulations are dictionaries, but
    # they are lists of lists for multifield simulations. Thus, if the 
    # current post-process object is a dictionary, corrects it and assu-
    # mes the provided dictionary is for displacement only

    if isinstance(post_processes, dict):

        post_processes = [["Displacement", post_processes]]

    # Does the same for the submesh

    if isinstance(post_processesSubmesh, dict):

        post_processesSubmesh = [["Displacement", post_processesSubmesh]]

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