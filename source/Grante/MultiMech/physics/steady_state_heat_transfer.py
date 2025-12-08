# Routine to store the physics of steady-state heat transfer

from fenics import *
import ufl_legacy as ufl
from dolfin import *

#     |-------------------------------------------------------------------------|
#     |   This code aims to apply a simple exemple of a heat transfer exercise  |
#     |               a plane wall to validate the FeNiCs prompt                |
#     |-------------------------------------------------------------------------|

# First, it is necessary to define the goemtry and the mesh (Box mesh)

polynomial_degree = 1

# Defines the lenght of the edges (in his case: parameters fo the exeercise 2.12 
# of the 7th ed. Heat and Mass Transfer Incropera


L = 0.1
W = 1.0
H = 1.0

# Defines the prescribed temperatues conditions 

T_hot = 600.0
T_cold = 400.0

# We are considering a total isotropic material, thus, the sencond order tensor 
# of the condutivity is gonna be k*I, where I is the identity tensor 

k = 100.0

I = Identity(3)

k_tensor = k*I

mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(L, H, W), 5, 10, 10)

# Defines the function Space

V = FunctionSpace(mesh, 'P', polynomial_degree)

# Boundary conditions 

tol = 1e-10

# Defines where each volume or facet is localized

left_facet = CompiledSubDomain("near(x[0], 0)")
right_facet = CompiledSubDomain("near(x[0], L)", L=L)
volume_1 = CompiledSubDomain("x[1]<= h + tol", h=H, tol=tol)

# Defines the number of each part defined previously

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(0)
left_facet.mark(boundary_markers, 1)
right_facet.mark(boundary_markers, 2)

volume_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
volume_markers.set_all(0)
volume_1.mark(volume_markers, 3)

# Defines the Dirichlet boundary consitions

bc_hot = DirichletBC(V, Constant(T_hot), left_facet)
bc_cold = DirichletBC(V, Constant(T_cold), right_facet)

# Put in a list

bc = [bc_hot, bc_cold]

# ************************************************************
#                         variational Form                   *
# ************************************************************

T = TrialFunction(V)

delta_T = TestFunction(V)

# To test, first, we can define the energy generation as constant zero

q_v = Constant(0.0)

# Recording coersivity, as shown by Larx Milgran, the bilinear and linear parts are:

# We have only one volume, thus:

dx = Measure("dx", domain=mesh, subdomain_data=volume_markers)

# bilinear part

a = dot(k_tensor*grad(T), grad(delta_T)) * dx

# linear part (if had heat flux prescribed, it would be necessary to apply more one part: 
# (heat_flux * delta_T * ds(number of the face)

l = q_v * delta_T * dx

# Solve 

T_solve = Function(V, name = "Temperature")

solve(a == l, T_solve, bc)

# ************************************************************
#                 Projection of the heat_flux                *
# ************************************************************

W = VectorFunctionSpace(mesh, 'P', polynomial_degree)

# We know that: heat_flux = -K * grad(T)

flux_equation = -1.0 * k_tensor * grad(T_solve)

# cg is a iterative method

flux_solve = project(flux_equation, W, solver_type="cg")
flux_solve.rename("Heat Flux", "Heat Flux Vector")

# ************************************************************
#                         Save in a file                     *
# ************************************************************

xdmf_file = XDMFFile("Results_heat_transfer.xdmf")

xdmf_file.write(T_solve, 0.0)
xdmf_file.write(flux_solve, 0.0)

xdmf_file.close()

# Routine to store the variational form and other accessories for a hy-
# perelastic Cauchy continuum in solid mechanics

from ..tool_box import mesh_handling_tools as mesh_tools

from ..tool_box import variational_tools

from ..tool_box import functional_tools

from ..tool_box import pseudotime_stepping_tools as newton_raphson_tools

from ...PythonicUtilities import programming_tools

# Defines a function to model a hyperelastic problem with a temperature
# field only

@programming_tools.optional_argumentsInitializer({'neumann_loads': 
lambda: [], 'dirichlet_loads': lambda: [], 'solution_name': lambda: [
"solution", "DNS"], 'volume_physGroupsSubmesh': lambda: [], ('post_pro'+
'cessesSubmesh'): lambda: dict(), 'dirichlet_boundaryConditions': lambda: 
dict(), "heat_generation_dict": lambda: dict()})

def steady_state_heat_transfer_temperature_based(constitutive_model, 
heat_flux_dictionary, maximum_loadingSteps, t_final, post_processes, 
mesh_fileName, solver_parameters, neumann_loads=None, dirichlet_loads=
None, polynomial_degree=2, quadrature_degree=2, t=0.0, 
volume_physGroupsSubmesh=None, post_processesSubmesh=None, 
solution_name=None, verbose=False, dirichlet_boundaryConditions=None,
heat_generation_dict=None):

    ####################################################################
    #                               Mesh                               #
    ####################################################################

    # Reads the mesh and constructs some fenics objects using the xdmf 
    # file

    mesh_dataClass = mesh_tools.read_mshMesh(mesh_fileName, 
    quadrature_degree=quadrature_degree, verbose=verbose)

    ####################################################################
    #                          Function space                          #
    ####################################################################

    # Assembles a dictionary of finite elements for the single primal 
    # field: temperature (Kelvin). Each field has a key and the corres-
    # ponding value is another dictionary, which has keys for necessary
    # information to create finite elements

    elements_dictionary = {"Temperature": {"field type": "scalar", "in"+
    "terpolation function": "CG", "polynomial degree": polynomial_degree
    }}

    # From the dictionary of elements, the finite elements are created,
    # then, all the rest is created: the function spaces, trial and test 
    # functions, solution function. Everything is split and named by ac-
    # cording to the element's name

    (solution_functionSpace, solution_new, fields_names, solution_fields, 
    variation_fields, delta_solution, fields_namesDict
    ) = functional_tools.construct_monolithicFunctionSpace(
    elements_dictionary, mesh_dataClass, verbose=verbose)

    ####################################################################
    #                        Boundary conditions                       #
    ####################################################################

    # Defines the boundary conditions and the list of temperature loads
    # using the dictionary of boundary conditions

    bc, dirichlet_loads = functional_tools.construct_DirichletBCs(
    dirichlet_boundaryConditions, fields_namesDict, 
    solution_functionSpace, mesh_dataClass, dirichlet_loads=
    dirichlet_loads)

    ####################################################################
    #                         Variational forms                        #
    ####################################################################

    # Constructs the variational form for the inner work

    internal_VarForm = variational_tools.steady_state_heat_internal_work(
    "Temperature", solution_fields, variation_fields, 
    constitutive_model, mesh_dataClass)

    # Constructs the variational forms for the traction work

    out_heat_flux_variational_form, neumann_loads = variational_tools.boundary_heat_flux_work(
    heat_flux_dictionary, "Temperature", solution_fields, 
    variation_fields, solution_new, fields_namesDict, mesh_dataClass, 
    neumann_loads)

    # Constructs the variational form for the work of heat generation

    heat_generation_variational_form, neumann_loads = variational_tools.heat_generation_work(
    heat_generation_dict, "Temperature", solution_fields, 
    variation_fields, solution_new, fields_namesDict, mesh_dataClass, 
    neumann_loads)

    ####################################################################
    #              Problem and solver parameters setting               #
    ####################################################################

    # Assembles the residual and the nonlinear problem object. Sets the
    # solver parameters too

    residual_form = (internal_VarForm-out_heat_flux_variational_form-
    heat_generation_variational_form)

    solver = functional_tools.set_nonlinearProblem(residual_form, 
    solution_new, delta_solution, bc, solver_parameters=
    solver_parameters)

    ####################################################################
    #                 Solution and pseudotime stepping                 #
    ####################################################################

    # Iterates through the pseudotime stepping algorithm

    newton_raphson_tools.newton_raphsonSingleField(solver, solution_new, 
    fields_namesDict, mesh_dataClass, constitutive_model, 
    post_processesDict=post_processes, post_processesSubmeshDict=
    post_processesSubmesh, neumann_loads=neumann_loads, dirichlet_loads=
    dirichlet_loads, solution_name=solution_name, 
    volume_physGroupsSubmesh=volume_physGroupsSubmesh, t=t, t_final=
    t_final, maximum_loadingSteps=maximum_loadingSteps)