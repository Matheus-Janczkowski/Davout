# Routine to compute the micropolar macroscale problem

from dolfin import *

from mpi4py import MPI

import ufl_legacy as ufl

import numpy as np

import matplotlib.pyplot as plt

from mshr import *

#import periodic_structure as mesher

import source.tool_box.mesh_handling_tools as mesh_tools

import source.tool_box.homogenization_tools as homogenization_tools

import source.tool_box.file_handling_tools as file_tools

import source.tool_box.functional_tools as functional_tools

import source.tool_box.boundary_conditions_tools as BCs_tools

import source.tool_box.tensor_tools as tensor_tools

import source.tool_box.variational_tools as variational_tools

# Defines the indices for Einstein summation notation

i, j, k, l = ufl.indices(4)

########################################################################
########################################################################
##                      User defined parameters                       ##
########################################################################
########################################################################

########################################################################
#                         Material properties                          #
########################################################################

# Sets a dictionary for material property for each physical group in the
# volumetric domain. The keys are the physical groups and the values are
# the material properties

mu_materials = dict()

mu_materials[1] = 26.12

mu_materials[2] = 26.12

mu_materials[3] = 26.12

# Defines some micropolar constitutive parameters

alpha = 0.0

beta = 0.0

gamma = 1e-12#5.22e-8 #5.22e-8 #((1e-5)**2)*2*mu # 43.265e-6

########################################################################
#                          Macro deformations                          #
########################################################################

# A set of macro deformations are stored in txt files. One must define
# their file names

macro_displacementName = "tests//micropolar//results//text//u_RVE_homogenized_intermediate_code"

macro_gradDisplacementName = "tests//micropolar//results//text//grad_u_RVE_homogenized_intermediate_code"

macro_microrotationName = "tests//micropolar//results//text//phi_RVE_homogenized_intermediate_code"

macro_gradMicrorotationName = "tests//micropolar//results//text//grad_phi_RVE_homogenized_intermediate_code"

########################################################################
#                                 Mesh                                 #
########################################################################

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

file_name = "tests//test_meshes//intervertebral_disc"

# Defines the physical groups that belongs to the RVE

volume_physGroupsRVE = [1]

########################################################################
#                            Function space                            #
########################################################################

# Defines the shape functions degree

polynomial_degree = 1

########################################################################
#                           Solver parameters                          #
########################################################################

# Sets some parameters

parameters["form_compiler"]["representation"] = "uflacs"
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Sets the solver parameters

linear_solver = "mumps"

relative_tolerance = 1e-10

absolute_tolerance = 1e-10

maximum_iterations = 10

# Sets the initial time

t = 0.0

# Sets the final pseudotime of the simulation

t_final = 1.0

# Sets the maximum number of steps of loading

maximum_loadingSteps = 11

########################################################################
#                          Boundary conditions                         #
########################################################################

# Defines a load expression

maximum_load = 4E1

load = Expression("(t/t_final)*maximum_load", t=t, t_final=t_final,
maximum_load=maximum_load, degree=0)

traction_boundary = as_vector([0.0, 0.0, load])

# Defines a dictionary of tractions

traction_dictionary = dict()

traction_dictionary[5] = traction_boundary

# Defines the boundary physical groups to apply fixed support boundary
# condition. This variable can be either a list of physical groups tags
# or simply a tag

fixed_supportPhysicalGroups = 4

########################################################################
########################################################################
##               Calculation: Mr. User, take care ahead!              ##
########################################################################
########################################################################

########################################################################
#                                 Mesh                                 #
########################################################################

# Reads the mesh and constructs some fenics objects using the xdmf file

(mesh, dx, ds, n, domain_meshCollection, domain_meshFunction, 
boundary_meshCollection, boundary_meshFunction) = mesh_tools.read_mshMesh(
file_name)

########################################################################
#                            Function space                            #
########################################################################

# Defines the finite element spaces for the displacement field, u, and 
# for the microrotation field, phi

U = VectorElement("CG", mesh.ufl_cell(), polynomial_degree)

V = VectorElement("CG", mesh.ufl_cell(), polynomial_degree)

# Define the mixed element for the monolithic solution

micropolar_mixedElement = MixedElement([U,V])
 
monolithic_functionSpace = FunctionSpace(mesh, micropolar_mixedElement)

########################################################################
#                         RVE post-processing                          #
########################################################################

# Creates the RVE submesh and automatically constructs the function spa-
# ces and the DOFs mapping between meshes

(RVE_submesh, altered_domainMeshFunction, UV_submesh, RVE_meshMapping, 
parent_meshMapping, sol_RVE, RVE_toParentCellMap, dx_submesh) = mesh_tools.create_submesh(
domain_meshCollection, volume_physGroupsRVE, 
monolithic_functionSpace)

########################################################################
#                          Micropolar tensors                          #
########################################################################

# Constructs the three-dimensional identity tensor 

I = Identity(3)

# Defines the curvature tensor

def curvature_tensor(phi):

    # Defines the permutation tensor for 3D euclidean space

    perm = ufl.PermutationSymbol(3)
    
    # Computes the micro-rotation tensor Rbar
     
    R_bar = tensor_tools.rotation_tensorEulerRodrigues(phi)
    
    # Computes the gradient of each component of R_bar to get a third 
    # order tensor, grad_Rbar

    grad_Rbar = grad(R_bar)

    # Constructs the third order curvature tensor K_third

    K_third = as_tensor(R_bar[j,i]*grad_Rbar[j,k,l], (i,k,l))

    # Contracts with the permutation tensor to form K_second

    K_second = 0.5*as_tensor(perm[i,j,k]*K_third[k,j,l], (i,l))

    return K_second

########################################################################
#                         Material properties                          #
########################################################################

# Converts the dictionary of shear modulus to a discontinuous Galerkin
# function space

mu = functional_tools.physical_groupToDGSpace(mu_materials, mesh, 
domain_meshFunction)

# Defines other material properties derived from the shear modulus (the
# mu Lamé parameter)

kappa = 0#-0.15*mu

lmbda = 63.84-(2*mu/3)

########################################################################
#                          Boundary conditions                         #
########################################################################

# Defines the boundary conditions

bc = BCs_tools.fixed_supportDirichletBC(monolithic_functionSpace,
boundary_meshFunction, boundary_physicalGroups=
fixed_supportPhysicalGroups)

########################################################################
#                        Constitutive modelling                        #
########################################################################

# Sets the Helmholtz free energy density to a neo-hookean isotropic mo-
# del. Uses the micropolar stretch as input

def psi_NH(V_bar):

    energy = 0.5*mu*(tr(V_bar*V_bar.T)-3.0)

    return energy

# Sets the volumetric part of the Helmholtz free energy density

def psi_vol(Vbar):

    J = det(Vbar)

    energy = ((lmbda*0.25)*((J**2)-1))-((lmbda*0.5)*ln(J))-(mu*ln(J))

    return energy

# Sets the energy density related to the curvature

def psi_k(K_curvature):

    energy = ((0.5*(alpha*(tr(K_curvature)**2)+(beta*tr(K_curvature*
    K_curvature))+(gamma*tr(K_curvature*K_curvature.T)))))

    return energy

def psi_hat(Vbar):

    energy = 0.25*kappa*(tr(Vbar*Vbar.T)- tr(Vbar*Vbar)) 

    return energy

# Defines the micropolar Kirchhoff stress tensor

def Kirchhoff_Stress(u, phi):

    # Sets the identity tensor and the deformation gradient

    I = Identity(3)

    F = grad(u) + I

    # Evaluates the rotation tensor given by the micropolar rotation

    R_bar = tensor_tools.rotation_tensorEulerRodrigues(phi)

    # Evaluates the micropolar stretch and the jacobian
    
    V_bar = F*(R_bar.T)

    J = det(V_bar)

    # Evaluates the curvature tensor and its push-forward

    K_curvature = curvature_tensor(phi)

    k_curvatureSpatial = R_bar*K_curvature*R_bar.T

    V_barTransposed = variable(V_bar.T)

    V_barVariable = variable(V_bar)

    # Evaluates the total energy density

    #psi_total = (psi_NH(V_barTransposed.T)+psi_vol(V_barTransposed.T)+
    #psi_hat(V_barTransposed.T)+psi_k(k_curvatureSpatial))

    # Evaluates the micropolar Kirchhoff stress

    #tau = J*V_bar*diff(psi_total,V_barTransposed)

    psi_total = (psi_NH(V_barVariable)+psi_vol(V_barVariable)+
    psi_hat(V_barVariable)+psi_k(k_curvatureSpatial))

    # Evaluates the micropolar Kirchhoff stress

    derivative_psi = diff(psi_total,V_barVariable)

    tau = J*V_bar*(derivative_psi.T)

    #tau = ((lmbda/2)*((J*J)-1)*I) + (mu*((V_bar*V_bar.T)-I)) + ((kappa/
    #2)*((V_bar*V_bar.T)-(V_bar*V_bar)))

    # Pulls back to the reference configuration

    return tau

# Definition of the Kirchhoff couple stress 

def Couple_Kirchhoff_Stress(u, phi):

    # Sets the identity tensor and the deformation gradient

    I = Identity(3)

    F = grad(u) + I

    # Evaluates the rotation tensor given by the micropolar rotation

    R_bar = tensor_tools.rotation_tensorEulerRodrigues(phi)

    # Evaluates the micropolar stretch and the jacobian
    
    V_bar = F*(R_bar.T)

    J = det(V_bar)

    # Evaluates the curvature tensor and its push-forward

    K_curvature = curvature_tensor(phi)

    k_curvatureSpatial = R_bar*K_curvature*R_bar.T
    
    k_curvatureSpatialTransposed = variable(k_curvatureSpatial.T)

    k_curvatureSpatialVariable = variable(k_curvatureSpatial)

    # Evaluates the total energy density

    #psi_total = (psi_NH(V_bar)+psi_vol(V_bar)+psi_hat(V_bar)+psi_k(
    #k_curvatureSpatialTransposed.T))

    #tau = J*V_bar*diff(psi_total,k_curvatureSpatialTransposed)

    psi_total = (psi_NH(V_bar)+psi_vol(V_bar)+psi_hat(V_bar)+psi_k(
    k_curvatureSpatialVariable))

    # Evaluates the couple micropolar Kirchhof stress

    derivative_psi = diff(psi_total, k_curvatureSpatialVariable)

    tau = J*V_bar*(derivative_psi.T)
    #tau = V_bar*((alpha*tr(k_curvatureSpatial)*I)+(beta*
    #k_curvatureSpatial) + (gamma*k_curvatureSpatial.T))

    # Pulls back to the reference configuration

    return tau

# Definition of the deformation gradient

def def_grad(u):

    I = Identity(3)

    return I + grad(u)

########################################################################
#                           Variational forms                          #
########################################################################

# Defines the trial and test functions

dsol = TrialFunction(monolithic_functionSpace) 

v = TestFunction(monolithic_functionSpace)

# Creates the function for the updated solution, i.e. the vector of pa-
# rameters

sol_new = Function(monolithic_functionSpace)

# Splits the solution and the test function into their respective fields

u_new, phi_new = split(sol_new)

vu, vphi = split(v)

# Initializes objects for the stresses at the reference configuration

stress = Kirchhoff_Stress(u_new,phi_new)

couple_stress = Couple_Kirchhoff_Stress(u_new,phi_new)

# Constructs the variational forms for the inner work

piola_transformation = inv(def_grad(u_new)).T

Int_du = inner(stress*piola_transformation, grad(vu))*dx 

Int_dphi = (inner(couple_stress*piola_transformation, grad(vphi))*dx)-(inner(stress, 
tensor_tools.skew_2OrderTensor(vphi))*dx) 

# Constructs the variational forms for the traction work

#l_du = dot(traction_boundary,vu)*ds(9)
l_du = variational_tools.traction_work(traction_dictionary, vu, ds)

#l_dphi = dot(moment,vphi)*ds(10)

F_Int = Int_du + Int_dphi - l_du #- l_dphi

Gain = derivative(F_Int , sol_new, dsol)

Res = NonlinearVariationalProblem(F_Int, sol_new, bc, J=Gain)

########################################################################
#                      Solver parameters setting                       #
########################################################################

solver = NonlinearVariationalSolver(Res)

solver.parameters["nonlinear_solver"] = "newton"

solver.parameters["newton_solver"]["linear_solver"] = linear_solver

solver.parameters["newton_solver"]["relative_tolerance"] = (
relative_tolerance)

solver.parameters["newton_solver"]["absolute_tolerance"] = (
absolute_tolerance)

solver.parameters["newton_solver"]["maximum_iterations"] = (
maximum_iterations)

"""

solver.parameters["newton_solver"]["preconditioner"] = (
preconditioner)

solver.parameters['newton_solver']['krylov_solver']['absolute_tole'+
'rance'] = krylov_absoluteTolerance

solver.parameters['newton_solver']['krylov_solver']['relative_tole'+
'rance'] = krylov_relativeTolerance

solver.parameters['newton_solver']['krylov_solver']['maximum_itera'+
'tions'] = krylov_maximumIterations

solver.parameters['newton_solver']['krylov_solver']['monitor_conve'+
'rgence'] = krylov_monitorConvergence"""

########################################################################
#                         Files initialization                         #
########################################################################

conc_u = File("tests//micropolar//results//graphics//u_intermediate_code.pvd")

conc_phi = File("tests//micropolar//results//graphics//phi_intermediate_code.pvd")

conc_u_RVE = File("tests//micropolar//results//graphics//u_RVE_intermediate_code.pvd")

conc_phi_RVE = File("tests//micropolar//results//graphics//phi_RVE_intermediate_code.pvd")

########################################################################
#                   Solution and pseudotime stepping                   #
########################################################################

# Initializes the pseudotime counter

time_counter = 0

# Evaluates the pseudotime step

delta_t = (t_final-t)/(maximum_loadingSteps-1)

# Initializes the lists to save information for the microscale

u_RVE_homogenized = []

grad_u_RVE_homogenized = []

phi_RVE_homogenized = []

grad_phi_RVE_homogenized =[]

# Iterates through the pseudotime stepping

while t<(t_final*1.0001):

    print("###########################################################"+
    "#############\n#                 Incremental step: "+str(
    time_counter+1)+"; current time: "+str(t)+"               #\n#####"+
    "#################################################################"+
    "##\n")

    # Solves the nonlinear variational problem 

    solver.solve()

    # Splits the solution in their corresponding fields

    u, phi = sol_new.split(deepcopy=True)

    u.rename("DNS Displacement", "DNS")

    phi.rename("DNS Micro-rotation", "DNS")

    # Appends the quantities to be sent to the microscale

    u_RVE_homogenized.append([t, homogenization_tools.get_homogenized(u, 
    dx).tolist()])

    grad_u_RVE_homogenized.append([t,
    homogenization_tools.get_homogenized_gradient(grad(u), dx).tolist()])

    phi_RVE_homogenized.append([t,homogenization_tools.get_homogenized(
    phi, dx).tolist()])

    grad_phi_RVE_homogenized.append([t,
    homogenization_tools.get_homogenized_gradient(grad(phi), dx).tolist(
    )])

    """

    # Iterates through the elements of the submesh

    for element in cells(RVE_submesh):

        # Gets the index of the element in the RVE submesh

        submesh_index = element.index()

        # Gets the index of the element in the parent mesh

        parent_index = RVE_toParentCellMap[submesh_index]

        # Translates the values of the solution using the DOFs mapping

        sol_RVE.vector()[U_rveDofMap.cell_dofs(submesh_index)] = sol_new.vector()[U_parentDofMap.cell_dofs(parent_index)] 
        sol_RVE.vector()[V_rveDofMap.cell_dofs(submesh_index)] = sol_new.vector()[V_parentDofMap.cell_dofs(parent_index)] 
    """

    # Takes the values of the solution to the submesh of the RVE

    sol_RVE = mesh_tools.field_parentToSubmesh(RVE_submesh, sol_RVE, 
    sol_new, RVE_toParentCellMap, RVE_meshMapping, parent_meshMapping)

    # Splits the solution in the RVE submesh

    u_RVE, phi_RVE = sol_RVE.split(deepcopy=True)

    u_RVE.rename("DNS-Subdomain Displacement", "DNS-Subdomain ")

    phi_RVE.rename("DNS-Subdomain Micro-rotation", "DNS-Subdomain ")

    # Updates the files

    conc_u << u

    conc_phi << phi

    conc_u_RVE << u_RVE

    conc_phi_RVE << phi_RVE

    # Updates the pseudo time variables, the load, and the counter

    t += delta_t

    load.t = t
    
    time_counter += 1

    if time_counter>=maximum_loadingSteps:

        print("The maximum number of loading steps,",
        maximum_loadingSteps, "has just been reached. Stops the simula"+
        "tion immediatly\n")

        break

    # Saves the microscale quantities as they are to be able to be re-
    # trieved later if the simulation crashes before its completion

    file_tools.list_toTxt(u_RVE_homogenized, macro_displacementName)

    file_tools.list_toTxt(grad_u_RVE_homogenized, 
    macro_gradDisplacementName)

    file_tools.list_toTxt(phi_RVE_homogenized, macro_microrotationName)

    file_tools.list_toTxt(grad_phi_RVE_homogenized, 
    macro_gradMicrorotationName)

print ("\nSimulation completed")