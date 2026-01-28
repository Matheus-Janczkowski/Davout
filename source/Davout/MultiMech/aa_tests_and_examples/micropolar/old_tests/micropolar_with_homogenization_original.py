# Import all necessary packages and modules

from dolfin import *
from mpi4py import MPI
import ufl_legacy as ufl
import numpy as np
import matplotlib.pyplot as plt
from mshr import *
import tests.micropolar.tools_homogenization as ht
import tests.micropolar.tools_io as tools_io

iter = 0
t = 0.0
Tf = 1.0
deltaT = 0.1

# Define the indices for Einstein summation notation
i, j, k, l = ufl.indices(4)

parameters["form_compiler"]["representation"] = "uflacs"
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 2

# Defines the name of the file to save the mesh in. Do not write the fi-
# le termination, e.g. .msh or .xdmf; both options will be saved automa-
# tically

file_name = "tests//test_meshes//micropolar_prism"#intervertebral_disc"#"periodic_beam"

# Initializes the mesh object and reads the xdmf file

mesh = Mesh()

print("\n-------------------------------------------------------------------------")
print(" -                            Reads the mesh                            -")

# Initializes a mesh value collection to store mesh data

data_meshCollection = MeshValueCollection("size_t", mesh, mesh.topology().dim())

# Reads the mesh with domain physical groups

#with XDMFFile(file_name+"_domain.xdmf") as infile:
with XDMFFile(mesh.mpi_comm(), file_name+"_domain.xdmf") as infile:

    infile.read(mesh)

    infile.read(data_meshCollection, "domain")

ct = MeshFunction("size_t", mesh, data_meshCollection)

data_meshCollection = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)

# Reads the mesh with surface physical groups

#with XDMFFile(file_name+"_boundary.xdmf") as infile:
with XDMFFile(mesh.mpi_comm(), file_name+"_boundary.xdmf") as infile:
   
    infile.read(data_meshCollection, "boundary")

# Converts the mesh value collections to a mesh functions, for mesh va-
# lue collections are low level and cannot be used for FEM integration
# and other higher level operations inside FEniCs

ft = MeshFunction("size_t", mesh, data_meshCollection)

dx = Measure("dx", domain=mesh, subdomain_data=ct)

ds = Measure("ds", domain=mesh, subdomain_data=ft)

n  = FacetNormal(mesh)

# dx(1) -> outer_fiber
# dx(2) -> outer_matrix
# dx(3) -> RVE_fiber
# dx(4) -> RVE_matrix

# ds(5) -> bottom
# ds(6) -> front
# ds(7) -> left
# ds(8) -> back
# ds(9) -> right
# ds(10) -> top

# Define the finite element spaces for the displacement field "u" and for the microrotation field "phi"

U = VectorElement("CG",mesh.ufl_cell(), 2) # Displacement, Q2

V = VectorElement("CG",mesh.ufl_cell(), 1) # Microrotation, Q1

# Define the mixed element for the monolithic solution

MicroPolarMixedElement = MixedElement([U,V])
 
UV = FunctionSpace(mesh, MicroPolarMixedElement)

UV_subspaces = UV.split()

print(UV.sub(0).dim())

print(UV.sub(1).dim())

print(UV_subspaces[0].dim())

print(UV_subspaces[1].dim())

########################################################################
#                         RVE post-processing                          #
########################################################################

# Creates a submesh for the RVE

RVE_submesh = MeshView.create(ct,1)

# Creates a function space for the displacement inside the submesh of 
# the RVE
 
UV_submesh = FunctionSpace(RVE_submesh, MicroPolarMixedElement)

sol_RVE = Function(UV_submesh)

# Creates the solution of the displacement field inside the RVE submesh

# Creates the mappings of the degrees of freedom in both function spaces

U_rveDofMap = UV_submesh.sub(0).dofmap()
V_rveDofMap = UV_submesh.sub(1).dofmap()

U_parentDofMap = UV.sub(0).dofmap()
V_parentDofMap = UV.sub(1).dofmap()

# Creates the mapping of elements from the submesh to the parent mesh, 
# i.e. given the element index in the submesh, it throws the index in 
# the parent mesh

RVE_toParentCellMap = RVE_submesh.topology().mapping()[mesh.id()
].cell_map()

###########################################################################################
###########################################################################################

Vm = FunctionSpace(mesh, "DG", 0)
vm = Function(Vm)

mu_materials = [26.12,26.12,26.12]

# Assign material properties based on region
for cell in cells(mesh):
    region_id = ct[cell]  # Get the region id from cf
    if (region_id == 1):
        vm.vector()[cell.index()] = mu_materials[0]
    elif (region_id == 2):
        vm.vector()[cell.index()] = mu_materials[1]
    if (region_id == 3):
        vm.vector()[cell.index()] = mu_materials[2]

# Define material parameters for the micropolar material
mu = vm
kappa = 0#-0.15*mu
alpha = 0.0
beta = 0.0
gamma = 1.18E0#1e-12#5.22e-8 #5.22e-8 #((1e-5)**2)*2*mu # 43.265e-6
lmbda = 63.84-(2*mu/3)
I = Identity(3)

maximum_load = 4E-1
load = Expression("(t/t_final)*maximum_load", t=t, t_final=Tf,
maximum_load=maximum_load, degree=0)

bc1 = DirichletBC(UV.sub(0), Constant((0.0, 0.0, 0.0)), ft, 2)#4)
bc2 = DirichletBC(UV.sub(1), Constant((0.0, 0.0, 0.0)), ft, 2)#4)

bc = [bc1, bc2]

traction_boundary = as_vector([load,0.0,0.0])

def safe_sqrt(a):
    return sqrt(a + 1.0e-12)


def phi_angle(phi):
    return safe_sqrt(dot(phi, phi))

def W_matrix(phi):
   
   perm = ufl.PermutationSymbol(3)
   
   W = as_tensor(perm[j,i,k]*phi[k], (i,j))

   return W

def MicroRotation_Tensor(phi):

    rotation_angle = phi_angle(phi)

    W_phi = W_matrix(phi)

    R_bar = ufl.cos(rotation_angle)*I + (ufl.sin(rotation_angle)/rotation_angle)*W_phi \
                + ((1-ufl.cos(rotation_angle))/(rotation_angle**2))*ufl.outer(phi,phi)

    return R_bar

def curvature_tensor(phi):

    # Define the permutation tensor for 3D
    perm = ufl.PermutationSymbol(3)
    
    # Compute the micro-rotation tensor `Rbar` 
    Rbar = MicroRotation_Tensor(phi)
    
    # Compute the gradient of each component of Rbar to get a third-order tensor `gradRbar`
    # Here we assume that Rbar has dimensions (3, 3), and we calculate the gradient manually
    gradRbar = grad(Rbar)

    # Construct the third-order curvature tensor Kthird
    Kthird = as_tensor(Rbar[j, i] * gradRbar[j,k,l], (i, k, l))

    # Contract with the permutation tensor to form Ksecond
    Ksecond = (1/2)*as_tensor(perm[i, j, k] * Kthird[k, j, l], (i, l))

    return Ksecond

def psi_vol(Vbar):

    J = det(Vbar)

    energy = ((lmbda/4)*((J**2) - 1)) - ((lmbda/2)*(ln(J))) - (mu*ln(J))

    return energy

def psi_NH(Vbar):

    energy = 0.5*mu*(tr(Vbar*Vbar.T)-3)
    return energy

def psi_hat(Vbar):

    energy = 0.25*kappa*(tr(Vbar*Vbar.T)- tr(Vbar*Vbar)) 

    return energy

def psi_k(K_curvature):

    energy = 0.5*(alpha*(tr(K_curvature)**2) + beta*tr(K_curvature*K_curvature) + gamma*tr(K_curvature*K_curvature.T))

    return energy


def Kirchhoff_Stress(u, phi):

    I = Identity(3)
    F = grad(u) + I

    R_bar = MicroRotation_Tensor(phi)
    
    V_bar = F*(R_bar.T)
    J = det(V_bar)
    """

    K_curvature = curvature_tensor(phi)
    k_curvature_spatial = R_bar*K_curvature*R_bar.T

    V_bar_trans = variable(V_bar.T)
    k_curvature_spatial = variable(k_curvature_spatial.T)

    psi_total = psi_NH(V_bar_trans.T) + psi_vol(V_bar_trans.T) +  psi_hat(V_bar_trans.T) + psi_k(k_curvature_spatial)

    tau = J*V_bar*diff(psi_total, V_bar_trans)#"""

    # The Eq. 58 and 59 are wrong in Bauer et al: the right-hand side are
    # the Cauchy stress tensors, not the Kirchhoff stresses. For this reason,
    # the expression below was multiplied by J

    tau = ((lmbda/2)*J*((J*J)-1)*I) + (mu*J*((V_bar*V_bar.T)-I)) + ((kappa/2)*J*((V_bar*V_bar.T)-(V_bar*V_bar)))

    return tau

def Couple_Kirchhoff_Stress(u, phi):

    I = Identity(3)
    F = grad(u) + I
    
    R_bar = MicroRotation_Tensor(phi)
    
    V_bar = F*(R_bar.T)
    J = det(V_bar)

    K_curvature = curvature_tensor(phi)
    k_curvature_spatial = R_bar*K_curvature*R_bar.T
    """
    V_bar = variable(V_bar.T)
    k_curvature_spatial_trans = variable(k_curvature_spatial.T)

    psi_total = psi_NH(V_bar) + psi_vol(V_bar) +  psi_hat(V_bar) + psi_k(k_curvature_spatial_trans.T)

    tau = J*V_bar*diff(psi_total, k_curvature_spatial_trans)#"""

    # The Eq. 58 and 59 are wrong in Bauer et al: the right-hand side are
    # the Cauchy stress tensors, not the Kirchhoff stresses. For this reason,
    # the expression below was multiplied by J

    tau = J*V_bar*((alpha*tr(k_curvature_spatial)*I)+(beta*k_curvature_spatial) + (gamma*k_curvature_spatial.T))

    return tau

def def_grad(u):
    I = Identity(3)
    return I + grad(u)

# The function that multiplied the PDE/residual is called test function. The unknown function u to be approximated is referred to as trial function.

dsol, v = TrialFunction(UV), TestFunction(UV)
vu, vphi = split(v)

sol_new = Function(UV)

print(len(sol_new.sub(0).vector().get_local()))

print(len(sol_new.sub(1).vector().get_local()))

u_new, phi_new = split(sol_new)

print(len(u_new.vector()))

print(len(phi_new.vector()))

invgrad = inv(def_grad(u_new)).T

stress = Kirchhoff_Stress(u_new,phi_new)
couple_stress = Couple_Kirchhoff_Stress(u_new,phi_new)

Int_du = inner(stress*invgrad, grad(vu))*dx 

Int_dphi = inner(couple_stress*invgrad, grad(vphi))*dx - inner(stress, W_matrix(vphi))*dx 

l_du = dot(traction_boundary,vu)*ds(4)#5)

#l_dphi = dot(moment,vphi)*ds(10)

F_Int = Int_du + Int_dphi - l_du #- l_dphi

Gain = derivative(F_Int , sol_new, dsol)
Res = NonlinearVariationalProblem(F_Int, sol_new, bc, J=Gain)
solver = NonlinearVariationalSolver(Res)
solver.parameters['nonlinear_solver'] = 'newton'
solver.parameters['newton_solver']['relative_tolerance'] = 1e-10
solver.parameters['newton_solver']['maximum_iterations'] = 10
solver.parameters['newton_solver']['linear_solver'] = 'mumps'
solver.parameters['newton_solver']['absolute_tolerance'] = 1e-10

conc_u = File("tests//micropolar//results//graphics//u_original.pvd")
conc_phi = File("tests//micropolar//results//graphics//phi_original.pvd")

conc_u_RVE = File("tests//micropolar//results//graphics//u_RVE_original.pvd")
conc_phi_RVE = File("tests//micropolar//results//graphics//phi_RVE_original.pvd")

u_RVE_homogenized = []
grad_u_RVE_homogenized = []

phi_RVE_homogenized = []
grad_phi_RVE_homogenized =[]

time_counter = 0

while t < (Tf*1.0001):

    print("\n-------------------------------------------------------------------------")
    print("-              Incremental step:", time_counter+1, "; current time:", t, "                 -")
    print("-------------------------------------------------------------------------\n")

    solver.solve()

    u,phi = sol_new.split(deepcopy=True)

    u.rename("DNS Displacement", "DNS")
    phi.rename("DNS Micro-rotation", "DNS")

    """
    u_RVE_homogenized.append([t,ht.get_homogenized(u, dx).tolist()])
    grad_u_RVE_homogenized.append([t,ht.get_homogenized_gradient(grad(u), dx).tolist()])
    phi_RVE_homogenized.append([t,ht.get_homogenized(phi, dx).tolist()])
    grad_phi_RVE_homogenized.append([t,ht.get_homogenized_gradient(grad(phi), dx).tolist()])

    # Iterates through the elements of the submesh

    for element in cells(RVE_submesh):

        # Gets the index of the element in the RVE submesh

        submesh_index = element.index()

        # Gets the index of the element in the parent mesh

        parent_index = RVE_toParentCellMap[submesh_index]

        # Translates the values of the solution using the DOFs mapping

        sol_RVE.vector()[U_rveDofMap.cell_dofs(submesh_index)] = sol_new.vector()[U_parentDofMap.cell_dofs(parent_index)] 
        sol_RVE.vector()[V_rveDofMap.cell_dofs(submesh_index)] = sol_new.vector()[V_parentDofMap.cell_dofs(parent_index)]

    u_RVE, phi_RVE = sol_RVE.split(deepcopy=True)

    u_RVE.rename("DNS-Subdomain Displacement", "DNS-Subdomain ")
    phi_RVE.rename("DNS-Subdomain Micro-rotation", "DNS-Subdomain ")
    conc_u_RVE << u_RVE
    conc_phi_RVE << phi_RVE"""

    conc_u << u
    conc_phi << phi

    t += deltaT
    load.t = t
    time_counter += 1

    tools_io.list_toTxt(u_RVE_homogenized, "tests//micropolar//results//text//u_RVE_homogenized_original")
    tools_io.list_toTxt(grad_u_RVE_homogenized, "tests//micropolar//results//text//grad_u_RVE_homogenized_original")
    tools_io.list_toTxt(phi_RVE_homogenized, "tests//micropolar//results//text//phi_RVE_homogenized_original")
    tools_io.list_toTxt(grad_phi_RVE_homogenized, "tests//micropolar//results//text//grad_phi_RVE_homogenized_original")

print ('Simulation Completed')

