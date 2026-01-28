from dolfin import *
import numpy as np

import source.constitutive_models.hyperelasticity.isotropic_hyperelasticity as constitutive_models

import tests.test_meshes.beam_gmsh as beam_gmsh

import source.tool_box.mesh_handling_tools as mesh_tools

# Beam geometry

ratio_Lb = 1.5E-1

gamma = 1.18E0

beta = 0.0

mu = 26.12

K_constitutive = 63.84

L = 1.0       # Length of beam
b = 0.1       # Side of square cross section

L = np.sqrt((beta+gamma)/(2*mu))

b = L/ratio_Lb

print("b=", b)

# Mesh resolution
"""nx, ny, nz = 6, 6, 20
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(b, b, 10*b), nx, ny, nz)

ds = Measure("ds", domain=mesh, subdomain_data=side_marker)

dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 3})

class SideYMinus(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

side_marker = MeshFunction("size_t", mesh, 2, 0)
SideYMinus().mark(side_marker, 1)"""

n_volumes = 1

#mesh_fileName = "tests//test_meshes//micropolar_beam_with_fibers"

mesh_fileName = "tests//test_meshes//micropolar_beam"

beam_gmsh.generate_micropolarBeam(mu, ratio_Lb, beta, gamma, 
mesh_fileName, n_volumes, transfinite=True)

(mesh, dx, ds, n, domain_meshCollection, domain_meshFunction, 
boundary_meshCollection, boundary_meshFunction,
domain_physGroupsNamesToTags, boundary_physGroupsNamesToTags
) = mesh_tools.read_mshMesh(mesh_fileName, quadrature_degree=2)#"""

# Function space for displacement
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)

# Boundary conditions
def clamp_boundary(x, on_boundary):
    return on_boundary and near(x[2], 0.0)

u_max = 1E-1

t_max = -2E6

t_final = 1.0

t = 0.0

displacement = Expression("u_max*(t/t_final)", u_max=u_max, t=t, t_final=t_final, degree=1)  # Adjust as needed

bc = [DirichletBC(V, Constant((0.0, 0.0, 0.0)), boundary_meshFunction, 5)]#,
    #DirichletBC(V.sub(2), displacement, boundary_meshFunction, 6)]

#bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), boundary_meshFunction, 5) clamp_boundary)

# Define traction (on the side y = -b/2)

max_steps = 11

delta_t = (t_final-t)/(max_steps-1)

traction_magnitude = Expression("t_max*(t/t_final)", t_max=t_max, t=t, t_final=t_final, degree=0)  # Adjust as needed

# Kinematics
du = TrialFunction(V)
v  = TestFunction(V)
u  = Function(V)

d = len(u)
I = Identity(d)
F = I + grad(u)
C = F.T * F
J = det(F)

# Material parameters (Neo-Hookean)

lmbda = K_constitutive-(2*mu/3)

dF = grad(v)

# Stored strain energy density (Neo-Hookean)

"""
C_variable = variable(C)

J_variable = sqrt(det(C_variable))

psi_variable = (mu/2)*(tr(C_variable) - 3) - mu*ln(J_variable) + (lmbda/2)*(ln(J_variable))**2

S = 2*diff(psi_variable, C_variable)

P = F*S"""

#"""
material_properties = dict()

E = ((mu/(lmbda+mu))*((2*mu)+(3*lmbda)))

nu = lmbda/(2*(lmbda+mu))

E = 100E6

nu = 0.4

material_properties["E"] = E 

material_properties["v"] = nu

constitutive_model = constitutive_models.Neo_Hookean(material_properties)

P = constitutive_model.first_piolaStress(u)#"""

traction_work = (dot(as_vector([0.0, traction_magnitude, 0.0]),v)*ds(6))

print(traction_work)

variational_form = (inner(P, dF)*dx)-traction_work

# Variational problem
#F_res = variational_form#derivative(Pi, u, v)
#J_form = derivative(F_res, u, du)

residual_derivative = derivative(variational_form , u, du)

Res = NonlinearVariationalProblem(variational_form, u, bc, J=
residual_derivative)

solver = NonlinearVariationalSolver(Res)

solver.parameters["nonlinear_solver"] = "newton"

solver.parameters["newton_solver"]["linear_solver"] = "cg"

solver.parameters["newton_solver"]["relative_tolerance"] = 1e-4#1e-3

solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-4#1e-3

solver.parameters["newton_solver"]["maximum_iterations"] = 10

file = File("tests//micropolar//Bauer_et_al//results//neo_hookean_beam.pvd")

for i in range(max_steps):

    print("\n#########################################################"+
    "###############\n#                                Step: "+str(i+1)+
    "                              #\n################################"+
    "########################################\n")

    traction_magnitude.t = t

    displacement.t = t

    solver.solve()

    # Solve nonlinear problem
    #solve(F_res == 0, u, bc, J=J_form,
    #    solver_parameters={"newton_solver": {
    #        "relative_tolerance": 1e-6,
    #        "maximum_iterations": 25
    #    }})
    
    t += delta_t

    # Save results
    file << u