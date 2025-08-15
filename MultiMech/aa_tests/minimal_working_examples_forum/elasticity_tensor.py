# Minimal working example for saving elasticity tensor at a point

from dolfin import *
import ufl_legacy as ufl
import numpy as np
import os

parameters["form_compiler"]["representation"] = "quadrature"

L, H, W = 1.0, 0.2, 0.3
mesh = BoxMesh(Point(0,0,0), Point(W,H,L), 5,5,5)

lower_facet = CompiledSubDomain("near(x[1], 0)")
left_facet = CompiledSubDomain("near(x[2], 0)")
right_facet = CompiledSubDomain("near(x[2], L)", L=L)
volume_1 = CompiledSubDomain("x[2]<=L*0.5", L=L)
volume_2 = CompiledSubDomain("x[2]>L*0.5", L=L)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_markers.set_all(0)
lower_facet.mark(boundary_markers, 2)
left_facet.mark(boundary_markers, 3)
right_facet.mark(boundary_markers, 6)

dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 2})
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# Constitutive model

class Neo_Hookean:

    def __init__(self, E, v):

        self.E = E
        self.v = v
        self.mu = Constant(self.E/(2*(1+self.v)))
        self.lmbda = Constant(self.v*self.E/((1+self.v)*(1-2*self.v)))

    def strain_energy(self, C):

        I1_C = ufl.tr(C)
        I2_C = ufl.det(C)
        J = ufl.sqrt(I2_C)
        
        psi_1 = (self.mu/2)*(I1_C - 3)

        ln_J = ufl.ln(J)
        psi_2 = -(self.mu*ln_J)+((self.lmbda*0.5)*((ln_J)**2))

        return psi_1+psi_2

    def first_piolaStress(self, u):

        I = Identity(3)
        F = grad(u)+I
        C = (F.T)*F
        C = variable(C)
        
        W = self.strain_energy(C)
        S = 2*diff(W,C)

        return F*S

    def second_piolaStress(self, u):

        I = Identity(3)
        F = grad(u)+I
        C = (F.T)*F
        C = variable(C)
        
        W = self.strain_energy(C)
        S = 2*diff(W,C)

        return S
    
    def second_elasticityTensor(self, u):

        I = Identity(3)
        F = grad(u)+I
        C = (F.T)*F
        C = variable(C)
        
        W = self.strain_energy(C)
        S = 2*diff(W,C)

        return diff(S, C)

constitutive_model = Neo_Hookean(1E6,0.3)
traction_vectors = {2: Constant([0.0, 0.0, 0.0]), 6: Constant([0.0, 0.0, 1E5])}

# Sets the forms

V = VectorFunctionSpace(mesh, "Lagrange", 2)
u_trial = TrialFunction(V)
v = TestFunction(V)
u_solution = Function(V)

bc = [DirichletBC(V, Constant((0.0, 0.0, 0.0)), boundary_markers, 3)]

external_work = 0.0

internal_work = inner(constitutive_model.first_piolaStress(u_solution),grad(v))*dx

for physical_group, traction in traction_vectors.items():
    external_work += dot(traction,v)*ds(physical_group)

# Solver
residual_form = internal_work-external_work
residual_derivative = derivative(residual_form , u_solution, u_trial)

Res = NonlinearVariationalProblem(residual_form, u_solution, bc, J=residual_derivative)
solver = NonlinearVariationalSolver(Res)
solver.solve()

file = XDMFFile(os.getcwd()+"//tests//minimal_working_examples_forum//displacement.xdmf")
file.write(u_solution)

# Gets the fourth order tensor
"""
W_tensor = TensorFunctionSpace(mesh, "CG", 1, shape=(3,3,3,3))
C_tensor = project(constitutive_model.second_elasticityTensor(u_solution), W_tensor)(Point(0.5*W,
0.5*H, 0.5*L))

txt_file = open(os.getcwd()+"//tests//minimal_working_examples_forum//dS_dC_tensor.txt", "w")
txt_file.write(str(C_tensor))
txt_file.close()"""

W_index = FunctionSpace(mesh, "DG", 1)
C_array = np.zeros((3,3,3,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                C_array[i,j,k,l] = project(constitutive_model.second_elasticityTensor(
                u_solution)[i,j,k,l], W_index, form_compiler_parameters={
                "quadrature_degree": 2})(Point(0.55*W, 0.55*H, 0.55*L))

txt_file = open(os.getcwd()+"//tests//minimal_working_examples_forum//dS_dC_array.txt", "w")
txt_file.write(str(C_array))
txt_file.close()

quadrature_element = VectorElement("Quadrature", mesh.ufl_cell(), degree=2,
quad_scheme="default")
C_array = np.zeros((3,3,3,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                C_array[i,j,k,l] = project(constitutive_model.second_elasticityTensor(
                u_solution)[i,j,k,l], quadrature_element)(Point(0.55*W, 0.55*H, 0.55*L))

txt_file = open(os.getcwd()+"//tests//minimal_working_examples_forum//dS_dC_array.txt", "w")
txt_file.write(str(C_array))
txt_file.close()

S_array = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        S_array[i,j] = project(constitutive_model.second_piolaStress(
        u_solution)[i,j], W_index, form_compiler_parameters={
        "quadrature_degree": 2})(Point(0.5*W, 0.5*H, 0.5*L))

txt_file = open(os.getcwd()+"//tests//minimal_working_examples_forum//S_stress.txt", "w")
txt_file.write(str(S_array))
txt_file.close()