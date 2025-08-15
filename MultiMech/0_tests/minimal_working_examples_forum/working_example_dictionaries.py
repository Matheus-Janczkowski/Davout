# Minimal working example with dictionaries and classes

from dolfin import *
import ufl_legacy as ufl

L, H, W = 1.0, 0.2, 0.3
mesh = BoxMesh(Point(0,0,0), Point(W,H,L), 10,10,10)

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

volume_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
volume_markers.set_all(0)
volume_1.mark(volume_markers, 7)
volume_2.mark(volume_markers, 8)

dx = Measure("dx", domain=mesh, subdomain_data=volume_markers, metadata={
"quadrature_degree": 2})
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

constitutive_models = {7: Neo_Hookean(1E6,0.3), 8: Neo_Hookean(0.5E6, 0.4)}
traction_vectors = {2: Constant([0.0, 0.0, 0.0]), 6: Constant([0.0, 0.0, 0.0])}

# Sets the forms

V = VectorFunctionSpace(mesh, "Lagrange", 2)
u_trial = TrialFunction(V)
v = TestFunction(V)
u_solution = Function(V)

bc = [DirichletBC(V, Constant((0.0, 0.0, 0.0)), boundary_markers, 3)]

internal_work = 0.0
external_work = 0.0

for physical_group, model in constitutive_models.items():
    internal_work += inner(model.first_piolaStress(u_solution),grad(v))*dx(physical_group)

for physical_group, traction in traction_vectors.items():
    external_work += dot(traction,v)*ds(physical_group)

# Solver
residual_form = internal_work-external_work
residual_derivative = derivative(residual_form , u_solution, u_trial)

Res = NonlinearVariationalProblem(residual_form, u_solution, bc, J=residual_derivative)

solver = NonlinearVariationalSolver(Res)