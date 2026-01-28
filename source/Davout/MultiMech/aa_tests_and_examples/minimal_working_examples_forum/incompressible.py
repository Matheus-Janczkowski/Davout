# Routine to experiment with incompressible hyperelasticity

from dolfin import *

import ufl_legacy as ufl

from ....PythonicUtilities.path_tools import get_parent_path_of_file

from ....MultiMech.tool_box.mesh_handling_tools import read_mshMesh

# Geometry information

L, H, W = 1.0, 0.2, 0.3

mesh = BoxMesh(Point(0,0,0), Point(W,H,L), 5,5,25)

# Sets the surface domains

lower_facet = CompiledSubDomain("near(x[1], 0)")

left_facet = CompiledSubDomain("near(x[2], 0)")

right_facet = CompiledSubDomain("near(x[2], L)", L=L)

# Sets the volumetric domains

volume_1 = CompiledSubDomain("x[2]<=L*0.5", L=L)

volume_2 = CompiledSubDomain("x[2]>L*0.5", L=L)

# Sets the boundaries

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

boundary_markers.set_all(0)

lower_facet.mark(boundary_markers, 2)

left_facet.mark(boundary_markers, 3)

right_facet.mark(boundary_markers, 6)

# Sets the integration measures

dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 2})

ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# Neumann boundary conditions

traction_vectors = {2: Constant([0.0, 0.0, 0.0]), 6: Constant([0.0, 0.0, 
5E5])}

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

# Evaluates the geometry volume

inv_V0 = 1/assemble(1.0*dx)

# Sets the constitutive model

constitutive_model = Neo_Hookean(1E6,0.3)

# Sets the function space

mixed_element = MixedElement([VectorElement("Lagrange", mesh.ufl_cell(),
2), FiniteElement("Lagrange", mesh.ufl_cell(), 1)], )

monolithic_functionSpace = FunctionSpace(mesh, mixed_element)

trial_functions = TrialFunction(monolithic_functionSpace)

monolithic_solution = Function(monolithic_functionSpace)

u, lmbda = split(monolithic_solution)

delta_u, delta_lambda = split(TestFunction(monolithic_functionSpace))

# Dirichlet boundary conditions

bc = [DirichletBC(monolithic_functionSpace.sub(0), Constant((0.0, 0.0, 
0.0)), boundary_markers, 3)]

# Variational form

external_work = 0.0

I = Identity(3)

F = grad(u)+I

F_invT = inv(F).T

J = det(F)

internal_work = ((inner(constitutive_model.first_piolaStress(u)+(lmbda*
inv_V0*J*F_invT), grad(delta_u))*dx)+(inv_V0*(((J-1)*delta_lambda)*dx)))

for physical_group, traction in traction_vectors.items():

    external_work += dot(traction, delta_u)*ds(physical_group)

# Solver

residual_form = internal_work-external_work

residual_derivative = derivative(residual_form , monolithic_solution, 
trial_functions)

Res = NonlinearVariationalProblem(residual_form, monolithic_solution, 
bc, J=residual_derivative)

solver = NonlinearVariationalSolver(Res)

solver.solve()

# Solution saving

u_solution, lambda_solution = monolithic_solution.split()

J = det(grad(u_solution)+I)

V_new = assemble(J*dx)

print("The ratio of the new volume by the volume of the reference conf"+
"iguration is "+str(inv_V0*V_new)+"\n")

u_solution.rename("Displacement", "DNS")

lambda_solution.rename("Pressure", "DNS")

file = XDMFFile(get_parent_path_of_file()+"//tests//displacement.xdmf")

file.write(u_solution)

file = XDMFFile(get_parent_path_of_file()+"//tests//pressure.xdmf")

file.write(lambda_solution)