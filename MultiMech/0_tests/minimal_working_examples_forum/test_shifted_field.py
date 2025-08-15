from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Define the mesh (rectangular domain)

d = 2
L, H = 1.0, 0.2  # Length and height
mesh = RectangleMesh(Point(0, 0), Point(L, H), 40, 10)

# Define function space for vector field (displacement)
u_element = VectorElement("CG", mesh.ufl_cell(), 2)

lambda_element = VectorElement("Real", mesh.ufl_cell(), 0)

mixed_element = MixedElement([u_element, lambda_element])

V = FunctionSpace(mesh, mixed_element)

# Define material properties
E = 210e9       # Young's modulus in Pascals
nu = 0.3        # Poisson's ratio

# Define plane strain elasticity parameters
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Defines the edges

right_edge = CompiledSubDomain("near(x[0], L)", L=L)
left_edge = CompiledSubDomain("near(x[0], 0)")
upper_edge = CompiledSubDomain("near(x[1], H)", H=H)
lower_edge = CompiledSubDomain("near(x[1], 0)")

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
right_edge.mark(boundary_markers, 1)
left_edge.mark(boundary_markers, 2)
upper_edge.mark(boundary_markers, 3)
lower_edge.mark(boundary_markers, 4)

ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

bc = DirichletBC(V.sub(0), Constant((0, 0)), boundary_markers, 2)

# Define the variational problem
u = TrialFunction(V)
v = TestFunction(V)

u_function = Function(V)

u_disp, lmbda_lagrange = split(u_function)

v_disp, v_lmbda = split(v)

I = Identity(2)               # Identity tensor

# Strain and stress definitions (plane strain)
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda * tr(epsilon(u)) * I + 2 * mu * epsilon(u)

# Define a custom expression for u(x + d)
class ShiftedFieldExpression(UserExpression):
    def __init__(self, field, shift, **kwargs):
        super().__init__(**kwargs)
        self.field = field
        self.shift = np.array(shift)

    def eval(self, value, x):
        # Calculate shifted coordinates
        shifted_x = x + self.shift
        value[:] = self.field(shifted_x)  # Evaluate field at shifted coordinates

    def value_shape(self):
        return (2,)  # Since we are working with a vector field in 2D

# Create an instance of the shifted field
u_shifted_expr = ShiftedFieldExpression(field=u_disp, shift=[0.0, H], degree=1)

# Interpolate the shifted field into the same function space
u_shifted = project(u_shifted_expr, V.sub(0))

# Weak form of the elasticity problem
f = Constant((0, 0.0*-9.81e3))  # Body force (gravity)
t = Constant((1e7, 0))      # Traction on right edge

# Variational form
a = (inner(sigma(u_disp), epsilon(v_disp)) * dx)-(dot(u_disp,v_lmbda)*dx)
L = (dot(f, v_disp) * dx + dot(t, v_disp) * ds(1))+(dot(lmbda_zlagrange,v_disp)*dx)

residual_form = a-L

residual_derivative = derivative(residual_form, u_function, u)

Res = NonlinearVariationalProblem(residual_form, u_function, bc, 
J=residual_derivative)

# Solve the problem
solver = NonlinearVariationalSolver(Res)

solver.solve()

# Extract displacement function from solution
u_disp_func, _ = u_function.split()

# Scale factor for visualization
scale = 1000

# Get mesh coordinates and displacements
V_coords = mesh.coordinates()
deformation = u_disp_func.compute_vertex_values(mesh).reshape((2, -1)).T
deformed_coords = V_coords + scale * deformation

# Plot original and deformed mesh
plt.figure(figsize=(8, 4))
plt.triplot(V_coords[:, 0], V_coords[:, 1], mesh.cells(), color='gray', label='Original mesh')
plt.triplot(deformed_coords[:, 0], deformed_coords[:, 1], mesh.cells(), color='red', label='Deformed mesh')
plt.legend()
plt.title(f"Deformed mesh (scaled deformation factor = {scale})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()